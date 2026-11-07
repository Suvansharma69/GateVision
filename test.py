# Enhanced Car Detection System with Fixed Duplicate Deletion and Optimized Streaming
from ultralytics import YOLO
import cv2
import pytesseract
import os
import numpy as np
import re
from collections import Counter, defaultdict, deque
import threading
import queue
import time
from datetime import datetime, timedelta
import json
from flask import Flask, render_template, Response, jsonify, send_from_directory, request, send_file
import math
import base64
import logging
from threading import Lock, RLock
import hashlib
import concurrent.futures
from functools import lru_cache
import gc

# Advanced AI/ML imports
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
import imutils

# MongoDB imports
from pymongo import MongoClient
from bson.objectid import ObjectId
import pymongo.errors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load all configuration from config.py (reads .env automatically)
from config import (
    RTSP_STREAM, DASHBOARD_DIR, SCREENSHOTS_DIR, CAR_IMAGES_DIR, PLATE_IMAGES_DIR,
    MODEL_PATH, YOLO_CAR_MODEL, TESSERACT_CMD,
    MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME,
    FLASK_HOST, FLASK_PORT
)

# Set Tesseract executable path
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    logger.warning("Could not set Tesseract path. Ensure it's in your system's PATH.")

# Optimized Performance Configuration - NO FRAME SKIPPING
MAX_QUEUE_SIZE = 5
PROCESSING_INTERVAL = 1  # Process every frame - NO SKIPPING
OCR_PROCESSING_INTERVAL = 1  # OCR on every frame for quality - NO SKIPPING
MEMORY_CLEANUP_INTERVAL = 200  # Increased for better performance
DUPLICATE_CHECK_INTERVAL = 10  # Check for duplicates every 10 frames

# Stream optimization settings - Enhanced Quality
RTSP_RECONNECT_TIMEOUT = 5
RTSP_READ_TIMEOUT = 1.0
MAX_CONSECUTIVE_FAILURES = 10
STREAM_BUFFER_SIZE = 2
TARGET_FPS = 30

# Enhanced Car detection parameters
CAR_DETECTION_CONFIDENCE = 0.35
MIN_CAR_WIDTH = 120
MIN_CAR_HEIGHT = 80
CAR_EXPANSION_RATIO = 0.15
FULL_CAR_CAPTURE_RATIO = 1.3

# Car classes from COCO dataset
CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Improved Entry/Exit Detection Configuration
VEHICLE_TRACKING_HISTORY = 25
MOVEMENT_THRESHOLD = 50
ORIENTATION_CONFIDENCE_THRESHOLD = 0.7
TEMPORAL_CONSISTENCY_FRAMES = 7
DIRECTION_ANALYSIS_FRAMES = 10

# Enhanced Duplicate cleanup settings
DUPLICATE_CLEANUP_INTERVAL = 180  # 3 minutes
AUTO_CLEANUP_ENABLED = True
DUPLICATE_THRESHOLD_SECONDS = 30  # Consider duplicates within 30 seconds

# Ensure directories exist
for directory in [SCREENSHOTS_DIR, CAR_IMAGES_DIR, PLATE_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask app
app = Flask(__name__)

class EnhancedDuplicateCleanupManager:
    """Enhanced manager for cleaning up duplicate vehicle records with improved logic"""
    def __init__(self, mongo_manager):
        self.mongo_manager = mongo_manager
        self.cleanup_lock = RLock()
        self.last_cleanup = time.time()
        self.cleanup_stats = {
            'total_cleanups': 0,
            'duplicates_removed': 0,
            'images_deleted': 0,
            'last_cleanup_time': None,
            'storage_freed_mb': 0.0,
            'plates_processed': 0,
            'errors': 0
        }

    def cleanup_duplicates(self, plate_number=None):
        """Enhanced duplicate cleanup - keeps only ONE record per plate number with proper image handling"""
        try:
            with self.cleanup_lock:
                if not self.mongo_manager.is_connected:
                    logger.warning("MongoDB not connected, skipping duplicate cleanup")
                    return {'status': 'error', 'message': 'Database not connected'}

                logger.info(f"🧹 Starting enhanced duplicate cleanup for: {'ALL PLATES' if not plate_number else plate_number}")
                
                # Build query
                query = {}
                if plate_number:
                    query = {"plate_number": plate_number.upper().strip()}

                # Enhanced aggregation pipeline to find duplicates with full record details
                pipeline = [
                    {"$match": query},
                    {"$group": {
                        "_id": "$plate_number",
                        "count": {"$sum": 1},
                        "records": {
                            "$push": {
                                "id": "$_id",
                                "timestamp": "$timestamp",
                                "car_image_path": "$car_image_path",
                                "plate_image_path": "$plate_image_path",
                                "detection_id": "$detection_id",
                                "confidence": "$confidence",
                                "entry_exit_status": "$entry_exit_status",
                                "vehicle_orientation": "$vehicle_orientation",
                                "orientation_confidence": "$orientation_confidence"
                            }
                        }
                    }},
                    {"$match": {"count": {"$gt": 1}}},  # Only plates with duplicates
                    {"$sort": {"count": -1}}  # Process plates with most duplicates first
                ]

                duplicate_groups = list(self.mongo_manager.collection.aggregate(pipeline))
                
                if not duplicate_groups:
                    logger.info("✅ No duplicates found")
                    return {
                        'status': 'success',
                        'message': 'No duplicates found',
                        'duplicates_processed': 0,
                        'records_removed': 0,
                        'images_deleted': 0,
                        'storage_freed_mb': 0.0
                    }

                total_removed = 0
                images_deleted = 0
                storage_freed = 0
                plates_processed = len(duplicate_groups)
                errors = 0

                logger.info(f"🔍 Found {plates_processed} plates with duplicates")

                for duplicate_group in duplicate_groups:
                    try:
                        plate_num = duplicate_group['_id']
                        records = duplicate_group['records']
                        duplicate_count = len(records)
                        
                        logger.info(f"🔄 Processing plate {plate_num}: {duplicate_count} total records")

                        # Sort records by multiple criteria to keep the BEST one
                        def record_score(record):
                            score = 0
                            # Prefer records with higher confidence
                            score += record.get('confidence', 0) * 10
                            # Prefer records with orientation confidence
                            score += record.get('orientation_confidence', 0) * 5
                            # Prefer records with both car and plate images
                            if record.get('car_image_path') and os.path.exists(record.get('car_image_path', '')):
                                score += 20
                            if record.get('plate_image_path') and os.path.exists(record.get('plate_image_path', '')):
                                score += 10
                            # Prefer more recent records (timestamp as tiebreaker)
                            if record.get('timestamp'):
                                score += record['timestamp'].timestamp() * 0.001
                            return score

                        # Sort records by score (best first) and keep the BEST one
                        records.sort(key=record_score, reverse=True)
                        best_record = records[0]
                        records_to_delete = records[1:]  # Delete all others

                        logger.info(f"📌 Keeping BEST record for {plate_num} (confidence: {best_record.get('confidence', 0):.1f})")
                        logger.info(f"🗑️ Deleting {len(records_to_delete)} duplicate records")

                        for record in records_to_delete:
                            try:
                                # Delete associated images first
                                car_image_path = record.get('car_image_path', '')
                                plate_image_path = record.get('plate_image_path', '')
                                
                                deleted_files = 0
                                file_sizes_freed = 0

                                # Delete car image
                                if car_image_path and os.path.exists(car_image_path):
                                    try:
                                        file_size = os.path.getsize(car_image_path)
                                        os.remove(car_image_path)
                                        deleted_files += 1
                                        file_sizes_freed += file_size
                                        logger.debug(f"   🗑️ Deleted car image: {os.path.basename(car_image_path)}")
                                    except Exception as e:
                                        logger.error(f"   ❌ Error deleting car image {car_image_path}: {e}")
                                        errors += 1

                                # Delete plate image
                                if plate_image_path and os.path.exists(plate_image_path):
                                    try:
                                        file_size = os.path.getsize(plate_image_path)
                                        os.remove(plate_image_path)
                                        deleted_files += 1
                                        file_sizes_freed += file_size
                                        logger.debug(f"   🗑️ Deleted plate image: {os.path.basename(plate_image_path)}")
                                    except Exception as e:
                                        logger.error(f"   ❌ Error deleting plate image {plate_image_path}: {e}")
                                        errors += 1

                                # Delete database record
                                delete_result = self.mongo_manager.collection.delete_one({"_id": record['id']})
                                
                                if delete_result.deleted_count > 0:
                                    total_removed += 1
                                    images_deleted += deleted_files
                                    storage_freed += file_sizes_freed
                                    logger.debug(f"   ✅ Deleted record: {record.get('detection_id', 'unknown')}")
                                else:
                                    logger.warning(f"   ⚠️ Record not found in database: {record.get('detection_id', 'unknown')}")
                                    errors += 1

                            except Exception as e:
                                logger.error(f"   ❌ Error deleting duplicate record for {plate_num}: {e}")
                                errors += 1

                        logger.info(f"✅ Completed cleanup for plate {plate_num}: removed {len(records_to_delete)} duplicates")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing duplicate group for plate {duplicate_group.get('_id', 'unknown')}: {e}")
                        errors += 1

                # Update cleanup statistics
                self.cleanup_stats['total_cleanups'] += 1
                self.cleanup_stats['duplicates_removed'] += total_removed
                self.cleanup_stats['images_deleted'] += images_deleted
                self.cleanup_stats['storage_freed_mb'] += storage_freed / (1024 * 1024)
                self.cleanup_stats['plates_processed'] += plates_processed
                self.cleanup_stats['errors'] += errors
                self.cleanup_stats['last_cleanup_time'] = datetime.now()
                self.last_cleanup = time.time()

                result = {
                    'status': 'success',
                    'duplicates_processed': plates_processed,
                    'records_removed': total_removed,
                    'images_deleted': images_deleted,
                    'storage_freed_mb': round(storage_freed / (1024 * 1024), 2),
                    'plate_number': plate_number or 'all plates',
                    'errors': errors,
                    'message': f'Cleanup completed: {total_removed} duplicate records removed, {images_deleted} images deleted'
                }

                logger.info(f"🧹✅ Enhanced duplicate cleanup completed:")
                logger.info(f"   📊 Plates processed: {plates_processed}")
                logger.info(f"   🗑️ Records removed: {total_removed}")
                logger.info(f"   🖼️ Images deleted: {images_deleted}")
                logger.info(f"   💾 Storage freed: {result['storage_freed_mb']:.2f} MB")
                logger.info(f"   ❌ Errors: {errors}")
                
                return result

        except Exception as e:
            logger.error(f"❌ Enhanced duplicate cleanup error: {e}")
            self.cleanup_stats['errors'] += 1
            return {
                'status': 'error', 
                'message': f'Cleanup failed: {str(e)}',
                'duplicates_processed': 0,
                'records_removed': 0,
                'images_deleted': 0,
                'storage_freed_mb': 0.0
            }

    def cleanup_orphaned_images(self):
        """Enhanced cleanup of orphaned image files with better safety checks"""
        try:
            with self.cleanup_lock:
                if not self.mongo_manager.is_connected:
                    logger.warning("MongoDB not connected, skipping orphan image cleanup")
                    return {'status': 'error', 'message': 'Database not connected'}

                logger.info("🧹 Starting enhanced orphaned image cleanup...")
                
                # Get all image paths from the database with better error handling
                db_images = set()
                try:
                    records_cursor = self.mongo_manager.collection.find(
                        {}, 
                        {"car_image_path": 1, "plate_image_path": 1},
                        no_cursor_timeout=True
                    ).batch_size(1000)
                    
                    record_count = 0
                    for record in records_cursor:
                        record_count += 1
                        if record.get("car_image_path"):
                            car_path = os.path.abspath(record["car_image_path"])
                            db_images.add(car_path)
                        if record.get("plate_image_path"):
                            plate_path = os.path.abspath(record["plate_image_path"])
                            db_images.add(plate_path)
                    
                    records_cursor.close()
                    logger.info(f"🔍 Scanned {record_count} database records, found {len(db_images)} image references")
                    
                except Exception as e:
                    logger.error(f"Error reading database image references: {e}")
                    return {'status': 'error', 'message': f'Database scan failed: {str(e)}'}

                # Get all image files from the filesystem with safety checks
                disk_images = set()
                total_disk_size = 0
                
                image_dirs = [CAR_IMAGES_DIR, PLATE_IMAGES_DIR]
                for directory in image_dirs:
                    if not os.path.exists(directory):
                        logger.warning(f"Directory does not exist: {directory}")
                        continue
                        
                    try:
                        for filename in os.listdir(directory):
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                full_path = os.path.abspath(os.path.join(directory, filename))
                                if os.path.isfile(full_path):
                                    disk_images.add(full_path)
                                    try:
                                        total_disk_size += os.path.getsize(full_path)
                                    except OSError:
                                        pass
                    except Exception as e:
                        logger.error(f"Error scanning directory {directory}: {e}")
                        continue

                logger.info(f"📁 Found {len(disk_images)} image files on disk ({total_disk_size / (1024*1024):.2f} MB total)")

                # Find orphaned images (exist on disk but not in database)
                orphaned_images = disk_images - db_images
                logger.info(f"🗑️ Found {len(orphaned_images)} orphaned images to delete")

                if not orphaned_images:
                    return {
                        'status': 'success',
                        'orphaned_images_found': 0,
                        'images_deleted': 0,
                        'storage_freed_mb': 0.0,
                        'message': 'No orphaned images found'
                    }

                # Delete orphaned images with error handling
                deleted_count = 0
                total_size_freed = 0
                errors = 0

                for image_path in orphaned_images:
                    try:
                        if os.path.exists(image_path):
                            file_size = os.path.getsize(image_path)
                            os.remove(image_path)
                            deleted_count += 1
                            total_size_freed += file_size
                            logger.debug(f"   🗑️ Deleted orphaned image: {os.path.basename(image_path)}")
                    except OSError as e:
                        logger.error(f"   ❌ Error deleting orphaned file {image_path}: {e}")
                        errors += 1
                
                storage_freed_mb = total_size_freed / (1024 * 1024)
                self.cleanup_stats['images_deleted'] += deleted_count
                self.cleanup_stats['storage_freed_mb'] += storage_freed_mb
                self.cleanup_stats['errors'] += errors
                
                result = {
                    'status': 'success',
                    'orphaned_images_found': len(orphaned_images),
                    'images_deleted': deleted_count,
                    'storage_freed_mb': round(storage_freed_mb, 2),
                    'errors': errors,
                    'message': f'Deleted {deleted_count} orphaned images, freed {storage_freed_mb:.2f} MB'
                }
                
                logger.info(f"🧹✅ Orphaned image cleanup completed:")
                logger.info(f"   🖼️ Images deleted: {deleted_count}")
                logger.info(f"   💾 Storage freed: {storage_freed_mb:.2f} MB")
                logger.info(f"   ❌ Errors: {errors}")
                
                return result

        except Exception as e:
            logger.error(f"❌ Orphaned image cleanup error: {e}")
            return {
                'status': 'error', 
                'message': f'Orphan cleanup failed: {str(e)}',
                'orphaned_images_found': 0,
                'images_deleted': 0,
                'storage_freed_mb': 0.0
            }

    def get_duplicate_statistics(self):
        """Get enhanced statistics about duplicate records"""
        try:
            if not self.mongo_manager.is_connected:
                return {'status': 'error', 'message': 'Database not connected'}

            # Enhanced pipeline to get comprehensive duplicate statistics
            pipeline = [
                {"$group": {
                    "_id": "$plate_number",
                    "count": {"$sum": 1},
                    "latest_timestamp": {"$max": "$timestamp"},
                    "earliest_timestamp": {"$min": "$timestamp"},
                    "avg_confidence": {"$avg": "$confidence"},
                    "total_records": {"$sum": 1},
                    "with_car_images": {"$sum": {"$cond": [{"$ne": ["$car_image_path", ""]}, 1, 0]}},
                    "with_plate_images": {"$sum": {"$cond": [{"$ne": ["$plate_image_path", ""]}, 1, 0]}}
                }},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}}
            ]

            duplicates = list(self.mongo_manager.collection.aggregate(pipeline))
            
            total_duplicate_plates = len(duplicates)
            total_duplicate_records = sum(d['count'] - 1 for d in duplicates)  # Subtract 1 to keep one of each
            total_records_all = sum(d['count'] for d in duplicates)
            
            # Get top 15 duplicates for detailed view
            top_duplicates = duplicates[:15]
            
            # Calculate storage impact
            estimated_storage_mb = total_duplicate_records * 0.5  # Rough estimate

            return {
                'status': 'success',
                'plates_with_duplicates': total_duplicate_plates,
                'total_duplicate_records': total_duplicate_records,
                'total_records_affected': total_records_all,
                'estimated_storage_impact_mb': round(estimated_storage_mb, 2),
                'top_duplicates': top_duplicates,
                'cleanup_stats': self.cleanup_stats.copy(),
                'last_analysis': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {
                'status': 'error', 
                'message': str(e),
                'plates_with_duplicates': 0,
                'total_duplicate_records': 0
            }

    def should_run_auto_cleanup(self):
        """Enhanced auto cleanup decision with better timing"""
        return (AUTO_CLEANUP_ENABLED and 
                time.time() - self.last_cleanup > DUPLICATE_CLEANUP_INTERVAL and
                self.mongo_manager.is_connected)

    def run_auto_cleanup(self):
        """Run automatic cleanup in background with better error handling"""
        def _cleanup():
            try:
                logger.info("🔄 Starting automatic enhanced duplicate cleanup...")
                result = self.cleanup_duplicates()
                if result['status'] == 'success':
                    logger.info(f"✅ Auto cleanup completed successfully: {result['records_removed']} duplicates removed")
                else:
                    logger.warning(f"⚠️ Auto cleanup completed with issues: {result.get('message', 'Unknown error')}")
            except Exception as e:
                logger.error(f"❌ Auto cleanup error: {e}")
                self.cleanup_stats['errors'] += 1
        
        cleanup_thread = threading.Thread(target=_cleanup, daemon=True, name="AutoCleanup")
        cleanup_thread.start()


class ImprovedVehicleOrientationDetector:
    """Enhanced vehicle orientation detector with better accuracy"""
    def __init__(self):
        self.vehicle_tracks = {}
        self.movement_history = deque(maxlen=VEHICLE_TRACKING_HISTORY)
        self.direction_vectors = deque(maxlen=DIRECTION_ANALYSIS_FRAMES)
        
        # Enhanced feature extractors
        try:
            self.edge_detector = cv2.createLineSegmentDetector()
        except:
            self.edge_detector = None
        
        logger.info("✅ Enhanced vehicle orientation detector initialized")
    
    def detect_vehicle_orientation_and_movement(self, vehicle_img, vehicle_bbox, full_frame, vehicle_id=None):
        """
        Enhanced orientation detection with improved accuracy and movement tracking
        Returns: ('front'/'back'/'unknown', confidence, movement_direction)
        """
        try:
            if vehicle_img is None or vehicle_img.size == 0:
                return 'unknown', 0.0, 'stationary'
            
            # Multi-modal analysis with enhanced scoring
            orientation_scores = {}
            confidence_scores = {}
            
            # Method 1: Enhanced plate position analysis
            plate_result = self._enhanced_plate_position_analysis(vehicle_img, vehicle_bbox, full_frame)
            orientation_scores['plate'] = plate_result[0]
            confidence_scores['plate'] = plate_result[1]
            
            # Method 2: Enhanced geometric feature analysis
            geometric_result = self._enhanced_geometric_analysis(vehicle_img)
            orientation_scores['geometry'] = geometric_result[0]
            confidence_scores['geometry'] = geometric_result[1]
            
            # Method 3: Advanced light detection
            light_result = self._advanced_light_detection(vehicle_img)
            orientation_scores['lights'] = light_result[0]
            confidence_scores['lights'] = light_result[1]
            
            # Method 4: Edge pattern analysis
            edge_result = self._edge_pattern_analysis(vehicle_img)
            orientation_scores['edges'] = edge_result[0]
            confidence_scores['edges'] = edge_result[1]
            
            # Method 5: Vehicle contour analysis
            contour_result = self._vehicle_contour_analysis(vehicle_img)
            orientation_scores['contour'] = contour_result[0]
            confidence_scores['contour'] = contour_result[1]
            
            # Method 6: Movement vector analysis (if tracking available)
            movement_direction = 'stationary'
            movement_confidence = 0.0
            if vehicle_id:
                movement_result = self._movement_vector_analysis(vehicle_bbox, vehicle_id)
                movement_direction = movement_result[0]
                movement_confidence = movement_result[1]
            
            # Enhanced weighted decision making
            front_score = 0.0
            back_score = 0.0
            total_weight = 0.0
            
            # Improved weights based on reliability
            weights = {
                'plate': 3.0,    # Highest weight for plate analysis
                'lights': 2.5,   # High weight for light detection
                'geometry': 2.0, # Medium-high weight for geometry
                'edges': 1.5,    # Medium weight for edges
                'contour': 1.0   # Base weight for contour
            }
            
            for method, orientation in orientation_scores.items():
                confidence = confidence_scores.get(method, 0.5)
                weight = weights.get(method, 1.0) * confidence
                
                if orientation == 'front':
                    front_score += weight
                elif orientation == 'back':
                    back_score += weight
                
                total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                front_score /= total_weight
                back_score /= total_weight
            
            # Determine final orientation with improved thresholds
            confidence_threshold = ORIENTATION_CONFIDENCE_THRESHOLD
            if front_score > back_score and front_score > confidence_threshold:
                final_orientation = 'front'
                final_confidence = min(0.95, front_score)
            elif back_score > front_score and back_score > confidence_threshold:
                final_orientation = 'back'
                final_confidence = min(0.95, back_score)
            else:
                final_orientation = 'unknown'
                final_confidence = max(front_score, back_score)
            
            # Apply temporal consistency
            final_orientation, final_confidence = self._apply_temporal_consistency(
                final_orientation, final_confidence, vehicle_id
            )
            
            return final_orientation, final_confidence, movement_direction
            
        except Exception as e:
            logger.error(f"Enhanced orientation detection error: {e}")
            return 'unknown', 0.0, 'stationary'
    
    def _enhanced_plate_position_analysis(self, vehicle_img, vehicle_bbox, full_frame):
        """Enhanced license plate position analysis with better accuracy"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale plate detection with improved parameters
            plate_regions = []
            
            # Enhanced contour-based detection
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations to enhance plate-like shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicle_height, vehicle_width = gray.shape
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w < 30 or h < 8:  # Too small
                    continue
                    
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Enhanced plate criteria with better aspect ratio range
                if (1.8 < aspect_ratio < 6.5 and 
                    w > vehicle_width * 0.08 and 
                    h > vehicle_height * 0.015 and
                    area > 80 and
                    w < vehicle_width * 0.9 and
                    h < vehicle_height * 0.3):
                    
                    vertical_position = y / vehicle_height
                    horizontal_center = (x + w/2) / vehicle_width
                    
                    # Calculate enhanced confidence based on multiple factors
                    confidence = 0.3
                    
                    # Position scoring
                    if 0.25 < horizontal_center < 0.75:  # Reasonably centered
                        confidence += 0.25
                    
                    # Aspect ratio scoring
                    if 2.5 < aspect_ratio < 5.0:  # Good plate aspect ratio
                        confidence += 0.25
                    
                    # Size scoring
                    size_score = min(1.0, (w * h) / (vehicle_width * vehicle_height * 0.1))
                    confidence += size_score * 0.2
                    
                    plate_regions.append((x, y, w, h, vertical_position, confidence))
            
            if not plate_regions:
                return 'unknown', 0.0
            
            # Analyze plate positions with better logic
            weighted_position = 0.0
            total_confidence = 0.0
            
            for region in plate_regions:
                _, _, _, _, v_pos, conf = region
                weighted_position += v_pos * conf
                total_confidence += conf
            
            if total_confidence > 0.1:
                avg_position = weighted_position / total_confidence
                
                # Enhanced position analysis
                # Front plates: typically in upper 65% of vehicle
                # Rear plates: typically in lower 55% of vehicle
                if avg_position < 0.65:
                    confidence = min(0.9, total_confidence * 1.2)
                    return 'front', confidence
                elif avg_position > 0.45:
                    confidence = min(0.9, total_confidence)
                    return 'back', confidence
            
            return 'unknown', total_confidence * 0.5
            
        except Exception as e:
            logger.error(f"Enhanced plate analysis error: {e}")
            return 'unknown', 0.0
    
    def _enhanced_geometric_analysis(self, vehicle_img):
        """Enhanced geometric feature analysis with improved accuracy"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Divide into more precise regions
            top_region = gray[:height//3, :]
            middle_region = gray[height//3:2*height//3, :]
            bottom_region = gray[2*height//3:, :]
            
            # Calculate enhanced texture complexity
            top_complexity = np.std(top_region)
            middle_complexity = np.std(middle_region)
            bottom_complexity = np.std(bottom_region)
            
            # Analyze horizontal features with better kernels
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
            horizontal_features = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_horizontal)
            
            top_horizontal = np.sum(horizontal_features[:height//3, :] > 0)
            bottom_horizontal = np.sum(horizontal_features[2*height//3:, :] > 0)
            
            # Enhanced scoring with multiple factors
            front_score = 0.0
            back_score = 0.0
            confidence = 0.4
            
            # Complexity analysis (front vehicles typically have more complex top regions)
            complexity_ratio = top_complexity / (bottom_complexity + 1)
            if complexity_ratio > 1.2:
                front_score += 0.3
                confidence += 0.15
            elif complexity_ratio < 0.8:
                back_score += 0.3
                confidence += 0.15
            
            # Horizontal features analysis
            horizontal_ratio = top_horizontal / (bottom_horizontal + 1)
            if horizontal_ratio > 1.3:
                front_score += 0.4
                confidence += 0.2
            elif horizontal_ratio < 0.7:
                back_score += 0.4
                confidence += 0.2
            
            # Edge density analysis with improved processing
            edges = cv2.Canny(gray, 50, 150)
            top_edges = np.sum(edges[:height//3, :] > 0)
            bottom_edges = np.sum(edges[2*height//3:, :] > 0)
            
            edge_ratio = top_edges / (bottom_edges + 1)
            if edge_ratio > 1.15:
                front_score += 0.25
                confidence += 0.1
            elif edge_ratio < 0.85:
                back_score += 0.25
                confidence += 0.1
            
            # Determine result
            if front_score > back_score and front_score > 0.3:
                return 'front', min(0.85, confidence)
            elif back_score > front_score and back_score > 0.3:
                return 'back', min(0.85, confidence)
            else:
                return 'unknown', confidence * 0.6
                
        except Exception as e:
            logger.error(f"Enhanced geometric analysis error: {e}")
            return 'unknown', 0.0
    
    def _advanced_light_detection(self, vehicle_img):
        """Advanced headlight/taillight detection with improved accuracy"""
        try:
            # Multi-color space analysis for better light detection
            hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2LAB)
            
            height, width = vehicle_img.shape[:2]
            
            # Enhanced bright light detection (headlights, daytime running lights)
            bright_mask1 = cv2.inRange(hsv, (0, 0, 180), (180, 100, 255))
            bright_mask2 = cv2.inRange(lab, (170, 0, 0), (255, 255, 255))
            bright_mask = cv2.bitwise_or(bright_mask1, bright_mask2)
            
            # Enhanced red light detection (taillights)
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (20, 255, 255))
            red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Enhanced orange/amber detection (turn signals)
            orange_mask = cv2.inRange(hsv, (8, 100, 120), (25, 255, 255))
            
            # White light detection (backup lights, license plate lights)
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
            
            # Analyze light distribution in regions
            top_quarter = height // 4
            top_half = height // 2
            bottom_quarter = 3 * height // 4
            
            # Count pixels in different regions
            bright_top = np.sum(bright_mask[:top_half, :] > 0)
            bright_bottom = np.sum(bright_mask[top_half:, :] > 0)
            
            red_top = np.sum(red_mask[:top_half, :] > 0)
            red_bottom = np.sum(red_mask[top_half:, :] > 0)
            
            orange_top = np.sum(orange_mask[:top_half, :] > 0)
            orange_bottom = np.sum(orange_mask[top_half:, :] > 0)
            
            white_top = np.sum(white_mask[:top_quarter, :] > 0)
            white_bottom = np.sum(white_mask[bottom_quarter:, :] > 0)
            
            # Enhanced scoring system
            front_score = 0.0
            back_score = 0.0
            confidence = 0.2
            
            # Bright lights analysis (headlights, DRLs)
            if bright_top > bright_bottom and bright_top > 100:
                front_score += 0.5
                confidence += 0.3
            elif bright_bottom > bright_top and bright_bottom > 80:
                back_score += 0.2
                confidence += 0.15
            
            # Red light analysis (strong indicator for rear)
            if red_bottom > red_top and red_bottom > 50:
                back_score += 0.6
                confidence += 0.4
            elif red_top > 0 and red_bottom == 0:
                front_score += 0.1  # Some vehicles have red accents in front
            
            # Orange/amber analysis (turn signals, side markers)
            if orange_top > orange_bottom and orange_top > 30:
                front_score += 0.25
                confidence += 0.15
            elif orange_bottom > orange_top and orange_bottom > 30:
                back_score += 0.25
                confidence += 0.15
            
            # White light analysis
            if white_top > 50:  # License plate lights, backup lights
                back_score += 0.2
                confidence += 0.1
            
            # Combined analysis
            total_lights = bright_top + bright_bottom + red_top + red_bottom
            if total_lights > 200:
                confidence += 0.1
            
            if front_score > back_score:
                return 'front', min(0.95, confidence + (front_score - back_score) * 0.5)
            elif back_score > front_score:
                return 'back', min(0.95, confidence + (back_score - front_score) * 0.5)
            else:
                return 'unknown', confidence * 0.7
                
        except Exception as e:
            logger.error(f"Advanced light detection error: {e}")
            return 'unknown', 0.0
    
    def _edge_pattern_analysis(self, vehicle_img):
        """Enhanced edge pattern analysis for vehicle orientation"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale edge detection with better parameters
            edges1 = cv2.Canny(gray, 40, 120)
            edges2 = cv2.Canny(gray, 60, 180)
            edges = cv2.bitwise_or(edges1, edges2)
            
            height, width = gray.shape
            
            # Enhanced line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25,
                                   minLineLength=width//12, maxLineGap=8)
            
            if lines is None:
                return 'unknown', 0.0
            
            # Analyze line orientations and positions with better categorization
            horizontal_lines_top = 0
            horizontal_lines_bottom = 0
            vertical_lines_top = 0
            vertical_lines_bottom = 0
            diagonal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle and position
                if x2 != x1:
                    angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                else:
                    angle = 90
                
                line_center_y = (y1 + y2) / 2
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Weight by line length for better analysis
                weight = min(1.0, line_length / (width * 0.2))
                
                if angle < 25:  # Horizontal lines
                    if line_center_y < height / 2:
                        horizontal_lines_top += weight
                    else:
                        horizontal_lines_bottom += weight
                elif angle > 65:  # Vertical lines
                    if line_center_y < height / 2:
                        vertical_lines_top += weight
                    else:
                        vertical_lines_bottom += weight
                else:  # Diagonal lines
                    diagonal_lines += weight
            
            # Enhanced analysis with better scoring
            front_score = 0.0
            back_score = 0.0
            confidence = 0.3
            
            # Horizontal line analysis (front vehicles often have more grille lines)
            h_ratio = horizontal_lines_top / (horizontal_lines_bottom + 0.1)
            if h_ratio > 1.3:
                front_score += 0.35
                confidence += 0.15
            elif h_ratio < 0.7:
                back_score += 0.3
                confidence += 0.1
            
            # Vertical line analysis
            v_ratio = vertical_lines_top / (vertical_lines_bottom + 0.1)
            if v_ratio > 1.2:
                front_score += 0.2
                confidence += 0.1
            elif v_ratio < 0.8:
                back_score += 0.2
                confidence += 0.1
            
            # Total line complexity
            total_lines = horizontal_lines_top + horizontal_lines_bottom + vertical_lines_top + vertical_lines_bottom
            if total_lines > 8:
                confidence += 0.15
            
            if front_score > back_score:
                return 'front', min(0.8, confidence + front_score * 0.5)
            elif back_score > front_score:
                return 'back', min(0.8, confidence + back_score * 0.5)
            else:
                return 'unknown', confidence * 0.6
                
        except Exception as e:
            logger.error(f"Edge pattern analysis error: {e}")
            return 'unknown', 0.0
    
    def _vehicle_contour_analysis(self, vehicle_img):
        """Enhanced vehicle contour shape analysis"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Enhanced contour finding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations to improve contour detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 'unknown', 0.0
            
            # Find the largest contour (likely the vehicle outline)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Enhanced contour analysis
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            
            if hull_area == 0:
                return 'unknown', 0.0
            
            solidity = contour_area / hull_area
            
            # Get bounding rectangle and analyze
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Analyze moments for shape characteristics
            moments = cv2.moments(largest_contour)
            
            # Enhanced confidence calculation
            confidence = 0.2
            
            if 0.7 < solidity < 0.95:  # Good vehicle-like solidity
                confidence += 0.2
            
            if 1.2 < aspect_ratio < 4.0:  # Reasonable car aspect ratio
                confidence += 0.2
            
            if contour_area > gray.size * 0.1:  # Significant portion of image
                confidence += 0.1
            
            # For now, return moderate confidence with unknown
            # This could be expanded with more sophisticated shape analysis
            return 'unknown', min(0.6, confidence)
            
        except Exception as e:
            logger.error(f"Contour analysis error: {e}")
            return 'unknown', 0.0
    
    def _movement_vector_analysis(self, vehicle_bbox, vehicle_id):
        """Enhanced vehicle movement analysis to determine direction"""
        try:
            current_time = time.time()
            bbox_center = ((vehicle_bbox[0] + vehicle_bbox[2]) / 2, 
                          (vehicle_bbox[1] + vehicle_bbox[3]) / 2)
            
            if vehicle_id not in self.vehicle_tracks:
                self.vehicle_tracks[vehicle_id] = deque(maxlen=DIRECTION_ANALYSIS_FRAMES)
            
            # Add current position to track
            self.vehicle_tracks[vehicle_id].append((bbox_center, current_time))
            
            track = self.vehicle_tracks[vehicle_id]
            
            if len(track) < 4:  # Need more points for reliable analysis
                return 'stationary', 0.0
            
            # Calculate movement vector with improved analysis
            positions = [pos for pos, _ in track]
            times = [t for _, t in track]
            
            # Calculate weighted movement direction (recent movements have more weight)
            dx_total = 0
            dy_total = 0
            total_weight = 0
            valid_movements = 0
            
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dt = times[i] - times[i-1]
                
                if dt > 0:
                    # Weight recent movements more heavily
                    weight = (i / len(positions)) ** 2
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance > 3:  # Minimum movement threshold
                        dx_total += (dx / dt) * weight
                        dy_total += (dy / dt) * weight
                        total_weight += weight
                        valid_movements += 1
            
            if valid_movements < 2 or total_weight == 0:
                return 'stationary', 0.0
            
            avg_dx = dx_total / total_weight
            avg_dy = dy_total / total_weight
            
            # Calculate movement magnitude and direction
            movement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
            
            if movement_magnitude < 5:  # Too slow to determine direction reliably
                return 'stationary', 0.0
            
            # Determine primary movement direction with improved logic
            if abs(avg_dx) > abs(avg_dy) * 1.5:  # Primarily horizontal movement
                if avg_dx > 0:
                    direction = 'moving_right'
                else:
                    direction = 'moving_left'
            elif abs(avg_dy) > abs(avg_dx) * 1.5:  # Primarily vertical movement
                if avg_dy > 0:
                    direction = 'moving_down'
                else:
                    direction = 'moving_up'
            else:  # Diagonal movement
                if avg_dx > 0 and avg_dy > 0:
                    direction = 'moving_down_right'
                elif avg_dx > 0 and avg_dy < 0:
                    direction = 'moving_up_right'
                elif avg_dx < 0 and avg_dy > 0:
                    direction = 'moving_down_left'
                else:
                    direction = 'moving_up_left'
            
            confidence = min(0.9, movement_magnitude / 30.0)
            return direction, confidence
            
        except Exception as e:
            logger.error(f"Movement analysis error: {e}")
            return 'stationary', 0.0
    
    def _apply_temporal_consistency(self, orientation, confidence, vehicle_id):
        """Enhanced temporal consistency with better smoothing"""
        try:
            if vehicle_id is None:
                return orientation, confidence
            
            # Store detection history per vehicle
            if vehicle_id not in self.vehicle_tracks:
                self.vehicle_tracks[vehicle_id] = deque(maxlen=TEMPORAL_CONSISTENCY_FRAMES)
            
            current_time = time.time()
            
            # Add current detection with timestamp
            self.vehicle_tracks[vehicle_id].append((orientation, confidence, current_time))
            
            track = self.vehicle_tracks[vehicle_id]
            
            if len(track) < 3:  # Need at least 3 detections
                return orientation, confidence
            
            # Analyze recent detections with time weighting
            recent_orientations = []
            total_weight = 0
            
            for o, c, t in track:
                if c > 0.2:  # Only consider reasonable confidence detections
                    # Weight recent detections more heavily
                    time_weight = max(0.1, 1.0 - (current_time - t) / 10.0)  # 10 second decay
                    confidence_weight = c
                    combined_weight = time_weight * confidence_weight
                    
                    recent_orientations.append((o, combined_weight))
                    total_weight += combined_weight
            
            if len(recent_orientations) < 2 or total_weight == 0:
                return orientation, confidence
            
            # Calculate weighted consensus
            orientation_scores = {'front': 0, 'back': 0, 'unknown': 0}
            for o, weight in recent_orientations:
                orientation_scores[o] += weight
            
            # Normalize scores
            for key in orientation_scores:
                orientation_scores[key] /= total_weight
            
            # Find the most consistent orientation
            best_orientation = max(orientation_scores.keys(), 
                                 key=lambda x: orientation_scores[x])
            best_score = orientation_scores[best_orientation]
            
            # Apply temporal consistency if there's strong agreement
            if best_score >= 0.6 and len(recent_orientations) >= 3:
                # High consistency - boost confidence
                boosted_confidence = min(0.95, confidence * 1.3 + best_score * 0.2)
                return best_orientation, boosted_confidence
            
            return orientation, confidence
            
        except Exception as e:
            logger.error(f"Temporal consistency error: {e}")
            return orientation, confidence
    
    def get_entry_exit_status(self, orientation, confidence, movement_direction=None):
        """
        FIXED entry/exit status.
        This version prioritizes the actual movement direction over the vehicle's orientation (front/back),
        which is a more reliable way to determine entry or exit.
        """
        # The overall confidence in the status is based on the detection/orientation confidence
        status_confidence = confidence

        # If movement is detected, it becomes the primary factor for determining entry/exit
        if movement_direction and movement_direction != 'stationary':
            # --- CONFIGURATION ASSUMPTION ---
            # Based on a typical camera setup, we assume movement towards the bottom or right of the
            # screen is "Entering" and movement towards the top or left is "Exiting".
            # This can be adjusted depending on the physical camera installation.
            entering_directions = ['moving_down', 'moving_right', 'moving_down_right', 'moving_down_left']
            exiting_directions = ['moving_up', 'moving_left', 'moving_up_right', 'moving_up_left']

            status = "Vehicle In Motion"  # Default status if direction is ambiguous
            
            if movement_direction in entering_directions:
                status = "Vehicle Entering"
            elif movement_direction in exiting_directions:
                status = "Vehicle Exiting"

            # Create a more descriptive status message
            orientation_desc = f"{orientation.title()} View" if orientation != 'unknown' else "Side View"
            movement_desc = movement_direction.replace("_", " ").title()
            
            final_status = f"{status} ({movement_desc}, {orientation_desc})"
            return final_status, status_confidence

        # --- FALLBACK LOGIC ---
        # If the vehicle is stationary or its movement is unclear, revert to the orientation-based logic.
        # This is useful for vehicles that have just appeared or come to a stop.
        if confidence < 0.4:  # Require slightly higher confidence if no movement is detected
            return 'Vehicle Detected', confidence

        if orientation == 'front':
            return 'Vehicle Entered (Front View, Stationary)', confidence
        elif orientation == 'back':
            return 'Vehicle Exited (Rear View, Stationary)', confidence
        else:
            return 'Vehicle Detected (Stationary)', confidence * 0.8


class OptimizedOCREngine:
    """Enhanced OCR engine with improved preprocessing and accuracy"""
    def __init__(self):
        self.cache_lock = threading.Lock()
        self.preprocessing_cache = {}
        self.valid_plate_pattern = re.compile(r'^[A-Z0-9]{4,10}$')  # More flexible pattern
        
        # Enhanced OCR configurations
        self.ocr_configs = {
            'fast': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'accurate': '--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'single_line': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'enhanced': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        }
        
    def enhanced_preprocess(self, img):
        """Enhanced image preprocessing for better OCR accuracy"""
        try:
            if img is None or img.size == 0:
                return img
                
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Enhanced resize with better interpolation
            h, w = gray.shape
            if w < 150:
                scale = 150 / w
                gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            elif w > 500:  # Also handle too large images
                scale = 500 / w
                gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Enhanced contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Sharpening with better kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure good contrast
            sharpened = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
            
            return sharpened
        except Exception as e:
            logger.error(f"Enhanced preprocessing error: {e}")
            return img

    def enhanced_ocr(self, img):
        """Enhanced OCR with better accuracy and caching"""
        try:
            if img is None or img.size == 0:
                return ""
            
            # Generate hash for caching
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            
            with self.cache_lock:
                if img_hash in self.preprocessing_cache:
                    return self.preprocessing_cache[img_hash]
            
            # Enhanced preprocessing
            processed = self.enhanced_preprocess(img)
            
            # Try multiple OCR approaches for better accuracy
            texts = []
            
            # Approach 1: Fast OCR
            try:
                text1 = pytesseract.image_to_string(processed, config=self.ocr_configs['fast']).strip()
                if text1:
                    texts.append(text1)
            except:
                pass
            
            # Approach 2: Enhanced OCR for better accuracy
            try:
                text2 = pytesseract.image_to_string(processed, config=self.ocr_configs['enhanced']).strip()
                if text2:
                    texts.append(text2)
            except:
                pass
            
            # Choose the best result
            best_text = ""
            best_score = 0
            
            for text in texts:
                cleaned = ''.join(c for c in text if c.isalnum()).upper()
                if self.is_valid_license_plate(cleaned):
                    score = len(cleaned) + (10 if 4 <= len(cleaned) <= 8 else 0)
                    if score > best_score:
                        best_score = score
                        best_text = cleaned
            
            # Cache if good result
            if best_text and self.is_valid_license_plate(best_text):
                with self.cache_lock:
                    # Limit cache size
                    if len(self.preprocessing_cache) > 100:
                        # Remove oldest 25% of entries
                        keys_to_remove = list(self.preprocessing_cache.keys())[:25]
                        for key in keys_to_remove:
                            del self.preprocessing_cache[key]
                    
                    self.preprocessing_cache[img_hash] = best_text
            
            return best_text
            
        except Exception as e:
            logger.error(f"Enhanced OCR error: {e}")
            return ""

    def is_valid_license_plate(self, text):
        """Enhanced license plate validation with better patterns"""
        if not text or len(text) < 4 or len(text) > 10:
            return False
        
        # Check for basic pattern
        if not self.valid_plate_pattern.match(text):
            return False
        
        # Additional validation rules
        # Must have at least one letter and one number
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if not (has_letter and has_number):
            return False
        
        # Reject common OCR errors
        error_patterns = ['III', '111', '000', 'OOO', '###']
        if any(pattern in text for pattern in error_patterns):
            return False
        
        return True


class EnhancedMongoDBManager:
    """Enhanced MongoDB manager with better error handling and performance"""
    def __init__(self, uri=MONGO_URI, db_name=MONGO_DB_NAME, collection_name=MONGO_COLLECTION_NAME):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connection_lock = RLock()
        self.is_connected = False
        self.write_queue = queue.Queue(maxsize=200)  # Larger queue
        self.batch_size = 15  # Larger batches for better performance
        self.batch_timeout = 2.0  # Longer timeout for better batching
        
        self._connect()
        self._start_batch_writer()

    def _connect(self):
        """Enhanced connection with better settings"""
        try:
            self.client = MongoClient(
                self.uri, 
                serverSelectionTimeoutMS=5000,  # Longer timeout
                maxPoolSize=50,  # More connections
                minPoolSize=10,
                maxIdleTimeMS=30000,  # Longer idle time
                waitQueueTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Create enhanced indexes for performance
            try:
                indexes = [
                    [("plate_number", 1), ("timestamp", -1)],
                    [("detection_id", 1)],
                    [("entry_exit_status", 1)],
                    [("vehicle_orientation", 1)],
                    [("timestamp", -1)],
                    [("plate_number", 1)],
                    [("confidence", -1)],
                    [("is_entry", 1)],
                    [("is_exit", 1)]
                ]
                
                for index in indexes:
                    try:
                        self.collection.create_index(index)
                    except pymongo.errors.OperationFailure:
                        pass  # Index might already exist
                        
            except Exception as e:
                logger.warning(f"Could not create some indexes: {e}")

            self.is_connected = True
            logger.info("✅ Enhanced MongoDB connected with optimized settings")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.is_connected = False
            return False

    def _start_batch_writer(self):
        """Start enhanced batch writer"""
        if self.is_connected:
            self.batch_writer_thread = threading.Thread(target=self._enhanced_batch_writer, daemon=True, name="BatchWriter")
            self.batch_writer_thread.start()

    def _enhanced_batch_writer(self):
        """Enhanced batch writer with better error handling and performance"""
        batch = []
        last_write = time.time()
        
        while True:
            try:
                try:
                    # Non-blocking get with timeout
                    item = self.write_queue.get(timeout=0.5)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                current_time = time.time()
                should_write = (len(batch) >= self.batch_size or 
                               (batch and current_time - last_write > self.batch_timeout))
                
                if should_write and self.is_connected and batch:
                    try:
                        # Use unordered bulk insert for better performance
                        result = self.collection.insert_many(batch, ordered=False)
                        logger.debug(f"💾 Batch saved {len(result.inserted_ids)} vehicle records")
                        batch.clear()
                        last_write = current_time
                        
                    except pymongo.errors.BulkWriteError as bwe:
                        # Handle duplicate key errors gracefully
                        duplicate_errors = 0
                        other_errors = 0
                        
                        for error in bwe.details.get('writeErrors', []):
                            if error['code'] == 11000:  # Duplicate key error
                                duplicate_errors += 1
                            else:
                                other_errors += 1
                        
                        if duplicate_errors > 0:
                            logger.debug(f"💾 Batch write: {duplicate_errors} duplicates ignored, {len(batch) - duplicate_errors} records saved")
                        
                        if other_errors > 0:
                            logger.warning(f"⚠️ Batch write: {other_errors} other errors occurred")
                        
                        batch.clear()
                        last_write = current_time
                        
                    except Exception as e:
                        logger.error(f"❌ Batch write error: {e}")
                        batch.clear()
                        # Try to reconnect on error
                        if not self._reconnect():
                            time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Batch writer error: {e}")
                time.sleep(1)

    def _reconnect(self):
        """Enhanced reconnection logic"""
        try:
            logger.info("🔄 Attempting MongoDB reconnection...")
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
            
            time.sleep(2)
            return self._connect()
            
        except Exception as e:
            logger.error(f"❌ Reconnection failed: {e}")
            return False

    def save_vehicle_record_async(self, plate_number, confidence, car_image_path, plate_image_path, 
                                bbox_info, vehicle_info=None, additional_data=None, 
                                entry_exit_status=None, vehicle_orientation=None, orientation_confidence=None):
        """Enhanced async record saving with better data validation"""
        try:
            timestamp = datetime.now()
            
            # Enhanced document structure with better validation
            document = {
                "plate_number": str(plate_number).strip().upper() if plate_number else "",
                "timestamp": timestamp,
                "confidence": float(confidence) if confidence else 0.0,
                "car_image_path": str(car_image_path).strip() if car_image_path else "",
                "plate_image_path": str(plate_image_path).strip() if plate_image_path else "",
                "detection_id": f"{plate_number}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{hash(str(bbox_info)) % 1000:03d}",
                "bbox_info": bbox_info or {},
                "vehicle_info": vehicle_info or {},
                "processing_method": "Enhanced Stream + Improved Entry/Exit Detection + Smart Duplicate Cleanup",
                "image_capture_enabled": True,
                "full_car_captured": bool(car_image_path and os.path.exists(car_image_path) if car_image_path else False),
                "plate_captured": bool(plate_image_path and os.path.exists(plate_image_path) if plate_image_path else False),
                "processed": True,
                "entry_exit_status": str(entry_exit_status) if entry_exit_status else "Vehicle Detected",
                "vehicle_orientation": str(vehicle_orientation) if vehicle_orientation else "unknown",
                "orientation_confidence": float(orientation_confidence) if orientation_confidence else 0.0,
                "is_entry": bool(entry_exit_status and "Enter" in str(entry_exit_status)),
                "is_exit": bool(entry_exit_status and "Exit" in str(entry_exit_status)),
                "movement_info": vehicle_info.get("movement_direction", "unknown") if vehicle_info else "unknown",
                "system_version": "Enhanced_v6.0",
                "quality_score": self._calculate_quality_score(confidence, orientation_confidence, vehicle_info)
            }
            
            # Add additional data if provided
            if additional_data and isinstance(additional_data, dict):
                document.update(additional_data)
            
            # Non-blocking queue put with timeout
            try:
                self.write_queue.put_nowait(document)
                return True
            except queue.Full:
                # Try to make space by removing older items
                try:
                    self.write_queue.get_nowait()  # Remove one old item
                    self.write_queue.put_nowait(document)  # Add new item
                    logger.debug("📝 Write queue full - replaced oldest item")
                    return True
                except queue.Empty:
                    logger.warning("📝 Write queue full - dropping record")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Async save error: {e}")
            return False

    def _calculate_quality_score(self, confidence, orientation_confidence, vehicle_info):
        """Calculate a quality score for the detection"""
        try:
            score = 0.0
            
            # Confidence score (0-40 points)
            if confidence:
                score += min(40, confidence * 0.4)
            
            # Orientation confidence (0-30 points)
            if orientation_confidence:
                score += min(30, orientation_confidence * 30)
            
            # Vehicle info quality (0-30 points)
            if vehicle_info:
                if vehicle_info.get("full_car_captured"):
                    score += 15
                if vehicle_info.get("type") and vehicle_info["type"] != "unknown":
                    score += 10
                if vehicle_info.get("car_confidence", 0) > 0.5:
                    score += 5
            
            return round(score, 1)
            
        except Exception:
            return 0.0

    def get_vehicle_records(self, limit=50, query=None):
        """Enhanced record retrieval with better filtering and error handling"""
        try:
            if not self.is_connected:
                return []
            
            search_filter = {}
            if query and query.strip():
                query = query.strip()
                search_filter = {
                    "$or": [
                        {"plate_number": {"$regex": query, "$options": "i"}},
                        {"detection_id": {"$regex": query, "$options": "i"}},
                        {"vehicle_info.type": {"$regex": query, "$options": "i"}},
                        {"entry_exit_status": {"$regex": query, "$options": "i"}},
                        {"vehicle_orientation": {"$regex": query, "$options": "i"}}
                    ]
                }
            
            # Enhanced query with better projection
            projection = {
                "plate_number": 1,
                "timestamp": 1,
                "confidence": 1,
                "car_image_path": 1,
                "plate_image_path": 1,
                "detection_id": 1,
                "bbox_info": 1,
                "vehicle_info": 1,
                "processing_method": 1,
                "full_car_captured": 1,
                "plate_captured": 1,
                "entry_exit_status": 1,
                "vehicle_orientation": 1,
                "orientation_confidence": 1,
                "is_entry": 1,
                "is_exit": 1,
                "movement_info": 1,
                "quality_score": 1
            }
            
            cursor = self.collection.find(search_filter, projection).sort("timestamp", -1).limit(limit)
            records = []
            
            for doc in cursor:
                record = {
                    "plate_number": doc.get("plate_number", ""),
                    "timestamp": doc.get("timestamp"),
                    "confidence": doc.get("confidence", 0.0),
                    "car_image_path": doc.get("car_image_path", ""),
                    "plate_image_path": doc.get("plate_image_path", ""),
                    "detection_id": doc.get("detection_id", ""),
                    "bbox_info": doc.get("bbox_info", {}),
                    "vehicle_info": doc.get("vehicle_info", {}),
                    "processing_method": doc.get("processing_method", "Enhanced Detection"),
                    "full_car_captured": doc.get("full_car_captured", False),
                    "plate_captured": doc.get("plate_captured", False),
                    "entry_exit_status": doc.get("entry_exit_status", "Vehicle Detected"),
                    "vehicle_orientation": doc.get("vehicle_orientation", "unknown"),
                    "orientation_confidence": doc.get("orientation_confidence", 0.0),
                    "is_entry": doc.get("is_entry", False),
                    "is_exit": doc.get("is_exit", False),
                    "movement_info": doc.get("movement_info", "unknown"),
                    "quality_score": doc.get("quality_score", 0.0)
                }
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"❌ Error getting vehicle records: {e}")
            return []

    def get_enhanced_stats(self):
        """Enhanced statistics with comprehensive performance data"""
        try:
            if not self.is_connected:
                return self._default_stats()
            
            # Enhanced aggregation pipeline for better performance
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_vehicles": {"$sum": 1},
                        "unique_plates": {"$addToSet": "$plate_number"},
                        "full_cars_captured": {
                            "$sum": {"$cond": [{"$eq": ["$full_car_captured", True]}, 1, 0]}
                        },
                        "plates_captured": {
                            "$sum": {"$cond": [{"$eq": ["$plate_captured", True]}, 1, 0]}
                        },
                        "entries": {
                            "$sum": {"$cond": [{"$eq": ["$is_entry", True]}, 1, 0]}
                        },
                        "exits": {
                            "$sum": {"$cond": [{"$eq": ["$is_exit", True]}, 1, 0]}
                        },
                        "front_detections": {
                            "$sum": {"$cond": [{"$eq": ["$vehicle_orientation", "front"]}, 1, 0]}
                        },
                        "back_detections": {
                            "$sum": {"$cond": [{"$eq": ["$vehicle_orientation", "back"]}, 1, 0]}
                        },
                        "unknown_orientation": {
                            "$sum": {"$cond": [{"$eq": ["$vehicle_orientation", "unknown"]}, 1, 0]}
                        },
                        "latest_timestamp": {"$max": "$timestamp"},
                        "avg_confidence": {"$avg": "$confidence"},
                        "avg_orientation_confidence": {"$avg": "$orientation_confidence"},
                        "avg_quality_score": {"$avg": "$quality_score"}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result and result[0]:
                stats = result[0]
                unique_plates_list = stats.get("unique_plates", [])
                
                return {
                    "total_vehicles": stats.get("total_vehicles", 0),
                    "unique_plates": len([p for p in unique_plates_list if p]),  # Filter empty plates
                    "full_cars_captured": stats.get("full_cars_captured", 0),
                    "car_images_captured": stats.get("full_cars_captured", 0),
                    "plate_images_captured": stats.get("plates_captured", 0),
                    "entries": stats.get("entries", 0),
                    "exits": stats.get("exits", 0),
                    "front_detections": stats.get("front_detections", 0),
                    "back_detections": stats.get("back_detections", 0),
                    "unknown_orientation": stats.get("unknown_orientation", 0),
                    "latest_detection": stats.get("latest_timestamp"),
                    "avg_confidence": round(stats.get("avg_confidence", 0.0), 1),
                    "avg_orientation_confidence": round(stats.get("avg_orientation_confidence", 0.0), 1),
                    "avg_quality_score": round(stats.get("avg_quality_score", 0.0), 1),
                    "database_status": "Connected"
                }
            else:
                return self._default_stats()
                
        except Exception as e:
            logger.error(f"❌ Error getting enhanced stats: {e}")
            return self._default_stats()

    def _default_stats(self):
        """Return enhanced default stats when database is unavailable"""
        return {
            "total_vehicles": 0,
            "unique_plates": 0,
            "full_cars_captured": 0,
            "car_images_captured": 0,
            "plate_images_captured": 0,
            "entries": 0,
            "exits": 0,
            "front_detections": 0,
            "back_detections": 0,
            "unknown_orientation": 0,
            "avg_confidence": 0.0,
            "avg_orientation_confidence": 0.0,
            "avg_quality_score": 0.0,
            "database_status": "Disconnected"
        }

    def close_connection(self):
        """Enhanced cleanup with better error handling"""
        try:
            logger.info("🔄 Closing MongoDB connection...")
            
            # Wait for batch writer to finish current batch
            if hasattr(self, 'batch_writer_thread') and self.batch_writer_thread.is_alive():
                # Give it some time to finish
                time.sleep(2)
            
            if self.client:
                self.client.close()
                self.is_connected = False
                logger.info("✅ MongoDB connection closed")
                
        except Exception as e:
            logger.error(f"❌ Error closing MongoDB connection: {e}")


class UltraStableRTSPReader:
    """Enhanced ultra-stable RTSP reader with better quality and reliability"""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame_lock = Lock()
        self.is_connected = False
        self.running = False
        self.consecutive_failures = 0
        self.last_frame_time = time.time()
        self.reconnect_attempts = 0
        
        # Enhanced buffering for quality
        self.latest_frame = None
        self.frame_buffer = queue.Queue(maxsize=3)  # Slightly larger buffer for quality
        
        # Performance monitoring
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_fps_calculation = time.time()
        self.current_fps = 0.0
        
        # Enhanced capture settings for quality
        self.width = 1920
        self.height = 1080
        self.target_fps = TARGET_FPS
        
        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize RTSP capture with enhanced quality settings"""
        logger.info(f"🔗 Connecting to enhanced ultra-stable RTSP stream: {self.rtsp_url}")
        
        try:
            # Try multiple backends for best compatibility
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.rtsp_url, backend)
                    
                    if self.cap.isOpened():
                        logger.info(f"✅ Connected using backend: {backend}")
                        break
                except:
                    continue
            
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                logger.error("❌ Failed to open RTSP stream with any backend")
                return False
            
            # Enhanced capture settings for quality - NO FRAME SKIPPING
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, STREAM_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Quality settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Test frame reading
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("❌ Cannot read frames from stream")
                self.cap.release()
                return False
            
            logger.info(f"✅ Enhanced ultra-stable RTSP connected - Frame size: {test_frame.shape}")
            self.is_connected = True
            
            # Start enhanced capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._enhanced_capture_loop, daemon=True, name="RTSPCapture")
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RTSP initialization error: {e}")
            self.is_connected = False
            return False

    def _enhanced_capture_loop(self):
        """Enhanced capture loop with better quality and no frame skipping"""
        logger.info("🎬 Starting enhanced capture loop - NO FRAME SKIPPING")
        
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                if not self.cap or not self.cap.isOpened():
                    logger.warning("📡 Stream disconnected, attempting enhanced reconnection...")
                    if not self._enhanced_reconnect():
                        time.sleep(2)
                        continue
                
                # Read frame with timeout handling - NO SKIPPING
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    logger.debug(f"⚠️ Frame read failed (attempt {self.consecutive_failures})")
                    
                    if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error("🚨 Too many consecutive failures, reconnecting...")
                        self._enhanced_reconnect()
                    time.sleep(0.02)
                    continue
                
                # Successful frame read
                self.consecutive_failures = 0
                self.frames_received += 1
                self.last_frame_time = current_time
                
                # Update frame with minimal locking
                with self.frame_lock:
                    self.latest_frame = frame.copy()  # Deep copy for thread safety
                
                # Enhanced buffer management - maintain quality
                try:
                    # If buffer is full, remove oldest frame but keep quality
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    
                    self.frame_buffer.put_nowait(frame.copy())
                    
                except queue.Full:
                    self.frames_dropped += 1
                
                # Calculate FPS
                if current_time - self.last_fps_calculation > 2.0:  # Calculate every 2 seconds
                    if self.frames_received > 0:
                        self.current_fps = self.frames_received / (current_time - self.last_fps_calculation)
                    self.frames_received = 0
                    self.last_fps_calculation = current_time
                
                # Maintain target FPS without dropping frames
                elapsed = current_time - last_frame_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = current_time
                
            except Exception as e:
                logger.error(f"❌ Enhanced capture error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)
        
        logger.warning("🎬 Enhanced capture loop stopped")

    def _enhanced_reconnect(self):
        """Enhanced reconnection with better error handling"""
        with self.frame_lock:
            if self.is_connected:
                self.is_connected = False
                logger.info("📡 Setting stream status to disconnected")
        
        try:
            self.reconnect_attempts += 1
            logger.info(f"🔄 Enhanced reconnection attempt #{self.reconnect_attempts}")
            
            # Clean shutdown of current connection
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            
            time.sleep(2)  # Wait before reconnection
            
            self.consecutive_failures = 0
            
            success = self._initialize_capture()
            if success:
                logger.info("✅ Enhanced reconnection successful")
            else:
                logger.error("❌ Enhanced reconnection failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Enhanced reconnection error: {e}")
            return False

    def get_frame(self):
        """Get the latest frame with enhanced quality"""
        try:
            # Try buffer first (most recent high-quality frame)
            try:
                frame = self.frame_buffer.get_nowait()
                return frame
            except queue.Empty:
                pass
            
            # Fallback to latest frame
            with self.frame_lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting frame: {e}")
            return None

    def is_healthy(self):
        """Enhanced health check"""
        return (self.is_connected and 
                self.consecutive_failures < MAX_CONSECUTIVE_FAILURES and
                time.time() - self.last_frame_time < 5.0)  # 5 second tolerance

    def get_stats(self):
        """Get enhanced performance statistics"""
        return {
            'connected': self.is_connected,
            'healthy': self.is_healthy(),
            'consecutive_failures': self.consecutive_failures,
            'reconnect_attempts': self.reconnect_attempts,
            'current_fps': round(self.current_fps, 1),
            'frames_dropped': self.frames_dropped,
            'last_frame_age': round(time.time() - self.last_frame_time, 1),
            'buffer_size': self.frame_buffer.qsize(),
            'target_fps': self.target_fps
        }

    def release(self):
        """Enhanced cleanup with better shutdown"""
        logger.info("🎬 Shutting down enhanced ultra-stable RTSP reader...")
        
        self.running = False
        
        # Wait for capture thread to finish
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=5.0)
        
        # Clean shutdown
        if hasattr(self, 'cap') and self.cap:
            try:
                self.cap.release()
            except:
                pass
        
        # Clear buffers
        try:
            while not self.frame_buffer.empty():
                self.frame_buffer.get_nowait()
        except:
            pass
        
        with self.frame_lock:
            self.latest_frame = None
        
        logger.info("🎬 Enhanced ultra-stable RTSP reader shutdown complete")


class EnhancedCarDetector:
    """Enhanced car detector with no frame skipping and smart duplicate cleanup"""
    def __init__(self, plate_model_path, car_model_path=YOLO_CAR_MODEL):
        # Load YOLO models with enhanced settings
        try:
            self.plate_model = YOLO(plate_model_path)
            self.car_model = YOLO(car_model_path)
            
            # GPU optimization
            if torch.cuda.is_available():
                self.plate_model.to('cuda')
                self.car_model.to('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🎮 GPU ACCELERATION ENABLED: {gpu_name}")
            else:
                logger.info("💻 CPU Processing Mode")
            
            # Model optimization
            self.plate_model.overrides['verbose'] = False
            self.car_model.overrides['verbose'] = False
            
            logger.info(f"✅ Enhanced YOLO models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO models: {e}")
            raise
        
        # Initialize enhanced components
        self.ocr_engine = OptimizedOCREngine()
        self.mongo_manager = EnhancedMongoDBManager()
        self.orientation_detector = ImprovedVehicleOrientationDetector()
        self.duplicate_manager = EnhancedDuplicateCleanupManager(self.mongo_manager)
        
        # Load existing plates for duplicate prevention
        self.captured_plates = set()
        self._load_existing_plates()
        
        # Enhanced detection parameters
        self.plate_confidence_threshold = 0.3  # Slightly higher for quality
        self.car_confidence_threshold = CAR_DETECTION_CONFIDENCE
        self.iou_threshold = 0.4
        
        # NO FRAME SKIPPING - Process every frame for quality
        self.frame_count = 0
        self.last_ocr_frame = 0
        self.processing_interval = PROCESSING_INTERVAL  # Always 1 - no skipping
        self.ocr_interval = OCR_PROCESSING_INTERVAL    # Always 1 - no skipping
        
        # Thread-safe collections
        self.captured_plates_lock = RLock()
        self.session_plates = set()
        self.recent_detections = deque(maxlen=50)  # Larger for better duplicate detection
        self.vehicle_records = []
        self.vehicle_records_lock = RLock()
        
        # Enhanced vehicle tracking
        self.vehicle_tracker = {}
        self.track_id_counter = 0
        
        # Enhanced background processing
        self.image_processor_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ImageProc")
        
        # Performance statistics
        self.stats_lock = RLock()
        self.detection_stats = {
            'total_detections': 0,
            'unique_plates': len(self.captured_plates),
            'full_cars_captured': 0,
            'car_images_captured': 0,
            'plate_images_captured': 0,
            'entries': 0,
            'exits': 0,
            'front_detections': 0,
            'back_detections': 0,
            'unknown_orientation': 0,
            'session_start': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_detection': 'None',
            'plates_list': [],
            'is_live': True,
            'mongodb_status': 'Connected' if self.mongo_manager.is_connected else 'Disconnected',
            'processing_mode': 'Enhanced Stream + No Frame Skipping + Smart Duplicate Cleanup',
            'duplicate_cleanups': 0
        }
        
        # Enhanced memory management
        self.last_cleanup = 0
        self.last_duplicate_cleanup = time.time()

    def _load_existing_plates(self):
        """Load existing plates from database with better error handling"""
        try:
            logger.info("📋 Loading existing plates from database...")
            records = self.mongo_manager.get_vehicle_records(limit=1000)  # Load more for better duplicate detection
            self.captured_plates = {record['plate_number'] for record in records 
                                  if record['plate_number'] and len(record['plate_number']) > 3}
            logger.info(f"📋 Loaded {len(self.captured_plates)} existing plates for duplicate prevention")
        except Exception as e:
            logger.error(f"❌ Error loading existing plates: {e}")
            self.captured_plates = set()

    def should_process_frame(self):
        """Enhanced frame processing decision - NO SKIPPING for quality"""
        self.frame_count += 1
        
        # Run enhanced duplicate cleanup periodically
        if (self.frame_count % (DUPLICATE_CHECK_INTERVAL * 30) == 0 and 
            self.duplicate_manager.should_run_auto_cleanup()):
            self.duplicate_manager.run_auto_cleanup()
            with self.stats_lock:
                self.detection_stats['duplicate_cleanups'] += 1
        
        # Process EVERY frame for maximum quality - NO SKIPPING
        if self.frame_count % self.processing_interval == 0:
            if self.frame_count - self.last_cleanup > MEMORY_CLEANUP_INTERVAL:
                self._enhanced_cleanup()
                self.last_cleanup = self.frame_count
            return True
        
        return True  # Always process for quality

    def should_run_ocr(self):
        """Enhanced OCR decision - NO SKIPPING for quality"""
        # Run OCR on EVERY frame for best accuracy - NO SKIPPING
        if self.frame_count - self.last_ocr_frame >= self.ocr_interval:
            self.last_ocr_frame = self.frame_count
            return True
        return True  # Always run OCR for quality

    def _enhanced_cleanup(self):
        """Enhanced memory cleanup with better management"""
        try:
            # Enhanced OCR cache management
            with self.ocr_engine.cache_lock:
                if len(self.ocr_engine.preprocessing_cache) > 150:
                    # Keep more recent entries for better performance
                    cache_items = list(self.ocr_engine.preprocessing_cache.items())
                    self.ocr_engine.preprocessing_cache = dict(cache_items[-75:])
            
            # Enhanced vehicle records management
            with self.vehicle_records_lock:
                if len(self.vehicle_records) > 100:
                    self.vehicle_records = self.vehicle_records[-50:]
            
            # Enhanced vehicle tracking cleanup
            current_time = time.time()
            old_tracks = [vid for vid, track in self.vehicle_tracker.items() 
                         if current_time - track.get('last_seen', 0) > 60]  # Longer timeout for quality
            for vid in old_tracks:
                del self.vehicle_tracker[vid]
            
            # Enhanced recent detections cleanup
            cutoff_time = current_time - 30  # 30 second window for duplicates
            self.recent_detections = deque([
                (key, timestamp) for key, timestamp in self.recent_detections
                if timestamp > cutoff_time
            ], maxlen=50)
            
            # Periodic garbage collection
            if self.frame_count % 500 == 0:
                gc.collect()
                
        except Exception as e:
            logger.error(f"❌ Enhanced cleanup error: {e}")

    def detect_cars_enhanced(self, frame):
        """Enhanced car detection with better accuracy and tracking"""
        try:
            # Run inference with enhanced parameters
            results = self.car_model(frame, 
                                   conf=self.car_confidence_threshold, 
                                   classes=CAR_CLASSES, 
                                   verbose=False,
                                   imgsz=640,
                                   half=True if torch.cuda.is_available() else False)
            
            cars = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Enhanced size filtering
                        width = x2 - x1
                        height = y2 - y1
                        
                        if (width >= MIN_CAR_WIDTH and height >= MIN_CAR_HEIGHT and
                            width < frame.shape[1] * 0.9 and height < frame.shape[0] * 0.8):  # Not too large
                            
                            # Enhanced vehicle ID assignment
                            vehicle_id = self._enhanced_vehicle_tracking((x1, y1, x2, y2))
                            
                            cars.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.get_vehicle_class_name(class_id),
                                'vehicle_id': vehicle_id,
                                'dimensions': {'width': width, 'height': height}
                            })
            
            return cars
        except Exception as e:
            logger.error(f"❌ Enhanced car detection error: {e}")
            return []

    def _enhanced_vehicle_tracking(self, bbox):
        """Enhanced vehicle tracking with better accuracy"""
        try:
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_time = time.time()
            
            # Find closest existing track with enhanced matching
            min_distance = float('inf')
            closest_id = None
            
            for vehicle_id, track_info in self.vehicle_tracker.items():
                if current_time - track_info['last_seen'] > 15:  # 15 second timeout
                    continue
                
                track_center = track_info['center']
                distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
                
                # Enhanced matching with size consideration
                size_similarity = 1.0
                if 'dimensions' in track_info:
                    track_width = track_info['dimensions']['width']
                    track_height = track_info['dimensions']['height']
                    current_width = x2 - x1
                    current_height = y2 - y1
                    
                    width_ratio = min(track_width, current_width) / max(track_width, current_width) if max(track_width, current_width) > 0 else 1.0
                    height_ratio = min(track_height, current_height) / max(track_height, current_height) if max(track_height, current_height) > 0 else 1.0
                    size_similarity = (width_ratio + height_ratio) / 2
                
                # Combined score for better matching
                score = distance * (2 - size_similarity)  # Lower is better
                
                if score < min_distance and distance < 150 and size_similarity > 0.5:
                    min_distance = score
                    closest_id = vehicle_id
            
            if closest_id is not None:
                # Update existing track with enhanced info
                self.vehicle_tracker[closest_id].update({
                    'center': center,
                    'bbox': bbox,
                    'last_seen': current_time,
                    'dimensions': {'width': x2 - x1, 'height': y2 - y1},
                    'track_length': self.vehicle_tracker[closest_id].get('track_length', 0) + 1
                })
                return closest_id
            else:
                # Create new track with enhanced info
                self.track_id_counter += 1
                new_id = f"vehicle_{self.track_id_counter:04d}"
                self.vehicle_tracker[new_id] = {
                    'center': center,
                    'bbox': bbox,
                    'last_seen': current_time,
                    'first_seen': current_time,
                    'dimensions': {'width': x2 - x1, 'height': y2 - y1},
                    'track_length': 1
                }
                return new_id
                
        except Exception as e:
            logger.error(f"❌ Enhanced vehicle tracking error: {e}")
            return f"vehicle_{int(time.time() * 1000) % 10000:04d}"

    def get_vehicle_class_name(self, class_id):
        """Enhanced vehicle class name mapping"""
        class_names = {
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck'
        }
        return class_names.get(class_id, 'vehicle')

    def find_plate_in_car_enhanced(self, frame, car_bbox, plate_detections):
        """Enhanced plate-in-car matching with better accuracy"""
        car_x1, car_y1, car_x2, car_y2 = car_bbox
        
        best_plate = None
        best_score = 0
        
        for plate in plate_detections:
            plate_x1, plate_y1, plate_x2, plate_y2 = plate['bbox']
            
            # Enhanced center point check
            plate_center_x = (plate_x1 + plate_x2) / 2
            plate_center_y = (plate_y1 + plate_y2) / 2
            
            # Check if plate center is within car bounds
            if (car_x1 <= plate_center_x <= car_x2 and car_y1 <= plate_center_y <= car_y2):
                
                # Calculate enhanced overlap score
                overlap_x1 = max(car_x1, plate_x1)
                overlap_y1 = max(car_y1, plate_y1)
                overlap_x2 = min(car_x2, plate_x2)
                overlap_y2 = min(car_y2, plate_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    plate_area = (plate_x2 - plate_x1) * (plate_y2 - plate_y1)
                    car_area = (car_x2 - car_x1) * (car_y2 - car_y1)
                    
                    if plate_area > 0 and car_area > 0:
                        overlap_ratio = overlap_area / plate_area
                        size_ratio = plate_area / car_area
                        
                        # Enhanced scoring with multiple factors
                        score = overlap_ratio * 0.7 + min(1.0, size_ratio * 20) * 0.3
                        
                        # Boost score for plates with good confidence
                        score *= (0.8 + plate['confidence'] * 0.2)
                        
                        if score > best_score:
                            best_score = score
                            best_plate = plate
        
        return best_plate

    def extract_and_enhance_car_image(self, frame, car_bbox):
        """Enhanced car image extraction with better quality"""
        try:
            x1, y1, x2, y2 = car_bbox
            frame_h, frame_w = frame.shape[:2]
            
            # Enhanced expansion with better logic
            car_w = x2 - x1
            car_h = y2 - y1
            
            # Adaptive expansion based on car size
            expand_ratio = min(0.3, max(0.1, 200 / car_w if car_w > 0 else 0.1))  # Smaller cars get more expansion
            expand_w = int(car_w * expand_ratio)
            expand_h = int(car_h * expand_ratio)
            
            full_car_x1 = max(0, x1 - expand_w)
            full_car_y1 = max(0, y1 - expand_h)
            full_car_x2 = min(frame_w, x2 + expand_w)
            full_car_y2 = min(frame_h, y2 + expand_h)
            
            car_img = frame[full_car_y1:full_car_y2, full_car_x1:full_car_x2]
            
            # Enhanced image processing for quality
            if car_img.size > 0:
                h, w = car_img.shape[:2]
                
                # Enhanced resize logic
                target_width = 600  # Higher target for better quality
                if w > 0 and w < target_width:
                    scale = target_width / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    car_img = cv2.resize(car_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                elif w > 1200:  # Avoid too large images
                    scale = 1200 / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    car_img = cv2.resize(car_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Enhanced image enhancement
                # Noise reduction
                car_img = cv2.bilateralFilter(car_img, 9, 75, 75)
                
                # Contrast enhancement
                lab = cv2.cvtColor(car_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                car_img = cv2.merge([l, a, b])
                car_img = cv2.cvtColor(car_img, cv2.COLOR_LAB2BGR)
                
                # Sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
                car_img = cv2.filter2D(car_img, -1, kernel)
            
            return car_img, (full_car_x1, full_car_y1, full_car_x2, full_car_y2)
            
        except Exception as e:
            logger.error(f"❌ Enhanced car extraction error: {e}")
            return None, None

    def save_images_enhanced(self, frame, plate_img, plate_bbox, plate_text, 
                           confidence, car_info):
        """Enhanced image saving with improved duplicate handling and quality"""
        def _save_enhanced():
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                
                car_image_path = ""
                full_car_bbox = None
                vehicle_orientation = "unknown"
                orientation_confidence = 0.0
                entry_exit_status = "Vehicle Detected"
                movement_direction = "stationary"
                quality_score = 0.0
                
                if car_info:
                    # Extract enhanced full car image
                    full_car_img, full_car_bbox = self.extract_and_enhance_car_image(
                        frame, car_info['bbox'])
                    
                    if full_car_img is not None and full_car_img.size > 0:
                        # Enhanced orientation detection
                        vehicle_id = car_info.get('vehicle_id')
                        orientation_result = self.orientation_detector.detect_vehicle_orientation_and_movement(
                            full_car_img, car_info['bbox'], frame, vehicle_id)
                        
                        vehicle_orientation = orientation_result[0]
                        orientation_confidence = orientation_result[1]
                        movement_direction = orientation_result[2]
                        
                        # Get enhanced entry/exit status
                        entry_exit_status, status_confidence = self.orientation_detector.get_entry_exit_status(
                            vehicle_orientation, orientation_confidence, movement_direction)
                        
                        # Calculate quality score
                        quality_score = self._calculate_detection_quality(
                            confidence, orientation_confidence, car_info, plate_text)
                        
                        # Update statistics
                        with self.stats_lock:
                            if "Enter" in entry_exit_status:
                                self.detection_stats['entries'] += 1
                            elif "Exit" in entry_exit_status:
                                self.detection_stats['exits'] += 1
                            
                            if vehicle_orientation == 'front':
                                self.detection_stats['front_detections'] += 1
                            elif vehicle_orientation == 'back':
                                self.detection_stats['back_detections'] += 1
                            else:
                                self.detection_stats['unknown_orientation'] += 1
                        
                        # Save enhanced car image
                        orientation_prefix = vehicle_orientation.upper()[:3]
                        confidence_str = f"{int(orientation_confidence*100):02d}"
                        quality_str = f"{int(quality_score):02d}"
                        car_filename = f"ENHANCED_{orientation_prefix}_{confidence_str}Q{quality_str}_{plate_text}_{timestamp}.jpg"
                        car_filepath = os.path.join(CAR_IMAGES_DIR, car_filename)
                        
                        # Save with high quality
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95,
                                       cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                        success = cv2.imwrite(car_filepath, full_car_img, encode_params)
                        
                        if success:
                            car_image_path = car_filepath
                            with self.stats_lock:
                                self.detection_stats['car_images_captured'] += 1
                                self.detection_stats['full_cars_captured'] += 1
                
                # Save enhanced plate image
                plate_image_path = ""
                if plate_img is not None and plate_img.size > 0:
                    plate_filename = f"PLATE_{plate_text}_{timestamp}.jpg"
                    plate_filepath = os.path.join(PLATE_IMAGES_DIR, plate_filename)
                    
                    # Enhanced plate processing
                    enhanced_plate = self._enhance_plate_image(plate_img)
                    
                    # Save with high quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 98,
                                   cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                    success = cv2.imwrite(plate_filepath, enhanced_plate, encode_params)
                    
                    if success:
                        plate_image_path = plate_filepath
                        with self.stats_lock:
                            self.detection_stats['plate_images_captured'] += 1
                
                # Enhanced vehicle info
                vehicle_info = {
                    "type": car_info.get('class_name', 'vehicle') if car_info else 'unknown',
                    "car_confidence": float(car_info.get('confidence', 0.0)) if car_info else 0.0,
                    "full_car_captured": bool(car_image_path),
                    "plate_captured": bool(plate_image_path),
                    "orientation": vehicle_orientation,
                    "orientation_confidence": float(orientation_confidence),
                    "movement_direction": movement_direction,
                    "vehicle_id": car_info.get('vehicle_id') if car_info else None,
                    "track_length": self.vehicle_tracker.get(car_info.get('vehicle_id'), {}).get('track_length', 0) if car_info else 0,
                    "quality_score": float(quality_score),
                    "car_dimensions": car_info.get('dimensions', {}) if car_info else {}
                }
                
                # Enhanced bbox info
                bbox_info = {
                    "plate_bbox": plate_bbox,
                    "car_bbox": car_info['bbox'] if car_info else [],
                    "full_car_bbox": full_car_bbox if full_car_bbox else [],
                    "frame_dimensions": [frame.shape[1], frame.shape[0]],
                    "expansion_applied": True,
                    "enhancement_applied": True
                }
                
                # Enhanced additional data
                additional_data = {
                    "car_filename": os.path.basename(car_image_path) if car_image_path else "",
                    "plate_filename": os.path.basename(plate_image_path) if plate_image_path else "",
                    "enhancement_level": "high_quality",
                    "capture_method": "Enhanced Stream + No Frame Skipping + Smart Duplicate Cleanup",
                    "orientation_detection_applied": True,
                    "movement_tracking_applied": True,
                    "quality_enhancement_applied": True,
                    "processing_time": "enhanced",
                    "duplicate_prevention_active": True
                }
                
                # Save to MongoDB with enhanced data
                self.mongo_manager.save_vehicle_record_async(
                    plate_text, confidence, car_image_path, plate_image_path,
                    bbox_info, vehicle_info, additional_data, 
                    entry_exit_status, vehicle_orientation, orientation_confidence
                )
                
                # Store enhanced local record
                vehicle_record = {
                    'plate_number': plate_text,
                    'timestamp': datetime.now(),
                    'confidence': float(confidence),
                    'car_image_path': car_image_path,
                    'plate_image_path': plate_image_path,
                    'detection_id': f"{plate_text}_{timestamp}",
                    'bbox_info': bbox_info,
                    'vehicle_info': vehicle_info,
                    'full_car_captured': bool(car_image_path),
                    'plate_captured': bool(plate_image_path),
                    'entry_exit_status': entry_exit_status,
                    'vehicle_orientation': vehicle_orientation,
                    'orientation_confidence': float(orientation_confidence),
                    'movement_direction': movement_direction,
                    'is_entry': "Enter" in entry_exit_status,
                    'is_exit': "Exit" in entry_exit_status,
                    'quality_score': float(quality_score)
                }
                
                with self.vehicle_records_lock:
                    self.vehicle_records.append(vehicle_record)
                    if len(self.vehicle_records) > 100:
                        self.vehicle_records = self.vehicle_records[-75:]
                
                logger.info(f"🚗📸 ENHANCED CAPTURE: {plate_text} ({entry_exit_status}) [Q:{quality_score:.0f}, C:{orientation_confidence:.2f}]")
                
            except Exception as e:
                logger.error(f"❌ Enhanced save error: {e}")
        
        # Submit to enhanced thread pool
        self.image_processor_executor.submit(_save_enhanced)

    def _enhance_plate_image(self, plate_img):
        """Enhanced plate image processing for better quality"""
        try:
            if plate_img is None or plate_img.size == 0:
                return plate_img
            
            # Convert to grayscale if needed
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                # Convert back to BGR for consistency
                plate_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Enhanced resize with better quality
            h, w = plate_img.shape[:2]
            if w > 0 and w < 200:
                scale = 200 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Enhanced noise reduction
            plate_img = cv2.bilateralFilter(plate_img, 9, 75, 75)
            
            # Enhanced contrast and brightness
            lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            plate_img = cv2.merge([l, a, b])
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_LAB2BGR)
            
            # Enhanced sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            plate_img = cv2.filter2D(plate_img, -1, kernel)
            
            return plate_img
            
        except Exception as e:
            logger.error(f"❌ Enhanced plate enhancement error: {e}")
            return plate_img

    def _calculate_detection_quality(self, confidence, orientation_confidence, car_info, plate_text):
        """Calculate enhanced quality score for the detection"""
        try:
            score = 0.0
            
            # Plate confidence (0-25 points)
            if confidence:
                score += min(25, confidence * 25)
            
            # Orientation confidence (0-25 points)
            if orientation_confidence:
                score += min(25, orientation_confidence * 25)
            
            # Car detection quality (0-25 points)
            if car_info:
                car_conf = car_info.get('confidence', 0)
                score += min(25, car_conf * 25)
                
                # Bonus for good car dimensions
                dimensions = car_info.get('dimensions', {})
                if dimensions:
                    width = dimensions.get('width', 0)
                    height = dimensions.get('height', 0)
                    if width > 150 and height > 100:
                        score += 5
            
            # Plate text quality (0-25 points)
            if plate_text and len(plate_text) >= 5:
                score += min(25, len(plate_text) * 3)
                
                # Bonus for good plate patterns
                if self.ocr_engine.is_valid_license_plate(plate_text):
                    score += 5
            
            return min(100.0, round(score, 1))
            
        except Exception:
            return 0.0

    def is_recent_detection_enhanced(self, plate_text, bbox):
        """Enhanced duplicate detection with better accuracy"""
        current_time = time.time()
        
        # Enhanced detection key with better spatial grouping
        bbox_x = (bbox[0] + bbox[2]) // 2
        bbox_y = (bbox[1] + bbox[3]) // 2
        detection_key = f"{plate_text}_{bbox_x//30}_{bbox_y//30}"  # 30-pixel grouping
        
        # Enhanced cleanup of old detections
        cutoff_time = current_time - DUPLICATE_THRESHOLD_SECONDS
        self.recent_detections = deque([
            (key, timestamp) for key, timestamp in self.recent_detections
            if timestamp > cutoff_time
        ], maxlen=50)
        
        # Enhanced duplicate check
        for key, timestamp in self.recent_detections:
            if key == detection_key and current_time - timestamp < DUPLICATE_THRESHOLD_SECONDS:
                return True
        
        self.recent_detections.append((detection_key, current_time))
        return False

    def is_already_captured(self, plate_text):
        """Enhanced captured plate check with smart duplicate prevention"""
        if not plate_text or len(plate_text) < 4:
            return True
        
        plate_text = plate_text.upper().strip()
        
        with self.captured_plates_lock:
            # Check both permanent and session captures
            return plate_text in self.captured_plates or plate_text in self.session_plates

    def mark_as_captured(self, plate_text):
        """Enhanced mark as captured with better tracking"""
        if not plate_text or len(plate_text) < 4:
            return
        
        plate_text = plate_text.upper().strip()
        
        with self.captured_plates_lock:
            self.captured_plates.add(plate_text)
            self.session_plates.add(plate_text)

    def process_frame_enhanced(self, frame, fps=30):
        """Enhanced frame processing with NO frame skipping and better quality"""
        annotated_frame = frame.copy()
        
        # Enhanced processing decision - ALWAYS process for quality
        should_process = self.should_process_frame()
        should_ocr = self.should_run_ocr()
        
        if not should_process:
            # This should rarely happen since we're not skipping frames
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(annotated_frame, f"🔥 ENHANCED QUALITY STREAM - {timestamp} | {fps:.1f} FPS",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return annotated_frame

        try:
            # Enhanced parallel detection with better parameters
            detected_cars = self.detect_cars_enhanced(frame)
            
            # Enhanced plate detection with better accuracy
            plate_results = self.plate_model(frame,
                                           conf=self.plate_confidence_threshold,
                                           iou=self.iou_threshold,
                                           imgsz=640,
                                           verbose=False,
                                           half=True if torch.cuda.is_available() else False)

        except Exception as e:
            logger.error(f"❌ Enhanced detection failed: {e}")
            cv2.putText(annotated_frame, "❌ DETECTION ERROR - RAW FEED", (100, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame

        # Enhanced frame annotation
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"🔥 ENHANCED QUALITY + NO SKIP - {timestamp} | {fps:.1f} FPS",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Enhanced plate detection extraction
        plate_detections = []

        for result in plate_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # Enhanced size filtering
                    width = x2 - x1
                    height = y2 - y1
                    
                    if (width >= 40 and height >= 12 and  # Larger minimum size for quality
                        width <= frame.shape[1] * 0.8 and height <= frame.shape[0] * 0.4):  # Not too large
                        
                        plate_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'dimensions': {'width': width, 'height': height}
                        })

        # Enhanced car visualization with tracking info
        for car in detected_cars:
            car_x1, car_y1, car_x2, car_y2 = car['bbox']
            
            # Enhanced car rectangle with quality indicators
            quality_color = (255, 0, 0) if car['confidence'] > 0.7 else (0, 255, 0)
            cv2.rectangle(annotated_frame, (car_x1, car_y1), (car_x2, car_y2), quality_color, 2)
            
            # Enhanced car labels
            car_label = f"{car['class_name'].upper()} {car['confidence']:.2f}"
            track_label = f"ID:{car.get('vehicle_id', 'N/A')}"
            quality_label = f"Track:{car.get('dimensions', {}).get('width', 0)}x{car.get('dimensions', {}).get('height', 0)}"
            
            cv2.putText(annotated_frame, car_label, (car_x1, car_y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
            cv2.putText(annotated_frame, track_label, (car_x1, car_y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
            cv2.putText(annotated_frame, quality_label, (car_x1, car_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        new_captures = 0

        # Enhanced car-plate association processing
        for car in detected_cars:
            associated_plate = self.find_plate_in_car_enhanced(frame, car['bbox'], plate_detections)
            
            if associated_plate:
                x1, y1, x2, y2 = associated_plate['bbox']
                confidence = associated_plate['confidence']
                plate_img = frame[y1:y2, x1:x2]
                
                plate_text = ""
                # Enhanced OCR - run more frequently for quality
                if should_ocr or confidence > 0.5:
                    plate_text = self.ocr_engine.enhanced_ocr(plate_img)

                # Enhanced duplicate and validity checking
                if (plate_text and
                    self.ocr_engine.is_valid_license_plate(plate_text) and
                    not self.is_already_captured(plate_text) and
                    not self.is_recent_detection_enhanced(plate_text, associated_plate['bbox'])):

                    self.mark_as_captured(plate_text)
                    self.save_images_enhanced(frame, plate_img, associated_plate['bbox'],
                                            plate_text, confidence * 100, car)
                    self.update_detection_stats_enhanced(plate_text)
                    new_captures += 1
                    
                    logger.info(f"🔥🚗 ENHANCED CAPTURE: {plate_text} ({car['class_name']}) [Track: {car.get('vehicle_id', 'N/A')}]")
                    
                    # Enhanced capture visualization
                    cv2.rectangle(annotated_frame, (car['bbox'][0] - 3, car['bbox'][1] - 3), 
                                (car['bbox'][2] + 3, car['bbox'][3] + 3), (0, 255, 255), 4)
                    cv2.putText(annotated_frame, "🔥 ENHANCED CAPTURED", 
                               (car['bbox'][0], car['bbox'][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Enhanced plate visualization
                display_text = plate_text if plate_text else f"OCR... {confidence:.2f}"
                plate_color = (0, 255, 0) if plate_text else (0, 255, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), plate_color, 2)
                cv2.putText(annotated_frame, display_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Enhanced standalone plate processing
        for plate in plate_detections:
            # Check if plate is already associated with a car
            already_processed = any(self.find_plate_in_car_enhanced(frame, car['bbox'], [plate]) 
                                  for car in detected_cars)
            
            if not already_processed:
                x1, y1, x2, y2 = plate['bbox']
                confidence = plate['confidence']
                plate_img = frame[y1:y2, x1:x2]

                plate_text = ""
                if should_ocr or confidence > 0.5:
                    plate_text = self.ocr_engine.enhanced_ocr(plate_img)

                # Enhanced standalone plate processing
                if (plate_text and
                    self.ocr_engine.is_valid_license_plate(plate_text) and
                    not self.is_already_captured(plate_text) and
                    not self.is_recent_detection_enhanced(plate_text, plate['bbox'])):
                    
                    self.mark_as_captured(plate_text)
                    self.save_images_enhanced(frame, plate_img, plate['bbox'],
                                            plate_text, confidence * 100, None)
                    self.update_detection_stats_enhanced(plate_text)
                    new_captures += 1
                    
                    logger.info(f"🔥📄 ENHANCED STANDALONE: {plate_text}")
                    cv2.rectangle(annotated_frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 0), 3)

                # Enhanced standalone plate visualization
                display_text = plate_text if plate_text else f"Analyzing... {confidence:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, display_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Enhanced status display
        with self.stats_lock:
            total_vehicles = len(self.captured_plates)
            session_count = len(self.session_plates)
            full_cars = self.detection_stats.get('full_cars_captured', 0)
            entries = self.detection_stats.get('entries', 0)
            exits = self.detection_stats.get('exits', 0)
            front_detections = self.detection_stats.get('front_detections', 0)
            back_detections = self.detection_stats.get('back_detections', 0)
            unknown_orientation = self.detection_stats.get('unknown_orientation', 0)
            duplicate_cleanups = self.detection_stats.get('duplicate_cleanups', 0)
            mongodb_status = "🟢" if self.mongo_manager.is_connected else "🔴"

        # Enhanced information display
        cv2.putText(annotated_frame,
                   f"📊 Session: {session_count} | Total: {total_vehicles} | Cars: {full_cars} | DB {mongodb_status} | Cleanups: {duplicate_cleanups}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.putText(annotated_frame,
                   f"🔥 🟢Entry: {entries} | 🔴Exit: {exits} | 👁️Front: {front_detections} | 🔙Back: {back_detections} | ❓Unknown: {unknown_orientation}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.putText(annotated_frame,
                   f"🎯 Detected: {len(detected_cars)} Cars, {len(plate_detections)} Plates | New: {new_captures} | Tracks: {len(self.vehicle_tracker)}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Enhanced processing status
        ocr_status = "🔍 ENHANCED OCR" if should_ocr else "⏸️ OCR"
        processing_status = "🔥 NO SKIP" if should_process else "⏯️ SKIP"
        cv2.putText(annotated_frame, f"Frame #{self.frame_count} | {processing_status} | {ocr_status}",
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def update_detection_stats_enhanced(self, plate_text):
        """Enhanced statistics update with better tracking"""
        with self.stats_lock:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.detection_stats['total_detections'] += 1
            self.detection_stats['unique_plates'] = len(self.captured_plates)
            self.detection_stats['last_detection'] = current_time
            
            # Enhanced plate list management
            found = False
            for plate_entry in self.detection_stats['plates_list']:
                if plate_entry['plate'] == plate_text:
                    plate_entry['count'] += 1
                    plate_entry['last_seen'] = current_time
                    found = True
                    break
            
            if not found:
                self.detection_stats['plates_list'].append({
                    'plate': plate_text,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'count': 1,
                    'confidence': 95.0,
                    'source': 'Enhanced Stream + No Frame Skipping + Smart Duplicate Cleanup',
                    'quality_score': 85.0
                })
                
                # Enhanced list management with better size control
                if len(self.detection_stats['plates_list']) > 150:
                    # Sort by last_seen and keep most recent
                    self.detection_stats['plates_list'].sort(key=lambda x: x['last_seen'], reverse=True)
                    self.detection_stats['plates_list'] = self.detection_stats['plates_list'][:100]

    def get_enhanced_stats(self):
        """Get comprehensive enhanced statistics with performance data"""
        with self.stats_lock:
            stats = self.detection_stats.copy()
            
            # Get enhanced MongoDB stats
            mongo_stats = self.mongo_manager.get_enhanced_stats()
            
            # Enhanced statistics integration
            stats.update({
                'session_plates_count': len(self.session_plates),
                'total_known_plates': len(self.captured_plates),
                'mongodb_total_plates': mongo_stats.get('total_vehicles', 0),
                'processing_interval': self.processing_interval,
                'ocr_interval': self.ocr_interval,
                'frame_count': self.frame_count,
                'active_vehicle_tracks': len(self.vehicle_tracker),
                'optimization_level': 'Enhanced Quality + No Frame Skipping + Smart Duplicate Cleanup',
                'mongodb_status': mongo_stats.get('database_status', 'Unknown'),
                'full_cars_captured': mongo_stats.get('full_cars_captured', 0),
                'plate_images_captured': mongo_stats.get('plate_images_captured', 0),
                'mongodb_entries': mongo_stats.get('entries', 0),
                'mongodb_exits': mongo_stats.get('exits', 0),
                'mongodb_front_detections': mongo_stats.get('front_detections', 0),
                'mongodb_back_detections': mongo_stats.get('back_detections', 0),
                'mongodb_unknown_orientation': mongo_stats.get('unknown_orientation', 0),
                'avg_confidence': mongo_stats.get('avg_confidence', 0.0),
                'avg_orientation_confidence': mongo_stats.get('avg_orientation_confidence', 0.0),
                'avg_quality_score': mongo_stats.get('avg_quality_score', 0.0),
                'color_preservation': True,
                'enhanced_duplicate_cleanup_enabled': True,
                'auto_cleanup_enabled': AUTO_CLEANUP_ENABLED,
                'performance_mode': 'enhanced_quality',
                'vehicle_tracking_enabled': True,
                'improved_entry_exit_detection': True,
                'frame_skipping_disabled': True,
                'quality_enhancement_enabled': True,
                'smart_duplicate_prevention': True,
                'enhanced_ocr_enabled': True,
                'duplicate_threshold_seconds': DUPLICATE_THRESHOLD_SECONDS
            })
            
            return stats

    def get_vehicle_records(self, limit=50, query=None):
        """Get enhanced vehicle records with comprehensive information"""
        try:
            # Get enhanced MongoDB records
            mongo_records = self.mongo_manager.get_vehicle_records(limit, query)
            
            # Get local records
            with self.vehicle_records_lock:
                local_records = self.vehicle_records[-limit:] if not query else []
            
            # Enhanced record combination with deduplication
            all_records = mongo_records + local_records
            seen_ids = set()
            unique_records = []
            
            for record in all_records:
                record_id = record.get('detection_id', '')
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    unique_records.append(record)
            
            # Enhanced sorting by timestamp and quality
            unique_records.sort(key=lambda x: (
                x.get('timestamp', datetime.min),
                x.get('quality_score', 0)
            ), reverse=True)
            
            return unique_records[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting enhanced vehicle records: {e}")
            return []

    def cleanup(self):
        """Enhanced cleanup with comprehensive shutdown"""
        logger.info("🧹 Cleaning up enhanced system...")
        
        try:
            # Enhanced thread pool shutdown
            if hasattr(self, 'image_processor_executor'):
                logger.info("🔄 Shutting down image processor...")
                self.image_processor_executor.shutdown(wait=True, timeout=10)
            
            # Enhanced MongoDB cleanup
            if self.mongo_manager:
                logger.info("🔄 Closing MongoDB connection...")
                self.mongo_manager.close_connection()
            
            # Enhanced cache cleanup
            with self.ocr_engine.cache_lock:
                self.ocr_engine.preprocessing_cache.clear()
            
            # Enhanced memory cleanup
            self.vehicle_tracker.clear()
            self.recent_detections.clear()
            
            with self.vehicle_records_lock:
                self.vehicle_records.clear()
            
            logger.info("✅ Enhanced cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Enhanced cleanup error: {e}")


# Global instances
reader = None
enhanced_processor = None

def generate_enhanced_frames():
    """Enhanced frame generator with no skipping and better quality"""
    global reader, enhanced_processor
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    fps_calculation_interval = 30  # More stable FPS calculation
    error_count = 0
    max_errors = 50
    
    logger.info("🔥 Starting enhanced frame generator with NO FRAME SKIPPING")
    
    while True:
        try:
            if not reader or not reader.is_healthy():
                logger.warning("⚠️ Stream unhealthy, showing enhanced error frame")
                error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(error_frame, "🚨 ENHANCED STREAM CONNECTION ISSUE", (350, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(error_frame, "Enhanced reconnection in progress...", (450, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1.0)
                continue
            
            frame = reader.get_frame()
            if frame is None:
                error_count += 1
                if error_count > max_errors:
                    logger.error("❌ Too many frame read errors")
                    break
                time.sleep(0.02)
                continue
            
            error_count = 0
            frame_count += 1
            fps_counter += 1
            
            # Enhanced FPS calculation
            if fps_counter >= fps_calculation_interval:
                current_time = time.time()
                time_diff = current_time - fps_start_time
                if time_diff > 0:
                    current_fps = fps_counter / time_diff
                fps_counter = 0
                fps_start_time = current_time
            
            # Enhanced frame processing - NO SKIPPING
            try:
                processed_frame = enhanced_processor.process_frame_enhanced(frame, current_fps)
            except Exception as e:
                logger.error(f"❌ Enhanced processing error: {e}")
                processed_frame = frame.copy()
                cv2.putText(processed_frame, "⚠️ ENHANCED PROCESSING ERROR - RAW FEED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Enhanced encoding with better quality
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 88,  # High quality
                               cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                               cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
                ret, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    logger.warning("⚠️ Enhanced frame encoding failed")
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"❌ Enhanced encoding error: {e}")
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"❌ Enhanced stream generation error: {e}")
            time.sleep(0.05)


# Enhanced Flask routes with better error handling
@app.route('/')
def index():
    try:
        return send_from_directory(DASHBOARD_DIR, 'dashboard.html')
    except Exception as e:
        logger.error(f"❌ Error serving dashboard: {e}")
        return f"Dashboard error: {e}", 500

@app.route('/<path:filename>')
def serve_static_files(filename):
    try:
        return send_from_directory(DASHBOARD_DIR, filename)
    except Exception as e:
        logger.error(f"❌ Error serving static file {filename}: {e}")
        return f"File error: {e}", 404

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_enhanced_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame',
                       headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                               'Pragma': 'no-cache',
                               'Expires': '0'})
    except Exception as e:
        logger.error(f"❌ Video feed error: {e}")
        return f"Video feed error: {e}", 500

@app.route('/api/enhanced_stats')
def get_enhanced_stats_route():
    """Get comprehensive enhanced statistics"""
    global enhanced_processor
    try:
        if enhanced_processor:
            stats = enhanced_processor.get_enhanced_stats()
            
            # Add enhanced stream health info
            if reader:
                stream_stats = reader.get_stats()
                stats.update({
                    'stream_connected': stream_stats['connected'],
                    'stream_healthy': stream_stats['healthy'],
                    'stream_reconnects': stream_stats['reconnect_attempts'],
                    'stream_current_fps': stream_stats['current_fps'],
                    'stream_frames_dropped': stream_stats['frames_dropped'],
                    'stream_buffer_size': stream_stats.get('buffer_size', 0),
                    'stream_target_fps': stream_stats.get('target_fps', TARGET_FPS)
                })
            return jsonify(stats)
        
        return jsonify({
            'total_detections': 0,
            'unique_plates': 0,
            'performance_mode': 'enhanced_quality',
            'stream_status': 'Enhanced processor not available',
            'error': 'System not initialized'
        })
    except Exception as e:
        logger.error(f"❌ Enhanced stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicle_records')
def get_vehicle_records_route():
    """Get enhanced vehicle records"""
    global enhanced_processor
    try:
        if enhanced_processor:
            limit = int(request.args.get('limit', 50))
            query = request.args.get('query', '').strip()
            
            records = enhanced_processor.get_vehicle_records(limit, query if query else None)
            return jsonify({
                'records': records,
                'total': len(records),
                'source': 'Enhanced System + No Frame Skipping + Smart Duplicate Cleanup',
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'limit': limit
            })
        
        return jsonify({
            'records': [], 
            'total': 0, 
            'error': 'Enhanced processor not available',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Vehicle records error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup_duplicates', methods=['POST'])
def cleanup_duplicates_route():
    """Enhanced duplicate cleanup with better error handling"""
    global enhanced_processor
    try:
        if not enhanced_processor:
            return jsonify({'error': 'Enhanced processor not available'}), 500
        
        data = request.get_json() or {}
        plate_number = data.get('plate_number', '').strip()
        
        logger.info(f"🧹 API cleanup request for: {'ALL PLATES' if not plate_number else plate_number}")
        
        if plate_number:
            result = enhanced_processor.duplicate_manager.cleanup_duplicates(plate_number)
        else:
            result = enhanced_processor.duplicate_manager.cleanup_duplicates()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Duplicate cleanup API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup_orphaned_images', methods=['POST'])
def cleanup_orphaned_images_route():
    """Enhanced orphaned image cleanup"""
    global enhanced_processor
    try:
        if not enhanced_processor:
            return jsonify({'error': 'Enhanced processor not available'}), 500
        
        logger.info("🧹 API orphaned image cleanup request")
        result = enhanced_processor.duplicate_manager.cleanup_orphaned_images()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Orphaned image cleanup API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/duplicate_statistics')
def get_duplicate_statistics_route():
    """Get enhanced duplicate statistics"""
    global enhanced_processor
    try:
        if not enhanced_processor:
            return jsonify({'error': 'Enhanced processor not available'}), 500
        
        result = enhanced_processor.duplicate_manager.get_duplicate_statistics()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Duplicate statistics API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_vehicles', methods=['POST'])
def search_vehicles_route():
    """Enhanced vehicle search"""
    global enhanced_processor
    try:
        if not enhanced_processor:
            return jsonify({'error': 'Enhanced processor not available'}), 500
        
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        limit = data.get('limit', 100)
        
        records = enhanced_processor.get_vehicle_records(limit, query if query else None)
        
        return jsonify({
            'records': records,
            'total': len(records),
            'query': query,
            'source': 'Enhanced Search + No Frame Skipping + Smart Duplicate Cleanup'
        })
        
    except Exception as e:
        logger.error(f"❌ Vehicle search API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Enhanced image serving with better security"""
    try:
        import urllib.parse
        decoded_path = urllib.parse.unquote(image_path)
        
        # Enhanced security check
        allowed_dirs = [CAR_IMAGES_DIR, PLATE_IMAGES_DIR, SCREENSHOTS_DIR]
        full_path = os.path.abspath(decoded_path)
        
        is_allowed = False
        for allowed_dir in allowed_dirs:
            allowed_dir_abs = os.path.abspath(allowed_dir)
            try:
                # Check if the requested path is a subpath of an allowed directory
                if os.path.commonpath([full_path, allowed_dir_abs]) == allowed_dir_abs:
                    is_allowed = True
                    break
            except ValueError:
                continue
        
        if not is_allowed:
            logger.warning(f"❌ Unauthorized image access attempt: {decoded_path}")
            return jsonify({'error': 'Access denied'}), 403
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return send_file(full_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        logger.error(f"❌ Error serving image: {e}")
        return jsonify({'error': 'Failed to serve image'}), 500

@app.route('/api/stream_health')
def get_stream_health_route():
    """Get enhanced stream health information"""
    global reader
    try:
        if reader:
            stats = reader.get_stats()
            return jsonify({
                'stream_health': stats,
                'status': 'healthy' if stats['healthy'] else 'unhealthy',
                'performance': 'enhanced_quality',
                'timestamp': datetime.now().isoformat(),
                'frame_skipping': False,
                'quality_mode': 'high'
            })
        return jsonify({'error': 'Enhanced stream reader not available'})
    except Exception as e:
        logger.error(f"❌ Stream health API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_info')
def get_system_info_route():
    """Get enhanced comprehensive system information"""
    global enhanced_processor
    
    try:
        system_info = {
            'system_name': 'Enhanced Vehicle Detection with Smart Duplicate Cleanup',
            'version': '6.0_Enhanced',
            'performance_mode': 'enhanced_quality',
            'frame_skipping_enabled': False,
            'features': {
                'enhanced_processing': True,
                'ultra_stable_rtsp_stream': True,
                'improved_entry_exit_detection': True,
                'vehicle_tracking': True,
                'multi_modal_orientation_analysis': True,
                'temporal_consistency': True,
                'movement_vector_analysis': True,
                'smart_duplicate_cleanup': True,
                'enhanced_duplicate_prevention': True,
                'auto_cleanup': AUTO_CLEANUP_ENABLED,
                'mongodb_storage': True,
                'color_preservation': True,
                'gpu_acceleration': torch.cuda.is_available(),
                'enhanced_ocr': True,
                'quality_enhancement': True,
                'no_frame_skipping': True,
                'smart_duplicate_detection': True,
                'comprehensive_image_enhancement': True
            },
            'performance_optimizations': {
                'processing_interval': PROCESSING_INTERVAL,
                'ocr_interval': OCR_PROCESSING_INTERVAL,
                'target_fps': TARGET_FPS,
                'duplicate_threshold_seconds': DUPLICATE_THRESHOLD_SECONDS,
                'enhanced_buffering': True,
                'quality_encoding': True,
                'concurrent_processing': True,
                'smart_memory_management': True
            }
        }
        
        if enhanced_processor:
            system_info.update({
                'processor_status': 'Enhanced Active',
                'mongodb_connected': enhanced_processor.mongo_manager.is_connected,
                'active_vehicle_tracks': len(enhanced_processor.vehicle_tracker),
                'duplicate_manager_active': True,
                'session_plates': len(enhanced_processor.session_plates),
                'total_plates': len(enhanced_processor.captured_plates)
            })
            
            if reader:
                stream_stats = reader.get_stats()
                system_info.update({
                    'stream_status': 'healthy' if stream_stats['healthy'] else 'unhealthy',
                    'stream_fps': stream_stats['current_fps'],
                    'frames_dropped': stream_stats['frames_dropped'],
                    'stream_buffer_size': stream_stats.get('buffer_size', 0)
                })
        
        return jsonify(system_info)
    except Exception as e:
        logger.error(f"❌ System info API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset_session')
def reset_session_route():
    """Enhanced session reset with comprehensive cleanup"""
    global enhanced_processor
    try:
        if enhanced_processor:
            with enhanced_processor.captured_plates_lock:
                old_count = len(enhanced_processor.session_plates)
                enhanced_processor.session_plates.clear()
                enhanced_processor.recent_detections.clear()
            
            with enhanced_processor.vehicle_records_lock:
                enhanced_processor.vehicle_records.clear()
            
            # Enhanced tracking cleanup
            enhanced_processor.vehicle_tracker.clear()
            enhanced_processor.track_id_counter = 0
            
            # Enhanced statistics reset
            with enhanced_processor.stats_lock:
                enhanced_processor.detection_stats['entries'] = 0
                enhanced_processor.detection_stats['exits'] = 0
                enhanced_processor.detection_stats['front_detections'] = 0
                enhanced_processor.detection_stats['back_detections'] = 0
                enhanced_processor.detection_stats['unknown_orientation'] = 0
                enhanced_processor.detection_stats['session_start'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return jsonify({
                'status': 'success',
                'message': f'Enhanced session reset - {old_count} session plates cleared',
                'features': 'Enhanced Quality Stream + No Frame Skipping + Smart Duplicate Cleanup Active',
                'old_session_count': old_count,
                'reset_timestamp': datetime.now().isoformat()
            })
        
        return jsonify({'status': 'error', 'message': 'Enhanced processor not available'})
    except Exception as e:
        logger.error(f"❌ Session reset API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def main():
    global reader, enhanced_processor
    
    try:
        logger.info("="*100)
        logger.info("🔥🚗 ENHANCED VEHICLE DETECTION SYSTEM")
        logger.info("🎯 Quality Stream + No Frame Skipping + Smart Duplicate Cleanup")
        logger.info("="*100)
        logger.info("🚀 ENHANCED FEATURES:")
        logger.info("   🔥 Enhanced quality RTSP stream processing (NO frame skipping)")
        logger.info("   ⚡ Lightning-fast automatic reconnection with better stability")
        logger.info("   📊 Real-time enhanced stream performance monitoring")
        logger.info("   🚗 Complete enhanced vehicle image capture with quality processing")
        logger.info("   🏷️ Enhanced license plate capture with advanced OCR")
        logger.info("   🎯 Improved entry/exit detection with multi-modal analysis")
        logger.info("   📍 Advanced vehicle tracking with enhanced movement vector analysis")
        logger.info("   🧹 Smart duplicate record cleanup with image management")
        logger.info("   💾 Asynchronous high-performance MongoDB storage")
        logger.info("="*100)
        
        logger.info("🔧 Initializing system components...")
        
        # Initialize the RTSP reader
        reader = UltraStableRTSPReader(RTSP_STREAM)
        
        # Initialize the car detection and processing engine
        enhanced_processor = EnhancedCarDetector(MODEL_PATH, YOLO_CAR_MODEL)
        
        logger.info("✅ System initialized successfully. Starting web server...")
        
        # Start the Flask web server
        # Use '0.0.0.0' to make it accessible on the network
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

    except Exception as e:
        logger.critical(f"💥 CRITICAL ERROR during initialization: {e}", exc_info=True)
    finally:
        logger.info("🛑 System shutting down...")
        if reader:
            reader.release()
        if enhanced_processor:
            enhanced_processor.cleanup()
        logger.info("👋 Goodbye!")


if __name__ == '__main__':
    main()