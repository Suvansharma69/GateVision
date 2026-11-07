# Enhanced Car Detection System with Optimized Stream and Improved Entry/Exit Detection
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
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Optimized Performance Configuration for Speed
MAX_QUEUE_SIZE = 3
PROCESSING_INTERVAL = 1  # Process every frame for smooth tracking
OCR_PROCESSING_INTERVAL = 2
FRAME_SKIP_THRESHOLD = 1
MEMORY_CLEANUP_INTERVAL = 100

# Stream optimization settings
RTSP_RECONNECT_TIMEOUT = 3
RTSP_READ_TIMEOUT = 0.5
MAX_CONSECUTIVE_FAILURES = 5
STREAM_BUFFER_SIZE = 1
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
VEHICLE_TRACKING_HISTORY = 20
MOVEMENT_THRESHOLD = 50
ORIENTATION_CONFIDENCE_THRESHOLD = 0.7
TEMPORAL_CONSISTENCY_FRAMES = 5
DIRECTION_ANALYSIS_FRAMES = 8

# Duplicate cleanup settings
DUPLICATE_CLEANUP_INTERVAL = 300
AUTO_CLEANUP_ENABLED = True

# Ensure directories exist
for directory in [SCREENSHOTS_DIR, CAR_IMAGES_DIR, PLATE_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask app
app = Flask(__name__)

class DuplicateCleanupManager:
    """Manager for cleaning up duplicate vehicle records and images"""
    def __init__(self, mongo_manager):
        self.mongo_manager = mongo_manager
        self.cleanup_lock = RLock()
        self.last_cleanup = time.time()
        self.cleanup_stats = {
            'total_cleanups': 0,
            'duplicates_removed': 0,
            'images_deleted': 0,
            'last_cleanup_time': None,
            'storage_freed_mb': 0
        }

    def cleanup_duplicates(self, plate_number=None):
        """Clean up duplicate records for a specific plate or all plates"""
        try:
            with self.cleanup_lock:
                if not self.mongo_manager.is_connected:
                    logger.warning("MongoDB not connected, skipping duplicate cleanup")
                    return {'status': 'error', 'message': 'Database not connected'}

                query = {}
                if plate_number:
                    query = {"plate_number": plate_number.upper().strip()}

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
                                "detection_id": "$detection_id"
                            }
                        }
                    }},
                    {"$match": {"count": {"$gt": 1}}}
                ]

                duplicates = list(self.mongo_manager.collection.aggregate(pipeline))

                total_removed = 0
                images_deleted = 0
                storage_freed = 0

                for duplicate_group in duplicates:
                    plate_num = duplicate_group['_id']
                    records = duplicate_group['records']

                    records.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                    records_to_delete = records[1:]

                    logger.info(f"🔄 Cleaning duplicates for plate {plate_num}: {len(records_to_delete)} duplicates found")

                    for record in records_to_delete:
                        try:
                            car_image_path = record.get('car_image_path', '')
                            plate_image_path = record.get('plate_image_path', '')

                            deleted_files = 0
                            file_sizes = 0

                            if car_image_path and os.path.exists(car_image_path):
                                try:
                                    file_size = os.path.getsize(car_image_path)
                                    os.remove(car_image_path)
                                    deleted_files += 1
                                    file_sizes += file_size
                                except Exception as e:
                                    logger.error(f"Error deleting car image {car_image_path}: {e}")

                            if plate_image_path and os.path.exists(plate_image_path):
                                try:
                                    file_size = os.path.getsize(plate_image_path)
                                    os.remove(plate_image_path)
                                    deleted_files += 1
                                    file_sizes += file_size
                                except Exception as e:
                                    logger.error(f"Error deleting plate image {plate_image_path}: {e}")

                            self.mongo_manager.collection.delete_one({"_id": record['id']})

                            total_removed += 1
                            images_deleted += deleted_files
                            storage_freed += file_sizes

                        except Exception as e:
                            logger.error(f"Error deleting duplicate record: {e}")

                self.cleanup_stats['total_cleanups'] += 1
                self.cleanup_stats['duplicates_removed'] += total_removed
                self.cleanup_stats['images_deleted'] += images_deleted
                self.cleanup_stats['storage_freed_mb'] += storage_freed / (1024 * 1024)
                self.cleanup_stats['last_cleanup_time'] = datetime.now()
                self.last_cleanup = time.time()

                result = {
                    'status': 'success',
                    'duplicates_processed': len(duplicates),
                    'records_removed': total_removed,
                    'images_deleted': images_deleted,
                    'storage_freed_mb': round(storage_freed / (1024 * 1024), 2),
                    'plate_number': plate_number or 'all plates'
                }

                logger.info(f"🧹 Duplicate cleanup completed: {total_removed} records removed")
                return result

        except Exception as e:
            logger.error(f"Duplicate cleanup error: {e}")
            return {'status': 'error', 'message': str(e)}

    def cleanup_orphaned_images(self):
        """Clean up image files that are no longer referenced in the database"""
        try:
            with self.cleanup_lock:
                if not self.mongo_manager.is_connected:
                    logger.warning("MongoDB not connected, skipping orphan image cleanup")
                    return {'status': 'error', 'message': 'Database not connected'}

                logger.info("🧹 Starting orphaned image cleanup...")
                
                # 1. Get all image paths from the database
                db_images = set()
                records = self.mongo_manager.collection.find({}, {"car_image_path": 1, "plate_image_path": 1})
                for record in records:
                    if record.get("car_image_path"):
                        db_images.add(record["car_image_path"])
                    if record.get("plate_image_path"):
                        db_images.add(record["plate_image_path"])
                
                logger.info(f"🔍 Found {len(db_images)} image references in the database.")

                # 2. Get all image paths from the filesystem
                disk_images = set()
                image_dirs = [CAR_IMAGES_DIR, PLATE_IMAGES_DIR]
                for directory in image_dirs:
                    if not os.path.exists(directory):
                        continue
                    for filename in os.listdir(directory):
                        full_path = os.path.join(directory, filename)
                        if os.path.isfile(full_path):
                            disk_images.add(full_path)

                logger.info(f"📁 Found {len(disk_images)} image files on disk.")

                # 3. Find and delete orphaned images
                orphaned_images = disk_images - db_images
                deleted_count = 0
                total_size_freed = 0

                logger.info(f"🗑️ Found {len(orphaned_images)} orphaned images to delete.")

                for image_path in orphaned_images:
                    try:
                        file_size = os.path.getsize(image_path)
                        os.remove(image_path)
                        deleted_count += 1
                        total_size_freed += file_size
                    except OSError as e:
                        logger.error(f"Error deleting orphaned file {image_path}: {e}")
                
                storage_freed_mb = round(total_size_freed / (1024 * 1024), 2)
                self.cleanup_stats['images_deleted'] += deleted_count
                self.cleanup_stats['storage_freed_mb'] += storage_freed_mb
                
                result = {
                    'status': 'success',
                    'orphaned_images_found': len(orphaned_images),
                    'images_deleted': deleted_count,
                    'storage_freed_mb': storage_freed_mb
                }
                
                logger.info(f"✅ Orphaned image cleanup completed. Deleted {deleted_count} files, freeing {storage_freed_mb} MB.")
                return result

        except Exception as e:
            logger.error(f"Orphaned image cleanup error: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_duplicate_statistics(self):
        """Get statistics about duplicate records"""
        try:
            if not self.mongo_manager.is_connected:
                return {'status': 'error', 'message': 'Database not connected'}

            pipeline = [
                {"$group": {
                    "_id": "$plate_number",
                    "count": {"$sum": 1},
                    "latest_timestamp": {"$max": "$timestamp"},
                    "earliest_timestamp": {"$min": "$timestamp"}
                }},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}}
            ]

            duplicates = list(self.mongo_manager.collection.aggregate(pipeline))
            
            total_duplicates = len(duplicates)
            total_duplicate_records = sum(d['count'] - 1 for d in duplicates)
            top_duplicates = duplicates[:10]

            return {
                'status': 'success',
                'plates_with_duplicates': total_duplicates,
                'total_duplicate_records': total_duplicate_records,
                'top_duplicates': top_duplicates,
                'cleanup_stats': self.cleanup_stats
            }

        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {'status': 'error', 'message': str(e)}

    def should_run_auto_cleanup(self):
        """Check if auto cleanup should run"""
        return (AUTO_CLEANUP_ENABLED and 
                time.time() - self.last_cleanup > DUPLICATE_CLEANUP_INTERVAL)

    def run_auto_cleanup(self):
        """Run automatic cleanup in background"""
        def _cleanup():
            try:
                logger.info("🔄 Starting automatic duplicate cleanup...")
                result = self.cleanup_duplicates()
                logger.info(f"✅ Auto cleanup completed: {result}")
            except Exception as e:
                logger.error(f"Auto cleanup error: {e}")
        
        threading.Thread(target=_cleanup, daemon=True).start()


class ImprovedVehicleOrientationDetector:
    """Significantly improved vehicle orientation detector with multi-modal analysis"""
    def __init__(self):
        self.vehicle_tracks = {}
        self.movement_history = deque(maxlen=VEHICLE_TRACKING_HISTORY)
        self.direction_vectors = deque(maxlen=DIRECTION_ANALYSIS_FRAMES)
        
        # Enhanced feature extractors
        self.edge_detector = cv2.createLineSegmentDetector()
        
        logger.info("✅ Improved vehicle orientation detector initialized")
    
    def detect_vehicle_orientation_and_movement(self, vehicle_img, vehicle_bbox, full_frame, vehicle_id=None):
        """
        Enhanced orientation detection with movement tracking and temporal consistency
        Returns: ('front'/'back'/'unknown', confidence, movement_direction)
        """
        try:
            if vehicle_img is None or vehicle_img.size == 0:
                return 'unknown', 0.0, 'stationary'
            
            # Multi-modal analysis
            orientation_scores = {}
            confidence_scores = {}
            
            # Method 1: Advanced plate position analysis
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
            
            # Weighted decision making
            front_score = 0.0
            back_score = 0.0
            total_weight = 0.0
            
            weights = {
                'plate': 2.5,    # Highest weight for plate analysis
                'lights': 2.0,   # High weight for light detection
                'geometry': 1.5, # Medium-high weight for geometry
                'edges': 1.2,    # Medium weight for edges
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
            
            # Determine final orientation
            if front_score > back_score and front_score > ORIENTATION_CONFIDENCE_THRESHOLD:
                final_orientation = 'front'
                final_confidence = front_score
            elif back_score > front_score and back_score > ORIENTATION_CONFIDENCE_THRESHOLD:
                final_orientation = 'back'
                final_confidence = back_score
            else:
                final_orientation = 'unknown'
                final_confidence = max(front_score, back_score)
            
            # Apply temporal consistency
            final_orientation, final_confidence = self._apply_temporal_consistency(
                final_orientation, final_confidence, vehicle_id
            )
            
            logger.debug(f"Orientation analysis: {final_orientation} (conf: {final_confidence:.2f}, movement: {movement_direction})")
            
            return final_orientation, final_confidence, movement_direction
            
        except Exception as e:
            logger.error(f"Enhanced orientation detection error: {e}")
            return 'unknown', 0.0, 'stationary'
    
    def _enhanced_plate_position_analysis(self, vehicle_img, vehicle_bbox, full_frame):
        """Enhanced license plate position analysis"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale plate detection
            plate_regions = []
            
            # Method 1: Contour-based detection
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicle_height, vehicle_width = gray.shape
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Refined plate criteria
                if (2.0 < aspect_ratio < 6.0 and 
                    w > vehicle_width * 0.1 and 
                    h > vehicle_height * 0.02 and
                    area > 100):
                    
                    vertical_position = y / vehicle_height
                    horizontal_center = (x + w/2) / vehicle_width
                    
                    # Calculate confidence based on position and characteristics
                    confidence = 0.5
                    
                    # Better positioned plates get higher confidence
                    if 0.3 < horizontal_center < 0.7:  # Centered horizontally
                        confidence += 0.2
                    
                    if 2.5 < aspect_ratio < 4.5:  # Good aspect ratio
                        confidence += 0.2
                    
                    plate_regions.append((x, y, w, h, vertical_position, confidence))
            
            if not plate_regions:
                return 'unknown', 0.0
            
            # Analyze plate positions
            weighted_position = 0.0
            total_confidence = 0.0
            
            for region in plate_regions:
                _, _, _, _, v_pos, conf = region
                weighted_position += v_pos * conf
                total_confidence += conf
            
            if total_confidence > 0:
                avg_position = weighted_position / total_confidence
                
                # Front plates: upper 70% of vehicle
                # Rear plates: lower 50% of vehicle
                if avg_position < 0.7:
                    return 'front', min(0.9, total_confidence)
                else:
                    return 'back', min(0.9, total_confidence)
            
            return 'unknown', 0.0
            
        except Exception as e:
            logger.error(f"Enhanced plate analysis error: {e}")
            return 'unknown', 0.0
    
    def _enhanced_geometric_analysis(self, vehicle_img):
        """Enhanced geometric feature analysis"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Divide into regions
            top_region = gray[:height//3, :]
            middle_region = gray[height//3:2*height//3, :]
            bottom_region = gray[2*height//3:, :]
            
            # Calculate texture complexity
            top_complexity = np.std(top_region)
            middle_complexity = np.std(middle_region)
            bottom_complexity = np.std(bottom_region)
            
            # Analyze horizontal features (grilles, bumpers)
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
            horizontal_features = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_horizontal)
            
            top_horizontal = np.sum(horizontal_features[:height//3, :] > 0)
            bottom_horizontal = np.sum(horizontal_features[2*height//3:, :] > 0)
            
            # Score calculation
            front_score = 0.0
            back_score = 0.0
            confidence = 0.5
            
            # Front vehicles typically have more complex top regions (headlights, grille)
            if top_complexity > bottom_complexity * 1.15:
                front_score += 0.3
                confidence += 0.1
            elif bottom_complexity > top_complexity * 1.15:
                back_score += 0.3
                confidence += 0.1
            
            # Horizontal features analysis
            if top_horizontal > bottom_horizontal * 1.2:
                front_score += 0.4
                confidence += 0.15
            elif bottom_horizontal > top_horizontal * 1.2:
                back_score += 0.4
                confidence += 0.15
            
            # Edge density analysis
            edges = cv2.Canny(gray, 50, 150)
            top_edges = np.sum(edges[:height//3, :] > 0)
            bottom_edges = np.sum(edges[2*height//3:, :] > 0)
            
            if top_edges > bottom_edges * 1.1:
                front_score += 0.2
            elif bottom_edges > top_edges * 1.1:
                back_score += 0.2
            
            if front_score > back_score:
                return 'front', min(0.9, confidence + front_score - back_score)
            elif back_score > front_score:
                return 'back', min(0.9, confidence + back_score - front_score)
            else:
                return 'unknown', confidence * 0.5
                
        except Exception as e:
            logger.error(f"Enhanced geometric analysis error: {e}")
            return 'unknown', 0.0
    
    def _advanced_light_detection(self, vehicle_img):
        """Advanced headlight/taillight detection"""
        try:
            # Multi-color space analysis
            hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2LAB)
            
            height, width = vehicle_img.shape[:2]
            
            # Detect bright regions (headlights)
            # Enhanced bright light detection
            bright_mask1 = cv2.inRange(hsv, (0, 0, 200), (180, 80, 255))
            bright_mask2 = cv2.inRange(lab, (180, 0, 0), (255, 255, 255))
            bright_mask = cv2.bitwise_or(bright_mask1, bright_mask2)
            
            # Detect red regions (taillights) - more comprehensive
            red_mask1 = cv2.inRange(hsv, (0, 120, 120), (15, 255, 255))
            red_mask2 = cv2.inRange(hsv, (165, 120, 120), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Detect yellow/orange (turn signals, parking lights)
            yellow_mask = cv2.inRange(hsv, (15, 100, 150), (35, 255, 255))
            orange_mask = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))
            
            # Analyze light distribution
            top_half = height // 2
            
            # Count pixels in different regions
            bright_top = np.sum(bright_mask[:top_half, :] > 0)
            bright_bottom = np.sum(bright_mask[top_half:, :] > 0)
            
            red_top = np.sum(red_mask[:top_half, :] > 0)
            red_bottom = np.sum(red_mask[top_half:, :] > 0)
            
            yellow_top = np.sum(yellow_mask[:top_half, :] > 0)
            yellow_bottom = np.sum(yellow_mask[top_half:, :] > 0)
            
            # Scoring system
            front_score = 0.0
            back_score = 0.0
            confidence = 0.3
            
            # Bright lights analysis
            if bright_top > bright_bottom and bright_top > 50:
                front_score += 0.4
                confidence += 0.2
            elif bright_bottom > bright_top and bright_bottom > 50:
                back_score += 0.2
                confidence += 0.1
            
            # Red light analysis (strong indicator for rear)
            if red_bottom > red_top and red_bottom > 30:
                back_score += 0.5
                confidence += 0.3
            elif red_top > red_bottom and red_top > 20:
                front_score += 0.1  # Some front vehicles have red accents
                confidence += 0.1
            
            # Yellow/orange analysis (turn signals)
            if yellow_top > yellow_bottom and yellow_top > 20:
                front_score += 0.2
                confidence += 0.1
            elif yellow_bottom > yellow_top and yellow_bottom > 20:
                back_score += 0.2
                confidence += 0.1
            
            # Light pattern analysis
            if bright_top > 100 and red_bottom < 20:  # Bright front, no rear reds
                front_score += 0.3
                confidence += 0.15
            elif red_bottom > 50 and bright_top < 50:  # Red rear, less bright front
                back_score += 0.4
                confidence += 0.2
            
            if front_score > back_score:
                return 'front', min(0.95, confidence + (front_score - back_score))
            elif back_score > front_score:
                return 'back', min(0.95, confidence + (back_score - front_score))
            else:
                return 'unknown', confidence * 0.6
                
        except Exception as e:
            logger.error(f"Advanced light detection error: {e}")
            return 'unknown', 0.0
    
    def _edge_pattern_analysis(self, vehicle_img):
        """Edge pattern analysis for vehicle orientation"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(gray, 30, 90)
            edges2 = cv2.Canny(gray, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            
            height, width = gray.shape
            
            # Line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                   minLineLength=width//10, maxLineGap=10)
            
            if lines is None:
                return 'unknown', 0.0
            
            # Analyze line orientations and positions
            horizontal_lines_top = 0
            horizontal_lines_bottom = 0
            vertical_lines = 0
            diagonal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                if x2 != x1:
                    angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                else:
                    angle = 90
                
                line_center_y = (y1 + y2) / 2
                
                if angle < 20:  # Horizontal lines
                    if line_center_y < height / 2:
                        horizontal_lines_top += 1
                    else:
                        horizontal_lines_bottom += 1
                elif angle > 70:  # Vertical lines
                    vertical_lines += 1
                else:  # Diagonal lines
                    diagonal_lines += 1
            
            # Analysis
            front_score = 0.0
            back_score = 0.0
            confidence = 0.4
            
            # Front vehicles often have more horizontal features in the top (grille)
            if horizontal_lines_top > horizontal_lines_bottom:
                front_score += 0.3
                confidence += 0.1
            elif horizontal_lines_bottom > horizontal_lines_top:
                back_score += 0.3
                confidence += 0.1
            
            # Vertical lines can indicate different features
            if vertical_lines > horizontal_lines_top + horizontal_lines_bottom:
                confidence += 0.1
            
            total_lines = horizontal_lines_top + horizontal_lines_bottom + vertical_lines
            if total_lines > 5:
                confidence += 0.1
            
            if front_score > back_score:
                return 'front', min(0.8, confidence + front_score)
            elif back_score > front_score:
                return 'back', min(0.8, confidence + back_score)
            else:
                return 'unknown', confidence * 0.7
                
        except Exception as e:
            logger.error(f"Edge pattern analysis error: {e}")
            return 'unknown', 0.0
    
    def _vehicle_contour_analysis(self, vehicle_img):
        """Vehicle contour shape analysis"""
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Find vehicle contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 'unknown', 0.0
            
            # Find the largest contour (likely the vehicle outline)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Analyze contour properties
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            
            if hull_area == 0:
                return 'unknown', 0.0
            
            solidity = contour_area / hull_area
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Analyze moments for shape characteristics
            moments = cv2.moments(largest_contour)
            
            # Analysis
            confidence = 0.3
            
            if solidity > 0.8:  # More solid shapes
                confidence += 0.1
            
            if 1.5 < aspect_ratio < 3.0:  # Reasonable car aspect ratio
                confidence += 0.1
            
            # For now, return moderate confidence with unknown
            # This could be expanded with more sophisticated shape analysis
            return 'unknown', confidence
            
        except Exception as e:
            logger.error(f"Contour analysis error: {e}")
            return 'unknown', 0.0
    
    def _movement_vector_analysis(self, vehicle_bbox, vehicle_id):
        """Analyze vehicle movement to determine direction"""
        try:
            current_time = time.time()
            bbox_center = ((vehicle_bbox[0] + vehicle_bbox[2]) / 2, 
                          (vehicle_bbox[1] + vehicle_bbox[3]) / 2)
            
            if vehicle_id not in self.vehicle_tracks:
                self.vehicle_tracks[vehicle_id] = deque(maxlen=DIRECTION_ANALYSIS_FRAMES)
            
            # Add current position to track
            self.vehicle_tracks[vehicle_id].append((bbox_center, current_time))
            
            track = self.vehicle_tracks[vehicle_id]
            
            if len(track) < 3:
                return 'stationary', 0.0
            
            # Calculate movement vector
            positions = [pos for pos, _ in track]
            times = [t for _, t in track]
            
            # Calculate average movement direction
            dx_total = 0
            dy_total = 0
            valid_movements = 0
            
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dt = times[i] - times[i-1]
                
                if dt > 0 and (abs(dx) > 5 or abs(dy) > 5):  # Minimum movement threshold
                    dx_total += dx / dt
                    dy_total += dy / dt
                    valid_movements += 1
            
            if valid_movements < 2:
                return 'stationary', 0.0
            
            avg_dx = dx_total / valid_movements
            avg_dy = dy_total / valid_movements
            
            # Calculate movement magnitude and direction
            movement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
            
            if movement_magnitude < 10:  # Too slow to determine direction
                return 'stationary', 0.0
            
            # Determine primary movement direction
            if abs(avg_dx) > abs(avg_dy):
                if avg_dx > 0:
                    direction = 'moving_right'
                else:
                    direction = 'moving_left'
            else:
                if avg_dy > 0:
                    direction = 'moving_down'
                else:
                    direction = 'moving_up'
            
            confidence = min(0.8, movement_magnitude / 50.0)
            return direction, confidence
            
        except Exception as e:
            logger.error(f"Movement analysis error: {e}")
            return 'stationary', 0.0
    
    def _apply_temporal_consistency(self, orientation, confidence, vehicle_id):
        """Apply temporal consistency to orientation detection"""
        try:
            if vehicle_id is None:
                return orientation, confidence
            
            # Store detection history per vehicle
            if vehicle_id not in self.vehicle_tracks:
                self.vehicle_tracks[vehicle_id] = deque(maxlen=TEMPORAL_CONSISTENCY_FRAMES)
            
            # Add current detection
            self.vehicle_tracks[vehicle_id].append((orientation, confidence, time.time()))
            
            track = self.vehicle_tracks[vehicle_id]
            
            if len(track) < 2:
                return orientation, confidence
            
            # Analyze recent detections
            recent_orientations = [o for o, c, t in track if c > 0.3]
            
            if len(recent_orientations) < 2:
                return orientation, confidence
            
            # Count orientations
            orientation_counts = {}
            confidence_sum = {}
            
            for o, c, t in track:
                if o not in orientation_counts:
                    orientation_counts[o] = 0
                    confidence_sum[o] = 0
                orientation_counts[o] += 1
                confidence_sum[o] += c
            
            # Find most consistent orientation
            best_orientation = max(orientation_counts.keys(), 
                                 key=lambda x: orientation_counts[x])
            
            if orientation_counts[best_orientation] >= len(track) * 0.6:
                # High consistency - boost confidence
                avg_confidence = confidence_sum[best_orientation] / orientation_counts[best_orientation]
                boosted_confidence = min(0.95, avg_confidence * 1.2)
                return best_orientation, boosted_confidence
            
            return orientation, confidence
            
        except Exception as e:
            logger.error(f"Temporal consistency error: {e}")
            return orientation, confidence
    
    def get_entry_exit_status(self, orientation, confidence, movement_direction=None):
        """Convert orientation and movement to entry/exit status with confidence"""
        if confidence < 0.4:
            return 'Vehicle Detected', confidence
        
        if orientation == 'front':
            if movement_direction and 'right' in movement_direction:
                return 'Vehicle Entering (Moving Right)', min(0.95, confidence * 1.1)
            elif movement_direction and 'left' in movement_direction:
                return 'Vehicle Entering (Moving Left)', min(0.95, confidence * 1.1)
            else:
                return 'Vehicle Entered', confidence
        elif orientation == 'back':
            if movement_direction and 'right' in movement_direction:
                return 'Vehicle Exiting (Moving Right)', min(0.95, confidence * 1.1)
            elif movement_direction and 'left' in movement_direction:
                return 'Vehicle Exiting (Moving Left)', min(0.95, confidence * 1.1)
            else:
                return 'Vehicle Exited', confidence
        else:
            return 'Vehicle Detected', confidence * 0.8


class OptimizedOCREngine:
    """Optimized OCR engine with enhanced preprocessing and caching"""
    def __init__(self):
        self.cache_lock = threading.Lock()
        self.preprocessing_cache = {}
        self.valid_plate_pattern = re.compile(r'^[A-Z0-9]{5,8}$')
        
        # Pre-compile OCR configurations for speed
        self.ocr_configs = {
            'fast': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'accurate': '--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'single_line': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        }
        
    def ultra_fast_preprocess(self, img):
        """Ultra-fast image preprocessing optimized for speed"""
        try:
            if img is None or img.size == 0:
                return img
                
            # Resize if too small for better OCR
            h, w = img.shape[:2]
            if w < 120:
                scale = 120 / w
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            
            # Quick sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpened = cv2.filter2D(img, -1, kernel)
            
            # Contrast enhancement
            enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
            
            return enhanced
        except Exception as e:
            logger.error(f"Ultra-fast preprocessing error: {e}")
            return img

    def lightning_fast_ocr(self, img):
        """Lightning-fast OCR with minimal processing"""
        try:
            if img is None or img.size == 0:
                return ""
            
            # Generate fast hash for caching
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            
            with self.cache_lock:
                if img_hash in self.preprocessing_cache:
                    return self.preprocessing_cache[img_hash]
            
            # Ultra-fast preprocessing
            processed = self.ultra_fast_preprocess(img)
            
            # Quick OCR with optimized config
            text = pytesseract.image_to_string(processed, config=self.ocr_configs['fast']).strip()
            
            # Clean text
            text = ''.join(c for c in text if c.isalnum()).upper()
            
            # Validate and cache if good
            if self.is_valid_license_plate(text):
                with self.cache_lock:
                    # Limit cache size
                    if len(self.preprocessing_cache) > 50:
                        # Remove oldest 25% of entries
                        keys_to_remove = list(self.preprocessing_cache.keys())[:12]
                        for key in keys_to_remove:
                            del self.preprocessing_cache[key]
                    
                    self.preprocessing_cache[img_hash] = text
                return text
            
            return ""
            
        except Exception as e:
            logger.error(f"Lightning OCR error: {e}")
            return ""

    def is_valid_license_plate(self, text):
        """Fast license plate validation"""
        if not text or len(text) < 5:
            return False
        return bool(self.valid_plate_pattern.match(text))


class EnhancedMongoDBManager:
    """Enhanced MongoDB manager with optimized batch processing"""
    def __init__(self, uri=MONGO_URI, db_name=MONGO_DB_NAME, collection_name=MONGO_COLLECTION_NAME):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connection_lock = RLock()
        self.is_connected = False
        self.write_queue = queue.Queue(maxsize=100)  # Prevent memory issues
        self.batch_size = 10  # Larger batches for better performance
        self.batch_timeout = 1.0  # Faster batch processing
        
        self._connect()
        self._start_batch_writer()

    def _connect(self):
        """Enhanced connection with optimized settings"""
        try:
            self.client = MongoClient(
                self.uri, 
                serverSelectionTimeoutMS=3000,  # Faster timeout
                maxPoolSize=20,  # More connections
                minPoolSize=5,
                maxIdleTimeMS=20000,  # Shorter idle time
                waitQueueTimeoutMS=2000
            )
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Create indexes for performance
            try:
                self.collection.create_index([("plate_number", 1), ("timestamp", -1)])
                self.collection.create_index("detection_id", unique=True)
                self.collection.create_index("entry_exit_status")
                self.collection.create_index("vehicle_orientation")
                self.collection.create_index("timestamp")
            except pymongo.errors.OperationFailure:
                logger.warning("Could not create indexes, they may already exist.")
            except Exception as e:
                logger.error(f"Error creating MongoDB indexes: {e}")

            
            self.is_connected = True
            logger.info("✅ Enhanced MongoDB connected with optimized batch processing")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.is_connected = False
            return False

    def _start_batch_writer(self):
        """Start optimized batch writer"""
        if self.is_connected:
            self.batch_writer_thread = threading.Thread(target=self._optimized_batch_writer, daemon=True)
            self.batch_writer_thread.start()

    def _optimized_batch_writer(self):
        """Optimized batch writer with better performance"""
        batch = []
        last_write = time.time()
        
        while True:
            try:
                try:
                    # Non-blocking get with shorter timeout
                    item = self.write_queue.get(timeout=0.2)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                current_time = time.time()
                should_write = (len(batch) >= self.batch_size or 
                               (batch and current_time - last_write > self.batch_timeout))
                
                if should_write and self.is_connected:
                    try:
                        if batch:
                            # Use unordered bulk insert for better performance
                            self.collection.insert_many(batch, ordered=False)
                            logger.debug(f"💾 Batch saved {len(batch)} vehicle records")
                            batch.clear()
                            last_write = current_time
                    except pymongo.errors.BulkWriteError as bwe:
                        # Filter out duplicate key errors and log others
                        # This handles cases where a record might be processed twice before DB write
                        duplicate_errors = [e for e in bwe.details['writeErrors'] if e['code'] == 11000]
                        if len(duplicate_errors) == len(bwe.details['writeErrors']):
                             logger.warning(f"Batch write had {len(duplicate_errors)} duplicate key errors (ignored).")
                        else:
                            logger.error(f"Batch write error (with non-duplicate issues): {bwe.details}")
                        batch.clear()
                    except Exception as e:
                        logger.error(f"Batch write error: {e}")
                        batch.clear()
                        # Try to reconnect
                        self._connect()
                
            except Exception as e:
                logger.error(f"Batch writer error: {e}")
                time.sleep(0.5)

    def save_vehicle_record_async(self, plate_number, confidence, car_image_path, plate_image_path, 
                                bbox_info, vehicle_info=None, additional_data=None, 
                                entry_exit_status=None, vehicle_orientation=None, orientation_confidence=None):
        """Save vehicle record with enhanced data"""
        try:
            timestamp = datetime.now()
            document = {
                "plate_number": str(plate_number).strip().upper(),
                "timestamp": timestamp,
                "confidence": float(confidence) if confidence else 0.0,
                "car_image_path": str(car_image_path) if car_image_path else "",
                "plate_image_path": str(plate_image_path) if plate_image_path else "",
                "detection_id": f"{plate_number}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}",
                "bbox_info": bbox_info or {},
                "vehicle_info": vehicle_info or {},
                "processing_method": "Ultra-Fast Stream + Improved Entry/Exit Detection",
                "image_capture_enabled": True,
                "full_car_captured": bool(car_image_path),
                "processed": True,
                "entry_exit_status": entry_exit_status or "Vehicle Detected",
                "vehicle_orientation": vehicle_orientation or "unknown",
                "orientation_confidence": float(orientation_confidence) if orientation_confidence else 0.0,
                "is_entry": entry_exit_status and "Enter" in entry_exit_status,
                "is_exit": entry_exit_status and "Exit" in entry_exit_status,
                "movement_info": vehicle_info.get("movement_direction", "unknown") if vehicle_info else "unknown"
            }
            
            if additional_data:
                document.update(additional_data)
            
            # Non-blocking queue put
            try:
                self.write_queue.put_nowait(document)
                return True
            except queue.Full:
                logger.warning("Write queue full - dropping record")
                return False
                
        except Exception as e:
            logger.error(f"Async save error: {e}")
            return False

    def get_vehicle_records(self, limit=50, query=None):
        """Get vehicle records with enhanced filtering"""
        try:
            if not self.is_connected:
                return []
            
            search_filter = {}
            if query:
                search_filter = {
                    "$or": [
                        {"plate_number": {"$regex": query, "$options": "i"}},
                        {"detection_id": {"$regex": query, "$options": "i"}},
                        {"vehicle_info.type": {"$regex": query, "$options": "i"}},
                        {"entry_exit_status": {"$regex": query, "$options": "i"}},
                        {"vehicle_orientation": {"$regex": query, "$options": "i"}}
                    ]
                }
            
            cursor = self.collection.find(search_filter).sort("timestamp", -1).limit(limit)
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
                    "entry_exit_status": doc.get("entry_exit_status", "Vehicle Detected"),
                    "vehicle_orientation": doc.get("vehicle_orientation", "unknown"),
                    "orientation_confidence": doc.get("orientation_confidence", 0.0),
                    "is_entry": doc.get("is_entry"),
                    "is_exit": doc.get("is_exit"),
                    "movement_info": doc.get("movement_info", "unknown")
                }
                records.append(record)
            
            return records
        except Exception as e:
            logger.error(f"Error getting vehicle records: {e}")
            return []

    def get_enhanced_stats(self):
        """Get comprehensive statistics with performance optimizations"""
        try:
            if not self.is_connected:
                return self._default_stats()
            
            # Use aggregation pipeline for better performance
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_vehicles": {"$sum": 1},
                        "unique_plates": {"$addToSet": "$plate_number"},
                        "full_cars_captured": {
                            "$sum": {"$cond": [{"$eq": ["$full_car_captured", True]}, 1, 0]}
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
                        "latest_timestamp": {"$max": "$timestamp"}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result:
                stats = result[0]
                return {
                    "total_vehicles": stats.get("total_vehicles", 0),
                    "unique_plates": len(stats.get("unique_plates", [])),
                    "full_cars_captured": stats.get("full_cars_captured", 0),
                    "car_images_captured": stats.get("full_cars_captured", 0),  # Same as full cars
                    "plate_images_captured": stats.get("total_vehicles", 0),  # Assume all have plates
                    "entries": stats.get("entries", 0),
                    "exits": stats.get("exits", 0),
                    "front_detections": stats.get("front_detections", 0),
                    "back_detections": stats.get("back_detections", 0),
                    "latest_detection": stats.get("latest_timestamp"),
                    "database_status": "Connected"
                }
            else:
                return self._default_stats()
                
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return self._default_stats()

    def _default_stats(self):
        """Return default stats when database is unavailable"""
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
            "database_status": "Disconnected"
        }

    def close_connection(self):
        """Clean shutdown"""
        try:
            if self.client:
                self.client.close()
                self.is_connected = False
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


class UltraStableRTSPReader:
    """Ultra-stable RTSP reader optimized for high performance and reliability"""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame_lock = Lock()
        self.is_connected = False
        self.running = False
        self.consecutive_failures = 0
        self.last_frame_time = time.time()
        self.reconnect_attempts = 0
        
        # Optimized buffering for speed
        self.latest_frame = None
        self.frame_buffer = queue.Queue(maxsize=2)  # Small buffer for speed
        
        # Performance monitoring
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_fps_calculation = time.time()
        self.current_fps = 0.0
        
        # Ultra-optimized capture settings
        self.width = 1920
        self.height = 1080
        self.target_fps = TARGET_FPS
        
        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize RTSP capture with maximum performance"""
        logger.info(f"🔗 Connecting to ultra-stable RTSP stream: {self.rtsp_url}")
        
        try:
            # Try FFMPEG first (usually fastest)
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                logger.warning("FFMPEG backend failed, trying default")
                self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                logger.error("❌ Failed to open RTSP stream")
                return False
            
            # Optimize capture settings for speed
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Test frame reading
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("❌ Cannot read frames from stream")
                self.cap.release()
                return False
            
            logger.info(f"✅ Ultra-stable RTSP connected successfully")
            self.is_connected = True
            
            # Start optimized capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._ultra_fast_capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RTSP initialization error: {e}")
            self.is_connected = False
            return False

    def _ultra_fast_capture_loop(self):
        """Ultra-fast frame capture optimized for performance"""
        logger.info("🎬 Starting ultra-fast capture loop")
        
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                if not self.cap or not self.cap.isOpened():
                    logger.warning("📡 Stream disconnected, attempting quick reconnection...")
                    if not self._quick_reconnect():
                        time.sleep(1)
                        continue
                
                # Read frame as fast as possible
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error("🚨 Too many consecutive failures, reconnecting...")
                        self._quick_reconnect()
                    time.sleep(0.01)
                    continue
                
                # Successful frame read
                self.consecutive_failures = 0
                self.frames_received += 1
                self.last_frame_time = current_time
                
                # Update frame with minimal locking
                with self.frame_lock:
                    self.latest_frame = frame
                
                # Update buffer (drop old frames for speed)
                try:
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()  # Drop old frame
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    self.frame_buffer.put_nowait(frame)
                except queue.Full:
                    self.frames_dropped += 1
                
                # Calculate FPS
                if current_time - self.last_fps_calculation > 1.0:
                    self.current_fps = self.frames_received / (current_time - self.last_fps_calculation)
                    self.frames_received = 0
                    self.last_fps_calculation = current_time
                
                # Dynamic sleep for target FPS
                elapsed = current_time - last_frame_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = current_time
                
            except Exception as e:
                logger.error(f"Ultra-fast capture error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)
        
        logger.warning("🎬 Ultra-fast capture loop stopped")

    def _quick_reconnect(self):
        """Quick reconnection with minimal delay"""
        with self.frame_lock:
             if self.is_connected:
                self.is_connected = False
                logger.info("Setting stream status to disconnected.")
        try:
            self.reconnect_attempts += 1
            logger.info(f"🔄 Quick reconnection attempt #{self.reconnect_attempts}")
            
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            
            time.sleep(1)  # Minimal delay
            
            self.consecutive_failures = 0
            
            return self._initialize_capture()
            
        except Exception as e:
            logger.error(f"Quick reconnection error: {e}")
            return False

    def get_frame(self):
        """Get the latest frame with maximum speed"""
        try:
            # Try buffer first (most recent)
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
            logger.error(f"Error getting frame: {e}")
            return None

    def is_healthy(self):
        """Check if stream is healthy"""
        return (self.is_connected and 
                self.consecutive_failures < MAX_CONSECUTIVE_FAILURES and
                time.time() - self.last_frame_time < 3.0)  # 3 second tolerance

    def get_stats(self):
        """Get performance statistics"""
        return {
            'connected': self.is_connected,
            'healthy': self.is_healthy(),
            'consecutive_failures': self.consecutive_failures,
            'reconnect_attempts': self.reconnect_attempts,
            'current_fps': round(self.current_fps, 1),
            'frames_dropped': self.frames_dropped,
            'last_frame_age': time.time() - self.last_frame_time
        }

    def release(self):
        """Clean shutdown"""
        logger.info("🎬 Shutting down ultra-stable RTSP reader...")
        
        self.running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        
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
        
        logger.info("🎬 Ultra-stable RTSP reader shutdown complete")


class UltraFastCarDetector:
    """Ultra-fast detector optimized for performance with improved entry/exit detection"""
    def __init__(self, plate_model_path, car_model_path=YOLO_CAR_MODEL):
        # Load YOLO models with optimization
        try:
            self.plate_model = YOLO(plate_model_path)
            self.car_model = YOLO(car_model_path)
            
            # GPU optimization
            if torch.cuda.is_available():
                self.plate_model.to('cuda')
                self.car_model.to('cuda')
                logger.info(f"🎮 GPU ACCELERATION: {torch.cuda.get_device_name(0)}")
            
            # Model optimization
            self.plate_model.overrides['verbose'] = False
            self.car_model.overrides['verbose'] = False
            
            logger.info(f"✅ Ultra-fast YOLO models loaded and optimized")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO models: {e}")
            raise
        
        # Initialize optimized components
        self.ocr_engine = OptimizedOCREngine()
        self.mongo_manager = EnhancedMongoDBManager()
        self.orientation_detector = ImprovedVehicleOrientationDetector()
        self.duplicate_manager = DuplicateCleanupManager(self.mongo_manager)
        
        # Load existing plates for duplicate prevention
        self.captured_plates = set()
        self._load_existing_plates()
        
        # Optimized detection parameters
        self.plate_confidence_threshold = 0.25  # Lower for more detections
        self.car_confidence_threshold = CAR_DETECTION_CONFIDENCE
        self.iou_threshold = 0.4
        
        # Ultra-fast processing intervals
        self.frame_count = 0
        self.last_ocr_frame = 0
        self.processing_interval = PROCESSING_INTERVAL
        self.ocr_interval = OCR_PROCESSING_INTERVAL
        
        # Thread-safe collections
        self.captured_plates_lock = RLock()
        self.session_plates = set()
        self.recent_detections = deque(maxlen=30)  # Smaller for speed
        self.vehicle_records = []
        self.vehicle_records_lock = RLock()
        
        # Vehicle tracking for improved entry/exit
        self.vehicle_tracker = {}
        self.track_id_counter = 0
        
        # Optimized background processing
        self.image_processor_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, 
                                                                               thread_name_prefix="ImageProc")
        
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
            'session_start': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_detection': 'None',
            'plates_list': [],
            'is_live': True,
            'mongodb_status': 'Connected' if self.mongo_manager.is_connected else 'Disconnected',
            'processing_mode': 'Ultra-Fast Stream + Improved Entry/Exit + Duplicate Cleanup'
        }
        
        # Memory management
        self.last_cleanup = 0
        self.last_duplicate_cleanup = time.time()

    def _load_existing_plates(self):
        """Load existing plates from database with timeout"""
        try:
            # Quick load with limit for performance
            records = self.mongo_manager.get_vehicle_records(limit=500)
            self.captured_plates = {record['plate_number'] for record in records if record['plate_number']}
            logger.info(f"📋 Loaded {len(self.captured_plates)} existing plates")
        except Exception as e:
            logger.error(f"Error loading existing plates: {e}")
            self.captured_plates = set()

    def should_process_frame(self):
        """Optimized frame processing decision"""
        self.frame_count += 1
        
        # Run auto duplicate cleanup less frequently for performance
        if self.frame_count % 300 == 0 and self.duplicate_manager.should_run_auto_cleanup():
            self.duplicate_manager.run_auto_cleanup()
        
        # Process every frame for smooth tracking
        if self.frame_count % self.processing_interval == 0:
            if self.frame_count - self.last_cleanup > MEMORY_CLEANUP_INTERVAL:
                self._quick_cleanup()
                self.last_cleanup = self.frame_count
            return True
        
        return False

    def should_run_ocr(self):
        """OCR decision with performance optimization"""
        if self.frame_count - self.last_ocr_frame >= self.ocr_interval:
            self.last_ocr_frame = self.frame_count
            return True
        return False

    def _quick_cleanup(self):
        """Quick memory cleanup for performance"""
        try:
            with self.ocr_engine.cache_lock:
                if len(self.ocr_engine.preprocessing_cache) > 30:
                    # Keep only most recent entries
                    cache_items = list(self.ocr_engine.preprocessing_cache.items())
                    self.ocr_engine.preprocessing_cache = dict(cache_items[-15:])
            
            with self.vehicle_records_lock:
                if len(self.vehicle_records) > 50:
                    self.vehicle_records = self.vehicle_records[-25:]
            
            # Clean old vehicle tracks
            current_time = time.time()
            old_tracks = [vid for vid, track in self.vehicle_tracker.items() 
                         if current_time - track.get('last_seen', 0) > 30]
            for vid in old_tracks:
                del self.vehicle_tracker[vid]
            
            if self.frame_count % 200 == 0:
                gc.collect()
                
        except Exception as e:
            logger.error(f"Quick cleanup error: {e}")

    def detect_cars_ultra_fast(self, frame):
        """Ultra-fast car detection with tracking"""
        try:
            # Run inference with optimized parameters
            results = self.car_model(frame, 
                                   conf=self.car_confidence_threshold, 
                                   classes=CAR_CLASSES, 
                                   verbose=False,
                                   imgsz=640,  # Smaller size for speed
                                   half=True if torch.cuda.is_available() else False)  # FP16 for speed
            
            cars = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Size filtering for performance
                        if (x2 - x1) >= MIN_CAR_WIDTH and (y2 - y1) >= MIN_CAR_HEIGHT:
                            # Assign or update vehicle ID for tracking
                            vehicle_id = self._get_or_assign_vehicle_id((x1, y1, x2, y2))
                            
                            cars.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.get_vehicle_class_name(class_id),
                                'vehicle_id': vehicle_id
                            })
            
            return cars
        except Exception as e:
            logger.error(f"Ultra-fast car detection error: {e}")
            return []

    def _get_or_assign_vehicle_id(self, bbox):
        """Assign or get existing vehicle ID for tracking"""
        try:
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_time = time.time()
            
            # Find closest existing track
            min_distance = float('inf')
            closest_id = None
            
            for vehicle_id, track_info in self.vehicle_tracker.items():
                if current_time - track_info['last_seen'] > 10:  # 10 second timeout
                    continue
                
                track_center = track_info['center']
                distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
                
                if distance < min_distance and distance < 100:  # 100 pixel threshold
                    min_distance = distance
                    closest_id = vehicle_id
            
            if closest_id is not None:
                # Update existing track
                self.vehicle_tracker[closest_id].update({
                    'center': center,
                    'bbox': bbox,
                    'last_seen': current_time
                })
                return closest_id
            else:
                # Create new track
                self.track_id_counter += 1
                new_id = f"vehicle_{self.track_id_counter}"
                self.vehicle_tracker[new_id] = {
                    'center': center,
                    'bbox': bbox,
                    'last_seen': current_time,
                    'first_seen': current_time
                }
                return new_id
                
        except Exception as e:
            logger.error(f"Vehicle tracking error: {e}")
            return f"vehicle_{int(time.time())}"

    def get_vehicle_class_name(self, class_id):
        """Get vehicle class name from COCO class ID"""
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'vehicle')

    def find_plate_in_car_fast(self, frame, car_bbox, plate_detections):
        """Fast plate-in-car matching"""
        car_x1, car_y1, car_x2, car_y2 = car_bbox
        
        best_plate = None
        best_score = 0
        
        for plate in plate_detections:
            plate_x1, plate_y1, plate_x2, plate_y2 = plate['bbox']
            
            # Quick center point check
            plate_center_x = (plate_x1 + plate_x2) / 2
            plate_center_y = (plate_y1 + plate_y2) / 2
            
            if (car_x1 <= plate_center_x <= car_x2 and car_y1 <= plate_center_y <= car_y2):
                # Calculate overlap score
                overlap_x1 = max(car_x1, plate_x1)
                overlap_y1 = max(car_y1, plate_y1)
                overlap_x2 = min(car_x2, plate_x2)
                overlap_y2 = min(car_y2, plate_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    plate_area = (plate_x2 - plate_x1) * (plate_y2 - plate_y1)
                    score = overlap_area / plate_area if plate_area > 0 else 0
                    
                    if score > best_score:
                        best_score = score
                        best_plate = plate
        
        return best_plate

    def extract_and_enhance_car_image_fast(self, frame, car_bbox):
        """Fast car image extraction with minimal processing"""
        try:
            x1, y1, x2, y2 = car_bbox
            frame_h, frame_w = frame.shape[:2]
            
            # Quick expansion
            car_w = x2 - x1
            car_h = y2 - y1
            
            expand_w = int(car_w * 0.2)  # Smaller expansion for speed
            expand_h = int(car_h * 0.2)
            
            full_car_x1 = max(0, x1 - expand_w)
            full_car_y1 = max(0, y1 - expand_h)
            full_car_x2 = min(frame_w, x2 + expand_w)
            full_car_y2 = min(frame_h, y2 + expand_h)
            
            car_img = frame[full_car_y1:full_car_y2, full_car_x1:full_car_x2]
            
            # Minimal enhancement for speed
            if car_img.size > 0:
                # Quick resize if too small
                h, w = car_img.shape[:2]
                if w < 400:
                    scale = 400 / w
                    car_img = cv2.resize(car_img, (int(w*scale), int(h*scale)), 
                                       interpolation=cv2.INTER_LINEAR)
                
                # Quick sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) * 0.5
                car_img = cv2.filter2D(car_img, -1, kernel)
            
            return car_img, (full_car_x1, full_car_y1, full_car_x2, full_car_y2)
            
        except Exception as e:
            logger.error(f"Fast car extraction error: {e}")
            return None, None

    def save_images_ultra_fast(self, frame, plate_img, plate_bbox, plate_text, 
                             confidence, car_info):
        """Ultra-fast image saving with improved entry/exit detection"""
        def _save_ultra_fast():
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                
                car_image_path = ""
                full_car_bbox = None
                vehicle_orientation = "unknown"
                orientation_confidence = 0.0
                entry_exit_status = "Vehicle Detected"
                movement_direction = "stationary"
                
                if car_info:
                    # Extract full car image
                    full_car_img, full_car_bbox = self.extract_and_enhance_car_image_fast(
                        frame, car_info['bbox'])
                    
                    if full_car_img is not None and full_car_img.size > 0:
                        # Improved orientation detection
                        vehicle_id = car_info.get('vehicle_id')
                        orientation_result = self.orientation_detector.detect_vehicle_orientation_and_movement(
                            full_car_img, car_info['bbox'], frame, vehicle_id)
                        
                        vehicle_orientation = orientation_result[0]
                        orientation_confidence = orientation_result[1]
                        movement_direction = orientation_result[2]
                        
                        # Get enhanced entry/exit status
                        entry_exit_status, _ = self.orientation_detector.get_entry_exit_status(
                            vehicle_orientation, orientation_confidence, movement_direction)
                        
                        # Update statistics
                        with self.stats_lock:
                            if "Enter" in entry_exit_status:
                                self.detection_stats['entries'] += 1
                            elif "Exit" in entry_exit_status:
                                self.detection_stats['exits'] += 1
                        
                        # Save car image with orientation info
                        orientation_prefix = vehicle_orientation.upper() if vehicle_orientation != "unknown" else "UNK"
                        confidence_str = f"{int(orientation_confidence*100)}"
                        car_filename = f"FULLCAR_{orientation_prefix}_{confidence_str}_{plate_text}_{timestamp}.jpg"
                        car_filepath = os.path.join(CAR_IMAGES_DIR, car_filename)
                        
                        success = cv2.imwrite(car_filepath, full_car_img, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 85])  # Slightly lower quality for speed
                        if success:
                            car_image_path = car_filepath
                            with self.stats_lock:
                                self.detection_stats['car_images_captured'] += 1
                                self.detection_stats['full_cars_captured'] += 1
                
                # Save plate image
                plate_image_path = ""
                if plate_img is not None and plate_img.size > 0:
                    plate_filename = f"PLATE_{plate_text}_{timestamp}.jpg"
                    plate_filepath = os.path.join(PLATE_IMAGES_DIR, plate_filename)
                    
                    # Quick plate enhancement
                    enhanced_plate = self._quick_enhance_plate(plate_img)
                    success = cv2.imwrite(plate_filepath, enhanced_plate,
                                        [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if success:
                        plate_image_path = plate_filepath
                        with self.stats_lock:
                            self.detection_stats['plate_images_captured'] += 1
                
                # Prepare vehicle info
                vehicle_info = {
                    "type": car_info.get('class_name', 'vehicle') if car_info else 'unknown',
                    "car_confidence": car_info.get('confidence', 0.0) if car_info else 0.0,
                    "full_car_captured": bool(car_image_path),
                    "orientation": vehicle_orientation,
                    "orientation_confidence": orientation_confidence,
                    "movement_direction": movement_direction,
                    "entry_exit_status": entry_exit_status,
                    "vehicle_id": car_info.get('vehicle_id') if car_info else None,
                    "car_dimensions": {
                        "width": full_car_bbox[2] - full_car_bbox[0] if full_car_bbox else 0,
                        "height": full_car_bbox[3] - full_car_bbox[1] if full_car_bbox else 0
                    }
                }
                
                bbox_info = {
                    "plate_bbox": plate_bbox,
                    "car_bbox": car_info['bbox'] if car_info else [],
                    "full_car_bbox": full_car_bbox if full_car_bbox else [],
                    "frame_dimensions": [frame.shape[1], frame.shape[0]]
                }
                
                # Save to MongoDB
                self.mongo_manager.save_vehicle_record_async(
                    plate_text, confidence, car_image_path, plate_image_path,
                    bbox_info, vehicle_info, {
                        "car_filename": os.path.basename(car_image_path) if car_image_path else "",
                        "plate_filename": os.path.basename(plate_image_path) if plate_image_path else "",
                        "enhancement_applied": True,
                        "capture_method": "Ultra-Fast Stream + Improved Entry/Exit Detection",
                        "orientation_detection_applied": True,
                        "movement_tracking_applied": True,
                        "processing_time": "ultra_fast"
                    }, entry_exit_status, vehicle_orientation, orientation_confidence
                )
                
                # Store local record
                vehicle_record = {
                    'plate_number': plate_text,
                    'timestamp': datetime.now(),
                    'confidence': confidence,
                    'car_image_path': car_image_path,
                    'plate_image_path': plate_image_path,
                    'detection_id': f"{plate_text}_{timestamp}",
                    'bbox_info': bbox_info,
                    'vehicle_info': vehicle_info,
                    'full_car_captured': bool(car_image_path),
                    'entry_exit_status': entry_exit_status,
                    'vehicle_orientation': vehicle_orientation,
                    'orientation_confidence': orientation_confidence,
                    'movement_direction': movement_direction,
                    'is_entry': "Enter" in entry_exit_status,
                    'is_exit': "Exit" in entry_exit_status
                }
                
                with self.vehicle_records_lock:
                    self.vehicle_records.append(vehicle_record)
                    if len(self.vehicle_records) > 75:  # Smaller buffer for performance
                        self.vehicle_records = self.vehicle_records[-50:]
                
                logger.info(f"🚗📸 Ultra-fast capture: {plate_text} ({entry_exit_status}) [Conf: {orientation_confidence:.2f}]")
                
            except Exception as e:
                logger.error(f"Ultra-fast save error: {e}")
        
        # Submit to thread pool
        self.image_processor_executor.submit(_save_ultra_fast)

    def _quick_enhance_plate(self, plate_img):
        """Quick plate enhancement for performance"""
        try:
            if plate_img is None or plate_img.size == 0:
                return plate_img
            
            # Quick resize if too small
            h, w = plate_img.shape[:2]
            if w < 150:
                scale = 150 / w
                plate_img = cv2.resize(plate_img, (int(w*scale), int(h*scale)), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Quick contrast enhancement
            enhanced = cv2.convertScaleAbs(plate_img, alpha=1.05, beta=5)
            
            return enhanced
        except Exception as e:
            logger.error(f"Quick plate enhancement error: {e}")
            return plate_img

    def is_recent_detection_fast(self, plate_text, bbox):
        """Fast duplicate detection with smaller history"""
        current_time = time.time()
        detection_key = f"{plate_text}_{bbox[0]//20}_{bbox[1]//20}"  # Coarser grouping for speed
        
        # Clean old detections efficiently
        cutoff_time = current_time - 8.0  # 8 second window
        self.recent_detections = deque([
            (key, timestamp) for key, timestamp in self.recent_detections
            if timestamp > cutoff_time
        ], maxlen=30)
        
        # Check for duplicates
        for key, timestamp in self.recent_detections:
            if key == detection_key and current_time - timestamp < 4.0:  # 4 second duplicate window
                return True
        
        self.recent_detections.append((detection_key, current_time))
        return False

    def is_already_captured(self, plate_text):
        """Fast captured plate check"""
        if not plate_text:
            return True
        
        plate_text = plate_text.upper().strip()
        
        with self.captured_plates_lock:
            return plate_text in self.captured_plates

    def mark_as_captured(self, plate_text):
        """Fast mark as captured"""
        if not plate_text:
            return
        
        plate_text = plate_text.upper().strip()
        
        with self.captured_plates_lock:
            self.captured_plates.add(plate_text)
            self.session_plates.add(plate_text)

    def process_frame_ultra_fast(self, frame, fps=30):
        """Ultra-fast frame processing with improved entry/exit detection"""
        annotated_frame = frame.copy()
        if not self.should_process_frame():
            # Show processing status even when skipping
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(annotated_frame, f"⚡ ULTRA-FAST STREAM + IMPROVED ENTRY/EXIT - {timestamp} | {fps:.1f} FPS",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with self.stats_lock:
                total_vehicles = len(self.captured_plates)
                session_count = len(self.session_plates)
                full_cars = self.detection_stats.get('full_cars_captured', 0)
                entries = self.detection_stats.get('entries', 0)
                exits = self.detection_stats.get('exits', 0)

            cv2.putText(annotated_frame,
                       f"📊 Live Tracking | Session: {session_count} | Total: {total_vehicles} | Cars: {full_cars} | 🟢Entry: {entries} | 🔴Exit: {exits}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            return annotated_frame

        try:
            # Ultra-fast parallel detection
            detected_cars = self.detect_cars_ultra_fast(frame)
            
            # Optimized plate detection
            plate_results = self.plate_model(frame,
                                           conf=self.plate_confidence_threshold,
                                           iou=self.iou_threshold,
                                           imgsz=640,  # Smaller for speed
                                           verbose=False,
                                           half=True if torch.cuda.is_available() else False)

        except Exception as e:
            logger.error(f"Ultra-fast detection failed: {e}")
            cv2.putText(annotated_frame, "DETECTION ERROR", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame

        # Annotate frame
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"⚡ ULTRA-FAST + IMPROVED ENTRY/EXIT - {timestamp} | {fps:.1f} FPS",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Extract plate detections quickly
        plate_detections = []
        run_ocr = self.should_run_ocr()

        for result in plate_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    if (x2 - x1) >= 35 and (y2 - y1) >= 10:
                        plate_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        })

        # Draw detected cars with tracking info
        for car in detected_cars:
            car_x1, car_y1, car_x2, car_y2 = car['bbox']
            cv2.rectangle(annotated_frame, (car_x1, car_y1), (car_x2, car_y2), (255, 0, 0), 2)
            car_label = f"{car['class_name'].upper()} {car['confidence']:.2f}"
            track_label = f"ID:{car.get('vehicle_id', 'N/A')}"
            cv2.putText(annotated_frame, car_label, (car_x1, car_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(annotated_frame, track_label, (car_x1, car_y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        new_captures = 0

        # Process cars with associated plates
        for car in detected_cars:
            associated_plate = self.find_plate_in_car_fast(frame, car['bbox'], plate_detections)
            if associated_plate:
                x1, y1, x2, y2 = associated_plate['bbox']
                confidence = associated_plate['confidence']
                plate_img = frame[y1:y2, x1:x2]
                
                plate_text = ""
                if run_ocr or confidence > 0.6:
                    plate_text = self.ocr_engine.lightning_fast_ocr(plate_img)

                if (plate_text and
                    self.ocr_engine.is_valid_license_plate(plate_text) and
                    not self.is_already_captured(plate_text) and
                    not self.is_recent_detection_fast(plate_text, associated_plate['bbox'])):

                    self.mark_as_captured(plate_text)
                    self.save_images_ultra_fast(frame, plate_img, associated_plate['bbox'],
                                              plate_text, confidence * 100, car)
                    self.update_detection_stats_fast(plate_text)
                    new_captures += 1
                    logger.info(f"⚡🚗 ULTRA CAPTURE: {plate_text} ({car['class_name']}) [Track: {car.get('vehicle_id', 'N/A')}]")
                    
                    car_x1, car_y1, car_x2, car_y2 = car['bbox']
                    cv2.rectangle(annotated_frame, (car_x1 - 2, car_y1 - 2), (car_x2 + 2, car_y2 + 2), (0, 255, 255), 3)
                    cv2.putText(annotated_frame, "⚡ CAPTURED", (car_x1, car_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                display_text = plate_text if plate_text else "..."
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Process standalone plates
        for plate in plate_detections:
            already_processed = any(self.find_plate_in_car_fast(frame, car['bbox'], [plate]) for car in detected_cars)
            if not already_processed:
                x1, y1, x2, y2 = plate['bbox']
                confidence = plate['confidence']
                plate_img = frame[y1:y2, x1:x2]

                plate_text = ""
                if run_ocr or confidence > 0.6:
                    plate_text = self.ocr_engine.lightning_fast_ocr(plate_img)

                if (plate_text and
                    self.ocr_engine.is_valid_license_plate(plate_text) and
                    not self.is_already_captured(plate_text) and
                    not self.is_recent_detection_fast(plate_text, plate['bbox'])):
                    
                    self.mark_as_captured(plate_text)
                    self.save_images_ultra_fast(frame, plate_img, plate['bbox'],
                                              plate_text, confidence * 100, None)
                    self.update_detection_stats_fast(plate_text)
                    new_captures += 1
                    logger.info(f"⚡📄 STANDALONE PLATE: {plate_text}")
                    cv2.rectangle(annotated_frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 0), 3)

                display_text = plate_text if plate_text else "..."
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Status display
        with self.stats_lock:
            total_vehicles = len(self.captured_plates)
            session_count = len(self.session_plates)
            full_cars = self.detection_stats.get('full_cars_captured', 0)
            entries = self.detection_stats.get('entries', 0)
            exits = self.detection_stats.get('exits', 0)
            mongodb_status = "🟢" if self.mongo_manager.is_connected else "🔴"

        cv2.putText(annotated_frame,
                   f"📊 Session: {session_count} | Total: {total_vehicles} | Cars: {full_cars} | DB {mongodb_status}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(annotated_frame,
                   f"⚡ 🟢Entry: {entries} | 🔴Exit: {exits} | Detected: {len(detected_cars)} Cars, {len(plate_detections)} Plates | New: {new_captures}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        ocr_status = "🔍 OCR" if run_ocr else "⏸️"
        cv2.putText(annotated_frame, f"Frame #{self.frame_count} | {ocr_status}",
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame


    def update_detection_stats_fast(self, plate_text):
        """Fast statistics update"""
        with self.stats_lock:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.detection_stats['total_detections'] += 1
            self.detection_stats['unique_plates'] = len(self.captured_plates)
            self.detection_stats['last_detection'] = current_time
            
            # Quick plate list update
            found = False
            for plate_entry in self.detection_stats['plates_list'][-20:]:  # Check only recent entries
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
                    'source': 'Ultra-Fast Stream + Improved Entry/Exit Detection'
                })
                
                # Keep list manageable
                if len(self.detection_stats['plates_list']) > 100:
                    self.detection_stats['plates_list'] = self.detection_stats['plates_list'][-75:]

    def get_enhanced_stats(self):
        """Get comprehensive statistics with performance data"""
        with self.stats_lock:
            stats = self.detection_stats.copy()
            
            mongo_stats = self.mongo_manager.get_enhanced_stats()
            
            stats.update({
                'session_plates_count': len(self.session_plates),
                'total_known_plates': len(self.captured_plates),
                'mongodb_total_plates': mongo_stats.get('total_vehicles', 0),
                'processing_interval': self.processing_interval,
                'ocr_interval': self.ocr_interval,
                'frame_count': self.frame_count,
                'active_vehicle_tracks': len(self.vehicle_tracker),
                'optimization_level': 'Ultra-Fast Stream + Improved Entry/Exit + Duplicate Cleanup',
                'mongodb_status': mongo_stats.get('database_status', 'Unknown'),
                'full_cars_captured': mongo_stats.get('full_cars_captured', 0),
                'plate_images_captured': mongo_stats.get('plate_images_captured', 0),
                'mongodb_entries': mongo_stats.get('entries', 0),
                'mongodb_exits': mongo_stats.get('exits', 0),
                'front_detections': mongo_stats.get('front_detections', 0),
                'back_detections': mongo_stats.get('back_detections', 0),
                'color_preservation': True,
                'duplicate_cleanup_enabled': True,
                'auto_cleanup_enabled': AUTO_CLEANUP_ENABLED,
                'performance_mode': 'ultra_fast',
                'vehicle_tracking_enabled': True,
                'improved_entry_exit_detection': True
            })
            
            return stats

    def get_vehicle_records(self, limit=50, query=None):
        """Get vehicle records with tracking information"""
        try:
            mongo_records = self.mongo_manager.get_vehicle_records(limit, query)
            
            with self.vehicle_records_lock:
                local_records = self.vehicle_records[-limit:] if not query else []
            
            all_records = mongo_records + local_records
            seen_ids = set()
            unique_records = []
            
            for record in all_records:
                record_id = record.get('detection_id', '')
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    unique_records.append(record)
            
            unique_records.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            return unique_records[:limit]
            
        except Exception as e:
            logger.error(f"Error getting vehicle records: {e}")
            return []

    def cleanup(self):
        """Enhanced cleanup"""
        logger.info("🧹 Cleaning up ultra-fast system...")
        
        if hasattr(self, 'image_processor_executor'):
            self.image_processor_executor.shutdown(wait=True)
        
        if self.mongo_manager:
            self.mongo_manager.close_connection()
        
        logger.info("✅ Ultra-fast cleanup completed")


# Global instances
reader = None
ultra_fast_processor = None

def generate_ultra_fast_frames():
    """Ultra-fast frame generator with optimized error handling"""
    global reader, ultra_fast_processor
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    fps_calculation_interval = 10  # Calculate FPS every 10 frames
    error_count = 0
    max_errors = 30
    
    while True:
        try:
            if not reader or not reader.is_healthy():
                logger.warning("⚠️ Stream unhealthy, showing error frame")
                error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(error_frame, "🚨 STREAM CONNECTION ISSUE", (400, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(error_frame, "Ultra-fast reconnection in progress...", (450, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)  # Faster retry
                continue
            
            frame = reader.get_frame()
            if frame is None:
                error_count += 1
                if error_count > max_errors:
                    logger.error("Too many frame read errors")
                    break
                time.sleep(0.01)
                continue
            
            error_count = 0
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS more frequently
            if fps_counter >= fps_calculation_interval:
                current_time = time.time()
                time_diff = current_time - fps_start_time
                if time_diff > 0:
                    current_fps = fps_counter / time_diff
                fps_counter = 0
                fps_start_time = current_time
            
            try:
                processed_frame = ultra_fast_processor.process_frame_ultra_fast(frame, current_fps)
            except Exception as e:
                logger.error(f"Ultra-fast processing error: {e}")
                processed_frame = frame
                cv2.putText(processed_frame, "⚠️ PROCESSING ERROR - RAW FEED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Ultra-fast encoding
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80,  # Good quality
                               cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                               cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
                ret, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    logger.warning("Frame encoding failed")
                    time.sleep(0.005)
                    
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                time.sleep(0.005)
                
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            time.sleep(0.05)


# Enhanced Flask routes
@app.route('/')
def index():
    # This assumes dashboard.html is at the root of DASHBOARD_DIR
    return send_from_directory(DASHBOARD_DIR, 'dashboard.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_ultra_fast_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0'})

@app.route('/api/enhanced_stats')
def get_enhanced_stats_route():
    """Get enhanced statistics with ultra-fast performance data"""
    global ultra_fast_processor
    if ultra_fast_processor:
        stats = ultra_fast_processor.get_enhanced_stats()
        
        # Add stream health info
        if reader:
            stream_stats = reader.get_stats()
            stats.update({
                'stream_connected': stream_stats['connected'],
                'stream_healthy': stream_stats['healthy'],
                'stream_reconnects': stream_stats['reconnect_attempts'],
                'stream_current_fps': stream_stats['current_fps'],
                'stream_frames_dropped': stream_stats['frames_dropped']
            })
        return jsonify(stats)
    
    return jsonify({
        'total_detections': 0,
        'unique_plates': 0,
        'performance_mode': 'ultra_fast',
        'stream_status': 'Processor not available'
    })

@app.route('/api/vehicle_records')
def get_vehicle_records_route():
    """Get vehicle records with tracking data"""
    global ultra_fast_processor
    if ultra_fast_processor:
        limit = int(request.args.get('limit', 50))
        query = request.args.get('query', '')
        
        records = ultra_fast_processor.get_vehicle_records(limit, query if query else None)
        return jsonify({
            'records': records,
            'total': len(records),
            'source': 'Ultra-Fast System + Improved Entry/Exit',
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({
        'records': [], 
        'total': 0, 
        'error': 'Processor not available',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/cleanup_duplicates', methods=['POST'])
def cleanup_duplicates_route():
    """Clean up duplicate records"""
    global ultra_fast_processor
    if not ultra_fast_processor:
        return jsonify({'error': 'Processor not available'}), 500
    
    try:
        data = request.get_json() or {}
        plate_number = data.get('plate_number', '').strip()
        
        if plate_number:
            result = ultra_fast_processor.duplicate_manager.cleanup_duplicates(plate_number)
        else:
            result = ultra_fast_processor.duplicate_manager.cleanup_duplicates()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup_orphaned_images', methods=['POST'])
def cleanup_orphaned_images_route():
    """Clean up orphaned image files"""
    global ultra_fast_processor
    if not ultra_fast_processor:
        return jsonify({'error': 'Processor not available'}), 500
    
    try:
        result = ultra_fast_processor.duplicate_manager.cleanup_orphaned_images()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/duplicate_statistics')
def get_duplicate_statistics_route():
    """Get duplicate statistics"""
    global ultra_fast_processor
    if not ultra_fast_processor:
        return jsonify({'error': 'Processor not available'}), 500
    
    try:
        result = ultra_fast_processor.duplicate_manager.get_duplicate_statistics()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_vehicles', methods=['POST'])
def search_vehicles_route():
    """Search vehicle records"""
    global ultra_fast_processor
    if not ultra_fast_processor:
        return jsonify({'error': 'Processor not available'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        records = ultra_fast_processor.get_vehicle_records(100, query if query else None)
        
        return jsonify({
            'records': records,
            'total': len(records),
            'query': query,
            'source': 'Ultra-Fast Search + Improved Entry/Exit'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Serve captured images"""
    try:
        import urllib.parse
        decoded_path = urllib.parse.unquote(image_path)
        
        allowed_dirs = [CAR_IMAGES_DIR, PLATE_IMAGES_DIR, SCREENSHOTS_DIR]
        # This is a security measure to prevent directory traversal attacks
        full_path = os.path.abspath(decoded_path)
        
        is_allowed = False
        for allowed_dir in allowed_dirs:
            if os.path.commonpath([full_path, os.path.abspath(allowed_dir)]) == os.path.abspath(allowed_dir):
                is_allowed = True
                break
        
        if not is_allowed:
            return jsonify({'error': 'Access denied'}), 403
        
        if os.path.exists(full_path):
            return send_file(full_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        return jsonify({'error': 'Failed to serve image'}), 500

@app.route('/api/stream_health')
def get_stream_health_route():
    """Get ultra-fast stream health"""
    global reader
    if reader:
        stats = reader.get_stats()
        return jsonify({
            'stream_health': stats,
            'status': 'healthy' if stats['healthy'] else 'unhealthy',
            'performance': 'ultra_fast',
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'error': 'Stream reader not available'})

@app.route('/api/system_info')
def get_system_info_route():
    """Get system information"""
    global ultra_fast_processor
    
    system_info = {
        'system_name': 'Ultra-Fast Vehicle Detection with Improved Entry/Exit',
        'version': '5.0',
        'performance_mode': 'ultra_fast',
        'features': {
            'ultra_fast_processing': True,
            'ultra_stable_rtsp_stream': True,
            'improved_entry_exit_detection': True,
            'vehicle_tracking': True,
            'multi_modal_orientation_analysis': True,
            'temporal_consistency': True,
            'movement_vector_analysis': True,
            'duplicate_cleanup': True,
            'auto_cleanup': AUTO_CLEANUP_ENABLED,
            'mongodb_storage': True,
            'color_preservation': True,
            'gpu_acceleration': torch.cuda.is_available(),
            'optimized_ocr': True
        },
        'performance_optimizations': {
            'processing_interval': PROCESSING_INTERVAL,
            'ocr_interval': OCR_PROCESSING_INTERVAL,
            'target_fps': TARGET_FPS,
            'minimal_buffering': True,
            'fast_reconnection': True,
            'optimized_encoding': True,
            'concurrent_processing': True
        }
    }
    
    if ultra_fast_processor:
        system_info.update({
            'processor_status': 'Ultra-Fast Active',
            'mongodb_connected': ultra_fast_processor.mongo_manager.is_connected,
            'active_vehicle_tracks': len(ultra_fast_processor.vehicle_tracker),
            'duplicate_manager_active': True
        })
        
        if reader:
            stream_stats = reader.get_stats()
            system_info.update({
                'stream_status': 'healthy' if stream_stats['healthy'] else 'unhealthy',
                'stream_fps': stream_stats['current_fps'],
                'frames_dropped': stream_stats['frames_dropped']
            })
    
    return jsonify(system_info)

@app.route('/api/reset_session')
def reset_session_route():
    """Reset current session data"""
    global ultra_fast_processor
    if ultra_fast_processor:
        with ultra_fast_processor.captured_plates_lock:
            old_count = len(ultra_fast_processor.session_plates)
            ultra_fast_processor.session_plates.clear()
            ultra_fast_processor.recent_detections.clear()
        
        with ultra_fast_processor.vehicle_records_lock:
            ultra_fast_processor.vehicle_records.clear()
        
        # Clear vehicle tracking
        ultra_fast_processor.vehicle_tracker.clear()
        ultra_fast_processor.track_id_counter = 0
        
        with ultra_fast_processor.stats_lock:
            ultra_fast_processor.detection_stats['entries'] = 0
            ultra_fast_processor.detection_stats['exits'] = 0
        
        return jsonify({
            'status': 'success',
            'message': f'Ultra-fast session reset - {old_count} plates cleared',
            'features': 'Ultra-Fast Stream + Improved Entry/Exit Active'
        })
    return jsonify({'status': 'error', 'message': 'Processor not available'})


def main():
    global reader, ultra_fast_processor
    
    try:
        logger.info("="*90)
        logger.info("⚡🚗 ULTRA-FAST VEHICLE DETECTION SYSTEM")
        logger.info("🎯 Optimized Stream + Improved Entry/Exit Detection")
        logger.info("="*90)
        logger.info("🚀 ULTRA-FAST FEATURES:")
        logger.info("   ⚡ Ultra-fast RTSP stream processing (optimized for speed)")
        logger.info("   🔄 Lightning-fast automatic reconnection")
        logger.info("   📊 Real-time stream performance monitoring")
        logger.info("   🚗 Complete vehicle image capture (ultra-fast)")
        logger.info("   🏷️ License plate close-up capture (lightning-fast OCR)")
        logger.info("   🎯 Improved entry/exit detection with multi-modal analysis")
        logger.info("   📍 Advanced vehicle tracking and movement analysis")
        logger.info("   🧠 Temporal consistency and orientation confidence")
        logger.info("   🧹 Intelligent duplicate cleanup")
        logger.info("   💾 Optimized MongoDB batch processing")
        logger.info("   🎨 Full color preservation")
        logger.info("   ⚡ GPU acceleration (if available)")
        logger.info("   🔥 Concurrent processing and ultra-fast encoding")
        logger.info("="*90)
        
        # GPU check
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 GPU ACCELERATION: {gpu_name}")
            logger.info(f"🔥 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("💻 CPU Processing (ultra-optimized)")
        
        # System checks
        logger.info("🔍 Ultra-fast system verification...")
        
        # Test RTSP connection
        test_cap = cv2.VideoCapture(RTSP_STREAM)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            logger.info(f"✅ RTSP Stream: {'Ultra-Fast Ready' if ret else 'Connected (initializing)'}")
        else:
            logger.error("❌ RTSP Stream: Connection failed")
        test_cap.release()
        
        # Test MongoDB
        try:
            test_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            test_client.admin.command('ping')
            test_client.close()
            logger.info("✅ MongoDB: Ultra-Fast Connected")
        except:
            logger.warning("⚠️ MongoDB: Connection issues (will retry)")
        
        # Create directories
        for dir_name, dir_path in [
            ("Ultra-Fast Car Images", CAR_IMAGES_DIR),
            ("Lightning Plate Images", PLATE_IMAGES_DIR),
            ("Screenshots", SCREENSHOTS_DIR)
        ]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 Created {dir_name} directory")
            else:
                logger.info(f"✅ {dir_name} directory ready")
        
        # Test models
        if os.path.exists(MODEL_PATH):
            logger.info(f"✅ License plate model: Ultra-Fast Ready")
        else:
            logger.error(f"❌ License plate model not found: {MODEL_PATH}")
            return
        
        logger.info(f"✅ Car detection model: Ultra-Fast Ready (YOLOv8n)")
        
        # Initialize ultra-fast system
        logger.info("⚡ Initializing ULTRA-FAST detection system...")
        ultra_fast_processor = UltraFastCarDetector(MODEL_PATH, YOLO_CAR_MODEL)
        
        # Initialize ultra-stable RTSP reader
        logger.info("📡 Initializing ULTRA-STABLE RTSP connection...")
        reader = UltraStableRTSPReader(RTSP_STREAM)
        
        # Shorter wait for ultra-fast initialization
        time.sleep(2)
        
        if not reader.is_connected:
            logger.error("❌ Failed to establish ultra-stable RTSP connection. Exiting.")
            return
        
        # Get initial statistics
        with ultra_fast_processor.captured_plates_lock:
            total_plates = len(ultra_fast_processor.captured_plates)
            session_plates = len(ultra_fast_processor.session_plates)
        
        mongo_stats = ultra_fast_processor.mongo_manager.get_enhanced_stats()
        
        logger.info("="*90)
        logger.info("⚡🚗🎯 ULTRA-FAST SYSTEM READY!")
        logger.info("="*90)
        logger.info(f"📊 Database plates: {total_plates}")
        logger.info(f"📊 Session plates: {session_plates}")
        logger.info(f"🚗 Full cars captured: {mongo_stats.get('full_cars_captured', 0)}")
        logger.info(f"🏷️ Plate images: {mongo_stats.get('plate_images_captured', 0)}")
        logger.info(f"🟢 Entries recorded: {mongo_stats.get('entries', 0)}")
        logger.info(f"🔴 Exits recorded: {mongo_stats.get('exits', 0)}")
        logger.info(f"🎯 Active vehicle tracks: {len(ultra_fast_processor.vehicle_tracker)}")
        logger.info(f"📡 Stream status: {'Ultra-Healthy' if reader.is_healthy() else 'Connected'}")
        logger.info(f"💾 MongoDB: {'Ultra-Fast Connected' if ultra_fast_processor.mongo_manager.is_connected else 'Retrying'}")
        logger.info(f"🌐 Ultra-Fast Dashboard: http://127.0.0.1:5000")
        logger.info("="*90)
        
        # Start Flask application
        logger.info("🌐 Starting ultra-fast web server...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, 
                use_reloader=False, use_debugger=False)
        
    except KeyboardInterrupt:
        logger.info("🛑 Ultra-fast system stopped by user")
    except Exception as e:
        logger.error(f"❌ Ultra-fast system error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🔄 Shutting down ultra-fast system...")
        if ultra_fast_processor:
            ultra_fast_processor.cleanup()
        if reader:
            reader.release()
        logger.info("✅ Ultra-fast system shutdown complete")


if __name__ == "__main__":
    main()