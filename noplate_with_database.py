# Enhanced Car Detection System with Entry/Exit Detection
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

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Enhanced Configuration
MODEL_PATH = "runs/detect/license_plate_detection/weights/best.pt"
YOLO_CAR_MODEL = "yolov8n.pt"  # Pre-trained YOLOv8 for car detection
VEHICLE_ORIENTATION_MODEL = "yolov8n.pt"  # We'll use keypoint detection for orientation
DASHBOARD_DIR = r"D:\vehicle_speed\no_plate"
SCREENSHOTS_DIR = os.path.join(DASHBOARD_DIR, "screenshots")
CAR_IMAGES_DIR = os.path.join(DASHBOARD_DIR, "car_images")
PLATE_IMAGES_DIR = os.path.join(DASHBOARD_DIR, "plate_images")
RTSP_STREAM = "rtsp://camera:Rj18cb4820!@10.10.97.12:554/"

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "enhanced_vehicle_db"
MONGO_COLLECTION_NAME = "vehicle_records"

# Enhanced Performance Configuration
MAX_QUEUE_SIZE = 2
PROCESSING_INTERVAL = 3
OCR_PROCESSING_INTERVAL = 6
FRAME_SKIP_THRESHOLD = 5
MEMORY_CLEANUP_INTERVAL = 100

# Enhanced Car detection parameters
CAR_DETECTION_CONFIDENCE = 0.4
MIN_CAR_WIDTH = 150
MIN_CAR_HEIGHT = 100
CAR_EXPANSION_RATIO = 0.15  # Expand car bbox by 15% for better capture
FULL_CAR_CAPTURE_RATIO = 1.5  # Multiplier for full car capture

# Car classes from COCO dataset
CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Entry/Exit Detection Configuration
ORIENTATION_CONFIDENCE_THRESHOLD = 0.6
ENTRY_EXIT_HISTORY_SIZE = 10  # Track recent detections for better accuracy

# Ensure directories exist
for directory in [SCREENSHOTS_DIR, CAR_IMAGES_DIR, PLATE_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask app
app = Flask(__name__)

class VehicleOrientationDetector:
    """Advanced vehicle orientation detector for entry/exit detection"""
    def __init__(self):
        self.orientation_history = deque(maxlen=ENTRY_EXIT_HISTORY_SIZE)
        self.vehicle_tracks = {}  # Track vehicles across frames
        
        # Load additional models for orientation detection
        try:
            # We'll use pose estimation and geometric analysis
            self.pose_model = None  # Can be initialized with pose estimation model
            logger.info("✅ Vehicle orientation detector initialized")
        except Exception as e:
            logger.error(f"❌ Orientation detector initialization error: {e}")
    
    def detect_vehicle_orientation(self, vehicle_img, vehicle_bbox, full_frame):
        """
        Detect if vehicle is front-facing (entering) or back-facing (exiting)
        Returns: 'front' (entering), 'back' (exiting), or 'unknown'
        """
        try:
            if vehicle_img is None or vehicle_img.size == 0:
                return 'unknown'
            
            # Method 1: License plate position analysis
            plate_orientation = self._analyze_plate_position(vehicle_img, vehicle_bbox, full_frame)
            
            # Method 2: Vehicle feature analysis
            feature_orientation = self._analyze_vehicle_features(vehicle_img)
            
            # Method 3: Geometric analysis
            geometric_orientation = self._analyze_vehicle_geometry(vehicle_img, vehicle_bbox)
            
            # Method 4: Headlight/Taillight detection
            light_orientation = self._detect_lights(vehicle_img)
            
            # Combine all methods for final decision
            orientations = [plate_orientation, feature_orientation, geometric_orientation, light_orientation]
            orientation_counts = {'front': 0, 'back': 0, 'unknown': 0}
            
            for orient in orientations:
                if orient in orientation_counts:
                    orientation_counts[orient] += 1
            
            # Determine final orientation
            if orientation_counts['front'] > orientation_counts['back']:
                final_orientation = 'front'
            elif orientation_counts['back'] > orientation_counts['front']:
                final_orientation = 'back'
            else:
                final_orientation = 'unknown'
            
            # Add to history for temporal consistency
            self.orientation_history.append(final_orientation)
            
            # Use historical data for more stable detection
            if len(self.orientation_history) >= 3:
                recent_orientations = list(self.orientation_history)[-3:]
                front_count = recent_orientations.count('front')
                back_count = recent_orientations.count('back')
                
                if front_count > back_count:
                    return 'front'
                elif back_count > front_count:
                    return 'back'
            
            return final_orientation
            
        except Exception as e:
            logger.error(f"Orientation detection error: {e}")
            return 'unknown'
    
    def _analyze_plate_position(self, vehicle_img, vehicle_bbox, full_frame):
        """Analyze license plate position to determine orientation"""
        try:
            # In most countries, front plates are positioned differently than rear plates
            # Front plates are usually higher and more centered
            # Rear plates are usually lower and may be offset
            
            # Get vehicle dimensions
            vx1, vy1, vx2, vy2 = vehicle_bbox
            vehicle_height = vy2 - vy1
            vehicle_width = vx2 - vx1
            
            # Look for license plate regions in the vehicle
            # Use edge detection directly on color image
            edges = cv2.Canny(vehicle_img, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            plate_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # License plate aspect ratio is typically 2:1 to 5:1
                if 1.5 < aspect_ratio < 6 and w > 30 and h > 8:
                    plate_regions.append((x, y, w, h, y/vehicle_height))  # Include relative position
            
            if not plate_regions:
                return 'unknown'
            
            # Analyze plate positions
            avg_vertical_position = sum(region[4] for region in plate_regions) / len(plate_regions)
            
            # Front plates are typically in upper 60% of vehicle
            # Rear plates are typically in lower 40% of vehicle
            if avg_vertical_position < 0.6:
                return 'front'
            else:
                return 'back'
                
        except Exception as e:
            logger.error(f"Plate position analysis error: {e}")
            return 'unknown'
    
    def _analyze_vehicle_features(self, vehicle_img):
        """Analyze vehicle features like grille, bumper patterns"""
        try:
            # Work with original color image
            processed_img = vehicle_img.copy()
            
            # Convert to single channel for morphological operations
            if len(processed_img.shape) == 3:
                single_channel = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            else:
                single_channel = processed_img
            
            # Front vehicles typically have more horizontal lines (grille)
            # Rear vehicles typically have more vertical/curved lines (taillights)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            horizontal_lines = cv2.morphologyEx(single_channel, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_score = np.sum(horizontal_lines > 0)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            vertical_lines = cv2.morphologyEx(single_channel, cv2.MORPH_OPEN, vertical_kernel)
            vertical_score = np.sum(vertical_lines > 0)
            
            # Front typically has more horizontal features
            if horizontal_score > vertical_score * 1.2:
                return 'front'
            elif vertical_score > horizontal_score * 1.2:
                return 'back'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Vehicle feature analysis error: {e}")
            return 'unknown'
    
    def _analyze_vehicle_geometry(self, vehicle_img, vehicle_bbox):
        """Analyze vehicle geometric properties"""
        try:
            # Work with original color image
            processed_img = vehicle_img.copy()
            
            # Convert to single channel for analysis
            if len(processed_img.shape) == 3:
                single_channel = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            else:
                single_channel = processed_img
            
            # Analyze the shape profile of the vehicle
            height, width = single_channel.shape
            
            # Calculate intensity gradients
            grad_x = cv2.Sobel(single_channel, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(single_channel, cv2.CV_64F, 0, 1, ksize=3)
            
            # Analyze upper and lower regions
            upper_region = single_channel[:height//3, :]
            lower_region = single_channel[2*height//3:, :]
            
            upper_complexity = np.std(upper_region)
            lower_complexity = np.std(lower_region)
            
            # Front of vehicles typically has more complexity in upper region (grille, headlights)
            # Rear of vehicles typically has more complexity in lower region (bumper, exhaust)
            if upper_complexity > lower_complexity * 1.1:
                return 'front'
            elif lower_complexity > upper_complexity * 1.1:
                return 'back'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Vehicle geometry analysis error: {e}")
            return 'unknown'
    
    def _detect_lights(self, vehicle_img):
        """Detect headlights vs taillights"""
        try:
            # Convert to HSV for better light detection
            hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
            
            # Detect bright regions (lights)
            bright_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
            
            # Detect red regions (taillights)
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Count bright and red pixels
            bright_pixels = np.sum(bright_mask > 0)
            red_pixels = np.sum(red_mask > 0)
            
            # If more red pixels, likely rear (taillights)
            # If more bright pixels, likely front (headlights)
            if red_pixels > bright_pixels and red_pixels > 100:
                return 'back'
            elif bright_pixels > red_pixels and bright_pixels > 150:
                return 'front'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Light detection error: {e}")
            return 'unknown'
    
    def get_entry_exit_status(self, orientation):
        """Convert orientation to entry/exit status"""
        if orientation == 'front':
            return 'Vehicle Entered'
        elif orientation == 'back':
            return 'Vehicle Exited'
        else:
            return 'Vehicle Detected'


class OptimizedOCREngine:
    """Optimized OCR engine with enhanced preprocessing - keeping original colors"""
    def __init__(self):
        self.cache_lock = threading.Lock()
        self.preprocessing_cache = {}
        self.valid_plate_pattern = re.compile(r'^[A-Z0-9]{5,8}$')
        
    def fast_preprocess(self, img):
        """Quick image preprocessing for OCR - keeping original colors"""
        try:
            # Keep original color image
            processed = img.copy()
            
            # Basic noise reduction on color image
            denoised = cv2.fastNlMeansDenoisingColored(processed)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
        except Exception as e:
            logger.error(f"Fast preprocessing error: {e}")
            return img

    def smart_ensemble_ocr(self, img, use_advanced=False):
        """Enhanced OCR with multiple methods"""
        try:
            # Generate cache key
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            
            # Check cache
            with self.cache_lock:
                if img_hash in self.preprocessing_cache:
                    return self.preprocessing_cache[img_hash]
            
            # Preprocess
            processed = self.fast_preprocess(img)
            
            # Basic OCR
            text = pytesseract.image_to_string(processed, 
                                             config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            # Clean and validate
            text = ''.join(c for c in text if c.isalnum()).upper()
            
            if self.is_valid_license_plate(text):
                with self.cache_lock:
                    self.preprocessing_cache[img_hash] = text
                return text
            
            return ""
            
        except Exception as e:
            logger.error(f"Smart OCR error: {e}")
            return ""

    def is_valid_license_plate(self, text):
        """Validate license plate format"""
        if not text:
            return False
        return bool(self.valid_plate_pattern.match(text))


class EnhancedMongoDBManager:
    """Enhanced MongoDB manager for vehicle records with entry/exit data"""
    def __init__(self, uri=MONGO_URI, db_name=MONGO_DB_NAME, collection_name=MONGO_COLLECTION_NAME):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connection_lock = RLock()
        self.is_connected = False
        self.write_queue = queue.Queue()
        self.batch_size = 5
        self.batch_timeout = 1.5
        
        self._connect()
        self._start_batch_writer()

    def _connect(self):
        """Enhanced connection with optimized settings"""
        try:
            self.client = MongoClient(
                self.uri, 
                serverSelectionTimeoutMS=5000,
                maxPoolSize=15,
                minPoolSize=3,
                maxIdleTimeMS=30000
            )
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Create enhanced indexes including entry/exit
            try:
                self.collection.create_index("plate_number")
                self.collection.create_index("timestamp")
                self.collection.create_index("detection_id")
                self.collection.create_index("entry_exit_status")
                self.collection.create_index("vehicle_orientation")
                self.collection.create_index([("plate_number", 1), ("timestamp", -1)])
            except:
                pass
            
            self.is_connected = True
            logger.info("✅ Enhanced MongoDB connected with entry/exit support")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.is_connected = False
            return False

    def _start_batch_writer(self):
        """Start enhanced batch writer"""
        if self.is_connected:
            self.batch_writer_thread = threading.Thread(target=self._batch_writer_loop, daemon=True)
            self.batch_writer_thread.start()

    def _batch_writer_loop(self):
        """Enhanced batch writer loop"""
        batch = []
        last_write = time.time()
        
        while True:
            try:
                try:
                    item = self.write_queue.get(timeout=0.3)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_write > self.batch_timeout)):
                    
                    if batch and self.is_connected:
                        try:
                            self.collection.insert_many(batch, ordered=False)
                            logger.info(f"💾 Enhanced batch saved {len(batch)} vehicle records with entry/exit data")
                            batch.clear()
                            last_write = current_time
                        except Exception as e:
                            logger.error(f"Enhanced batch write error: {e}")
                            batch.clear()
                
            except Exception as e:
                logger.error(f"Enhanced batch writer error: {e}")
                time.sleep(1)

    def save_vehicle_record_async(self, plate_number, confidence, car_image_path, plate_image_path, 
                                bbox_info, vehicle_info=None, additional_data=None, 
                                entry_exit_status=None, vehicle_orientation=None):
        """Save enhanced vehicle record with entry/exit data"""
        try:
            document = {
                "plate_number": str(plate_number).strip().upper(),
                "timestamp": datetime.now(),
                "confidence": float(confidence) if confidence else 0.0,
                "car_image_path": str(car_image_path) if car_image_path else "",
                "plate_image_path": str(plate_image_path) if plate_image_path else "",
                "detection_id": f"{plate_number}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}",
                "bbox_info": bbox_info or {},
                "vehicle_info": vehicle_info or {},
                "processing_method": "Enhanced Full Car Capture with Entry/Exit Detection",
                "image_capture_enabled": True,
                "full_car_captured": bool(car_image_path),
                "processed": True,
                # New entry/exit fields
                "entry_exit_status": entry_exit_status or "Vehicle Detected",
                "vehicle_orientation": vehicle_orientation or "unknown",
                "is_entry": entry_exit_status == "Vehicle Entered" if entry_exit_status else None,
                "is_exit": entry_exit_status == "Vehicle Exited" if entry_exit_status else None
            }
            
            if additional_data:
                document.update(additional_data)
            
            if not self.write_queue.full():
                self.write_queue.put(document)
                return True
            else:
                logger.warning("Enhanced write queue full")
                return False
                
        except Exception as e:
            logger.error(f"Enhanced async save error: {e}")
            return False

    def get_vehicle_records(self, limit=50, query=None):
        """Get vehicle records with enhanced filtering including entry/exit"""
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
                    "processing_method": doc.get("processing_method", "Enhanced Full Car Detection"),
                    "full_car_captured": doc.get("full_car_captured", False),
                    # Entry/exit fields
                    "entry_exit_status": doc.get("entry_exit_status", "Vehicle Detected"),
                    "vehicle_orientation": doc.get("vehicle_orientation", "unknown"),
                    "is_entry": doc.get("is_entry"),
                    "is_exit": doc.get("is_exit")
                }
                records.append(record)
            
            return records
        except Exception as e:
            logger.error(f"Error getting vehicle records: {e}")
            return []

    def get_enhanced_stats(self):
        """Get comprehensive statistics including entry/exit data"""
        try:
            if not self.is_connected:
                return {
                    "total_vehicles": 0,
                    "unique_plates": 0,
                    "full_cars_captured": 0,
                    "entries": 0,
                    "exits": 0,
                    "database_status": "Disconnected"
                }
            
            # Get total records
            total = self.collection.estimated_document_count()
            
            # Get unique plates
            unique_plates = len(self.collection.distinct("plate_number"))
            
            # Count records with car images
            car_images_count = self.collection.count_documents({"car_image_path": {"$ne": ""}})
            
            # Count records with full car capture
            full_car_count = self.collection.count_documents({"full_car_captured": True})
            
            # Count records with plate images
            plate_images_count = self.collection.count_documents({"plate_image_path": {"$ne": ""}})
            
            # Count entries and exits
            entries_count = self.collection.count_documents({"is_entry": True})
            exits_count = self.collection.count_documents({"is_exit": True})
            
            # Get orientation statistics
            front_count = self.collection.count_documents({"vehicle_orientation": "front"})
            back_count = self.collection.count_documents({"vehicle_orientation": "back"})
            
            # Get recent activity (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            recent_count = self.collection.count_documents({"timestamp": {"$gte": yesterday}})
            
            # Get latest record
            latest_record = self.collection.find_one(sort=[("timestamp", -1)])
            latest_timestamp = latest_record.get("timestamp") if latest_record else None
            
            return {
                "total_vehicles": total,
                "unique_plates": unique_plates,
                "car_images_captured": car_images_count,
                "full_cars_captured": full_car_count,
                "plate_images_captured": plate_images_count,
                "entries": entries_count,
                "exits": exits_count,
                "front_detections": front_count,
                "back_detections": back_count,
                "recent_detections_24h": recent_count,
                "latest_detection": latest_timestamp,
                "database_status": "Connected"
            }
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return {
                "total_vehicles": 0,
                "unique_plates": 0,
                "full_cars_captured": 0,
                "entries": 0,
                "exits": 0,
                "database_status": f"Error: {str(e)}"
            }

    def close_connection(self):
        """Clean shutdown"""
        try:
            if self.client:
                self.client.close()
                self.is_connected = False
        except Exception as e:
            logger.error(f"Error closing enhanced connection: {e}")


class EnhancedFullCarDetector:
    """Enhanced detector for full car capture with license plates and entry/exit detection"""
    def __init__(self, plate_model_path, car_model_path=YOLO_CAR_MODEL):
        # Load YOLO models
        try:
            self.plate_model = YOLO(plate_model_path)
            self.car_model = YOLO(car_model_path)  # Pre-trained car detection model
            
            if torch.cuda.is_available():
                self.plate_model.to('cuda')
                self.car_model.to('cuda')
            logger.info(f"✅ Enhanced YOLO models loaded (Plate + Car + Entry/Exit)")
        except Exception as e:
            logger.error(f"❌ Failed to load enhanced YOLO models: {e}")
            raise
        
        # Initialize components
        self.ocr_engine = OptimizedOCREngine()
        self.mongo_manager = EnhancedMongoDBManager()
        self.orientation_detector = VehicleOrientationDetector()  # New orientation detector
        
        # Load existing plates
        self.captured_plates = set()
        self._load_existing_plates()
        
        # Enhanced detection parameters
        self.plate_confidence_threshold = 0.3
        self.car_confidence_threshold = CAR_DETECTION_CONFIDENCE
        self.iou_threshold = 0.4
        
        # Processing intervals
        self.frame_count = 0
        self.last_ocr_frame = 0
        self.processing_interval = PROCESSING_INTERVAL
        self.ocr_interval = OCR_PROCESSING_INTERVAL
        
        # Thread-safe collections
        self.captured_plates_lock = RLock()
        self.session_plates = set()
        self.recent_detections = deque(maxlen=50)
        self.vehicle_records = []
        self.vehicle_records_lock = RLock()
        
        # Background processors
        self.image_processor_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Enhanced statistics including entry/exit
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
            'processing_mode': 'Enhanced Full Car + Plate + Entry/Exit Detection'
        }
        
        # Memory management
        self.last_cleanup = 0

    def _load_existing_plates(self):
        """Load existing plates from enhanced database"""
        try:
            records = self.mongo_manager.get_vehicle_records(limit=1000)
            self.captured_plates = {record['plate_number'] for record in records if record['plate_number']}
            logger.info(f"📋 Loaded {len(self.captured_plates)} existing plates from enhanced database")
        except Exception as e:
            logger.error(f"Error loading existing plates: {e}")
            self.captured_plates = set()

    def should_process_frame(self):
        """Enhanced frame processing decision"""
        self.frame_count += 1
        
        if self.frame_count % self.processing_interval == 0:
            if self.frame_count - self.last_cleanup > MEMORY_CLEANUP_INTERVAL:
                self._cleanup_memory()
                self.last_cleanup = self.frame_count
            return True
        
        return False

    def should_run_ocr(self):
        """Enhanced OCR decision"""
        if self.frame_count - self.last_ocr_frame >= self.ocr_interval:
            self.last_ocr_frame = self.frame_count
            return True
        return False

    def _cleanup_memory(self):
        """Enhanced memory cleanup"""
        try:
            with self.ocr_engine.cache_lock:
                if len(self.ocr_engine.preprocessing_cache) > 30:
                    self.ocr_engine.preprocessing_cache.clear()
            
            # Limit vehicle records in memory
            with self.vehicle_records_lock:
                if len(self.vehicle_records) > 100:
                    self.vehicle_records = self.vehicle_records[-50:]
            
            gc.collect()
        except Exception as e:
            logger.error(f"Enhanced memory cleanup error: {e}")

    def detect_cars_in_frame(self, frame):
        """Detect cars using YOLO car model"""
        try:
            results = self.car_model(frame, conf=self.car_confidence_threshold, 
                                   classes=CAR_CLASSES, verbose=False)
            
            cars = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Size filtering
                        if (x2 - x1) >= MIN_CAR_WIDTH and (y2 - y1) >= MIN_CAR_HEIGHT:
                            cars.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.get_vehicle_class_name(class_id)
                            })
            
            return cars
        except Exception as e:
            logger.error(f"Car detection error: {e}")
            return []

    def get_vehicle_class_name(self, class_id):
        """Get vehicle class name from COCO class ID"""
        class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return class_names.get(class_id, 'vehicle')

    def find_plate_in_car(self, frame, car_bbox, plate_detections):
        """Find license plate within car boundary"""
        car_x1, car_y1, car_x2, car_y2 = car_bbox
        
        best_plate = None
        best_overlap = 0
        
        for plate in plate_detections:
            plate_x1, plate_y1, plate_x2, plate_y2 = plate['bbox']
            
            # Check if plate center is within car bounds
            plate_center_x = (plate_x1 + plate_x2) / 2
            plate_center_y = (plate_y1 + plate_y2) / 2
            
            if (car_x1 <= plate_center_x <= car_x2 and 
                car_y1 <= plate_center_y <= car_y2):
                
                # Calculate overlap ratio
                overlap_x1 = max(car_x1, plate_x1)
                overlap_y1 = max(car_y1, plate_y1)
                overlap_x2 = min(car_x2, plate_x2)
                overlap_y2 = min(car_y2, plate_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    plate_area = (plate_x2 - plate_x1) * (plate_y2 - plate_y1)
                    overlap_ratio = overlap_area / plate_area if plate_area > 0 else 0
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_plate = plate
        
        return best_plate

    def extract_full_car_image(self, frame, car_bbox, expansion_ratio=FULL_CAR_CAPTURE_RATIO):
        """Extract full car image with enhanced capture area"""
        try:
            x1, y1, x2, y2 = car_bbox
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate car dimensions
            car_w = x2 - x1
            car_h = y2 - y1
            
            # Enhanced expansion for full car capture
            expand_w = int(car_w * expansion_ratio * 0.3)
            expand_h = int(car_h * expansion_ratio * 0.3)
            
            # Calculate expanded bounding box
            full_car_x1 = max(0, x1 - expand_w)
            full_car_y1 = max(0, y1 - expand_h)
            full_car_x2 = min(frame_w, x2 + expand_w)
            full_car_y2 = min(frame_h, y2 + expand_h)
            
            # Extract full car region
            full_car_img = frame[full_car_y1:full_car_y2, full_car_x1:full_car_x2]
            full_car_bbox = (full_car_x1, full_car_y1, full_car_x2, full_car_y2)
            
            return full_car_img, full_car_bbox
            
        except Exception as e:
            logger.error(f"Error extracting full car image: {e}")
            return None, None

    def enhance_full_car_image(self, car_img):
        """Enhance full car image for better visibility and orientation detection"""
        try:
            if car_img is None or car_img.size == 0:
                return car_img
            
            # Resize if too small for better visibility
            h, w = car_img.shape[:2]
            min_width, min_height = 600, 400
            
            if w < min_width or h < min_height:
                scale = max(min_width/w, min_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                car_img = cv2.resize(car_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Enhanced image processing for better orientation detection
            # 1. Improve contrast and brightness
            enhanced = cv2.convertScaleAbs(car_img, alpha=1.15, beta=15)
            
            # 2. Apply CLAHE for better local contrast
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. Apply slight sharpening for better feature detection
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. Blend for natural look
            result = cv2.addWeighted(enhanced, 0.75, sharpened, 0.25, 0)
            
            return result
        except Exception as e:
            logger.error(f"Full car image enhancement error: {e}")
            return car_img

    def save_enhanced_full_car_images(self, frame, plate_img, plate_bbox, plate_text, 
                                    confidence, car_info):
        """Save full car and plate images with enhanced processing including entry/exit detection"""
        def _save_async():
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                
                # Extract and save full car image
                car_image_path = ""
                full_car_bbox = None
                vehicle_orientation = "unknown"
                entry_exit_status = "Vehicle Detected"
                
                if car_info:
                    full_car_img, full_car_bbox = self.extract_full_car_image(frame, car_info['bbox'])
                    
                    if full_car_img is not None and full_car_img.size > 0:
                        # Detect vehicle orientation (front/back) for entry/exit determination
                        vehicle_orientation = self.orientation_detector.detect_vehicle_orientation(
                            full_car_img, car_info['bbox'], frame)
                        entry_exit_status = self.orientation_detector.get_entry_exit_status(vehicle_orientation)
                        
                        # Update entry/exit statistics
                        with self.stats_lock:
                            if entry_exit_status == "Vehicle Entered":
                                self.detection_stats['entries'] += 1
                            elif entry_exit_status == "Vehicle Exited":
                                self.detection_stats['exits'] += 1
                        
                        # Save enhanced full car image with orientation info in filename
                        orientation_prefix = vehicle_orientation.upper() if vehicle_orientation != "unknown" else "UNK"
                        car_filename = f"FULLCAR_{orientation_prefix}_{plate_text}_{timestamp}.jpg"
                        car_filepath = os.path.join(CAR_IMAGES_DIR, car_filename)
                        
                        # Enhance full car image
                        car_enhanced = self.enhance_full_car_image(full_car_img)
                        success = cv2.imwrite(car_filepath, car_enhanced, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if success:
                            car_image_path = car_filepath
                            with self.stats_lock:
                                self.detection_stats['car_images_captured'] += 1
                                self.detection_stats['full_cars_captured'] += 1
                
                # Save enhanced plate close-up image
                plate_image_path = ""
                if plate_img is not None and plate_img.size > 0:
                    plate_filename = f"PLATE_{plate_text}_{timestamp}.jpg"
                    plate_filepath = os.path.join(PLATE_IMAGES_DIR, plate_filename)
                    
                    # Enhance plate image (keep original colors)
                    plate_enhanced = self.enhance_plate_image(plate_img)
                    success = cv2.imwrite(plate_filepath, plate_enhanced,
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        plate_image_path = plate_filepath
                        with self.stats_lock:
                            self.detection_stats['plate_images_captured'] += 1
                
                # Prepare enhanced vehicle info with entry/exit data
                vehicle_info = {
                    "type": car_info.get('class_name', 'vehicle') if car_info else 'unknown',
                    "car_confidence": car_info.get('confidence', 0.0) if car_info else 0.0,
                    "full_car_captured": bool(car_image_path),
                    "car_dimensions": {
                        "width": full_car_bbox[2] - full_car_bbox[0] if full_car_bbox else 0,
                        "height": full_car_bbox[3] - full_car_bbox[1] if full_car_bbox else 0
                    },
                    "orientation": vehicle_orientation,
                    "entry_exit_status": entry_exit_status
                }
                
                # Enhanced bbox info
                bbox_info = {
                    "plate_bbox": plate_bbox,
                    "car_bbox": car_info['bbox'] if car_info else [],
                    "full_car_bbox": full_car_bbox if full_car_bbox else [],
                    "frame_dimensions": [frame.shape[1], frame.shape[0]]
                }
                
                # Save to MongoDB with enhanced data including entry/exit information
                self.mongo_manager.save_vehicle_record_async(
                    plate_text, confidence, car_image_path, plate_image_path,
                    bbox_info, vehicle_info, {
                        "car_filename": os.path.basename(car_image_path) if car_image_path else "",
                        "plate_filename": os.path.basename(plate_image_path) if plate_image_path else "",
                        "enhancement_applied": True,
                        "capture_method": "Full Car Detection with Entry/Exit",
                        "orientation_detection_applied": True,
                        "color_preservation": True
                    }, entry_exit_status, vehicle_orientation
                )
                
                # Add to local records
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
                    'is_entry': entry_exit_status == "Vehicle Entered",
                    'is_exit': entry_exit_status == "Vehicle Exited"
                }
                
                with self.vehicle_records_lock:
                    self.vehicle_records.append(vehicle_record)
                    if len(self.vehicle_records) > 100:
                        self.vehicle_records = self.vehicle_records[-50:]
                
                logger.info(f"🚗📸 Full car capture with entry/exit: {plate_text} ({entry_exit_status}) -> Full Car: {bool(car_image_path)}, Plate: {bool(plate_image_path)}")
                
            except Exception as e:
                logger.error(f"Enhanced full car save error: {e}")
        
        # Submit to background executor
        self.image_processor_executor.submit(_save_async)

    def enhance_plate_image(self, plate_img):
        """Enhance plate image keeping original colors"""
        try:
            if plate_img is None or plate_img.size == 0:
                return plate_img
            
            # Keep original color image
            enhanced = plate_img.copy()
            
            # Resize for better visibility
            h, w = enhanced.shape[:2]
            if w < 200:
                scale = 200 / w
                enhanced = cv2.resize(enhanced, (int(w*scale), int(h*scale)))
            
            return enhanced
        except Exception as e:
            logger.error(f"Plate image enhancement error: {e}")
            return plate_img

    def is_recent_detection(self, plate_text, bbox):
        """Enhanced duplicate detection"""
        current_time = time.time()
        detection_key = f"{plate_text}_{bbox[0]}_{bbox[1]}"
        
        # Clean old detections
        self.recent_detections = deque([
            (key, timestamp) for key, timestamp in self.recent_detections
            if current_time - timestamp < 10.0  # Longer window for full car detection
        ], maxlen=50)
        
        # Check for recent duplicates
        for key, timestamp in self.recent_detections:
            if key == detection_key and current_time - timestamp < 5.0:
                return True
        
        self.recent_detections.append((detection_key, current_time))
        return False

    def is_already_captured(self, plate_text):
        """Check if plate already captured"""
        if not plate_text:
            return True
        
        plate_text = plate_text.upper().strip()
        
        with self.captured_plates_lock:
            return plate_text in self.captured_plates

    def mark_as_captured(self, plate_text):
        """Mark plate as captured"""
        if not plate_text:
            return
        
        plate_text = plate_text.upper().strip()
        
        with self.captured_plates_lock:
            self.captured_plates.add(plate_text)
            self.session_plates.add(plate_text)

    def process_frame_enhanced(self, frame, fps=30):
        """Enhanced frame processing with full car detection and entry/exit detection"""
        # Skip processing if not time yet
        if not self.should_process_frame():
            annotated_frame = frame.copy()
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(annotated_frame, f"🚗 FULL CAR + ENTRY/EXIT DETECTION - {timestamp} | {fps:.1f} FPS", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add enhanced stats with entry/exit
            with self.stats_lock:
                total_vehicles = len(self.captured_plates)
                session_count = len(self.session_plates)
                full_cars = self.detection_stats.get('full_cars_captured', 0)
                entries = self.detection_stats.get('entries', 0)
                exits = self.detection_stats.get('exits', 0)
            
            cv2.putText(annotated_frame, 
                       f"📊 Session: {session_count} | Total: {total_vehicles} | Full Cars: {full_cars} | 🟢Entries: {entries} | 🔴Exits: {exits}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            return annotated_frame
        
        # Run enhanced detection
        try:
            # Detect cars first
            detected_cars = self.detect_cars_in_frame(frame)
            
            # Detect license plates
            plate_results = self.plate_model(frame, conf=self.plate_confidence_threshold, 
                                           iou=self.iou_threshold, imgsz=640, verbose=False)
        except Exception as e:
            logger.error(f"Enhanced detection failed: {e}")
            return frame
        
        annotated_frame = frame.copy()
        
        # Add enhanced header with entry/exit info
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"🚗🔄 FULL VEHICLE + ENTRY/EXIT DETECTION - {timestamp} | {fps:.1f} FPS", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Process plate detections
        plate_detections = []
        run_ocr = self.should_run_ocr()
        
        for result in plate_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Size filtering for plates
                    if (x2 - x1) < 40 or (y2 - y1) < 12:
                        continue
                    
                    plate_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        
        # Draw detected cars first
        for car in detected_cars:
            car_x1, car_y1, car_x2, car_y2 = car['bbox']
            cv2.rectangle(annotated_frame, (car_x1, car_y1), (car_x2, car_y2), (255, 0, 0), 3)
            
            # Label car
            car_label = f"{car['class_name'].upper()} {car['confidence']:.2f}"
            cv2.putText(annotated_frame, car_label, (car_x1, car_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        new_captures = 0
        
        # Process each detected car and find associated plates
        for car in detected_cars:
            # Find plate within this car
            associated_plate = self.find_plate_in_car(frame, car['bbox'], plate_detections)
            
            if associated_plate:
                x1, y1, x2, y2 = associated_plate['bbox']
                confidence = associated_plate['confidence']
                
                # Draw plate detection
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract plate region
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue
                
                plate_text = ""
                
                # Enhanced OCR processing
                if run_ocr or confidence > 0.7:
                    try:
                        use_advanced = confidence > 0.8
                        plate_text = self.ocr_engine.smart_ensemble_ocr(plate_img, use_advanced)
                    except Exception as ocr_error:
                        logger.error(f"Enhanced OCR error: {ocr_error}")
                        plate_text = ""
                
                # Process valid plates with full car capture and entry/exit detection
                if (plate_text and 
                    self.ocr_engine.is_valid_license_plate(plate_text) and 
                    not self.is_already_captured(plate_text) and
                    not self.is_recent_detection(plate_text, associated_plate['bbox'])):
                    
                    # Mark as captured
                    self.mark_as_captured(plate_text)
                    
                    # Enhanced full car image saving with entry/exit detection
                    self.save_enhanced_full_car_images(frame, plate_img, associated_plate['bbox'], 
                                                     plate_text, confidence*100, car)
                    self.update_detection_stats(plate_text)
                    
                    new_captures += 1
                    logger.info(f"🚗⚡ FULL CAR CAPTURE WITH ENTRY/EXIT: {plate_text} ({car['class_name']}) (#{len(self.session_plates)})")
                    
                    # Highlight new capture with full car outline and entry/exit indicator
                    car_x1, car_y1, car_x2, car_y2 = car['bbox']
                    cv2.rectangle(annotated_frame, (car_x1-3, car_y1-3), (car_x2+3, car_y2+3), (0, 255, 255), 4)
                    cv2.rectangle(annotated_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 3)
                    cv2.putText(annotated_frame, "🚗 FULL CAR + ENTRY/EXIT CAPTURED", 
                               (car_x1, car_y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Enhanced display information with entry/exit status
                display_text = plate_text if plate_text else "SCANNING..."
                label = f"{display_text}"
                
                # Add enhanced label with background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1+text_size[0]+10, y1), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (x1+5, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show confidence
                cv2.putText(annotated_frame, f"{confidence:.2f}", 
                           (x2-50, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Enhanced capture status with entry/exit info
                if plate_text:
                    if plate_text in self.session_plates:
                        cv2.putText(annotated_frame, "✅ SESSION + 🚗 FULL CAR + 🔄 ENTRY/EXIT", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    elif plate_text in self.captured_plates:
                        cv2.putText(annotated_frame, "✅ DATABASE", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Process standalone plates (not associated with detected cars)
        for plate in plate_detections:
            # Check if this plate was already processed with a car
            already_processed = False
            for car in detected_cars:
                if self.find_plate_in_car(frame, car['bbox'], [plate]):
                    already_processed = True
                    break
            
            if not already_processed:
                x1, y1, x2, y2 = plate['bbox']
                confidence = plate['confidence']
                
                # Draw standalone plate detection
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract plate region
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue
                
                plate_text = ""
                
                # Enhanced OCR processing
                if run_ocr or confidence > 0.7:
                    try:
                        plate_text = self.ocr_engine.smart_ensemble_ocr(plate_img, False)
                    except Exception as ocr_error:
                        logger.error(f"Standalone OCR error: {ocr_error}")
                        plate_text = ""
                
                # Process valid standalone plates
                if (plate_text and 
                    self.ocr_engine.is_valid_license_plate(plate_text) and 
                    not self.is_already_captured(plate_text) and
                    not self.is_recent_detection(plate_text, plate['bbox'])):
                    
                    # Mark as captured
                    self.mark_as_captured(plate_text)
                    
                    # Save with estimated car region (no entry/exit detection for standalone plates)
                    self.save_enhanced_full_car_images(frame, plate_img, plate['bbox'], 
                                                     plate_text, confidence*100, None)
                    self.update_detection_stats(plate_text)
                    
                    new_captures += 1
                    logger.info(f"📄⚡ PLATE ONLY CAPTURE: {plate_text} (#{len(self.session_plates)})")
                    
                    # Highlight standalone plate capture
                    cv2.rectangle(annotated_frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 0), 3)
                    cv2.putText(annotated_frame, "📄 PLATE CAPTURED", 
                               (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Display information for standalone plates
                display_text = plate_text if plate_text else "SCANNING..."
                label = f"{display_text}"
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1+text_size[0]+10, y1), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (x1+5, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Enhanced status display with entry/exit statistics
        with self.stats_lock:
            total_vehicles = len(self.captured_plates)
            session_count = len(self.session_plates)
            full_cars = self.detection_stats.get('full_cars_captured', 0)
            entries = self.detection_stats.get('entries', 0)
            exits = self.detection_stats.get('exits', 0)
            mongodb_status = "🟢" if self.mongo_manager.is_connected else "🔴"
            
        cv2.putText(annotated_frame, 
                   f"📊 Full Car + Entry/Exit Mode | Session: {session_count} | Total: {total_vehicles} | Full Cars: {full_cars} | MongoDB {mongodb_status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Entry/Exit statistics
        cv2.putText(annotated_frame, 
                   f"🔄 🟢Entries: {entries} | 🔴Exits: {exits} | Cars Detected: {len(detected_cars)} | Plates: {len(plate_detections)} | New: {new_captures}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Enhanced processing status
        ocr_status = "🔍 OCR ON" if run_ocr else "⏸ OCR SKIP"
        cv2.putText(annotated_frame, f"⚡ Frame #{self.frame_count} | {ocr_status} | Full Car + Entry/Exit Detection Active", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return annotated_frame

    def update_detection_stats(self, plate_text):
        """Update enhanced detection statistics"""
        with self.stats_lock:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.detection_stats['total_detections'] += 1
            self.detection_stats['unique_plates'] = len(self.captured_plates)
            self.detection_stats['last_detection'] = current_time
            
            # Update plates list
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
                    'source': 'Full Car Detection with Entry/Exit'
                })

    def get_enhanced_stats(self):
        """Get comprehensive enhanced statistics including entry/exit data"""
        with self.stats_lock:
            stats = self.detection_stats.copy()
            
            # Add MongoDB stats
            mongo_stats = self.mongo_manager.get_enhanced_stats()
            
            stats.update({
                'session_plates_count': len(self.session_plates),
                'total_known_plates': len(self.captured_plates),
                'mongodb_total_plates': mongo_stats.get('total_vehicles', 0),
                'processing_interval': self.processing_interval,
                'ocr_interval': self.ocr_interval,
                'frame_count': self.frame_count,
                'optimization_level': 'Full Car + Plate + Entry/Exit Detection',
                'mongodb_status': mongo_stats.get('database_status', 'Unknown'),
                'full_cars_captured': mongo_stats.get('full_cars_captured', 0),
                'plate_images_captured': mongo_stats.get('plate_images_captured', 0),
                'mongodb_entries': mongo_stats.get('entries', 0),
                'mongodb_exits': mongo_stats.get('exits', 0),
                'front_detections': mongo_stats.get('front_detections', 0),
                'back_detections': mongo_stats.get('back_detections', 0),
                'color_preservation': True
            })
            
            return stats

    def get_vehicle_records(self, limit=50, query=None):
        """Get vehicle records with full car images and entry/exit data"""
        try:
            # Get from MongoDB (primary source)
            mongo_records = self.mongo_manager.get_vehicle_records(limit, query)
            
            # Merge with local records if needed
            with self.vehicle_records_lock:
                local_records = self.vehicle_records[-limit:] if not query else []
            
            # Combine and deduplicate
            all_records = mongo_records + local_records
            seen_ids = set()
            unique_records = []
            
            for record in all_records:
                record_id = record.get('detection_id', '')
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    unique_records.append(record)
            
            # Sort by timestamp (newest first)
            unique_records.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            return unique_records[:limit]
            
        except Exception as e:
            logger.error(f"Error getting vehicle records: {e}")
            return []

    def cleanup(self):
        """Enhanced cleanup"""
        logger.info("🧹 Cleaning up full car detector with entry/exit...")
        
        if hasattr(self, 'image_processor_executor'):
            self.image_processor_executor.shutdown(wait=True)
        
        if self.mongo_manager:
            self.mongo_manager.close_connection()
        
        logger.info("✅ Full car cleanup completed")


class UltraFastRTSPReader:
    """Ultra-fast RTSP reader optimized for real-time processing"""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame_lock = Lock()
        self.is_connected = False
        self.running = False
        
        # Minimal buffering for fastest response
        self.frame_queue = queue.Queue(maxsize=3)
        self.latest_frame = None
        self.current_frame = None
        
        # Optimized stream settings
        self.width = 1920
        self.height = 1080
        self.target_fps = 30
        
        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize RTSP capture with optimized settings"""
        logger.info(f"🔗 Attempting to connect to: {self.rtsp_url}")
        
        try:
            # Try different backends for RTSP
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    logger.info(f"Trying backend: {backend}")
                    self.cap = cv2.VideoCapture(self.rtsp_url, backend)
                    
                    # Basic connection test
                    if self.cap.isOpened():
                        # Try to read a test frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"✅ Connection successful with backend {backend}")
                            break
                        else:
                            logger.warning(f"Backend {backend} opened but no frames")
                            self.cap.release()
                            self.cap = None
                    else:
                        logger.warning(f"Backend {backend} failed to open stream")
                        
                except Exception as e:
                    logger.error(f"Backend {backend} error: {e}")
            
            if self.cap is None or not self.cap.isOpened():
                logger.error("❌ All backends failed")
                return False
            
            # Set optimal camera parameters
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Start capture thread
            self.is_connected = True
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            return False

    def _capture_loop(self):
        """Ultra-fast frame capture loop"""
        logger.info("🎬 Starting capture loop")
        frame_counter = 0
        
        while self.running:
            try:
                if not self.cap.isOpened():
                    logger.error("❌ Stream disconnected")
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.001)
                    continue
                
                # Update latest frame with minimal locking
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Keep frame queue minimal
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                frame_counter += 1
                if frame_counter % 100 == 0:
                    logger.debug(f"Captured {frame_counter} frames")
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)
        
        logger.warning("🎬 Capture thread stopped")

    def get_frame(self):
        """Get the latest frame with minimal latency"""
        try:
            # Try queue first
            try:
                return self.frame_queue.get_nowait()
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

    def release(self):
        """Clean shutdown"""
        logger.info("🎬 Releasing RTSP reader...")
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        logger.info("🎬 RTSP reader released")


# Global instances
reader = None
full_car_processor = None

def generate_enhanced_frames():
    """Enhanced frame generator with full car detection and entry/exit detection"""
    global reader, full_car_processor
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    fps_calculation_interval = 15
    
    while True:
        try:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            if fps_counter >= fps_calculation_interval:
                current_time = time.time()
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # Enhanced processing with entry/exit detection
            try:
                processed_frame = full_car_processor.process_frame_enhanced(frame, current_fps)
            except Exception as e:
                logger.error(f"Full car processing error: {e}")
                processed_frame = frame
                cv2.putText(processed_frame, "⚠️ FULL CAR + ENTRY/EXIT PROCESSING ERROR", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Optimized encoding
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80,
                               cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                ret, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Full car encoding error: {e}")
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Full car stream generation error: {e}")
            time.sleep(0.001)


# Enhanced Flask routes with entry/exit functionality
@app.route('/')
def index():
    return send_from_directory(DASHBOARD_DIR, 'dashboard.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_enhanced_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0'})

@app.route('/api/enhanced_stats')
def get_enhanced_stats():
    """Get enhanced statistics including full car images and entry/exit data"""
    global full_car_processor
    if full_car_processor:
        return jsonify(full_car_processor.get_enhanced_stats())
    return jsonify({
        'total_detections': 0,
        'unique_plates': 0,
        'full_cars_captured': 0,
        'plate_images_captured': 0,
        'entries': 0,
        'exits': 0,
        'mongodb_status': 'Processor not available',
        'color_preservation': True
    })

@app.route('/api/vehicle_records')
def get_vehicle_records():
    """Get vehicle records with full car images and entry/exit data"""
    global full_car_processor
    if full_car_processor:
        limit = int(request.args.get('limit', 50))
        query = request.args.get('query', '')
        
        records = full_car_processor.get_vehicle_records(limit, query if query else None)
        return jsonify({
            'records': records,
            'total': len(records),
            'source': 'Full Car Detection System with Entry/Exit - Color Preserved',
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({
        'records': [], 
        'total': 0, 
        'error': 'Full car processor not available',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search_vehicles', methods=['POST'])
def search_vehicles():
    """Search vehicle records including entry/exit status"""
    global full_car_processor
    if not full_car_processor:
        return jsonify({'error': 'Full car processor not available'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        records = full_car_processor.get_vehicle_records(100, query if query else None)
        
        return jsonify({
            'records': records,
            'total': len(records),
            'query': query,
            'source': 'Full Car Search with Entry/Exit - Color Preserved'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/entry_exit_stats')
def get_entry_exit_stats():
    """Get entry/exit specific statistics"""
    global full_car_processor
    if full_car_processor and full_car_processor.mongo_manager.is_connected:
        try:
            # Get detailed entry/exit statistics
            pipeline = [
                {
                    "$group": {
                        "_id": "$entry_exit_status",
                        "count": {"$sum": 1},
                        "latest": {"$max": "$timestamp"},
                        "earliest": {"$min": "$timestamp"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            result = list(full_car_processor.mongo_manager.collection.aggregate(pipeline))
            
            # Get hourly entry/exit distribution for today
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            hourly_pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": today_start},
                        "entry_exit_status": {"$in": ["Vehicle Entered", "Vehicle Exited"]}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "hour": {"$hour": "$timestamp"},
                            "status": "$entry_exit_status"
                        },
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id.hour": 1}}
            ]
            
            hourly_result = list(full_car_processor.mongo_manager.collection.aggregate(hourly_pipeline))
            
            # Format hourly data
            hourly_data = {}
            for item in hourly_result:
                hour = item['_id']['hour']
                status = item['_id']['status']
                if hour not in hourly_data:
                    hourly_data[hour] = {'entries': 0, 'exits': 0}
                
                if status == "Vehicle Entered":
                    hourly_data[hour]['entries'] = item['count']
                elif status == "Vehicle Exited":
                    hourly_data[hour]['exits'] = item['count']
            
            return jsonify({
                'entry_exit_summary': result,
                'hourly_distribution': hourly_data,
                'source': 'Entry/Exit Analysis - Color Preserved',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting entry/exit stats: {e}")
            return jsonify({'error': 'Failed to get entry/exit statistics'})
    
    return jsonify({'error': 'Database not available'})

@app.route('/api/orientation_stats')
def get_orientation_stats():
    """Get vehicle orientation statistics"""
    global full_car_processor
    if full_car_processor and full_car_processor.mongo_manager.is_connected:
        try:
            # Get orientation distribution
            pipeline = [
                {
                    "$group": {
                        "_id": "$vehicle_orientation",
                        "count": {"$sum": 1},
                        "latest": {"$max": "$timestamp"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            result = list(full_car_processor.mongo_manager.collection.aggregate(pipeline))
            
            return jsonify({
                'orientation_distribution': result,
                'source': 'Vehicle Orientation Analysis - Color Preserved',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting orientation stats: {e}")
            return jsonify({'error': 'Failed to get orientation statistics'})
    
    return jsonify({'error': 'Database not available'})

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Serve captured images (full car and plate images)"""
    try:
        # Decode the image path
        import urllib.parse
        decoded_path = urllib.parse.unquote(image_path)
        
        # Security check - ensure path is within allowed directories
        allowed_dirs = [CAR_IMAGES_DIR, PLATE_IMAGES_DIR, SCREENSHOTS_DIR]
        full_path = os.path.abspath(decoded_path)
        
        is_allowed = False
        for allowed_dir in allowed_dirs:
            if full_path.startswith(os.path.abspath(allowed_dir)):
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

@app.route('/api/mongodb_stats')
def get_mongodb_stats():
    """Get MongoDB specific statistics including entry/exit"""
    global full_car_processor
    if full_car_processor and full_car_processor.mongo_manager:
        stats = full_car_processor.mongo_manager.get_enhanced_stats()
        return jsonify(stats)
    return jsonify({'error': 'MongoDB not available'})

@app.route('/api/search_plates', methods=['POST'])
def search_plates():
    """Legacy endpoint for plate search (compatibility)"""
    return search_vehicles()

@app.route('/api/reset_session')
def reset_session():
    """Reset current session data"""
    global full_car_processor
    if full_car_processor:
        with full_car_processor.captured_plates_lock:
            old_count = len(full_car_processor.session_plates)
            full_car_processor.session_plates.clear()
            full_car_processor.recent_detections.clear()
        
        with full_car_processor.vehicle_records_lock:
            full_car_processor.vehicle_records.clear()
        
        # Reset entry/exit stats for session
        with full_car_processor.stats_lock:
            full_car_processor.detection_stats['entries'] = 0
            full_car_processor.detection_stats['exits'] = 0
        
        return jsonify({
            'status': 'success',
            'message': f'Full car session reset - {old_count} plates cleared',
            'enhancement_level': 'Full Car + Plate + Entry/Exit Detection - Color Preserved'
        })
    return jsonify({'status': 'error', 'message': 'Full car processor not available'})

@app.route('/api/stats')
def get_stats():
    """Legacy stats endpoint (compatibility)"""
    return get_enhanced_stats()

@app.route('/api/export_vehicles')
def export_vehicles():
    """Export vehicle data with full car information and entry/exit data"""
    global full_car_processor
    if full_car_processor:
        records = full_car_processor.get_vehicle_records(1000)
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'total_records': len(records),
            'source': 'Full Car Vehicle Detection System with Entry/Exit - Color Preserved',
            'features': [
                'Complete vehicle image capture (original colors)',
                'License plate close-up capture (original colors)',
                'Enhanced OCR recognition',
                'Vehicle type classification',
                'Entry/Exit detection',
                'Vehicle orientation detection',
                'MongoDB persistent storage',
                'Full car boundary detection',
                'Color preservation in all image captures'
            ],
            'records': records
        }
        
        return jsonify(export_data)
    
    return jsonify({'error': 'Full car processor not available'})

@app.route('/api/system_info')
def get_system_info():
    """Get full car system information with entry/exit capabilities"""
    global full_car_processor
    
    system_info = {
        'system_name': 'Full Car Vehicle Detection System with Entry/Exit Detection - Color Preserved',
        'version': '4.1',
        'features': {
            'full_car_capture': True,
            'car_type_detection': True,
            'plate_image_capture': True,
            'entry_exit_detection': True,
            'vehicle_orientation_detection': True,
            'mongodb_storage': True,
            'real_time_processing': True,
            'enhanced_ocr': True,
            'duplicate_detection': True,
            'vehicle_classification': True,
            'color_preservation': True
        },
        'directories': {
            'car_images': CAR_IMAGES_DIR,
            'plate_images': PLATE_IMAGES_DIR,
            'screenshots': SCREENSHOTS_DIR
        },
        'processing_intervals': {
            'frame_processing': PROCESSING_INTERVAL,
            'ocr_processing': OCR_PROCESSING_INTERVAL
        },
        'car_detection': {
            'confidence_threshold': CAR_DETECTION_CONFIDENCE,
            'min_car_width': MIN_CAR_WIDTH,
            'min_car_height': MIN_CAR_HEIGHT,
            'supported_vehicle_types': ['car', 'motorcycle', 'bus', 'truck']
        },
        'entry_exit_detection': {
            'orientation_methods': [
                'License plate position analysis',
                'Vehicle feature analysis',
                'Geometric analysis',
                'Headlight/Taillight detection'
            ],
            'confidence_threshold': ORIENTATION_CONFIDENCE_THRESHOLD,
            'history_size': ENTRY_EXIT_HISTORY_SIZE
        },
        'image_processing': {
            'grayscale_conversion': False,
            'color_preservation': True,
            'enhancement_applied': True
        }
    }
    
    if full_car_processor:
        system_info.update({
            'processor_status': 'Active',
            'mongodb_connected': full_car_processor.mongo_manager.is_connected,
            'total_captured_plates': len(full_car_processor.captured_plates),
            'session_plates': len(full_car_processor.session_plates),
            'frame_count': full_car_processor.frame_count,
            'full_cars_captured': full_car_processor.detection_stats.get('full_cars_captured', 0),
            'session_entries': full_car_processor.detection_stats.get('entries', 0),
            'session_exits': full_car_processor.detection_stats.get('exits', 0)
        })
    else:
        system_info['processor_status'] = 'Not Available'
    
    return jsonify(system_info)

@app.route('/api/vehicle_types')
def get_vehicle_types():
    """Get detected vehicle types statistics"""
    global full_car_processor
    if full_car_processor and full_car_processor.mongo_manager.is_connected:
        try:
            # Get vehicle type distribution from MongoDB
            pipeline = [
                {"$group": {
                    "_id": "$vehicle_info.type",
                    "count": {"$sum": 1},
                    "latest": {"$max": "$timestamp"}
                }},
                {"$sort": {"count": -1}}
            ]
            
            result = list(full_car_processor.mongo_manager.collection.aggregate(pipeline))
            
            vehicle_types = []
            for item in result:
                vehicle_types.append({
                    'type': item['_id'] if item['_id'] else 'unknown',
                    'count': item['count'],
                    'latest_detection': item['latest']
                })
            
            return jsonify({
                'vehicle_types': vehicle_types,
                'total_types': len(vehicle_types),
                'source': 'Full Car Detection Analysis with Entry/Exit - Color Preserved'
            })
            
        except Exception as e:
            logger.error(f"Error getting vehicle types: {e}")
            return jsonify({'error': 'Failed to get vehicle types'})
    
    return jsonify({'vehicle_types': [], 'error': 'Database not available'})

@app.route('/api/recent_captures')
def get_recent_captures():
    """Get recent captures with full car images and entry/exit data"""
    global full_car_processor
    if full_car_processor:
        try:
            # Get last 20 captures
            records = full_car_processor.get_vehicle_records(20)
            
            # Filter to only include records with full car images
            full_car_records = [
                record for record in records 
                if record.get('full_car_captured', False) and record.get('car_image_path')
            ]
            
            return jsonify({
                'recent_captures': full_car_records,
                'total': len(full_car_records),
                'source': 'Full Car Recent Captures with Entry/Exit - Color Preserved'
            })
            
        except Exception as e:
            logger.error(f"Error getting recent captures: {e}")
            return jsonify({'error': 'Failed to get recent captures'})
    
    return jsonify({'recent_captures': [], 'error': 'Processor not available'})

def main():
    global reader, full_car_processor
    
    try:
        logger.info("="*80)
        logger.info("🚗 STARTING FULL CAR VEHICLE DETECTION SYSTEM WITH ENTRY/EXIT")
        logger.info("🔄 Complete Vehicle + License Plate + Entry/Exit Detection - COLOR PRESERVED")
        logger.info("="*80)
        logger.info("🎯 Enhanced Features:")
        logger.info("   • Complete vehicle image capture (ORIGINAL COLORS)")
        logger.info("   • Vehicle type classification (car, truck, bus, motorcycle)")
        logger.info("   • License plate close-up capture (ORIGINAL COLORS)")
        logger.info("   • 🔄 Entry/Exit detection (Front = Entered, Back = Exited)")
        logger.info("   • Vehicle orientation detection")
        logger.info("   • Enhanced tabular data display with entry/exit status")
        logger.info("   • Advanced MongoDB integration with entry/exit tracking")
        logger.info("   • Real-time entry/exit statistics")
        logger.info("   • Smart vehicle-plate association")
        logger.info("   • Background image processing")
        logger.info("   • Multi-method orientation detection")
        logger.info("   • 🎨 COLOR PRESERVATION - No grayscale conversion")
        logger.info("="*80)
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU ACCELERATION: Enabled ({torch.cuda.get_device_name(0)})")
        else:
            logger.info("💻 Running on CPU (optimized for full car + entry/exit processing)")
        
        # Quick system checks
        logger.info("🔍 Full car + entry/exit system checks...")
        
        # Test RTSP
        test_cap = cv2.VideoCapture(RTSP_STREAM)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                logger.info("✅ RTSP stream: Ready for full car + entry/exit processing")
            else:
                logger.warning("⚠️ RTSP stream: Connected but no frames")
        else:
            logger.error("❌ RTSP stream: Connection failed")
        test_cap.release()
        
        # Test MongoDB
        try:
            test_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            test_client.admin.command('ping')
            test_client.close()
            logger.info("✅ Full car + Entry/Exit MongoDB: Ready")
        except:
            logger.warning("⚠️ Full car + Entry/Exit MongoDB: Connection issues (will retry)")
        
        # Test directories
        for dir_name, dir_path in [
            ("Full Car Images (Color)", CAR_IMAGES_DIR),
            ("Plate Images (Color)", PLATE_IMAGES_DIR),
            ("Screenshots", SCREENSHOTS_DIR)
        ]:
            if os.path.exists(dir_path):
                logger.info(f"✅ {dir_name} directory: {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 Created {dir_name} directory: {dir_path}")
        
        # Test YOLO models
        try:
            # Test plate detection model
            if os.path.exists(MODEL_PATH):
                logger.info(f"✅ Plate detection model: {MODEL_PATH}")
            else:
                logger.error(f"❌ Plate detection model not found: {MODEL_PATH}")
                return
            
            # Test car detection model (will be downloaded if needed)
            logger.info(f"✅ Car detection model: {YOLO_CAR_MODEL} (will download if needed)")
            
        except Exception as e:
            logger.error(f"❌ Model check failed: {e}")
            return
        
        # Initialize full car detection system with entry/exit
        logger.info("🔧 Initializing full car detection system with entry/exit + color preservation...")
        full_car_processor = EnhancedFullCarDetector(MODEL_PATH, YOLO_CAR_MODEL)
        
        # Initialize RTSP reader
        reader = UltraFastRTSPReader(RTSP_STREAM)
        
        # Brief initialization wait
        time.sleep(2)
        
        if not reader.is_connected:
            logger.error("❌ Failed to connect to RTSP stream")
            return
        
        # Display full car + entry/exit startup summary
        with full_car_processor.captured_plates_lock:
            total_plates = len(full_car_processor.captured_plates)
            session_plates = len(full_car_processor.session_plates)
        
        mongo_stats = full_car_processor.mongo_manager.get_enhanced_stats()
        
        logger.info("="*80)
        logger.info("🚗🔄🎨 FULL CAR + ENTRY/EXIT + COLOR PRESERVATION SYSTEM READY!")
        logger.info("="*80)
        logger.info(f"📊 Database plates: {total_plates}")
        logger.info(f"📊 Session plates: {session_plates}")
        logger.info(f"🚗 Full cars in DB: {mongo_stats.get('full_cars_captured', 0)}")
        logger.info(f"🏷️ Plate images in DB: {mongo_stats.get('plate_images_captured', 0)}")
        logger.info(f"🟢 Entries in DB: {mongo_stats.get('entries', 0)}")
        logger.info(f"🔴 Exits in DB: {mongo_stats.get('exits', 0)}")
        logger.info(f"➡️ Front detections: {mongo_stats.get('front_detections', 0)}")
        logger.info(f"⬅️ Back detections: {mongo_stats.get('back_detections', 0)}")
        logger.info(f"💾 Full car + Entry/Exit MongoDB: {'Connected' if full_car_processor.mongo_manager.is_connected else 'Retrying'}")
        logger.info(f"🤖 OCR Engine: Smart Ensemble (Enhanced - Color Preserved)")
        logger.info(f"🚗 Car Detection: YOLO v8 (Multi-class)")
        logger.info(f"🔄 Entry/Exit Detection: Multi-method orientation analysis")
        logger.info(f"🎨 Image Processing: Original colors preserved (No grayscale)")
        logger.info(f"🎯 Processing: Every {full_car_processor.processing_interval} frames")
        logger.info(f"🔍 OCR Interval: Every {full_car_processor.ocr_interval} frames")
        logger.info(f"🌐 Full Car + Entry/Exit Dashboard: http://localhost:5000")
        logger.info("="*80)
        logger.info("🚗🔄🎨 ENHANCED FEATURES ACTIVE:")
        logger.info("   • Complete vehicle image capture (ORIGINAL COLORS)")
        logger.info("   • Vehicle type classification")
        logger.info("   • License plate close-up capture (ORIGINAL COLORS)")
        logger.info("   • 🔄 Entry/Exit detection (Front/Back orientation)")
        logger.info("   • Real-time entry/exit statistics")
        logger.info("   • Enhanced tabular display with entry/exit status")
        logger.info("   • Advanced MongoDB storage with entry/exit tracking")
        logger.info("   • Vehicle-plate association")
        logger.info("   • Multi-method orientation detection:")
        logger.info("     - License plate position analysis")
        logger.info("     - Vehicle feature detection")
        logger.info("     - Geometric analysis")
        logger.info("     - Headlight/Taillight detection")
        logger.info("   • Background image processing")
        logger.info("   • Smart duplicate prevention")
        logger.info("   • GPU-accelerated processing")
        logger.info("   • Entry/Exit bracketed display in plate number")
        logger.info("   • 🎨 COLOR PRESERVATION: All images saved in original colors")
        logger.info("   • 🚫 NO GRAYSCALE CONVERSION: License plates retain natural colors")
        logger.info("="*80)
        
        # Start Flask with full car + entry/exit settings
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, 
                use_reloader=False, use_debugger=False)
        
    except KeyboardInterrupt:
        logger.info("🛑 Full car + entry/exit system stopped by user")
    except Exception as e:
        logger.error(f"❌ Full car + entry/exit system error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🔄 Shutting down full car + entry/exit system...")
        if full_car_processor:
            full_car_processor.cleanup()
        if reader:
            reader.release()
        logger.info("✅ Full car + entry/exit system shutdown complete")

if __name__ == "__main__":
    main()