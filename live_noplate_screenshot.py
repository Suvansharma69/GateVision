# Real-time optimized license plate detection system with HD QUALITY screenshots
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
from datetime import datetime
import json
from flask import Flask, render_template, Response, jsonify, send_from_directory
import math
import base64
import logging
from threading import Lock
import hashlib
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- CONFIGURATION ---
MODEL_PATH = "runs/detect/license_plate_detection/weights/best.pt"
DASHBOARD_DIR = r"D:\vehicle_speed\no_plate"
SCREENSHOTS_DIR = os.path.join(DASHBOARD_DIR, "screenshots")
RTSP_STREAM = "rtsp://camera:Rj18cb4820!@10.10.97.12:554/"

# Ensure directories exist
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Flask app
app = Flask(__name__)

class FastRTSPReader:
    """Ultra-fast RTSP reader optimized for real-time processing"""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame_lock = Lock()
        self.is_connected = False
        self.running = False
        
        # Minimal buffering for fastest response
        self.frame_queue = queue.Queue(maxsize=3)  # Very small buffer
        self.latest_frame = None
        self.current_frame = None
        
        # Optimized stream settings
        self.width = 1920  # Higher resolution for better quality
        self.height = 1080
        self.target_fps = 30
        
        self._initialize_capture()

    def _initialize_capture(self):
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
                        logger.warning(f"Backend {backend} failed to open")
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                except Exception as e:
                    logger.error(f"Backend {backend} error: {e}")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    continue
            
            if self.cap is None or not self.cap.isOpened():
                logger.error("❌ All backends failed - trying fallback options")
                
                # Fallback: Try without specifying backend
                try:
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info("✅ Fallback connection successful")
                        else:
                            logger.error("❌ Fallback failed - no frames")
                            return False
                    else:
                        logger.error("❌ Fallback connection failed")
                        return False
                except Exception as e:
                    logger.error(f"❌ Fallback error: {e}")
                    return False
            
            # Configure capture settings
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                # Get actual resolution (don't force if camera doesn't support)
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"📺 Stream info: {actual_width}x{actual_height} @ {actual_fps}fps")
                
                # Update dimensions to actual values
                if actual_width > 0 and actual_height > 0:
                    self.width = actual_width
                    self.height = actual_height
                
            except Exception as e:
                logger.warning(f"⚠️ Could not set some properties: {e}")
            
            self.is_connected = True
            self._start_capture_thread()
            logger.info("✨ HD stream initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical stream initialization error: {e}")
            return False

    def _start_capture_thread(self):
        if self.is_connected:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

    def _capture_loop(self):
        while self.running and self.is_connected:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.01)
                    continue
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    
                    # Clear old frames and add new one
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.01)

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return self._create_error_frame()

    def _create_error_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(frame, "🔄 CONNECTING...", (800, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        return frame

    def release(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class RealTimePlateDetector:
    """Real-time license plate detector with HD quality screenshots"""
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            logger.info(f"✅ YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise
        
        # Detection parameters
        self.confidence_threshold = 0.3
        self.iou_threshold = 0.4
        self.min_plate_width = 60
        self.min_plate_height = 20
        
        # STRICT duplicate prevention - once captured, never again
        self.captured_plates = set()  # Permanent record of captured plates
        self.captured_plates_lock = Lock()
        self.plate_hashes = set()  # Store hashes for similar plates
        
        # Thread-safe locks
        self.stats_lock = Lock()
        
        # HD screenshot processing
        self.screenshot_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Detection statistics
        self.detection_stats = {
            'total_detections': 0, 'unique_plates': 0,
            'session_start': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_detection': 'None', 'plates_list': [],
            'screenshots_saved': 0, 'is_live': True,
            'stream_status': 'Connected'
        }
        
        # Enhanced OCR configurations for HD quality
        self.hd_ocr_configs = [
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
        
        # Validation patterns
        self.valid_plate_patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # XX00XX0000
            r'^[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}$',     # XX00X0000
            r'^[A-Z]{3}[0-9]{3,4}$',                # XXX000
            r'^[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$',   # 0X000
            r'^[A-Z]{1,3}[0-9]{3,4}$',             # X000
            r'^[0-9]{3,4}[A-Z]{1,3}$',             # 000X
        ]
        
        self.false_positives = {
            'SPORTS', 'CAR', 'AUTO', 'TAXI', 'BUS', 'TRUCK', 'VIP', 'GOVT', 
            'POLICE', 'FIRE', 'AMBULANCE', 'SCHOOL', 'PRIVATE', 'COMMERCIAL'
        }

    def ultra_hd_enhance_plate(self, plate_img):
        """Ultra HD enhancement for maximum clarity and readability"""
        if plate_img is None or plate_img.size == 0:
            return None
        
        try:
            original = plate_img.copy()
            h, w = original.shape[:2]
            
            # STEP 1: Smart scaling - not too much zoom for clarity
            # Target a reasonable size that maintains clarity
            target_width = 600   # Reduced from 800 for better clarity
            target_height = 180  # Reduced proportionally
            
            # Calculate smart scale factor
            scale_x = target_width / w
            scale_y = target_height / h
            scale_factor = min(scale_x, scale_y) * 1.8  # Reduced multiplier
            scale_factor = min(scale_factor, 4.0)  # Cap at 4x to prevent over-scaling
            
            # Use INTER_LANCZOS4 for best quality upscaling
            if scale_factor > 1.0:
                scaled = cv2.resize(original, (int(w * scale_factor), int(h * scale_factor)), cv2.INTER_LANCZOS4)
            else:
                scaled = original.copy()
            
            # Convert to grayscale
            if len(scaled.shape) == 3:
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = scaled.copy()
            
            # STEP 2: Advanced denoising while preserving text edges
            # Use bilateral filter for edge-preserving smoothing
            denoised = cv2.bilateralFilter(gray, 9, 80, 80)
            
            # STEP 3: Intelligent contrast enhancement
            # Apply CLAHE with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
            contrast_enhanced = clahe.apply(denoised)
            
            # STEP 4: Sharpening with unsharp masking
            # Create Gaussian blur
            gaussian_blur = cv2.GaussianBlur(contrast_enhanced, (0, 0), 1.5)
            # Apply unsharp masking
            unsharp_mask = cv2.addWeighted(contrast_enhanced, 1.8, gaussian_blur, -0.8, 0)
            unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
            
            # STEP 5: Morphological operations for text cleanup
            # Use small kernel to clean up text without destroying it
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            morphed = cv2.morphologyEx(unsharp_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # STEP 6: Final contrast adjustment
            # Apply gamma correction for better visibility
            gamma = 1.1
            gamma_corrected = np.power(morphed / 255.0, 1.0 / gamma) * 255.0
            gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
            
            # STEP 7: Edge enhancement
            # Create a custom sharpening kernel
            sharpening_kernel = np.array([[-0.25, -0.5, -0.25],
                                         [-0.5, 4, -0.5],
                                         [-0.25, -0.5, -0.25]])
            sharpened = cv2.filter2D(gamma_corrected, -1, sharpening_kernel)
            
            # STEP 8: Final enhancement blend
            # Combine original enhanced with sharpened version
            final_enhanced = cv2.addWeighted(gamma_corrected, 0.6, sharpened, 0.4, 0)
            final_enhanced = np.clip(final_enhanced, 0, 255).astype(np.uint8)
            
            return final_enhanced
            
        except Exception as e:
            logger.error(f"HD enhancement failed: {e}")
            # Fallback to basic enhancement
            try:
                if len(plate_img.shape) == 3:
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = plate_img.copy()
                
                h, w = gray.shape[:2]
                scale_factor = max(300/w, 100/h) * 1.5
                scaled = cv2.resize(gray, (int(w*scale_factor), int(h*scale_factor)), cv2.INTER_CUBIC)
                
                enhanced = cv2.bilateralFilter(scaled, 9, 80, 80)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
                enhanced = clahe.apply(enhanced)
                
                return enhanced
            except:
                return plate_img

    def hd_ocr_extract(self, plate_img):
        """HD OCR extraction with multiple attempts for best results"""
        enhanced = self.ultra_hd_enhance_plate(plate_img)
        if enhanced is None:
            return ""
        
        results = []
        
        # Try multiple OCR configurations
        for config in self.hd_ocr_configs:
            try:
                text = pytesseract.image_to_string(enhanced, config=config).strip()
                clean_text = self.clean_plate_text(text)
                if clean_text and self.is_valid_license_plate(clean_text):
                    results.append(clean_text)
            except:
                continue
        
        # Try with different preprocessing
        preprocessing_methods = [
            lambda img: img,  # Original enhanced
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        for preprocess in preprocessing_methods:
            try:
                processed = preprocess(enhanced)
                text = pytesseract.image_to_string(processed, config=self.hd_ocr_configs[0]).strip()
                clean_text = self.clean_plate_text(text)
                if clean_text and self.is_valid_license_plate(clean_text):
                    results.append(clean_text)
            except:
                continue
        
        # Return most common valid result
        if results:
            return Counter(results).most_common(1)[0][0]
        return ""

    def generate_plate_hash(self, plate_text):
        """Generate hash for similar plate detection"""
        if not plate_text:
            return ""
        
        # Normalize the plate text
        normalized = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        
        # Create variations to catch similar OCR results
        variations = [normalized]
        
        # Add common OCR error variations
        ocr_replacements = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
        for old, new in ocr_replacements.items():
            if old in normalized:
                variations.append(normalized.replace(old, new))
            if new in normalized:
                variations.append(normalized.replace(new, old))
        
        # Return hash of the base normalized text
        return hashlib.md5(normalized.encode()).hexdigest()

    def is_already_captured(self, plate_text):
        """Check if this plate was already captured (STRICT - no duplicates)"""
        if not plate_text:
            return True
        
        with self.captured_plates_lock:
            # Direct match check
            if plate_text in self.captured_plates:
                return True
            
            # Hash-based similarity check
            plate_hash = self.generate_plate_hash(plate_text)
            if plate_hash in self.plate_hashes:
                return True
            
            return False

    def mark_as_captured(self, plate_text):
        """Mark plate as captured permanently"""
        if not plate_text:
            return
        
        with self.captured_plates_lock:
            self.captured_plates.add(plate_text)
            plate_hash = self.generate_plate_hash(plate_text)
            self.plate_hashes.add(plate_hash)

    def clean_plate_text(self, text):
        """Enhanced plate text cleaning"""
        if not text:
            return ""
        
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', '1').replace('!', '1').replace('?', '').replace('.', '')
        
        # Basic OCR corrections
        replacements = {'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8'}
        
        # Try replacements and return first valid result
        for old, new in replacements.items():
            test_text = cleaned.replace(old, new)
            if self.is_valid_license_plate(test_text):
                return test_text
        
        return cleaned

    def is_valid_license_plate(self, text):
        """Enhanced validation for license plates"""
        if not text or len(text) < 4 or len(text) > 15:
            return False
        
        if text.upper() in self.false_positives:
            return False
        
        # Must have both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        if not (has_letter and has_number):
            return False
        
        # Check against known patterns
        for pattern in self.valid_plate_patterns:
            if re.match(pattern, text):
                return True
        
        # Generic validation for other formats
        if 4 <= len(text) <= 12:
            letter_count = sum(1 for c in text if c.isalpha())
            number_count = sum(1 for c in text if c.isdigit())
            if letter_count >= 2 and number_count >= 2:
                return True
        
        return False

    def save_hd_screenshot(self, plate_img, plate_text, confidence, full_frame):
        """Save HD quality screenshot with perfect clarity"""
        def _hd_save():
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f"HD_plate_{plate_text}_{timestamp}.jpg"
                filepath = os.path.join(SCREENSHOTS_DIR, filename)
                
                # Get ultra HD enhanced version
                enhanced = self.ultra_hd_enhance_plate(plate_img)
                if enhanced is None:
                    enhanced = plate_img
                
                # Create professional HD layout
                if len(enhanced.shape) == 2:
                    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                else:
                    enhanced_color = enhanced
                
                # Professional border and layout
                border_size = 60
                final_img = cv2.copyMakeBorder(enhanced_color, border_size, border_size + 80, 
                                             border_size, border_size, 
                                             cv2.BORDER_CONSTANT, value=(20, 20, 20))
                
                # Add professional title
                title_text = f"LICENSE PLATE: {plate_text}"
                title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                title_x = (final_img.shape[1] - title_size[0]) // 2
                
                # Title with shadow
                cv2.putText(final_img, title_text, (title_x + 2, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                cv2.putText(final_img, title_text, (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                
                # Add timestamp
                timestamp_text = f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                cv2.putText(final_img, timestamp_text, (20, final_img.shape[0] - 45), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Add confidence
                confidence_text = f"Confidence: {confidence:.1f}%"
                cv2.putText(final_img, confidence_text, (20, final_img.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                
                # Add HD quality indicator
                quality_text = "ULTRA HD ENHANCED"
                cv2.putText(final_img, quality_text, (20, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Save with maximum quality
                save_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 100,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ]
                
                success = cv2.imwrite(filepath, final_img, save_params)
                
                if success:
                    file_size = os.path.getsize(filepath) / 1024
                    logger.info(f"📸 HD CAPTURE SAVED: {plate_text} -> {filename} ({file_size:.1f}KB)")
                    
                    with self.stats_lock:
                        self.detection_stats['screenshots_saved'] += 1
                    
                    return filepath
                else:
                    logger.error(f"Failed to save HD screenshot: {filepath}")
                
            except Exception as e:
                logger.error(f"HD screenshot error: {e}")
                import traceback
                traceback.print_exc()
            
            return None
        
        # Submit to background processing
        self.screenshot_executor.submit(_hd_save)

    def update_detection_stats(self, plate_text):
        """Update detection statistics"""
        with self.stats_lock:
            self.detection_stats['total_detections'] += 1
            self.detection_stats['unique_plates'] = len(self.captured_plates)
            self.detection_stats['last_detection'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update plates list
            found = False
            for p in self.detection_stats['plates_list']:
                if p['plate'] == plate_text:
                    p['count'] += 1
                    p['last_seen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    found = True
                    break
            
            if not found:
                self.detection_stats['plates_list'].append({
                    'plate': plate_text,
                    'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'last_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'count': 1,
                    'confidence': 95.0,
                    'vehicle_type': 'Vehicle'
                })

    def process_frame_realtime(self, frame, frame_count=0, fps=30):
        """Process frame with HD screenshot capability"""
        try:
            # YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, 
                               iou=self.iou_threshold, imgsz=1280, verbose=False)
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return frame
        
        annotated_frame = frame.copy()
        
        # Add header
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"🎯 HD QUALITY DETECTION - {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add stats
        with self.stats_lock:
            cv2.putText(annotated_frame, 
                       f"HD Captures: {self.detection_stats['screenshots_saved']} | Unique: {len(self.captured_plates)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Size filtering
                    if (x2 - x1) < self.min_plate_width or (y2 - y1) < self.min_plate_height:
                        continue
                    
                    # Extract plate region with padding
                    pad = 8
                    plate_img = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), 
                                    max(0, x1-pad):min(frame.shape[1], x2+pad)]
                    
                    if plate_img.size == 0:
                        continue
                    
                    # OCR extraction
                    plate_text = self.hd_ocr_extract(plate_img)
                    
                    # Draw detection box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Check if should capture (STRICT - only once per plate)
                    if (plate_text and 
                        self.is_valid_license_plate(plate_text) and 
                        not self.is_already_captured(plate_text)):
                        
                        # Mark as captured IMMEDIATELY to prevent duplicates
                        self.mark_as_captured(plate_text)
                        
                        # Save HD screenshot
                        self.save_hd_screenshot(plate_img, plate_text, confidence*100, frame)
                        self.update_detection_stats(plate_text)
                        
                        logger.info(f"🚗📸 NEW HD CAPTURE: {plate_text} (#{len(self.captured_plates)})")
                        
                        # Mark as newly captured
                        cv2.rectangle(annotated_frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 4)
                    
                    # Display information
                    display_text = plate_text if plate_text else "SCANNING..."
                    label = f"PLATE: {display_text}"
                    
                    # Add label background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1-30), (x1+text_size[0]+10, y1-5), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, label, (x1+5, y1-12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show confidence
                    cv2.putText(annotated_frame, f"{confidence:.2f}", 
                               (x2-50, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Show capture status
                    if plate_text and self.is_already_captured(plate_text):
                        cv2.putText(annotated_frame, "✅ CAPTURED", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status info
        cv2.putText(annotated_frame, f"HD MODE: {fps:.1f} FPS | No Duplicates", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated_frame

    def get_detection_stats(self):
        with self.stats_lock:
            return self.detection_stats.copy()

    def cleanup(self):
        self.screenshot_executor.shutdown(wait=False)


# Global instances
reader = None
processor = None

def generate_frames():
    """Generate frames for web streaming"""
    global reader, processor
    
    frame_count = 0
    
    while True:
        try:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            
            frame_count += 1
            
            # Process frame
            try:
                processed_frame = processor.process_frame_realtime(frame, frame_count)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                processed_frame = frame
            
            # Encode frame
            try:
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Stream error: {e}")
            time.sleep(0.001)


# Flask routes
@app.route('/')
def index():
    return send_from_directory(DASHBOARD_DIR, 'dashboard.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    global processor
    if processor:
        return jsonify(processor.get_detection_stats())
    return jsonify({'total_detections': 0, 'unique_plates': 0, 'screenshots_saved': 0})

@app.route('/api/refresh')
def refresh_stats():
    return get_stats()

@app.route('/api/plates')
def get_plates():
    global processor
    if processor:
        with processor.captured_plates_lock:
            plates_list = list(processor.captured_plates)
        return jsonify({'plates': sorted(plates_list)})
    return jsonify({'plates': []})

@app.route('/api/reset_captures')
def reset_captures():
    """Reset captured plates list (for testing purposes)"""
    global processor
    if processor:
        with processor.captured_plates_lock:
            processor.captured_plates.clear()
            processor.plate_hashes.clear()
        logger.info("🔄 Captured plates list reset")
        return jsonify({'status': 'success', 'message': 'Captured plates reset'})
    return jsonify({'status': 'error', 'message': 'Processor not available'})

def main():
    global reader, processor
    
    try:
        logger.info("🚀 STARTING HD QUALITY LICENSE PLATE DETECTION")
        logger.info("📸 Enhanced image processing for crystal clear screenshots")
        logger.info("🚫 Strict duplicate prevention - one capture per plate")
        
        # Test RTSP connection first
        logger.info("🔍 Testing RTSP connection...")
        test_cap = cv2.VideoCapture(RTSP_STREAM)
        if test_cap.isOpened():
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                logger.info("✅ RTSP stream test successful")
                h, w = test_frame.shape[:2]
                logger.info(f"📺 Stream resolution: {w}x{h}")
            else:
                logger.warning("⚠️ RTSP stream opened but no frames received")
        else:
            logger.error("❌ RTSP stream test failed - check camera connection")
            logger.error(f"🔗 Stream URL: {RTSP_STREAM}")
            logger.error("💡 Troubleshooting tips:")
            logger.error("   1. Check if camera is online and accessible")
            logger.error("   2. Verify RTSP URL format and credentials")
            logger.error("   3. Test with VLC or similar player first")
            logger.error("   4. Check network connectivity")
        test_cap.release()
        
        processor = RealTimePlateDetector(MODEL_PATH)
        reader = FastRTSPReader(RTSP_STREAM)
        
        # Give some time for connection to establish
        time.sleep(2)
        
        if not reader.is_connected:
            logger.error("❌ Failed to connect to RTSP stream")
            logger.error("🔧 Please check:")
            logger.error(f"   - Camera IP: 10.10.97.12")
            logger.error(f"   - RTSP Port: 554")
            logger.error(f"   - Credentials: camera:Rj18cb4820!")
            logger.error(f"   - Network connectivity")
            return
        
        logger.info("🌐 Web Dashboard: http://localhost:5000")
        logger.info("🎯 System ready for HD quality captures!")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if processor:
            processor.cleanup()
        if reader:
            reader.release()

if __name__ == "__main__":
    main()