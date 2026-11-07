"""
config.py — Central configuration loader for the Vehicle License Plate Detection System.

All sensitive values are read from environment variables first.
If a .env file exists in the project root, it is loaded automatically.
Fall back to the defaults shown below when no env var is set.

HOW TO USE:
  1. Copy .env.example → .env
  2. Fill in your camera URL, MongoDB URI, and paths
  3. Never commit .env to git (it's in .gitignore)
"""

import os

# ---------------------------------------------------------------------------
# Optional: auto-load a .env file if python-dotenv is installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed → rely on real environment variables

# ---------------------------------------------------------------------------
# Camera / Stream
# ---------------------------------------------------------------------------
RTSP_STREAM = os.environ.get(
    "RTSP_STREAM",
    "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/"   # ← replace in .env
)

# ---------------------------------------------------------------------------
# Paths  (default = current working directory → portable across machines)
# ---------------------------------------------------------------------------
DASHBOARD_DIR   = os.environ.get("DASHBOARD_DIR",   os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(DASHBOARD_DIR, "screenshots")
CAR_IMAGES_DIR  = os.path.join(DASHBOARD_DIR, "car_images")
PLATE_IMAGES_DIR= os.path.join(DASHBOARD_DIR, "plate_images")

MODEL_PATH      = os.environ.get("MODEL_PATH",      "runs/detect/license_plate_detection/weights/best.pt")
YOLO_CAR_MODEL  = os.environ.get("YOLO_CAR_MODEL",  "yolov8n.pt")

# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # Windows default
)

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_URI             = os.environ.get("MONGO_URI",              "mongodb://localhost:27017")
MONGO_DB_NAME         = os.environ.get("MONGO_DB_NAME",         "enhanced_vehicle_db")
MONGO_COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "vehicle_records")

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", 5000))
