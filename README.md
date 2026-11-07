# 🚗 Vehicle License Plate Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=data:image/svg+xml;base64,)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green?logo=mongodb)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A production-ready, real-time license plate detection and vehicle management system.**  
Detects license plates from live RTSP camera streams, reads them with OCR, tracks vehicle entry/exit, and displays everything on a live web dashboard — all backed by MongoDB.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the System](#-running-the-system)
- [Web Dashboard](#-web-dashboard)
- [Database Schema](#-database-schema)
- [Model Training](#-model-training)
- [Scripts Reference](#-scripts-reference)
- [Troubleshooting](#-troubleshooting)

---

## 🔭 Overview

This project is an end-to-end **automated vehicle surveillance system** that:

1. Connects to an **RTSP IP camera** stream.
2. Runs **YOLOv8** to detect vehicles (cars, trucks, buses, motorcycles) and isolate license plates in every frame.
3. Applies a **multi-pass OCR ensemble** (Tesseract + image pre-processing) to read the plate text.
4. Determines whether the vehicle is **entering** or **exiting** using multi-method orientation analysis.
5. Saves a **full-colour vehicle image + plate close-up** to disk and stores structured records in **MongoDB**.
6. Presents all data in real time on a **Flask web dashboard** accessible from any browser on the network.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RTSP IP Camera                          │
└────────────────────────┬────────────────────────────────────┘
                         │  RTSP Stream
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              UltraStableRTSPReader (threaded)                │
│  • Multi-backend connection (FFmpeg / GStreamer / fallback)  │
│  • Auto-reconnect on drop                                    │
│  • Minimal-buffer frame queue (< 3 frames)                  │
└────────────────────────┬────────────────────────────────────┘
                         │  Raw frames
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          EnhancedCarDetector / UltraFastCarDetector          │
│  ┌──────────────────┐   ┌────────────────────────────────┐  │
│  │  YOLOv8n (COCO)  │   │  Custom YOLOv8 License-Plate   │  │
│  │  Vehicle detect  │──▶│  Detector (best.pt)            │  │
│  └──────────────────┘   └──────────────┬───────────────── ┘  │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌──────────────┐        ┌──────────────────────────┐       │
│  │ Orientation  │        │  OCR Ensemble             │       │
│  │ Detector     │        │  (Tesseract + pre-proc)   │       │
│  │ front / back │        └──────────────────────────┘       │
│  └──────────────┘                                           │
└───────┬──────────────────────────┬──────────────────────────┘
        │  Images to disk          │  Structured records
        ▼                          ▼
 ┌────────────┐           ┌─────────────────┐
 │ car_images │           │    MongoDB       │
 │ plate_imgs │           │  enhanced_       │
 │screenshots │           │  vehicle_db      │
 └────────────┘           └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  Flask Dashboard │
                          │  :5000           │
                          └─────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎥 **Live RTSP Stream** | Connects to any RTSP camera; auto-reconnects on drop |
| 🚗 **Vehicle Detection** | YOLOv8n COCO — detects car, truck, bus, motorcycle |
| 🔖 **Plate Detection** | Custom-trained YOLOv8 model (20 epochs, 640 px) |
| 📝 **OCR** | Multi-method Tesseract ensemble with image pre-processing |
| 🔄 **Entry / Exit** | Plate-position + geometry + headlight analysis |
| 🎨 **Colour Preservation** | All captured images saved in original colour (no greyscale) |
| 📸 **Image Capture** | Full-car image + plate close-up saved per detection |
| 🗄 **MongoDB Storage** | Async writes; indexed on plate, timestamp, detection_id |
| 🧹 **Duplicate Cleanup** | Automatic dedup of repeated detections within configurable window |
| 🌐 **Web Dashboard** | Real-time Flask UI with live video, stats, records & analytics |
| ⚡ **GPU Acceleration** | Runs on CUDA if available; falls back to CPU |
| 🧵 **Multi-threaded** | Stream reader, detector, and Flask server run in separate threads |

---

## 📁 Project Structure

```
no_plate/
│
├── noplate_with_database.py      # ⭐ MAIN: Full system (RTSP + YOLO + OCR + Entry/Exit + MongoDB + Flask)
├── test.py                       # Enhanced version with duplicate cleanup & optimised streaming
├── exp.py                        # Experimental version with ultra-fast processing mode
│
├── live_noplate_screenshot.py    # Standalone: HD screenshots from live stream
├── live_stream.py                # Standalone: Raw RTSP stream viewer
├── detect_no_plate.py            # Standalone: Offline video-file inference
│
├── train_model.py                # Train/fine-tune the license plate YOLO model
├── mongodb_setup.py              # Setup helper: seed sample data, test connection
│
├── data.yaml                     # YOLO dataset config (paths, classes)
├── dashboard.html                # Frontend: Entry/Exit web dashboard (served by Flask)
│
├── requirements.txt              # Python dependencies
├── .gitignore
│
├── runs/                         # YOLO training artefacts
│   └── detect/
│       └── license_plate_detection/
│           ├── weights/
│           │   └── best.pt       # 🏆 Trained license plate model
│           ├── results.csv
│           ├── results.png
│           ├── F1_curve.png
│           ├── PR_curve.png
│           └── confusion_matrix.png
│
├── screenshots/                  # (auto-created) HD plate screenshots
├── car_images/                   # (auto-created) Full-car colour images
└── plate_images/                 # (auto-created) Plate close-ups
```

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10/11 (Linux also works with path adjustments)
- **Python**: 3.10 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but strongly recommended)

### External Software
| Software | Purpose | Download |
|---|---|---|
| [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) | License plate text recognition | Install to `C:\Program Files\Tesseract-OCR\` |
| [MongoDB Community](https://www.mongodb.com/try/download/community) | Vehicle record database | Run as a service on `localhost:27017` |
| [Git LFS](https://git-lfs.github.com/) | Tracking large model files (optional) | Required if cloning `best.pt` via LFS |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/vehicle-license-plate-detection.git
cd vehicle-license-plate-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. Install Tesseract OCR (Windows)

Download and install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).  
Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 5. Start MongoDB

```bash
# Windows Service (if installed as a service)
net start MongoDB

# Or manually
mongod --dbpath "C:\data\db"
```

### 6. Obtain the Trained Model

The pre-trained license plate detection model (`best.pt`) lives at:
```
runs/detect/license_plate_detection/weights/best.pt
```
If it is not included in the clone (large file), either:
- Download it from the [Releases page](https://github.com/YOUR_USERNAME/vehicle-license-plate-detection/releases), **or**
- Train it yourself (see [Model Training](#-model-training)).

---

## ⚙️ Configuration

Open `noplate_with_database.py` (or `test.py`) and update the top-level constants:

```python
# ── Camera ──────────────────────────────────────────────
RTSP_STREAM   = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/"

# ── Paths ───────────────────────────────────────────────
DASHBOARD_DIR = r"D:\vehicle_speed\no_plate"   # project root
MODEL_PATH    = "runs/detect/license_plate_detection/weights/best.pt"
YOLO_CAR_MODEL = "yolov8n.pt"

# ── MongoDB ─────────────────────────────────────────────
MONGO_URI             = "mongodb://localhost:27017"
MONGO_DB_NAME         = "enhanced_vehicle_db"
MONGO_COLLECTION_NAME = "vehicle_records"

# ── Tesseract (Windows path) ─────────────────────────────
# (in the script)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## ▶️ Running the System

### Full System (Recommended)

```bash
python noplate_with_database.py
```

This starts:
- RTSP stream reader
- Vehicle + plate detection pipeline
- Entry/exit orientation analysis
- MongoDB writer
- Flask dashboard on `http://localhost:5000`

### Alternative Variants

```bash
# Enhanced version with duplicate cleanup
python test.py

# Ultra-fast experimental version
python exp.py

# Screenshot-only mode (no DB)
python live_noplate_screenshot.py

# Raw stream viewer
python live_stream.py

# Offline video file inference
python detect_no_plate.py
```

---

## 🌐 Web Dashboard

Once running, open your browser and go to:

```
http://localhost:5000        # local
http://<SERVER_IP>:5000      # from another device on the same network
```

### Dashboard Features

- 📺 **Live video feed** with bounding boxes
- 📊 **Real-time statistics** — total detections, unique plates, entries, exits
- 📋 **Vehicle records table** — plate number, timestamp, entry/exit status, confidence
- 🖼 **Image gallery** — full-car and plate close-up thumbnails
- 🔍 **Plate search** — search historical records
- 🧹 **Session reset** — clear session data without touching the DB

### REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/video_feed` | MJPEG live stream |
| GET | `/api/stats` | JSON detection statistics |
| GET | `/api/plates` | List of all detected plate numbers |
| GET | `/api/reset_captures` | Reset in-memory plate list |
| GET | `/api/reset_session` | Reset session counters & tracks |
| GET | `/api/cleanup_duplicates` | Trigger duplicate cleanup |
| GET | `/api/system_info` | System health & stream stats |
| GET | `/screenshots/<file>` | Serve saved screenshot images |
| GET | `/car_images/<file>` | Serve full-car images |
| GET | `/plate_images/<file>` | Serve plate close-up images |

---

## 🗄 Database Schema

Each detected vehicle creates one document in MongoDB:

```json
{
  "_id": "ObjectId",
  "plate_number": "MH12AB1234",
  "timestamp": "2026-11-07T17:30:00",
  "confidence": 92.4,
  "orientation": "front",
  "entry_exit": "entry",
  "vehicle_type": "car",
  "car_image_path": "car_images/car_MH12AB1234_20261107_173000_123.jpg",
  "plate_image_path": "plate_images/plate_MH12AB1234_20261107_173000_123.jpg",
  "detection_id": "MH12AB1234_20261107_173000_123",
  "enhancement_applied": true,
  "detection_method": "YOLO + OCR Ensemble",
  "image_quality": "Ultra HD Enhanced",
  "created_at": "2026-11-07T17:30:00"
}
```

**Indexes:** `plate_number`, `timestamp`, `created_at`, `confidence`, `detection_id` (unique)

---

## 🏋️ Model Training

To train or re-train the license plate detection model on your own dataset:

### 1. Prepare Dataset

Place your YOLOv8-formatted dataset into:
```
License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
├── test/images/
└── test/labels/
```

> The included dataset has been downloaded from [Roboflow Universe](https://universe.roboflow.com/) — **License Plate Recognition v4** (640 px, 3× augmented).

### 2. Update `data.yaml`

```yaml
train: path/to/train/images
val:   path/to/valid/images
test:  path/to/test/images
nc: 1
names: ['License_Plate']
```

### 3. Run Training

```bash
python train_model.py
```

Default config: **20 epochs, batch=8, imgsz=640, AMP disabled**.  
Best weights saved to `runs/detect/license_plate_detection/weights/best.pt`.

---

## 📜 Scripts Reference

| Script | Purpose |
|---|---|
| `noplate_with_database.py` | **Main production script** — full pipeline with Entry/Exit + colour preservation |
| `test.py` | Enhanced version — adds improved duplicate cleanup and no-frame-skip mode |
| `exp.py` | Ultra-fast experimental version — optimised for speed with vehicle tracker |
| `live_noplate_screenshot.py` | HD screenshots only — no DB, no entry/exit |
| `live_stream.py` | Raw RTSP stream display — connection test utility |
| `detect_no_plate.py` | Offline inference on a saved video file |
| `train_model.py` | Fine-tune YOLOv8 on your license plate dataset |
| `mongodb_setup.py` | Seed sample data, test MongoDB connection, view collection info |

---

## 🛠 Troubleshooting

### RTSP stream not connecting
- Test the URL in VLC first: `Media → Open Network Stream → paste URL`
- Ensure the camera is on the same network/subnet
- Try different OpenCV backends: `cv2.CAP_FFMPEG`, `cv2.CAP_GSTREAMER`

### Tesseract not found
```
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
Adjust this path to your actual Tesseract install location.

### MongoDB connection refused
```bash
# Check if MongoDB is running
Get-Service MongoDB     # Windows PowerShell
systemctl status mongod # Linux

# Start it
net start MongoDB       # Windows
sudo systemctl start mongod # Linux
```

### `best.pt` model not found
Run `python train_model.py` to train the model, or download it from the [Releases](https://github.com/YOUR_USERNAME/vehicle-license-plate-detection/releases) page.

### CUDA / GPU errors
Verify your CUDA version matches your PyTorch build:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## 📄 License

This project is released under the **MIT License**.  
See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection backbone
- [Roboflow Universe](https://universe.roboflow.com/) — license plate dataset
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR engine
- [MongoDB](https://www.mongodb.com/) — document database
- [Flask](https://flask.palletsprojects.com/) — web dashboard framework

---

<div align="center">
Made with ❤️ | Real-time vehicle intelligence
</div>
