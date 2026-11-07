<div align="center">

# 🚗 GateVision

### Smart Vehicle Entry/Exit Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-7B2FBE)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)](https://flask.palletsprojects.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?logo=mongodb&logoColor=white)](https://mongodb.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7CA18)](LICENSE)

**GateVision** watches your gate 24/7. It reads every license plate from a live camera stream, decides whether a vehicle is entering or exiting, records the exact timestamp, and stores the full-colour vehicle image — all searchable in real time from any browser.

</div>

---

## 📋 Table of Contents

- [What GateVision Does](#-what-gatevision-does)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration-env)
- [Running GateVision](#-running-gatevision)
- [Web Dashboard](#-web-dashboard)
- [Database Schema](#-database-schema)
- [Model Training](#-model-training)
- [Scripts Reference](#-scripts-reference)
- [Troubleshooting](#-troubleshooting)

---

## 🔭 What GateVision Does

GateVision is an **end-to-end automated gate intelligence system** that:

1. **Connects** to any RTSP IP camera stream
2. **Detects** every vehicle (car, truck, bus, motorcycle) using **YOLOv8**
3. **Reads** the license plate text with a smart **multi-pass OCR ensemble**
4. **Determines** entry vs. exit using multi-method vehicle orientation analysis
5. **Timestamps** every event with second-level precision
6. **Saves** a full-colour vehicle image + plate close-up to disk
7. **Stores** structured records in **MongoDB** for instant querying
8. **Displays** everything live on a sleek **Flask web dashboard**

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     RTSP IP Camera                           │
└──────────────────────┬───────────────────────────────────────┘
                       │  RTSP Stream
                       ▼
┌──────────────────────────────────────────────────────────────┐
│          UltraStableRTSPReader  (threaded reader)            │
│  • Multi-backend: FFmpeg / GStreamer / fallback              │
│  • Auto-reconnect on drop                                    │
│  • Minimal-buffer frame queue (< 3 frames latency)          │
└──────────────────────┬───────────────────────────────────────┘
                       │  Raw frames
                       ▼
┌──────────────────────────────────────────────────────────────┐
│          EnhancedCarDetector  (main processing engine)       │
│                                                              │
│   YOLOv8n COCO ──▶ Vehicle Bbox ──▶  YOLOv8 best.pt        │
│   (car/truck/       (cropped)         (plate region)         │
│    bus/moto)                               │                 │
│                                           ▼                  │
│   Orientation ◀── Multi-method      OCR Ensemble            │
│   Detector         analysis          (Tesseract +            │
│   (entry/exit)     (plate pos,        pre-processing)        │
│                     headlights,                              │
│                     geometry)                                │
└──────┬───────────────────────────────┬────────────────────── ┘
       │  Images to disk               │  Structured records
       ▼                               ▼
 ┌───────────────┐            ┌─────────────────────┐
 │  car_images/  │            │   MongoDB            │
 │  plate_images/│            │   enhanced_          │
 │  screenshots/ │            │   vehicle_db         │
 └───────────────┘            └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   Flask Dashboard     │
                              │   :5000               │
                              │   (live video +       │
                              │    records + search)  │
                              └──────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎥 **Live RTSP Stream** | Connects to any IP camera; auto-reconnects on drop |
| 🚗 **Vehicle Detection** | YOLOv8n COCO — car, truck, bus, motorcycle |
| 🔖 **Plate Detection** | Custom-trained YOLOv8 (20 epochs, 640 px, Roboflow dataset) |
| 📝 **Smart OCR** | Multi-pass Tesseract ensemble with adaptive image pre-processing |
| 🕐 **Timestamping** | Every detection logged with exact date & time |
| 🔄 **Entry / Exit** | 4-method orientation analysis: plate position + geometry + headlights + features |
| 🎨 **Colour Preservation** | All images saved in original colour — no greyscale conversion |
| 📸 **Dual Capture** | Full vehicle image + plate close-up saved per detection |
| 🗄 **MongoDB Storage** | Async writes; indexed for fast plate/time queries |
| 🧹 **Auto Dedup** | Smart duplicate cleanup within configurable time window |
| 🌐 **Web Dashboard** | Real-time Flask UI: live video, stats, records table, image gallery, search |
| ⚡ **GPU Ready** | CUDA-accelerated if NVIDIA GPU available; CPU fallback |
| 🔒 **Secure Config** | All credentials in `.env` — nothing hardcoded in source |
| 🧵 **Multi-threaded** | Stream reader, detector, Flask server run independently |

---

## 📁 Project Structure

```
GateVision/
│
├── noplate_with_database.py      ⭐ MAIN: Full pipeline (RTSP + YOLO + OCR + Entry/Exit + MongoDB + Flask)
├── test.py                       Enhanced: adds smart duplicate cleanup + no-frame-skip mode
├── exp.py                        Experimental: ultra-fast processing with vehicle tracker
│
├── live_noplate_screenshot.py    Standalone: HD screenshots from live stream (no DB)
├── live_stream.py                Utility: raw RTSP viewer / connection tester
├── detect_no_plate.py            Offline: run inference on a saved video file
│
├── train_model.py                Train / fine-tune the license plate YOLO model
├── mongodb_setup.py              DB helper: seed sample data, test connection
│
├── config.py                     ⚙️  Central config — reads from .env or env vars
├── .env.example                  📋 Copy → .env, fill in your credentials
│
├── data.yaml                     YOLO dataset config (relative paths)
├── dashboard.html                Frontend: live Entry/Exit dashboard (served by Flask)
│
├── requirements.txt              Python dependencies
├── .gitignore
├── LICENSE
│
├── runs/detect/
│   └── license_plate_detection/
│       ├── weights/
│       │   └── best.pt           🏆 Trained license plate model (~5.4 MB)
│       ├── results.csv / results.png
│       ├── F1_curve.png / PR_curve.png / P_curve.png / R_curve.png
│       ├── confusion_matrix.png / confusion_matrix_normalized.png
│       └── labels.jpg / val_batch*.jpg
│
├── screenshots/                  (auto-created) plate screenshots
├── car_images/                   (auto-created) full-vehicle colour images
└── plate_images/                 (auto-created) plate close-ups
```

---

## 🔧 Prerequisites

### System Requirements
| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU works) | NVIDIA with CUDA 11.8+ |

### External Software

| Software | Purpose | Install |
|---|---|---|
| [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) | License plate text recognition | Windows: installer to `C:\Program Files\Tesseract-OCR\`<br>Linux: `sudo apt install tesseract-ocr` |
| [MongoDB Community](https://www.mongodb.com/try/download/community) | Vehicle record database | Run as service on `localhost:27017` |

---

## 🚀 Installation

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/GateVision.git
cd GateVision
```

### 2. Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU users** — install PyTorch with CUDA first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. Configure `.env`

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your values (see [Configuration](#-configuration-env) below).

### 5. Start MongoDB

```bash
# Windows service
net start MongoDB

# Manual
mongod --dbpath "C:\data\db"
```

### 6. Get the Trained Model

The model `runs/detect/license_plate_detection/weights/best.pt` (~5.4 MB) is excluded from git due to size.

**Option A** — Download from [Releases](https://github.com/YOUR_USERNAME/GateVision/releases) page  
**Option B** — Train yourself: `python train_model.py` (see [Model Training](#-model-training))

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` → `.env` and fill in your values:

```dotenv
# Your RTSP camera stream
RTSP_STREAM=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/

# Project root (default = current directory, usually fine as-is)
# DASHBOARD_DIR=C:\path\to\GateVision

# Tesseract OCR executable
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe   # Windows
# TESSERACT_CMD=/usr/bin/tesseract                              # Linux

# MongoDB connection
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=enhanced_vehicle_db
MONGO_COLLECTION_NAME=vehicle_records

# Web dashboard port
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

> **Never commit `.env`** — it's already in `.gitignore`. Only `.env.example` (with no real values) is tracked.

---

## ▶️ Running GateVision

### Full System (Recommended)

```bash
python noplate_with_database.py
```

Starts: RTSP reader → detection pipeline → entry/exit analysis → MongoDB writer → Flask dashboard

### Open Dashboard

```
http://localhost:5000           # local machine
http://<SERVER_IP>:5000         # any device on the same network
```

### Alternative Modes

```bash
python test.py                  # Enhanced: duplicate cleanup + no frame skip
python exp.py                   # Ultra-fast: optimised for high-FPS streams
python live_noplate_screenshot.py  # Screenshots only (no DB)
python live_stream.py           # Raw RTSP viewer
python detect_no_plate.py input.mp4 output.mp4   # Offline video
```

---

## 🌐 Web Dashboard

| Panel | What you see |
|---|---|
| 📺 Live Feed | Annotated video with bounding boxes |
| 📊 Stats | Total detections, unique plates, entries, exits |
| 📋 Records Table | Plate number, timestamp, entry/exit status, confidence, vehicle type |
| 🖼 Image Gallery | Full-car image + plate close-up per detection |
| 🔍 Search | Instant plate number lookup across all records |

### REST API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /video_feed` | MJPEG live stream |
| `GET /api/stats` | Detection statistics (JSON) |
| `GET /api/plates` | All detected plate numbers |
| `GET /api/reset_session` | Clear session counters |
| `GET /api/cleanup_duplicates` | Trigger duplicate cleanup |
| `GET /api/system_info` | Stream health + system info |
| `GET /screenshots/<file>` | Serve saved screenshots |
| `GET /car_images/<file>` | Serve full-car images |
| `GET /plate_images/<file>` | Serve plate close-ups |

---

## 🗄 Database Schema

Each detected vehicle creates one document in MongoDB (`enhanced_vehicle_db.vehicle_records`):

```json
{
  "_id": "ObjectId",
  "plate_number": "MH12AB1234",
  "timestamp": "2026-11-07T17:30:00",
  "confidence": 92.4,
  "orientation": "front",
  "entry_exit": "entry",
  "vehicle_type": "car",
  "car_image_path": "car_images/car_MH12AB1234_20261107_173000.jpg",
  "plate_image_path": "plate_images/plate_MH12AB1234_20261107_173000.jpg",
  "detection_id": "MH12AB1234_20261107_173000_123",
  "enhancement_applied": true,
  "detection_method": "YOLO + OCR Ensemble",
  "created_at": "2026-11-07T17:30:00"
}
```

**Indexes:** `plate_number`, `timestamp`, `created_at`, `confidence`, `detection_id` (unique)

---

## 🏋️ Model Training

### 1. Place Dataset

```
License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/
├── train/images/ & labels/
├── valid/images/ & labels/
└── test/images/  & labels/
```

> Dataset source: [Roboflow Universe — License Plate Recognition v4](https://universe.roboflow.com/) (640 px, 3× augmented)

### 2. Train

```bash
python train_model.py
```

Default: **20 epochs · batch=8 · imgsz=640 · AMP off**  
Best weights → `runs/detect/license_plate_detection/weights/best.pt`

---

## 📜 Scripts Reference

| Script | Role |
|---|---|
| `noplate_with_database.py` | **Main production script** — full GateVision pipeline with colour preservation |
| `test.py` | Enhanced version with improved duplicate dedup and no-frame-skip streaming |
| `exp.py` | Ultra-fast version — higher FPS, vehicle tracker, optimised for speed |
| `live_noplate_screenshot.py` | HD screenshot capture only — no DB, no entry/exit |
| `live_stream.py` | Raw RTSP viewer — use for camera connection testing |
| `detect_no_plate.py` | Offline inference — run on a saved video file |
| `train_model.py` | Fine-tune YOLOv8 on your license plate dataset |
| `mongodb_setup.py` | Seed sample data, test connection, view collection info |
| `config.py` | Central config loader — reads `.env` → env vars → defaults |

---

## 🛠 Troubleshooting

**RTSP stream not connecting**
```
1. Test URL in VLC: Media → Open Network Stream
2. Verify camera is on the same network
3. Check RTSP_STREAM value in your .env file
```

**Tesseract not found**
```
# Update TESSERACT_CMD in .env to your actual path
# Linux: sudo apt install tesseract-ocr
# macOS: brew install tesseract
```

**MongoDB connection refused**
```bash
net start MongoDB        # Windows
sudo systemctl start mongod  # Linux
```

**`best.pt` model not found**
```
python train_model.py    # train it yourself
# or download from the Releases page
```

**GPU / CUDA errors**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Reinstall PyTorch matching your CUDA version from pytorch.org
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection backbone
- [Roboflow Universe](https://universe.roboflow.com/) — license plate dataset
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR engine
- [MongoDB](https://www.mongodb.com/) — document database
- [Flask](https://flask.palletsprojects.com/) — web dashboard

---

<div align="center">
<b>GateVision</b> — Know every vehicle, every moment. 🚗🕐
</div>
