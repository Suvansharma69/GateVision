from ultralytics import YOLO
import cv2
import os
import sys

# Load config (reads .env automatically)
from config import MODEL_PATH, DASHBOARD_DIR

# --- Video paths (pass as CLI args or set defaults here) ---
# Usage: python detect_no_plate.py [input_video] [output_video]
video_path  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DASHBOARD_DIR, "sample.mp4")
output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DASHBOARD_DIR, "output.mp4")
model_path  = MODEL_PATH

# Load the trained YOLO model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output video saved at: {output_path}")