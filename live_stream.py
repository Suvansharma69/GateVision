import cv2
import os
import time

# Load RTSP URL from config (reads .env automatically)
from config import RTSP_STREAM

print(f"🔗 Connecting to: {RTSP_STREAM}")

# Create VideoCapture object
cap = cv2.VideoCapture(RTSP_STREAM)
if not cap.isOpened():
    print("Error: Could not open RTSP stream")
    exit()

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Optional: Set additional properties
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Re-check if the stream is opened after setting properties
if not cap.isOpened():
    print("Error: Could not open RTSP stream after setting properties")
    exit()

print("RTSP stream opened successfully")

try:
    while True:
        # cap.read() blocks until a new frame is ready
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame. Stream may have ended.")
            break
        
        # Display the frame
        cv2.imshow('RTSP Stream', frame)
        
        # Press 'q' to quit. The waitKey(1) provides a necessary small delay
        # for the window to process events.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("Stream interrupted by user")
    
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed")