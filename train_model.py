from ultralytics import YOLO
import cv2
import os
import yaml

# Define dataset configuration
dataset_config = {
    'train': 'D:/vehicle_speed/no_plate/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/train/images',
    'val': 'D:/vehicle_speed/no_plate/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/valid/images',
    'test': 'D:/vehicle_speed/no_plate/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/test/images',
    'nc': 1,
    'names': ['License_Plate']
}

# Save dataset configuration to a YAML file
with open('data.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

# Function to perform inference on a single image
def detect_license_plate(image_path, output_path):
    # Load the trained model
    trained_model = YOLO('runs/detect/license_plate_detection/weights/best.pt')
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Perform inference
    results = trained_model.predict(image_path, conf=0.4, iou=0.3)
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Draw rectangle around detected license plate
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save the output image
    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 nano model

    # Train the model
    model.train(
        data='data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        name='license_plate_detection',
        augment=True,
        exist_ok=True,
        amp=False,
        workers=4  # Reduced for stability
    )

    print("Model training completed. Use detect_license_plate() for inference.")