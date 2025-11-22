from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/train/sku110k_retail_detection/weights/best.pt")

# Run prediction on an image
results = model.predict(
    source="test.jpg", 
    conf=0.25,
    save=True
)

print("✅ Prediction complete! Check the 'runs/detect' folder.")
