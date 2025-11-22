import cv2
from ultralytics import YOLO
import torch

# Load your trained model (or use pretrained)
# For your custom trained model replace yolov8s.pt with your best.pt
model = YOLO("yolov8s.pt")  

# Use GPU if available
device = 0 if torch.cuda.is_available() else 'cpu'

# Open laptop camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Live CCTV Tracking Started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking (gives each object an ID)
    results = model.track(frame, persist=True, device=device)

    # Draw bounding boxes + IDs
    annotated_frame = results[0].plot()

    # Show live feed window
    cv2.imshow("Live Shop CCTV - Object & Customer Tracking", annotated_frame)

    # Quit when 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
