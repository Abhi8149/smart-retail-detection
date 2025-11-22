"""
Quick Script to Test YOLO Model on a Single Image
"""

import cv2
from ultralytics import YOLO
import sys

def test_image(image_path, model_path="yolov8s.pt", conf_threshold=0.25):
    """
    Test YOLO model on a single image
    
    Args:
        image_path: Path to the image file
        model_path: Path to YOLO model (default: yolov8s.pt)
        conf_threshold: Confidence threshold for detections
    """
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Load image
    print("Loading image...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Run detection
    print("\nRunning detection...")
    results = model(image, conf=conf_threshold, verbose=True)
    
    # Get detections
    detections = results[0].boxes
    print(f"\n✓ Found {len(detections)} objects")
    
    # Print detection details
    if len(detections) > 0:
        print("\nDetected Objects:")
        print("-" * 60)
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            print(f"{i+1}. {class_name.upper()}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
            print()
            
            # Draw on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("\n⚠ No objects detected!")
        print("Try:")
        print("  - Lowering confidence threshold (use --conf 0.1)")
        print("  - Using better lighting in the image")
        print("  - Ensuring objects are clearly visible")
    
    # Save result
    output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
    cv2.imwrite(output_path, image)
    print(f"\n✓ Result saved to: {output_path}")
    
    # Display result
    print("\nDisplaying result (press any key to close)...")
    cv2.imshow("Detection Result - Press any key to close", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Default values
    IMAGE_PATH = "sku110k-1\test\images\test_6_jpg.rf.db8bf2307c118a2e534dd109f39d366c.jpg"  # Change this to your image
    MODEL_PATH = "runs/train/sku110k_retail_detection/weights/best.pt"      # Or use your trained model: "runs/train/sku110k_retail_detection/weights/best.pt"
    CONF_THRESHOLD = 0.25
    
    # Check command line arguments
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        MODEL_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        CONF_THRESHOLD = float(sys.argv[3])
    
    # Run test
    test_image(IMAGE_PATH, MODEL_PATH, CONF_THRESHOLD)
