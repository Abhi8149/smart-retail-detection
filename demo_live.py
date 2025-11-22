"""
Quick Demo - Live Retail Detection
Works immediately with pretrained model - no training required!
"""

import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║     QUICK DEMO - LIVE RETAIL DETECTION                     ║
║     Using Pretrained Model (No Training Required)          ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Use pretrained model - works immediately!
    print("Loading pretrained YOLOv8 model...")
    model = YOLO("yolov8s.pt")
    
    # Check GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("✓ Camera opened successfully")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  SPACE - Take screenshot")
    print("  Q     - Quit")
    print("="*60 + "\n")
    print("Starting detection...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.3, device=device, verbose=False)
        
        # Count detections
        num_items = len(results[0].boxes)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Add overlay with statistics
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "LIVE DETECTION DEMO", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Items Detected: {num_items}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press Q to quit, SPACE for screenshot", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("Quick Demo - Live Retail Detection (Press Q to quit)", annotated_frame)
        
        frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord(' '):
            # Take screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"✓ Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    print("\nTo use the full system with comparison mode:")
    print("  python live_retail_detection.py")
    print("\nFor complete guide, see: LIVE_CAMERA_GUIDE.md")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
