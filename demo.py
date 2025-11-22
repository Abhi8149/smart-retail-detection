"""
Quick Demo Script - Test retail detection with webcam or sample images
"""

import cv2
from pathlib import Path
from retail_detection import EnhancedRetailMonitor

def capture_from_webcam():
    """Capture reference and current images from webcam"""
    print("\n=== WEBCAM CAPTURE MODE ===\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None, None
    
    print("Instructions:")
    print("1. Position camera to capture shelf")
    print("2. Press 'R' to capture REFERENCE image")
    print("3. Make changes to shelf (misplace items, remove items)")
    print("4. Press 'C' to capture CURRENT image")
    print("5. Press 'Q' to quit\n")
    
    reference_img = None
    current_img = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display instructions on frame
        display = frame.copy()
        cv2.putText(display, "Press 'R' for Reference, 'C' for Current, 'Q' to Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if reference_img is not None:
            cv2.putText(display, "Reference: CAPTURED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if current_img is not None:
            cv2.putText(display, "Current: CAPTURED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Webcam Capture', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r') or key == ord('R'):
            reference_img = frame.copy()
            cv2.imwrite("shelf_reference.jpg", reference_img)
            print("✓ Reference image captured and saved!")
            
        elif key == ord('c') or key == ord('C'):
            current_img = frame.copy()
            cv2.imwrite("shelf_current.jpg", current_img)
            print("✓ Current image captured and saved!")
            
            if reference_img is not None:
                print("\n✓ Both images captured! Ready to analyze.")
                break
            else:
                print("⚠ Please capture reference image first (press 'R')")
                
        elif key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return "shelf_reference.jpg" if reference_img is not None else None, \
           "shelf_current.jpg" if current_img is not None else None


def check_sample_images():
    """Check if sample images exist"""
    ref_path = Path("shelf_reference.jpg")
    curr_path = Path("shelf_current.jpg")
    
    if ref_path.exists() and curr_path.exists():
        return str(ref_path), str(curr_path)
    return None, None


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║             RETAIL DETECTION - DEMO MODE                   ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check for model
    model_path = "runs/train/sku110k_retail_detection/weights/best.pt"
    if not Path(model_path).exists():
        print(f"⚠ Trained model not found at: {model_path}")
        print("Using pretrained YOLOv8 model instead...")
        model_path = "yolov8n.pt"
    
    # Check for existing images
    ref_img, curr_img = check_sample_images()
    
    if ref_img and curr_img:
        print(f"✓ Found existing images:")
        print(f"  Reference: {ref_img}")
        print(f"  Current: {curr_img}")
        print("\nOptions:")
        print("  1. Use existing images")
        print("  2. Capture new images from webcam")
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            ref_img, curr_img = capture_from_webcam()
    else:
        print("No sample images found.")
        print("\nOptions:")
        print("  1. Capture from webcam")
        print("  2. Exit (add your own images first)")
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            ref_img, curr_img = capture_from_webcam()
        else:
            print("\nPlease add 'shelf_reference.jpg' and 'shelf_current.jpg'")
            print("Then run this script again.")
            return
    
    if not ref_img or not curr_img:
        print("\n✗ Could not get both images. Exiting.")
        return
    
    # Run detection
    print("\n" + "="*60)
    print("Starting Detection...")
    print("="*60)
    
    try:
        monitor = EnhancedRetailMonitor(
            model_path=model_path,
            similarity_threshold=0.85,
            spatial_threshold=100
        )
        
        results = monitor.compare_shelves(
            ref_image_path=ref_img,
            curr_image_path=curr_img,
            output_path="demo_result.jpg"
        )
        
        print("\n✓ Demo complete!")
        print("Check 'demo_result.jpg' for annotated output")
        
    except Exception as e:
        print(f"\n✗ Error during detection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
