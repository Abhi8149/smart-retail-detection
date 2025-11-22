"""
Simple script to capture reference and current shelf images from webcam
Use this to create proper before/after images for testing
"""

import cv2
from datetime import datetime

def capture_images():
    print("""
╔════════════════════════════════════════════════════════════╗
║     SHELF IMAGE CAPTURE TOOL                               ║
║     Create reference and current images for testing        ║
╚════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
1. First, arrange your shelf/products PERFECTLY (reference state)
2. Press 'R' to capture REFERENCE image
3. Then, move/remove some items to simulate issues
4. Press 'C' to capture CURRENT image
5. Press 'Q' to quit

Both images will be saved and ready for detection!
    """)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✓ Camera opened. Position your shelf in frame...")
    print("\nControls:")
    print("  R - Capture Reference image")
    print("  C - Capture Current image")
    print("  Q - Quit")
    
    reference_captured = False
    current_captured = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text overlay
        display = frame.copy()
        cv2.putText(display, "Press 'R' for Reference, 'C' for Current, 'Q' to Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if reference_captured:
            cv2.putText(display, "Reference: CAPTURED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Reference: NOT captured", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if current_captured:
            cv2.putText(display, "Current: CAPTURED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Current: NOT captured", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Capture Shelf Images", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shelf_reference_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n✓ REFERENCE image saved: {filename}")
            print("  Now move/remove some items to simulate issues...")
            reference_captured = True
            
        elif key == ord('c'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shelf_current_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n✓ CURRENT image saved: {filename}")
            current_captured = True
            
            if reference_captured:
                print("\n" + "="*60)
                print("✓ Both images captured!")
                print("Now you can run:")
                print("  python retail_detection.py")
                print("="*60)
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✓ Camera closed")


if __name__ == "__main__":
    capture_images()
