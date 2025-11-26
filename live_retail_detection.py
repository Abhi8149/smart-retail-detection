"""
Live Retail Detection System with Camera
Real-time detection of in-stock and out-of-stock items using webcam
Similar to hand gesture recognition - works with live camera feed
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from PIL import Image
from datetime import datetime
from collections import deque
import json

class LiveRetailDetector:
    def __init__(self, model_path, reference_image_path=None):
        """
        Initialize Live Retail Detector
        
        Args:
            model_path: Path to trained YOLO model
            reference_image_path: Optional reference shelf image for comparison
        """
        print(f"Loading YOLO model from: {model_path}")
        self.detector = YOLO(model_path)
        
        # Load Feature Extractor (ResNet50)
        print("Loading feature extractor...")
        self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = self.feature_extractor.to(self.device)
        print(f"Using device: {self.device}")
        
        # Image preprocessing for ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Reference shelf data
        self.reference_items = None
        self.reference_loaded = False
        
        if reference_image_path:
            self.load_reference_shelf(reference_image_path)
        
        # Detection settings
        self.conf_threshold = 0.25  # Lower threshold = more detections
        self.similarity_threshold = 0.85
        self.spatial_threshold = 100
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_items': 0,
            'in_stock': 0,
            'out_of_stock': 0,
            'misplaced': 0,
            'extra': 0
        }
        
        # Modes
        self.mode = "detection"  # detection, comparison, calibration
        self.show_help = False

    def load_reference_shelf(self, image_path):
        """Load and process reference shelf image"""
        print(f"\nLoading reference shelf from: {image_path}")
        ref_img = cv2.imread(image_path)
        
        if ref_img is None:
            print(f"Warning: Could not load reference image from {image_path}")
            return False
        
        self.reference_items = self.detect_and_extract(ref_img)
        self.reference_loaded = True
        print(f"Reference shelf loaded with {len(self.reference_items)} items")
        return True

    def get_embedding(self, image_crop):
        """Extract feature embedding from a product image crop"""
        if image_crop is None or image_crop.size == 0:
            return None
            
        try:
            img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature_vector = self.feature_extractor(input_tensor)
            
            return feature_vector.flatten().cpu().numpy()
        except Exception as e:
            return None

    def detect_and_extract(self, image):
        """Run YOLO detection and extract features"""
        results = self.detector(image, conf=self.conf_threshold, verbose=False)
        detections = []
        
        h, w, _ = image.shape
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            product_crop = image[y1:y2, x1:x2]
            
            if product_crop.size == 0:
                continue

            embedding = self.get_embedding(product_crop)
            
            if embedding is not None:
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1+x2)/2, (y1+y2)/2),
                    "embedding": embedding,
                    "confidence": confidence,
                })
            
        return detections

    def compare_with_reference(self, current_items):
        """Compare current frame with reference shelf"""
        if not self.reference_loaded or self.reference_items is None:
            return None, None, None, None
        
        correctly_placed = []
        misplaced_items = []
        missing_items = []
        extra_items = []
        
        matched_current_items = set()
        
        # Compare each reference item with current items
        for ref_item in self.reference_items:
            rx, ry = ref_item['center']
            best_match = None
            best_similarity = -1
            min_distance = float('inf')
            
            for curr_idx, curr in enumerate(current_items):
                if curr_idx in matched_current_items:
                    continue
                    
                cx, cy = curr['center']
                pixel_dist = np.sqrt((rx - cx)**2 + (ry - cy)**2)
                
                if pixel_dist < self.spatial_threshold:
                    similarity = 1 - cosine(ref_item['embedding'], curr['embedding'])
                    
                    if pixel_dist < min_distance or (pixel_dist == min_distance and similarity > best_similarity):
                        min_distance = pixel_dist
                        best_similarity = similarity
                        best_match = (curr_idx, curr)
            
            if best_match:
                curr_idx, matched_item = best_match
                matched_current_items.add(curr_idx)
                
                if best_similarity >= self.similarity_threshold:
                    correctly_placed.append((matched_item, best_similarity))
                else:
                    misplaced_items.append((matched_item, best_similarity))
            else:
                missing_items.append(ref_item)
        
        # Check for extra items
        for curr_idx, curr_item in enumerate(current_items):
            if curr_idx not in matched_current_items:
                extra_items.append(curr_item)
        
        return correctly_placed, misplaced_items, missing_items, extra_items

    def draw_detections(self, frame, items, color, label):
        """Draw bounding boxes for detected items"""
        for item in items:
            if isinstance(item, tuple):
                item, similarity = item[0], item[1]
                x1, y1, x2, y2 = item['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({similarity:.2f})", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                x1, y1, x2, y2 = item['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_ui(self, frame):
        """Draw user interface overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for statistics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "LIVE RETAIL DETECTION", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mode
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(frame, mode_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistics
        y_offset = 110
        if self.mode == "comparison" and self.reference_loaded:
            cv2.putText(frame, f"Total Items: {self.stats['total_items']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"In Stock: {self.stats['in_stock']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            cv2.putText(frame, f"Misplaced: {self.stats['misplaced']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Out of Stock: {self.stats['out_of_stock']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Extra: {self.stats['extra']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        else:
            cv2.putText(frame, f"Detected: {self.stats['total_items']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls hint
        cv2.putText(frame, "Press 'H' for Help", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Help menu
        if self.show_help:
            self.draw_help_menu(frame)

    def draw_help_menu(self, frame):
        """Draw help menu overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Create semi-transparent background
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Help text
        help_text = [
            "KEYBOARD CONTROLS",
            "",
            "H - Toggle this help menu",
            "D - Detection mode (basic)",
            "C - Comparison mode (with reference)",
            "R - Load/Reload reference shelf",
            "S - Save current frame as reference",
            "SPACE - Capture screenshot",
            "P - Pause/Resume",
            "+/- - Adjust confidence threshold",
            "Q - Quit",
            "",
            "Click anywhere to close help"
        ]
        
        y_start = h//4 + 40
        for i, text in enumerate(help_text):
            if text == "" or text.startswith("KEYBOARD"):
                cv2.putText(frame, text, (w//4 + 20, y_start + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, text, (w//4 + 20, y_start + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        start_time = cv2.getTickCount()
        
        # Detect items in current frame
        current_items = self.detect_and_extract(frame)
        self.stats['total_items'] = len(current_items)
        
        if self.mode == "comparison" and self.reference_loaded:
            # Compare with reference
            correct, misplaced, missing, extra = self.compare_with_reference(current_items)
            
            # Update statistics
            self.stats['in_stock'] = len(correct) if correct else 0
            self.stats['misplaced'] = len(misplaced) if misplaced else 0
            self.stats['out_of_stock'] = len(missing) if missing else 0
            self.stats['extra'] = len(extra) if extra else 0
            
            # Draw detections with colors
            if correct:
                self.draw_detections(frame, correct, (0, 255, 0), "OK")
            if misplaced:
                self.draw_detections(frame, misplaced, (0, 0, 255), "MISPLACED")
            if extra:
                self.draw_detections(frame, extra, (255, 0, 255), "EXTRA")
            
            # Draw missing items (from reference)
            for missing_item in (missing or []):
                x1, y1, x2, y2 = missing_item['bbox']
                # Show where item should be (ghost box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "OUT OF STOCK", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            # Simple detection mode - just show all detected items
            for item in current_items:
                x1, y1, x2, y2 = item['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Product ({item['confidence']:.2f})", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)
        self.fps_buffer.append(fps)
        
        # Draw UI
        self.draw_ui(frame)
        
        return frame

    def run(self, camera_id=0, width=1280, height=720):
        """
        Run live detection with camera
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            width: Camera resolution width
            height: Camera resolution height
        """
        print("\n" + "="*60)
        print("LIVE RETAIL DETECTION SYSTEM")
        print("="*60)
        print(f"Camera ID: {camera_id}")
        print(f"Resolution: {width}x{height}")
        print(f"Device: {self.device}")
        
        if self.reference_loaded:
            print(f"Reference loaded: {len(self.reference_items)} items")
            print("Mode: COMPARISON (with reference)")
        else:
            print("Mode: DETECTION (no reference)")
        
        print("\nPress 'H' for keyboard controls")
        print("="*60 + "\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("✓ Camera opened successfully")
        print("Starting live detection...\n")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                self.frame_count += 1
            else:
                # Show paused frame
                cv2.putText(processed_frame, "PAUSED", (processed_frame.shape[1]//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Display
            cv2.imshow("Live Retail Detection - Press Q to quit", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('d'):
                self.mode = "detection"
                print("Switched to DETECTION mode")
            elif key == ord('c'):
                if self.reference_loaded:
                    self.mode = "comparison"
                    print("Switched to COMPARISON mode")
                else:
                    print("No reference loaded! Press 'S' to save current frame as reference")
            elif key == ord('r'):
                # Reload reference
                ref_path = input("\nEnter reference image path: ")
                if self.load_reference_shelf(ref_path):
                    self.mode = "comparison"
            elif key == ord('s'):
                # Save current frame as reference
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ref_filename = f"reference_shelf_{timestamp}.jpg"
                cv2.imwrite(ref_filename, frame)
                print(f"\n✓ Saved current frame as: {ref_filename}")
                if self.load_reference_shelf(ref_filename):
                    self.mode = "comparison"
            elif key == ord(' '):
                # Capture screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"\n✓ Screenshot saved: {filename}")
            elif key == ord('p'):
                paused = not paused
                print("\nPAUSED" if paused else "\nRESUMED")
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.9, self.conf_threshold + 0.05)
                print(f"\nConfidence threshold: {self.conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                print(f"\nConfidence threshold: {self.conf_threshold:.2f}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {self.frame_count}")
        avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        print(f"Average FPS: {avg_fps:.2f}")
        print("="*60)


def main():
    """Main function to run live retail detection"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     LIVE RETAIL DETECTION SYSTEM                           ║
║     Real-time Stock Monitoring with Camera                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    MODEL_PATH = "runs/train/sku110k_retail_detection/weights/best.pt" 
    
    REFERENCE_IMAGE = None
    
    CAMERA_ID = 0  
    RESOLUTION = (1280, 720)  
    
    try:
        # Initialize detector
        detector = LiveRetailDetector(
            model_path=MODEL_PATH,
            reference_image_path=REFERENCE_IMAGE
        )
        
        # Run live detection
        detector.run(
            camera_id=CAMERA_ID,
            width=RESOLUTION[0],
            height=RESOLUTION[1]
        )
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Model exists at specified path")
        print("  2. Camera is connected and accessible")
        print("  3. Reference image exists (if specified)")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
