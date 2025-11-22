"""
Enhanced Retail Detection System
Detects misplaced items and out-of-stock scenarios using YOLO + Feature Matching
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from PIL import Image
import json
from datetime import datetime
from pathlib import Path

class EnhancedRetailMonitor:
    def __init__(self, model_path, similarity_threshold=0.85, spatial_threshold=100):
        """
        Initialize the Enhanced Retail Monitor
        
        Args:
            model_path: Path to trained YOLO model
            similarity_threshold: Cosine similarity threshold (0.0-1.0) for product matching
            spatial_threshold: Maximum pixel distance to consider items in same location
        """
        print(f"Loading YOLO model from: {model_path}")
        self.detector = YOLO(model_path)
        
        # Load Feature Extractor (ResNet50 for better features)
        print("Loading feature extractor (ResNet50)...")
        self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer to get feature embeddings
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
        
        # Thresholds
        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold
        
        # Detection results storage
        self.results_log = []

    def get_embedding(self, image_crop):
        """Extract feature embedding from a product image crop"""
        if image_crop is None or image_crop.size == 0:
            return None
            
        try:
            # Convert OpenCV BGR to PIL RGB
            img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature_vector = self.feature_extractor(input_tensor)
            
            return feature_vector.flatten().cpu().numpy()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def detect_and_extract(self, image, conf_threshold=0.25):
        """
        Run YOLO detection and extract features for every item found
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for YOLO detections
            
        Returns:
            List of detections with bounding boxes, centers, and embeddings
        """
        results = self.detector(image, conf=conf_threshold, verbose=False)
        detections = []
        
        h, w, _ = image.shape
        
        for box in results[0].boxes:
            # Get coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Ensure valid crop boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            product_crop = image[y1:y2, x1:x2]
            
            if product_crop.size == 0:
                continue

            # Extract visual features
            embedding = self.get_embedding(product_crop)
            
            if embedding is not None:
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1+x2)/2, (y1+y2)/2),
                    "embedding": embedding,
                    "confidence": confidence,
                    "area": (x2-x1) * (y2-y1)
                })
            
        return detections

    def compare_shelves(self, ref_image_path, curr_image_path, output_path=None):
        """
        Compare reference shelf with current shelf to detect misplaced and out-of-stock items
        
        Args:
            ref_image_path: Path to reference (ideal) shelf image
            curr_image_path: Path to current shelf image
            output_path: Optional path to save annotated output image
            
        Returns:
            Dictionary with detection results and statistics
        """
        print(f"\n{'='*60}")
        print("RETAIL SHELF ANALYSIS")
        print(f"{'='*60}")
        
        # Load images
        print(f"Loading reference image: {ref_image_path}")
        ref_img = cv2.imread(ref_image_path)
        if ref_img is None:
            raise ValueError(f"Could not load reference image: {ref_image_path}")
            
        print(f"Loading current image: {curr_image_path}")
        curr_img = cv2.imread(curr_image_path)
        if curr_img is None:
            raise ValueError(f"Could not load current image: {curr_image_path}")

        # Detect items on both shelves
        print("\nDetecting items on REFERENCE shelf...")
        ref_items = self.detect_and_extract(ref_img)
        print(f"  → Found {len(ref_items)} items")
        
        print("Detecting items on CURRENT shelf...")
        curr_items = self.detect_and_extract(curr_img)
        print(f"  → Found {len(curr_items)} items")

        # Analysis results
        correctly_placed = []
        misplaced_items = []
        missing_items = []
        extra_items = []
        
        # Track which current items have been matched
        matched_current_items = set()
        
        # Create output image
        output_img = curr_img.copy()
        
        print(f"\n{'='*60}")
        print("ANALYZING SHELF LAYOUT...")
        print(f"{'='*60}")
        
        # Compare each reference item with current items
        for idx, ref_item in enumerate(ref_items):
            rx, ry = ref_item['center']
            best_match = None
            best_similarity = -1
            min_distance = float('inf')
            
            # Find potential matches in current shelf
            for curr_idx, curr in enumerate(curr_items):
                if curr_idx in matched_current_items:
                    continue
                    
                cx, cy = curr['center']
                # Calculate spatial distance
                pixel_dist = np.sqrt((rx - cx)**2 + (ry - cy)**2)
                
                # Check if within spatial threshold
                if pixel_dist < self.spatial_threshold:
                    # Calculate visual similarity
                    similarity = 1 - cosine(ref_item['embedding'], curr['embedding'])
                    
                    if pixel_dist < min_distance or (pixel_dist == min_distance and similarity > best_similarity):
                        min_distance = pixel_dist
                        best_similarity = similarity
                        best_match = (curr_idx, curr)
            
            if best_match:
                curr_idx, matched_item = best_match
                matched_current_items.add(curr_idx)
                x1, y1, x2, y2 = matched_item['bbox']
                
                if best_similarity >= self.similarity_threshold:
                    # CORRECTLY PLACED - Green box
                    correctly_placed.append({
                        'location': (x1, y1, x2, y2),
                        'similarity': best_similarity,
                        'spatial_distance': min_distance
                    })
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_img, f"OK ({best_similarity:.2f})", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # MISPLACED - Red box
                    misplaced_items.append({
                        'location': (x1, y1, x2, y2),
                        'similarity': best_similarity,
                        'spatial_distance': min_distance
                    })
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(output_img, f"MISPLACED ({best_similarity:.2f})", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # MISSING/OUT OF STOCK - Yellow box
                x1, y1, x2, y2 = ref_item['bbox']
                missing_items.append({
                    'expected_location': (x1, y1, x2, y2)
                })
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(output_img, "OUT OF STOCK", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Check for extra items (not in reference)
        for curr_idx, curr_item in enumerate(curr_items):
            if curr_idx not in matched_current_items:
                x1, y1, x2, y2 = curr_item['bbox']
                extra_items.append({
                    'location': (x1, y1, x2, y2),
                    'confidence': curr_item['confidence']
                })
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(output_img, "EXTRA", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'reference_image': ref_image_path,
            'current_image': curr_image_path,
            'statistics': {
                'total_reference_items': len(ref_items),
                'total_current_items': len(curr_items),
                'correctly_placed': len(correctly_placed),
                'misplaced': len(misplaced_items),
                'out_of_stock': len(missing_items),
                'extra_items': len(extra_items)
            },
            'details': {
                'correctly_placed': correctly_placed,
                'misplaced_items': misplaced_items,
                'missing_items': missing_items,
                'extra_items': extra_items
            }
        }
        
        # Print summary
        self._print_summary(results)
        
        # Add summary text to image
        self._add_summary_overlay(output_img, results['statistics'])
        
        # Save or display output
        if output_path:
            cv2.imwrite(output_path, output_img)
            print(f"\n✓ Annotated image saved to: {output_path}")
        
        # Display the result
        cv2.imshow("Retail Shelf Analysis", output_img)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save results to JSON
        self._save_results(results)
        
        return results

    def _print_summary(self, results):
        """Print formatted analysis summary"""
        stats = results['statistics']
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Reference Items: {stats['total_reference_items']}")
        print(f"Current Items:   {stats['total_current_items']}")
        print(f"\n✓ Correctly Placed: {stats['correctly_placed']}")
        print(f"✗ Misplaced:        {stats['misplaced']}")
        print(f"⚠ Out of Stock:     {stats['out_of_stock']}")
        print(f"+ Extra Items:      {stats['extra_items']}")
        print(f"{'='*60}")
        
        # Calculate accuracy
        if stats['total_reference_items'] > 0:
            accuracy = (stats['correctly_placed'] / stats['total_reference_items']) * 100
            print(f"\nShelf Accuracy: {accuracy:.1f}%")
        
        # Alert level
        if stats['misplaced'] > 0 or stats['out_of_stock'] > 0:
            print("\n⚠ ALERT: Immediate restocking or reorganization required!")
        else:
            print("\n✓ All items are correctly placed!")

    def _add_summary_overlay(self, image, stats):
        """Add text overlay with statistics to the image"""
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add text
        y_offset = 40
        cv2.putText(image, "SHELF ANALYSIS SUMMARY", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(image, f"Correctly Placed: {stats['correctly_placed']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
        cv2.putText(image, f"Misplaced: {stats['misplaced']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += 25
        cv2.putText(image, f"Out of Stock: {stats['out_of_stock']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
        cv2.putText(image, f"Extra Items: {stats['extra_items']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def _save_results(self, results):
        """Save detection results to JSON file"""
        output_dir = Path("detection_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {output_file}")

    def batch_analyze(self, reference_image, current_images_dir, output_dir="batch_results"):
        """
        Analyze multiple current images against a single reference
        
        Args:
            reference_image: Path to reference shelf image
            current_images_dir: Directory containing current shelf images
            output_dir: Directory to save results
        """
        current_dir = Path(current_images_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        current_images = [f for f in current_dir.iterdir() 
                         if f.suffix.lower() in image_extensions]
        
        print(f"\nBatch Analysis: {len(current_images)} images to process")
        
        all_results = []
        for img_path in current_images:
            print(f"\nProcessing: {img_path.name}")
            output_img_path = output_path / f"annotated_{img_path.name}"
            
            try:
                results = self.compare_shelves(
                    reference_image, 
                    str(img_path), 
                    str(output_img_path)
                )
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        # Save batch summary
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n✓ Batch analysis complete. Results saved to: {output_path}")
        return all_results


def main():
    """Main function to run the retail detection system"""
    
    # Configuration
    MODEL_PATH = "runs/train/sku110k_retail_detection/weights/best.pt"
    # If you haven't trained yet, you can use the pretrained model:
    # MODEL_PATH = "yolov8n.pt"
    
    # UPDATE THESE WITH YOUR IMAGE FILENAMES
    REFERENCE_IMAGE = "sku110k-1/test/images/test_7_jpg.rf.68e62b4db300c23cb69519286f0f9b13.jpg"  # Your reference/ideal shelf
    CURRENT_IMAGE = "sku110k-1/test/images/test_7_jpg.rf.68e62b4db300c23cb69519286f0f9b13.jpg"     # Your current shelf to analyze
    OUTPUT_IMAGE = "shelf_analysis_result.jpg"
    
    # Advanced settings
    SIMILARITY_THRESHOLD = 0.85  # Higher = stricter matching (0.0-1.0)
    SPATIAL_THRESHOLD = 100      # Maximum pixels apart to consider same location
    
    print("""
╔════════════════════════════════════════════════════════════╗
║        ENHANCED RETAIL DETECTION SYSTEM                    ║
║        Misplaced & Out-of-Stock Detection                  ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize monitor
    try:
        monitor = EnhancedRetailMonitor(
            model_path=MODEL_PATH,
            similarity_threshold=SIMILARITY_THRESHOLD,
            spatial_threshold=SPATIAL_THRESHOLD
        )
        
        # Run analysis
        results = monitor.compare_shelves(
            ref_image_path=REFERENCE_IMAGE,
            curr_image_path=CURRENT_IMAGE,
            output_path=OUTPUT_IMAGE
        )
        
        # Optionally, run batch analysis
        # results = monitor.batch_analyze(
        #     reference_image=REFERENCE_IMAGE,
        #     current_images_dir="current_images/",
        #     output_dir="batch_analysis_results/"
        # )
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Model exists at specified path (or train using main.py)")
        print("  2. Reference and current images exist")
        print("\nUpdate the paths in the script and try again.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
