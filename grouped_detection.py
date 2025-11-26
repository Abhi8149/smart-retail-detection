"""
Grouped Product Detection with Color-Coded Analysis
Groups similar items and shows different products with different colors
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from PIL import Image
from collections import defaultdict
import colorsys


class GroupedProductDetector:
    def __init__(self, model_path, similarity_threshold=0.85):
        """
        Initialize detector with product grouping capability
        
        Args:
            model_path: Path to trained YOLO model
            similarity_threshold: Threshold for grouping similar items (0.0-1.0)
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
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.similarity_threshold = similarity_threshold
        self.conf_threshold = 0.25
        
        # Color palette for different product groups
        self.colors = self._generate_colors(20)
        
    def _generate_colors(self, n):
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
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
    
    def detect_products(self, image):
        """Detect all products in image and extract features"""
        results = self.detector(image, conf=self.conf_threshold, verbose=False)
        detections = []
        
        h, w, _ = image.shape
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Ensure coordinates are within image bounds
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
                    "group_id": None
                })
        
        return detections
    
    def group_similar_products(self, detections):
        """Group similar products together based on visual similarity"""
        if not detections:
            return []
        
        groups = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            # Start a new group
            current_group = [i]
            used.add(i)
            
            # Find similar items
            for j, other_det in enumerate(detections):
                if j in used:
                    continue
                
                # Calculate similarity
                similarity = 1 - cosine(det['embedding'], other_det['embedding'])
                
                if similarity >= self.similarity_threshold:
                    current_group.append(j)
                    used.add(j)
            
            groups.append(current_group)
        
        # Assign group IDs
        for group_id, group in enumerate(groups):
            for idx in group:
                detections[idx]['group_id'] = group_id
        
        return groups
    
    def draw_grouped_boxes(self, image, detections, groups):
        """Draw color-coded bounding boxes for grouped products"""
        overlay = image.copy()
        
        # Create group statistics
        group_info = []
        for group_id, group in enumerate(groups):
            count = len(group)
            color = self.colors[group_id % len(self.colors)]
            
            avg_conf = np.mean([detections[idx]['confidence'] for idx in group])
            
            group_info.append({
                'id': group_id,
                'count': count,
                'color': color,
                'confidence': avg_conf,
                'items': [detections[idx] for idx in group]
            })
            
            # Draw all boxes in this group with the same color
            for idx in group:
                det = detections[idx]
                x1, y1, x2, y2 = det['bbox']
                
                # Draw filled rectangle with transparency
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Draw border
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"Group {group_id+1}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
                cv2.putText(image, label, (x1+5, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay for transparency effect
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        return group_info
    
    def draw_group_bounding_box(self, image, group_items, color, group_id):
        """Draw a single bounding box around all items in a group"""
        # Find the overall bounding box for the group
        all_x1 = [item['bbox'][0] for item in group_items]
        all_y1 = [item['bbox'][1] for item in group_items]
        all_x2 = [item['bbox'][2] for item in group_items]
        all_y2 = [item['bbox'][3] for item in group_items]
        
        group_x1 = min(all_x1) - 5
        group_y1 = min(all_y1) - 5
        group_x2 = max(all_x2) + 5
        group_y2 = max(all_y2) + 5
        
        # Draw group bounding box
        cv2.rectangle(image, (group_x1, group_y1), (group_x2, group_y2), color, 4)
        
        # Draw label
        label = f"Product Type {group_id+1} ({len(group_items)} items)"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (group_x1, group_y1-text_h-15), (group_x1+text_w+15, group_y1), color, -1)
        cv2.putText(image, label, (group_x1+7, group_y1-7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_statistics_panel(self, image, group_info):
        """Draw statistics panel on image"""
        h, w = image.shape[:2]
        
        # Create semi-transparent panel
        panel_height = min(300, h - 50)
        panel = np.zeros((panel_height, 350, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "PRODUCT ANALYSIS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(panel, (10, 40), (340, 40), (255, 255, 255), 2)
        
        # Group statistics
        y_offset = 70
        cv2.putText(panel, f"Total Product Types: {len(group_info)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        total_items = sum(g['count'] for g in group_info)
        cv2.putText(panel, f"Total Items: {total_items}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 40
        
        # Individual group info
        for i, group in enumerate(sorted(group_info, key=lambda x: x['count'], reverse=True)):
            if y_offset > panel_height - 30:
                break
            
            # Color indicator
            cv2.rectangle(panel, (10, y_offset-15), (30, y_offset-5), group['color'], -1)
            
            # Group info
            text = f"Type {group['id']+1}: {group['count']} items"
            cv2.putText(panel, text, (40, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
        
        # Add panel to image
        overlay = image.copy()
        overlay[25:25+panel_height, w-375:w-25] = panel
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    
    def analyze_image(self, image_path, output_path=None, group_box_mode=True):
        print(f"\n{'='*70}")
        print(f"ANALYZING: {image_path}")
        print(f"{'='*70}\n")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f" Error: Could not load image from {image_path}")
            return
        
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Detect products
        print("Detecting products...")
        detections = self.detect_products(image)
        print(f"Found {len(detections)} products")
        
        if not detections:
            print("⚠ No products detected!")
            return
        
        # Group similar products
        print("Grouping similar products...")
        groups = self.group_similar_products(detections)
        print(f"✓ Identified {len(groups)} different product types")
        
        # Draw results
        result_image = image.copy()
        
        if group_box_mode:
            # Draw single box per group
            for group_id, group in enumerate(groups):
                color = self.colors[group_id % len(self.colors)]
                group_items = [detections[idx] for idx in group]
                self.draw_group_bounding_box(result_image, group_items, color, group_id)
        else:
            # Draw individual boxes with group colors
            group_info = self.draw_grouped_boxes(result_image, detections, groups)
        
        # Create group info for statistics
        group_info = []
        for group_id, group in enumerate(groups):
            group_info.append({
                'id': group_id,
                'count': len(group),
                'color': self.colors[group_id % len(self.colors)],
                'confidence': np.mean([detections[idx]['confidence'] for idx in group])
            })
        
        # Draw statistics
        self.draw_statistics_panel(result_image, group_info)
        
        # Print summary
        print(f"\n{'='*70}")
        print("DETECTION SUMMARY")
        print(f"{'='*70}")
        for i, info in enumerate(sorted(group_info, key=lambda x: x['count'], reverse=True)):
            print(f"Product Type {info['id']+1}: {info['count']} items (Confidence: {info['confidence']:.2%})")
        print(f"{'='*70}\n")
        
        # Save result
        if output_path is None:
            output_path = image_path.replace('.jpg', '_grouped.jpg').replace('.png', '_grouped.png')
        
        cv2.imwrite(output_path, result_image)
        print(f"✓ Result saved to: {output_path}")
        
        # Display
        print("\nDisplaying result (press any key to close)...")
        display_img = cv2.resize(result_image, None, fx=0.8, fy=0.8)
        cv2.imshow("Grouped Product Detection - Press any key to close", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return group_info


def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     GROUPED PRODUCT DETECTION SYSTEM                         ║
║     Similar items in same color, different items in          ║
║     different colors                                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    MODEL_PATH = "runs/train/sku110k_retail_detection/weights/best.pt"
    IMAGE_PATH = "image.png"
    OUTPUT_PATH = "grouped_detection_result.png"
    
    # Similarity threshold: Higher = stricter grouping (only very similar items grouped)
    #                       Lower = looser grouping (more items grouped together)
    SIMILARITY_THRESHOLD = 0.85
    
    # Drawing mode: True = single box per group, False = individual boxes with same color
    GROUP_BOX_MODE = True
    
    try:
        # Initialize detector
        detector = GroupedProductDetector(
            model_path=MODEL_PATH,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        
        # Analyze image
        detector.analyze_image(
            image_path=IMAGE_PATH,
            output_path=OUTPUT_PATH,
            group_box_mode=GROUP_BOX_MODE
        )
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Model exists at specified path")
        print("  2. Image file exists")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
