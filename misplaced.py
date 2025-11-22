import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from PIL import Image

class RetailMonitor:
    def __init__(self, model_path):
        # 1. Load the YOLO model we trained in the previous step
        self.detector = YOLO(model_path)
        
        # 2. Load a Feature Extractor (ResNet18)
        # This "brain" looks at an item and turns it into a mathematical vector (embedding).
        # If two items are different (e.g., Coke vs Pepsi), their vectors will be different.
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.eval() # Set to evaluation mode
        
        # Preprocessing for ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image_crop):
        """Turns a cropped image of a product into a feature vector."""
        # Convert OpenCV BGR to PIL RGB
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            feature_vector = self.feature_extractor(input_tensor)
        
        return feature_vector.flatten().numpy()

    def detect_and_extract(self, image):
        """Runs YOLO detection and extracts features for every item found."""
        results = self.detector(image, verbose=False)
        detections = []
        
        h, w, _ = image.shape
        
        for box in results[0].boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the detected product
            # Ensure we don't crop outside image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            product_crop = image[y1:y2, x1:x2]
            
            if product_crop.size == 0: continue

            # Get visual features
            embedding = self.get_embedding(product_crop)
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1+x2)/2, (y1+y2)/2),
                "embedding": embedding
            })
            
        return detections

    def compare_shelves(self, ref_image_path, curr_image_path):
        print("Loading images...")
        ref_img = cv2.imread(ref_image_path)
        curr_img = cv2.imread(curr_image_path)

        print("Detecting items on Reference Shelf...")
        ref_items = self.detect_and_extract(ref_img)
        
        print("Detecting items on Current Shelf...")
        curr_items = self.detect_and_extract(curr_img)

        misplaced_count = 0
        missing_count = 0
        
        # --- COMPARISON LOGIC ---
        # We try to match every item in the Reference image to an item in the Current image
        # based on physical location.
        
        for ref_item in ref_items:
            matched_item = None
            min_dist = float('inf')
            
            rx, ry = ref_item['center']
            
            # Find the closest item in the current image (Spatial Matching)
            for curr in curr_items:
                cx, cy = curr['center']
                # Euclidean distance between centers
                pixel_dist = np.sqrt((rx - cx)**2 + (ry - cy)**2)
                
                # Threshold: items must be within 50 pixels to be considered the "same slot"
                if pixel_dist < 50 and pixel_dist < min_dist:
                    min_dist = pixel_dist
                    matched_item = curr
            
            if matched_item:
                # If we found an item in the same spot, check if it looks the same
                similarity = 1 - cosine(ref_item['embedding'], matched_item['embedding'])
                
                # Similarity Threshold (0.0 to 1.0). 
                # If < 0.85, it's likely a different product (Misplaced).
                if similarity < 0.85:
                    misplaced_count += 1
                    x1, y1, x2, y2 = matched_item['bbox']
                    # Draw RED box for Misplaced
                    cv2.rectangle(curr_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(curr_img, "MISPLACED", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw GREEN box for Correct
                    x1, y1, x2, y2 = matched_item['bbox']
                    cv2.rectangle(curr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # If no item is found in that spot, it is Missing (Out of Stock)
                missing_count += 1
                x1, y1, x2, y2 = ref_item['bbox']
                # Draw YELLOW dashed box on the current image to show where it SHOULD be
                cv2.rectangle(curr_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(curr_img, "MISSING", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show result
        print(f"Analysis Complete: {misplaced_count} Misplaced, {missing_count} Missing")
        cv2.imshow("Shelf Analysis", curr_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- RUN THE MONITOR ---
    # 1. Use the trained model from the previous script
    # Update this path after training (e.g., runs/detect/sku110k_retail_model/weights/best.pt)
MODEL_PATH = "yolov8n.pt" 
    
monitor = RetailMonitor(MODEL_PATH)
    
    # 2. Provide two images (Reference and Current)
    # Ensure these images are taken from the same angle!
monitor.compare_shelves("shelf_reference.jpg", "shelf_current.jpg")