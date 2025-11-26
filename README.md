# Smart Retail Detection System
## Detect Misplaced Items & Out-of-Stock Products

This system uses YOLOv8 + Deep Learning to automatically detect:
- ✅ **Correctly placed items**
- ❌ **Misplaced products** (wrong item in wrong location)
- ⚠️ **Out-of-stock items** (missing products)
- ➕ **Extra items** (unexpected products)

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)
```powershell
python main.py
```
This will:
- Download the SKU-110K retail dataset
- Train a YOLOv8 model for product detection
- Save the trained model to `runs/train/sku110k_retail_detection/weights/best.pt`
- Training takes 2-4 hours depending on your GPU

### Step 3: Run Detection

#### Basic Detection
Place your images in the project folder:
- `shelf_reference.jpg` - The ideal/reference shelf layout
- `shelf_current.jpg` - Current shelf to analyze

Then run:
```powershell
python retail_detection.py
```

#### Batch Processing
To analyze multiple images at once, organize them in a folder and use batch mode (uncomment in `retail_detection.py`):
```python
monitor.batch_analyze(
    reference_image="shelf_reference.jpg",
    current_images_dir="current_images/",
    output_dir="batch_results/"
)
```

---

## 📊 How It Works

### 1. **Object Detection (YOLO)**
   - Detects all products on the shelf with bounding boxes
   - Trained on SKU-110K dataset (11,762 densely packed retail images)

### 2. **Feature Extraction (ResNet50)**
   - Extracts visual features from each detected product
   - Creates a unique "fingerprint" for each item

### 3. **Spatial Matching**
   - Compares reference shelf with current shelf
   - Matches items based on location (spatial proximity)

### 4. **Visual Comparison**
   - Uses cosine similarity to compare product features
   - Detects if the product is different (misplaced)

### 5. **Stock Detection**
   - Identifies gaps where products should be but aren't found
   - Flags out-of-stock situations

---

## 🎯 Detection Categories

| Category | Color | Description |
|----------|-------|-------------|
| **Correctly Placed** | 🟢 Green | Item matches reference at expected location |
| **Misplaced** | 🔴 Red | Wrong product at this location |
| **Out of Stock** | 🟡 Yellow | Expected item is missing |
| **Extra Item** | 🟣 Purple | Item found but not in reference |

---

## ⚙️ Configuration

Edit these parameters in `retail_detection.py`:

### Model Path
```python
MODEL_PATH = "runs/train/sku110k_retail_detection/weights/best.pt"
```

### Detection Sensitivity
```python
SIMILARITY_THRESHOLD = 0.85  # Higher = stricter (0.0-1.0)
# 0.80 = Lenient (allows more variation)
# 0.85 = Balanced (recommended)
# 0.90 = Strict (very precise matching)

SPATIAL_THRESHOLD = 100  # Max pixels apart to match
# 50 = Tight matching (small movements)
# 100 = Balanced (recommended)
# 150 = Loose matching (allows repositioning)
```

---

## 📁 Project Structure

```
actual smart retail/
├── main.py                    # Training script
├── retail_detection.py        # Enhanced detection system
├── misplaced.py              # Original detection script (legacy)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── shelf_reference.jpg       # Reference image (you provide)
├── shelf_current.jpg         # Current image (you provide)
├── runs/
│   └── train/
│       └── sku110k_retail_detection/
│           └── weights/
│               └── best.pt   # Trained model
└── detection_results/        # Analysis results (JSON + images)
```

---

## 📸 Image Requirements

### For Best Results:
1. **Same Angle**: Take reference and current images from the same position
2. **Good Lighting**: Ensure consistent, bright lighting
3. **High Resolution**: Use at least 640x640 pixels
4. **Clear View**: Avoid obstructions or extreme angles
5. **Stable Camera**: Use a tripod or fixed mounting for consistency

### Example Setup:
- Mount camera at eye level, 1-2 meters from shelf
- Use consistent time of day (same lighting)
- Capture entire shelf section in frame
- Take reference photo during ideal stocking

---

## 🔍 Output Files

### 1. Annotated Image
- Visual output with colored bounding boxes
- Labels showing detection status
- Summary overlay with statistics

### 2. JSON Results (`detection_results/analysis_TIMESTAMP.json`)
```json
{
  "timestamp": "2025-11-20T10:30:00",
  "statistics": {
    "total_reference_items": 50,
    "total_current_items": 48,
    "correctly_placed": 45,
    "misplaced": 2,
    "out_of_stock": 3,
    "extra_items": 0
  },
  "details": { ... }
}
```

---

## 🎓 Training Tips

### Model Selection (in main.py)
- `yolov8n.pt` - Fastest, lowest accuracy (good for testing)
- `yolov8s.pt` - Fast, decent accuracy
- `yolov8m.pt` - **Balanced (recommended)** ⭐
- `yolov8l.pt` - Slower, high accuracy
- `yolov8x.pt` - Slowest, highest accuracy

### Training Parameters
```python
TRAIN_CONFIG = {
    'epochs': 100,        # More epochs = better accuracy (but slower)
    'batch': 16,          # Reduce if GPU memory error (try 8 or 4)
    'patience': 20,       # Early stopping if no improvement
    'imgsz': 640,         # Image size (640 is standard)
}
```

### GPU vs CPU
- **With GPU**: Training takes 2-4 hours
- **Without GPU**: Training takes 12-24 hours
- Detection runs fast on both (GPU preferred)

---

## 🐛 Troubleshooting

### Issue: "Could not load model"
**Solution**: Train the model first using `python main.py`

### Issue: "Could not load image"
**Solution**: Check image paths are correct and files exist

### Issue: GPU Memory Error
**Solution**: Reduce batch size in `main.py`:
```python
'batch': 8,  # or even 4
```

### Issue: Poor Detection Accuracy
**Solutions**:
1. Ensure images are taken from same angle
2. Improve lighting consistency
3. Train for more epochs (increase to 150-200)
4. Use larger model (yolov8l.pt)
5. Lower similarity threshold to 0.80

### Issue: Too Many False Positives
**Solutions**:
1. Increase similarity threshold to 0.90
2. Decrease spatial threshold to 50
3. Improve image quality

---

## 📝 Usage Examples

### Example 1: Single Image Analysis
```python
from retail_detection import EnhancedRetailMonitor

monitor = EnhancedRetailMonitor("best.pt")
results = monitor.compare_shelves(
    "reference.jpg", 
    "current.jpg",
    "output.jpg"
)
print(f"Misplaced: {results['statistics']['misplaced']}")
print(f"Out of Stock: {results['statistics']['out_of_stock']}")
```

### Example 2: Batch Analysis
```python
monitor = EnhancedRetailMonitor("best.pt")
results = monitor.batch_analyze(
    reference_image="reference.jpg",
    current_images_dir="daily_captures/",
    output_dir="daily_analysis/"
)
```

### Example 3: Custom Thresholds
```python
# Stricter detection
monitor = EnhancedRetailMonitor(
    "best.pt",
    similarity_threshold=0.90,  # Very strict
    spatial_threshold=50         # Must be close
)
```

---

## 🎯 Use Cases

1. **Daily Store Audits**: Automate shelf compliance checking
2. **Inventory Management**: Real-time out-of-stock detection
3. **Planogram Compliance**: Verify products match layout plan
4. **Loss Prevention**: Detect unauthorized product placement
5. **Quality Control**: Ensure proper product organization

---

## 🔄 Workflow

```
┌─────────────────┐
│ Train Model     │ <- First time only (main.py)
│ (2-4 hours)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Capture Images  │ <- Take reference & current photos
│                 │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Run Detection   │ <- retail_detection.py
│ (few seconds)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Review Results  │ <- Check annotated images & JSON
│                 │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Take Action     │ <- Restock/reorganize as needed
│                 │
└─────────────────┘
```

---

## 📞 Support

If you encounter issues:
1. Check image paths and file existence
2. Verify model is trained and exists
3. Review console output for error messages
4. Check GPU/CPU compatibility
5. Ensure all dependencies are installed

---

## 🚀 Next Steps

1. ✅ Install dependencies
2. ✅ Train the model
3. ✅ Capture reference shelf image
4. ✅ Capture current shelf images
5. ✅ Run detection
6. ✅ Analyze results
7. ✅ Integrate into workflow

Good luck with your smart retail detection system! 🛒📦
