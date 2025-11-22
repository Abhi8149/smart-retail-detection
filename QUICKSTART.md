# 🚀 QUICK START GUIDE

## 30-Second Setup

### 1️⃣ Install Everything
```powershell
pip install -r requirements.txt
```

### 2️⃣ Validate System
```powershell
python validate.py
```

### 3️⃣ Train Model (2-4 hours, one-time only)
```powershell
python main.py
```

### 4️⃣ Run Detection
```powershell
python retail_detection.py
```

---

## 📁 What You Need

### Required Files:
- ✅ `shelf_reference.jpg` - Your ideal shelf layout
- ✅ `shelf_current.jpg` - Current shelf to check

### How to Get Them:
**Option A:** Use your own images (same angle, good lighting)
**Option B:** Capture with webcam using demo script:
```powershell
python demo.py
```

---

## 🎯 Expected Output

### Console Output:
```
RETAIL SHELF ANALYSIS
==================================================
Reference Items: 50
Current Items:   48

✓ Correctly Placed: 45
✗ Misplaced:        2
⚠ Out of Stock:     3
+ Extra Items:      0

Shelf Accuracy: 90.0%
⚠ ALERT: Immediate restocking required!
```

### Visual Output:
- **Green boxes** → Correctly placed items ✅
- **Red boxes** → Misplaced items ❌
- **Yellow boxes** → Out of stock ⚠️
- **Purple boxes** → Extra items ➕

### Files Created:
- `shelf_analysis_result.jpg` - Annotated image
- `detection_results/analysis_TIMESTAMP.json` - Detailed results

---

## ⚙️ Adjust Settings (Optional)

Edit `retail_detection.py`:

```python
# Stricter matching (fewer false positives)
SIMILARITY_THRESHOLD = 0.90
SPATIAL_THRESHOLD = 50

# Lenient matching (catch more issues)
SIMILARITY_THRESHOLD = 0.80
SPATIAL_THRESHOLD = 150
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not load model" | Run `python main.py` to train |
| "Could not load image" | Check file paths and names |
| GPU memory error | Reduce batch size in main.py to 8 or 4 |
| Poor accuracy | Use same camera angle, better lighting |

---

## 📞 Need Help?

Run validation to diagnose issues:
```powershell
python validate.py
```

---

## 🎓 Learning Path

1. **Day 1**: Setup + Training
   - Install dependencies
   - Train model (let it run overnight)

2. **Day 2**: Testing
   - Capture test images
   - Run detection
   - Tune parameters

3. **Day 3**: Production
   - Integrate into workflow
   - Set up batch processing
   - Monitor results

---

## 💡 Pro Tips

1. **Consistent Camera Position**: Mount camera in fixed position
2. **Good Lighting**: Avoid shadows and glare
3. **Regular Updates**: Retake reference when layout changes
4. **Batch Processing**: Process multiple stores/shelves at once
5. **Log Results**: Track trends over time with JSON outputs

---

**Ready?** Start with: `python validate.py`
