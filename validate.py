"""
Utility script for testing and validating the retail detection system
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n=== CHECKING DEPENDENCIES ===\n")
    
    dependencies = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'scipy': 'scipy',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'roboflow': 'roboflow',
        'ultralytics': 'ultralytics'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_gpu():
    """Check if GPU is available for PyTorch"""
    print("\n=== CHECKING GPU AVAILABILITY ===\n")
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"✓ GPU Available: {device}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        
        # Test GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU Memory: {total_memory:.2f} GB")
            
            if total_memory < 4:
                print("  ⚠ Warning: Less than 4GB GPU memory. Training may be slow.")
                print("  Recommendation: Reduce batch size to 4-8")
        except:
            pass
        
        return True
    else:
        print("✗ No GPU detected - using CPU")
        print("  Training will be much slower (12-24 hours)")
        print("  Recommendation: Use Google Colab or cloud GPU")
        return False


def check_model():
    """Check if trained model exists"""
    print("\n=== CHECKING MODEL ===\n")
    
    model_path = Path("runs/train/sku110k_retail_detection/weights/best.pt")
    
    if model_path.exists():
        print(f"✓ Trained model found: {model_path}")
        
        # Try to load model
        try:
            model = YOLO(str(model_path))
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    else:
        print(f"✗ Trained model not found: {model_path}")
        print("\n  You need to train the model first:")
        print("  python main.py")
        return False


def check_images():
    """Check if sample images exist"""
    print("\n=== CHECKING SAMPLE IMAGES ===\n")
    
    ref_img = Path("shelf_reference.jpg")
    curr_img = Path("shelf_current.jpg")
    
    status = True
    
    if ref_img.exists():
        img = cv2.imread(str(ref_img))
        if img is not None:
            h, w = img.shape[:2]
            print(f"✓ Reference image found: {w}x{h} pixels")
            if w < 640 or h < 640:
                print(f"  ⚠ Warning: Image resolution is low. Recommend at least 640x640")
        else:
            print(f"✗ Reference image exists but cannot be read")
            status = False
    else:
        print(f"✗ Reference image not found: {ref_img}")
        status = False
    
    if curr_img.exists():
        img = cv2.imread(str(curr_img))
        if img is not None:
            h, w = img.shape[:2]
            print(f"✓ Current image found: {w}x{h} pixels")
            if w < 640 or h < 640:
                print(f"  ⚠ Warning: Image resolution is low. Recommend at least 640x640")
        else:
            print(f"✗ Current image exists but cannot be read")
            status = False
    else:
        print(f"✗ Current image not found: {curr_img}")
        status = False
    
    if not status:
        print("\n  Add your images or use demo.py to capture them")
    
    return status


def test_detection():
    """Run a quick detection test"""
    print("\n=== TESTING DETECTION ===\n")
    
    # Check if images exist
    if not Path("shelf_reference.jpg").exists():
        print("✗ No test images available")
        print("  Run demo.py to capture test images first")
        return False
    
    try:
        print("Loading model...")
        model_path = "runs/train/sku110k_retail_detection/weights/best.pt"
        if not Path(model_path).exists():
            model_path = "yolov8n.pt"
            print(f"  Using pretrained model: {model_path}")
        
        model = YOLO(model_path)
        
        print("Running detection on sample image...")
        img = cv2.imread("shelf_reference.jpg")
        results = model(img, verbose=False)
        
        num_detections = len(results[0].boxes)
        print(f"✓ Detection successful: {num_detections} objects detected")
        
        if num_detections == 0:
            print("  ⚠ Warning: No objects detected. Model may need training.")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("RETAIL DETECTION SYSTEM - VALIDATION REPORT")
    print("="*60)
    
    results = {
        'dependencies': check_dependencies(),
        'gpu': check_gpu(),
        'model': check_model(),
        'images': check_images(),
    }
    
    # Run detection test if all checks pass
    if all([results['dependencies'], results['images']]):
        results['detection'] = test_detection()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL SYSTEMS READY!")
        print("\nYou can now run:")
        print("  - python main.py (to train the model)")
        print("  - python retail_detection.py (to run detection)")
        print("  - python demo.py (for interactive demo)")
    else:
        print("\n⚠ SOME ISSUES DETECTED")
        print("\nPlease fix the issues above before proceeding.")
        
        # Provide specific recommendations
        if not results['dependencies']:
            print("\n1. Install missing dependencies:")
            print("   pip install -r requirements.txt")
        
        if not results.get('gpu', True):
            print("\n2. GPU not available - training will be slower")
            print("   Consider using cloud GPU (Google Colab, AWS, etc.)")
        
        if not results.get('model', True):
            print("\n3. Train the model:")
            print("   python main.py")
        
        if not results.get('images', True):
            print("\n4. Add sample images or capture them:")
            print("   python demo.py")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main validation function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "deps":
            check_dependencies()
        elif command == "gpu":
            check_gpu()
        elif command == "model":
            check_model()
        elif command == "images":
            check_images()
        elif command == "test":
            test_detection()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  python validate.py deps    - Check dependencies")
            print("  python validate.py gpu     - Check GPU availability")
            print("  python validate.py model   - Check model")
            print("  python validate.py images  - Check sample images")
            print("  python validate.py test    - Run detection test")
            print("  python validate.py         - Run full validation")
    else:
        generate_test_report()


if __name__ == '__main__':
    main()
