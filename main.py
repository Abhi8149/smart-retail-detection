from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "9EijMRU6dTfrTPl2Y8tc"
PROJECT_NAME = "sku110k-toc01"
VERSION_NUMBER = 1

TRAIN_CONFIG = {
    'epochs': 10,          
    'imgsz': 512,         
    'batch': 16,           
    'patience': 5,
    'save_period': 5,
    'device': 0,
    'workers': 4,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'augment': True,
}

def main():
    # --- GPU CHECK ---
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU Available: {gpu_name}")
        print(f"  CUDA Version: {torch.version.cuda}")
        TRAIN_CONFIG['device'] = 0
    else:
        print("⚠ No GPU detected - using CPU")
        TRAIN_CONFIG['device'] = 'cpu'

    print("=" * 60 + "\n")

    # --- DOWNLOAD DATASET ---
    print("Downloading dataset (1GB+) if not found...")

    dataset_path = "sku110k-1"
    data_yaml_path = os.path.join(dataset_path, "data.yaml")

    if os.path.exists(data_yaml_path):
        print(f"✓ Dataset already exists at: {dataset_path}")
        dataset_location = dataset_path
    else:
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("sku-fzp1x").project(PROJECT_NAME)
        dataset = project.version(VERSION_NUMBER).download("yolov8")
        dataset_location = dataset.location
        print(f"✓ Dataset downloaded to: {dataset_location}")

    # --- VERIFY YAML ---
    final_yaml_path = os.path.join(dataset_location, "data.yaml")
    if not os.path.exists(final_yaml_path):
        raise FileNotFoundError(
            f"Dataset configuration not found at: {final_yaml_path}\n"
            "Please recheck the dataset folder."
        )

    # --- MODEL SELECTION ---
    print("Loading YOLOv8s model (best for 4070)...")
    model = YOLO("yolov8s.pt")

    # --- TRAINING ---
    print("\nStarting training with optimized GPU parameters...")
    print(f"Using dataset: {final_yaml_path}\n")

    results = model.train(
        data=final_yaml_path,
        epochs=TRAIN_CONFIG['epochs'],
        imgsz=TRAIN_CONFIG['imgsz'],
        batch=TRAIN_CONFIG['batch'],
        patience=TRAIN_CONFIG['patience'],
        save_period=TRAIN_CONFIG['save_period'],
        device=TRAIN_CONFIG['device'],
        workers=TRAIN_CONFIG['workers'],
        optimizer=TRAIN_CONFIG['optimizer'],
        lr0=TRAIN_CONFIG['lr0'],
        augment=TRAIN_CONFIG['augment'],
        project='runs/train',
        name='sku110k_retail_detection',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )

    # --- VALIDATION ---
    print("\nValidating model...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

    # --- EXPORT ---
    best_model_path = "runs/train/sku110k_retail_detection/weights/best.pt"
    print("\nTraining complete!")
    print(f"Best model saved at: {best_model_path}")

    return best_model_path


if __name__ == "__main__":
    main()
