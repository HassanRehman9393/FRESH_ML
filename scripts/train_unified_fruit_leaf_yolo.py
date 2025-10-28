"""
Train Unified YOLO Model with Fruits + Leaves
==============================================

Trains YOLO model to detect:
- 4 fruit types: mango, orange, guava, grapefruit  
- 1 object type: leaf

This enables separate analytics for fruits vs leaves in drone imagery.

Usage:
    python train_unified_fruit_leaf_yolo.py

Requirements:
    - Unified dataset in data/unified_fruit_leaf/
    - Structure:
        train/images/*.jpg
        train/labels/*.txt (YOLO format)
        val/images/*.jpg
        val/labels/*.txt
        test/images/*.jpg
        test/labels/*.txt
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

# Configuration
CONFIG = {
    'model': 'yolov8m.pt',  # Medium model (good balance of speed vs accuracy)
    'data': 'data/unified_fruit_leaf/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'workers': 4,
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'name': 'unified_fruit_leaf_detection',
    'patience': 20,  # Early stopping patience
    'save': True,
    'save_period': 10,  # Save checkpoint every 10 epochs
    'cache': True,  # Cache images for faster training
    'amp': True,  # Automatic Mixed Precision
    'augment': True,  # Use data augmentation
}

# Data YAML content
DATA_YAML = """
# Unified Fruit + Leaf Detection Dataset
# ========================================

# Dataset root directory
path: ../data/unified_fruit_leaf

# Splits
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 5

# Class names
names:
  0: mango
  1: orange
  2: guava
  3: grapefruit
  4: leaf

# Dataset info
download: false  # Manual dataset preparation required
"""

def create_data_yaml():
    """Create data.yaml configuration file"""
    data_yaml_path = Path('data/unified_fruit_leaf/data.yaml')
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(DATA_YAML)
    print(f"✅ Created data.yaml at: {data_yaml_path}")
    return data_yaml_path

def validate_dataset(data_yaml_path):
    """Validate dataset structure before training"""
    base_path = data_yaml_path.parent
    
    # Check required directories
    required_dirs = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    print("\n" + "="*70)
    print("📁 DATASET VALIDATION")
    print("="*70)
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        
        if exists:
            # Count files
            if 'images' in dir_path:
                count = len(list(full_path.glob('*.jpg'))) + len(list(full_path.glob('*.png')))
            else:  # labels
                count = len(list(full_path.glob('*.txt')))
            print(f"{status} {dir_path}: {count} files")
        else:
            print(f"{status} {dir_path}: NOT FOUND")
            all_exist = False
    
    print("="*70)
    
    if not all_exist:
        raise FileNotFoundError(
            "Dataset directories not found! Please prepare the dataset first.\n"
            "Expected structure:\n"
            "  data/unified_fruit_leaf/\n"
            "    ├── train/images/*.jpg\n"
            "    ├── train/labels/*.txt\n"
            "    ├── val/images/*.jpg\n"
            "    ├── val/labels/*.txt\n"
            "    ├── test/images/*.jpg\n"
            "    └── test/labels/*.txt\n"
        )
    
    print("✅ Dataset validation passed!\n")

def train_unified_yolo():
    """Train unified YOLO model"""
    print("\n" + "="*70)
    print("🚀 UNIFIED FRUIT + LEAF DETECTION TRAINING")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Model: {CONFIG['model']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch']}")
    print(f"Image size: {CONFIG['imgsz']}")
    print("="*70 + "\n")
    
    # Create data.yaml
    data_yaml_path = create_data_yaml()
    
    # Validate dataset
    validate_dataset(data_yaml_path)
    
    # Initialize YOLO
    print("📦 Loading YOLO model...")
    model = YOLO(CONFIG['model'])
    print("✅ Model loaded!\n")
    
    # Train
    print("🏋️ Starting training...")
    print("-" * 70)
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        workers=CONFIG['workers'],
        device=CONFIG['device'],
        name=CONFIG['name'],
        patience=CONFIG['patience'],
        save=CONFIG['save'],
        save_period=CONFIG['save_period'],
        cache=CONFIG['cache'],
        amp=CONFIG['amp'],
        augment=CONFIG['augment'],
        verbose=True,
        
        # Augmentation parameters
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10,   # Rotation
        translate=0.1,  # Translation
        scale=0.5,    # Scale
        shear=0.0,    # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,   # Flip up-down
        fliplr=0.5,   # Flip left-right
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.0,    # Mixup augmentation
    )
    
    # Validate
    print("\n" + "="*70)
    print("🧪 VALIDATING TRAINED MODEL")
    print("="*70)
    metrics = model.val()
    
    # Print results
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Results:")
    print(f"   mAP@50:    {metrics.box.map50:.3f}")
    print(f"   mAP@50-95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall:    {metrics.box.mr:.3f}")
    print(f"\n💾 Best model: {model.trainer.best}")
    print(f"📁 Output directory: runs/detect/{CONFIG['name']}")
    print("="*70)
    
    # Export model (optional)
    print("\n💾 Exporting model to ONNX format...")
    try:
        model.export(format='onnx')
        print("✅ Model exported successfully!")
    except Exception as e:
        print(f"⚠️ Export failed: {e}")
    
    return model, metrics

def print_usage_instructions():
    """Print instructions for using the trained model"""
    print("\n" + "="*70)
    print("📚 NEXT STEPS")
    print("="*70)
    print("\n1. Copy best model to pipeline:")
    print(f"   cp runs/detect/{CONFIG['name']}/weights/best.pt models/yolo_unified_fruit_leaf_best.pt")
    
    print("\n2. Update pipeline configuration:")
    print("   - Edit pipeline/pipeline_config.py")
    print("   - Update YOLO_MODEL_PATH to point to new model")
    print("   - Update class names: ['mango', 'orange', 'guava', 'grapefruit', 'leaf']")
    
    print("\n3. Update predictor.py:")
    print("   - Add leaf detection handling")
    print("   - Route leaves to disease detection (skip classification)")
    print("   - Update analytics to show separate fruit/leaf counts")
    
    print("\n4. Test the updated pipeline:")
    print("   python -m pipeline.predictor --image test_image.jpg")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        model, metrics = train_unified_yolo()
        print_usage_instructions()
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 TIP: Prepare your dataset first using:")
        print("   - Annotate leaf images with bounding boxes")
        print("   - Combine with existing fruit dataset")
        print("   - Use LabelImg, CVAT, or Roboflow for annotation")
        print("   - Expected format: YOLO (class_id x_center y_center width height)")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        raise
