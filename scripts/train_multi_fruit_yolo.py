"""
FRESH ML - Multi-Class YOLO v8 Training Script
Train single model for 4-class fruit detection: mango, grapefruit, guava, orange
Local training version - no Google Drive required
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def check_system():
    """Check system and GPU availability"""
    print("🔍 System Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detected: {device_name}")
        print(f"💾 GPU Memory: {memory_gb:.2f} GB")
        return True
    else:
        print("⚠️  No GPU detected, using CPU (training will be slower)")
        return False

def verify_dataset():
    """Verify dataset structure"""
    dataset_path = Path('data/unified/multi_fruit_detection')
    
    print(f"Dataset path: {dataset_path}")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        print("Please ensure multi_fruit_detection folder exists in data/unified/")
        return False, None
    
    print("✅ Dataset directory found!")
    
    # Check required directories
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    all_exist = all((dataset_path / d).exists() for d in required_dirs)
    
    if all_exist:
        print("✅ All required directories present!")
    else:
        print("⚠️ Some required directories are missing!")
        for d in required_dirs:
            status = "✅" if (dataset_path / d).exists() else "❌"
            print(f"{status} {d}")
        return False, None
    
    return True, dataset_path

def create_yaml_config(dataset_path):
    """Create data.yaml configuration"""
    yaml_content = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['mango', 'grapefruit', 'guava', 'orange']
    }
    
    # Save yaml file
    yaml_file = dataset_path / 'data.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("✅ Created data.yaml configuration")
    print(f"Classes: {yaml_content['nc']}")
    print(f"Names: {yaml_content['names']}")
    
    return yaml_file

def train_multi_fruit_yolo():
    """Train YOLO v8 model for multi-fruit detection"""
    
    print("🍎 FRESH ML - Multi-Class YOLO Training (Local)")
    print("=" * 50)
    
    # Check system
    has_gpu = check_system()
    
    # Verify dataset
    dataset_ok, dataset_path = verify_dataset()
    if not dataset_ok:
        return None, None
    
    # Create YAML configuration
    yaml_file = create_yaml_config(dataset_path)
    
    # Initialize YOLO model
    print("\n📦 Loading YOLO v8 model...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8 Nano model loaded successfully!")
    
    # Start training
    print("\n🚀 Starting YOLO training...")
    
    try:
        # Train the model with same parameters as notebook
        results = model.train(
            data=str(yaml_file),
            epochs=100,
            imgsz=640,
            batch=16 if has_gpu else 8,
            name='multi_fruit_detection_v1',
            patience=10,
            save_period=10,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=True,
            plots=True
        )
        
        print("🎉 Training completed!")
        print("📁 Results saved in: runs/detect/multi_fruit_detection_v1/")
        
        # Load and evaluate best model
        best_model_path = 'runs/detect/multi_fruit_detection_v1/weights/best.pt'
        
        if os.path.exists(best_model_path):
            print("\n� Evaluating trained model...")
            best_model = YOLO(best_model_path)
            
            # Evaluate on validation set
            metrics = best_model.val()
            
            print("\n📊 MODEL PERFORMANCE METRICS")
            print("=" * 60)
            print(f"mAP@0.5:        {metrics.box.map50:.4f}")
            print(f"mAP@0.5-0.95:   {metrics.box.map:.4f}")
            print(f"Precision:      {metrics.box.mp:.4f}")
            print(f"Recall:         {metrics.box.mr:.4f}")
            print("=" * 60)
            
            # Performance assessment
            print("\n🎯 TRAINING QUALITY ASSESSMENT:")
            if metrics.box.map50 > 0.85 and metrics.box.map > 0.65:
                print("✅ Model is WELL-TRAINED and ready for deployment!")
            elif metrics.box.map50 > 0.70:
                print("⚠️ Model shows GOOD performance but could be improved")
            else:
                print("❌ Model needs MORE TRAINING or data improvement")
            
            return best_model_path, metrics
        else:
            print("❌ Best model weights not found!")
            return None, None
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    model_path, results = train_multi_fruit_yolo()
    
    if model_path:
        print(f"\n🎯 Next Steps:")
        print(f"   1. Check training results in 'runs/detect/multi_fruit_detection_v1'")
        print(f"   2. Use model for inference: YOLO('{model_path}')")
        print(f"   3. Validate performance meets >95% mAP50 target")
    else:
        print(f"\n❌ Training failed. Check error messages above.")