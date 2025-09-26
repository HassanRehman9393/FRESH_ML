"""
FRESH ML - Multi-Class YOLO v8 Training Script
Train single model for 4-class fruit detection: mango, grapefruit, guava, orange
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import time

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"🚀 GPU detected: {device}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  No GPU detected, using CPU (training will be slower)")
        return False

def train_multi_fruit_yolo():
    """Train YOLO v8 model for multi-fruit detection"""
    
    print("🍎 FRESH ML - Multi-Class YOLO Training")
    print("=" * 50)
    
    # Check GPU
    has_gpu = check_gpu()
    device = 'cuda' if has_gpu else 'cpu'
    
    # Load model
    print("\n📦 Loading YOLO v8 model...")
    model = YOLO('yolov8n.pt')  # Start with nano model for speed
    print(f"✅ Model loaded on {device}")
    
    # Training configuration
    config_path = "data/unified/multi_fruit_detection/data.yaml"
    
    # Verify config exists
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\n📋 Training Configuration:")
    print(f"   Dataset: {config_path}")
    print(f"   Classes: {config['nc']} ({', '.join(config['names'])})")
    print(f"   Train images: ~12,967")
    print(f"   Val images: ~2,779")
    print(f"   Test images: ~2,781")
    
    # Training parameters
    training_params = {
        'data': config_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16 if has_gpu else 8,  # Adjust batch size based on hardware
        'name': 'multi_fruit_detection_v1',
        'patience': 15,  # Early stopping
        'save_period': 10,  # Save checkpoint every 10 epochs
        'device': device,
        'workers': 4,
        'project': 'runs/detect',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'box': 7.5,  # Box loss weight
        'cls': 0.5,   # Classification loss weight
        'dfl': 1.5,   # DFL loss weight
    }
    
    print(f"\n🔧 Training Parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"⏰ Estimated time: {2-4 if has_gpu else 8-12} hours")
    print("📊 Monitor progress in 'runs/detect/multi_fruit_detection_v1'")
    
    start_time = time.time()
    
    try:
        # Train the model
        results = model.train(**training_params)
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"⏰ Training time: {training_time/3600:.2f} hours")
        
        # Validation results
        print(f"\n📊 Final Results:")
        print(f"   mAP50: {results.box.map50:.4f}")
        print(f"   mAP50-95: {results.box.map:.4f}")
        
        # Save model info
        model_path = f"runs/detect/multi_fruit_detection_v1/weights/best.pt"
        print(f"💾 Best model saved: {model_path}")
        
        # Test inference
        print(f"\n🧪 Testing model inference...")
        test_model = YOLO(model_path)
        
        # Quick validation
        val_results = test_model.val(data=config_path)
        print(f"✅ Validation mAP50: {val_results.box.map50:.4f}")
        
        return model_path, val_results
        
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