"""
YOLO Detection Test Script
=========================

Test script to verify the YOLO detection pipeline works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import torch
from pipeline.detection.yolo_detector import YOLODetector
from pathlib import Path

def test_yolo_detector():
    """Test the YOLO detector with a sample image"""
    
    # Configuration
    model_path = "models/yolo_detection_best.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing YOLO Detector on device: {device}")
    print(f"Model path: {model_path}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Initialize detector
        print("\n🔄 Initializing YOLO detector...")
        detector = YOLODetector(
            model_path=model_path,
            device=device,
            confidence_threshold=0.5,
            iou_threshold=0.45
        )
        
        # Print model info
        print("\n📋 Model Information:")
        model_info = detector.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Test with a dummy image if no real image available
        print("\n🖼️  Testing with dummy image...")
        dummy_image = cv2.imread("download.png") if Path("download.png").exists() else None
        
        if dummy_image is None:
            # Create a dummy image for testing
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            print("  Using randomly generated test image")
        else:
            print("  Using existing download.png image")
        
        # Run detection
        print("\n🔍 Running detection...")
        detections = detector.detect(dummy_image)
        
        print(f"\n✅ Detection completed successfully!")
        print(f"📊 Results: {len(detections)} fruits detected")
        
        for i, detection in enumerate(detections):
            print(f"  Fruit {i+1}:")
            print(f"    Type: {detection['fruit_type']}")
            print(f"    Confidence: {detection['confidence']:.3f}")
            print(f"    Bounding Box: {detection['bbox']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting YOLO Detection Test")
    print("=" * 50)
    
    success = test_yolo_detector()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ YOLO Detection Pipeline Test PASSED!")
    else:
        print("❌ YOLO Detection Pipeline Test FAILED!")