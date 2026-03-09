"""
Test: YOLO Model Loading
=========================
Verify that the new YOLOv11s model loads correctly with proper class mapping.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.detection.yolo_detector import YOLODetector
import torch

def test_model_loads():
    """Test that the YOLOv11s model loads successfully"""
    print("🧪 Testing YOLO Model Loading...")
    print("=" * 70)
    
    try:
        # Initialize detector
        detector = YOLODetector(
            model_path="models_cache/yolo_detection_best.pt",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Verify model loaded
        assert detector.model is not None, "Model failed to load"
        print("✅ Model object created successfully")
        
        # Verify class mapping for new YOLOv11s model
        assert detector.class_names[0] == "grapefruit", f"Class 0 should be 'grapefruit', got '{detector.class_names[0]}'"
        assert detector.class_names[1] == "guava", f"Class 1 should be 'guava', got '{detector.class_names[1]}'"
        assert detector.class_names[2] == "mango", f"Class 2 should be 'mango', got '{detector.class_names[2]}'"
        assert detector.class_names[3] == "orange", f"Class 3 should be 'orange', got '{detector.class_names[3]}'"
        print("✅ Class mapping verified correctly")
        
        # Get model info
        info = detector.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Model Path: {info['model_path']}")
        print(f"   Device: {info['device']}")
        print(f"   Classes: {info['class_names']}")
        print(f"   Confidence Threshold: {info['confidence_threshold']}")
        print(f"   IoU Threshold: {info['iou_threshold']}")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - Model loaded successfully with correct mapping!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loads()
    sys.exit(0 if success else 1)
