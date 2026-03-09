"""
Test: Class Mapping Verification
=================================
Comprehensive verification that all class IDs map to correct fruit names.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.detection.yolo_detector import YOLODetector
import torch

def test_class_mapping():
    """Test that class mapping matches YOLOv11s training configuration"""
    print("🧪 Testing Class Mapping Verification...")
    print("=" * 70)
    
    try:
        # Initialize detector
        detector = YOLODetector(
            model_path="models_cache/yolo_detection_best.pt",
            device=torch.device("cpu")  # Use CPU for faster init in tests
        )
        
        # Expected mapping from fruit-detection.v2i.yolov11 dataset
        expected_mapping = {
            0: "grapefruit",
            1: "guava",
            2: "mango",
            3: "orange"
        }
        
        print("\n📋 Verifying Class Mapping:")
        print("-" * 70)
        print(f"{'Class ID':<12} {'Expected':<15} {'Actual':<15} {'Status':<10}")
        print("-" * 70)
        
        all_correct = True
        for class_id, expected_name in expected_mapping.items():
            actual_name = detector.class_names[class_id]
            status = "✅ PASS" if actual_name == expected_name else "❌ FAIL"
            
            if actual_name != expected_name:
                all_correct = False
            
            print(f"{class_id:<12} {expected_name:<15} {actual_name:<15} {status:<10}")
        
        print("-" * 70)
        
        # Additional validation
        assert len(detector.class_names) == 4, f"Expected 4 classes, got {len(detector.class_names)}"
        print(f"\n✅ Total classes: {len(detector.class_names)} (correct)")
        
        # Verify no unknown classes
        valid_fruits = ["grapefruit", "guava", "mango", "orange"]
        for class_name in detector.class_names.values():
            assert class_name in valid_fruits, f"Invalid fruit class: {class_name}"
        print(f"✅ All class names are valid fruits")
        
        if all_correct:
            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED - Class mapping is correct!")
            print("=" * 70)
            print("\n💡 Migration successful! Old → New mapping:")
            print("   0: mango → grapefruit")
            print("   1: orange → guava")
            print("   2: guava → mango")
            print("   3: grapefruit → orange")
            return True
        else:
            print("\n❌ TESTS FAILED - Class mapping is incorrect!")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_class_mapping()
    sys.exit(0 if success else 1)
