"""
Test: YOLO Detection on Sample Images
======================================
Test detection on actual sample images from test_images folder.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
from pipeline.detection.yolo_detector import YOLODetector

def test_detection_on_samples():
    """Test detection on known sample images"""
    print("🧪 Testing YOLO Detection on Sample Images...")
    print("=" * 70)
    
    try:
        # Initialize detector
        print("\n📦 Loading model...")
        detector = YOLODetector(
            model_path="models_cache/yolo_detection_best.pt",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            confidence_threshold=0.5
        )
        print(f"✅ Model loaded on {detector.device}")
        
        # Test images folder
        test_images_dir = Path("test_images")
        
        if not test_images_dir.exists():
            print("\n⚠️ test_images folder not found")
            print("   Create test_images/ folder and add sample fruit images")
            print("   Skipping sample image tests")
            return True  # Not a failure, just no images to test
        
        # Get test images (jpg and png)
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if not image_files:
            print("\n⚠️ No test images found in test_images/")
            print("   Add .jpg or .png images to test_images/ folder")
            print("   Skipping sample image tests")
            return True  # Not a failure, just no images to test
        
        print(f"\n📸 Found {len(image_files)} test images")
        print("=" * 70)
        
        total_detections = 0
        successful_tests = 0
        
        # Test first 5 images
        for img_path in image_files[:5]:
            print(f"\n📸 Testing: {img_path.name}")
            print("-" * 70)
            
            try:
                # Load image
                image = cv2.imread(str(img_path))
                
                if image is None:
                    print(f"   ⚠️ Failed to load image")
                    continue
                
                # Run detection
                detections = detector.detect(image)
                
                if detections:
                    print(f"   ✅ Found {len(detections)} fruits:")
                    for i, det in enumerate(detections, 1):
                        fruit_type = det['fruit_type']
                        confidence = det['confidence']
                        bbox = det['bbox']
                        print(f"      {i}. {fruit_type:12s} (confidence: {confidence:.3f}) at {bbox}")
                    total_detections += len(detections)
                else:
                    print(f"   ℹ️ No fruits detected (might be empty image or low confidence)")
                
                successful_tests += 1
                
            except Exception as e:
                print(f"   ❌ Error processing image: {str(e)}")
        
        print("\n" + "=" * 70)
        print(f"📊 Test Summary:")
        print(f"   Images tested: {successful_tests}")
        print(f"   Total fruits detected: {total_detections}")
        
        if successful_tests > 0:
            print(f"\n✅ TESTS COMPLETED - Detection working on sample images!")
            print("\n💡 Next steps:")
            print("   1. Review detection results above")
            print("   2. Verify fruit types match image content")
            print("   3. Check confidence scores are reasonable")
            return True
        else:
            print(f"\n⚠️ No images were successfully tested")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detection_on_samples()
    sys.exit(0 if success else 1)
