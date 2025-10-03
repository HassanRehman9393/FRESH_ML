"""
Classification Test Script
==========================

Test script to verify the ripeness classification pipeline works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from PIL import Image
from pipeline.classification.ripeness_classifier import RipenessClassifier
from pathlib import Path

def test_classification():
    """Test the ripeness classifier with a sample image"""
    
    # Configuration
    model_path = "models/classification_best.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing Ripeness Classifier on device: {device}")
    print(f"Model path: {model_path}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Initialize classifier
        print("\n🔄 Initializing ripeness classifier...")
        classifier = RipenessClassifier(
            model_path=model_path,
            device=device
        )
        
        # Print model info
        print("\n📋 Model Information:")
        model_info = classifier.get_model_info()
        for key, value in model_info.items():
            if key != 'class_names' and key != 'ripeness_mapping':  # Skip long dicts for readability
                print(f"  {key}: {value}")
        
        print(f"  Available classes: {len(model_info['class_names'])}")
        
        # Test with a dummy image
        print("\n🖼️  Testing with dummy image...")
        
        # Create a test image (224x224 RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Also test with PIL Image
        pil_image = Image.fromarray(test_image)
        
        print("  Testing with numpy array...")
        result1 = classifier.classify(test_image)
        
        print("  Testing with PIL Image...")
        result2 = classifier.classify(pil_image)
        
        print(f"\n✅ Classification completed successfully!")
        
        # Display results
        print(f"\n📊 Results (Numpy Array):")
        print(f"  Ripeness Level: {result1['ripeness_level']}")
        print(f"  Detailed Class: {result1['detailed_class']}")
        print(f"  Confidence: {result1['confidence']:.4f}")
        print(f"  Class ID: {result1['class_id']}")
        
        print(f"\n📊 Results (PIL Image):")
        print(f"  Ripeness Level: {result2['ripeness_level']}")
        print(f"  Detailed Class: {result2['detailed_class']}")
        print(f"  Confidence: {result2['confidence']:.4f}")
        print(f"  Class ID: {result2['class_id']}")
        
        # Test fruit-specific classes
        print(f"\n🍊 Fruit-specific ripeness classes:")
        for fruit in ['mango', 'orange', 'guava', 'grapefruit']:
            classes = classifier.get_fruit_specific_classes(fruit)
            print(f"  {fruit}: {classes}")
        
        # Test batch classification
        print(f"\n📦 Testing batch classification...")
        batch_images = [test_image, pil_image, test_image]
        batch_results = classifier.classify_batch(batch_images)
        print(f"  Processed {len(batch_results)} images in batch")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🍎 Starting Ripeness Classification Test")
    print("=" * 50)
    
    success = test_classification()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Ripeness Classification Pipeline Test PASSED!")
    else:
        print("❌ Ripeness Classification Pipeline Test FAILED!")