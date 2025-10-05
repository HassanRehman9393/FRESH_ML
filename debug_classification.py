"""
Classification Debug Script
===========================

This script helps debug classification issues by testing the model
with specific images and analyzing the results.
"""

import cv2
import numpy as np
from pipeline.classification.ripeness_classifier import RipenessClassifier
from pipeline.pipeline_config import PipelineConfig
import torch
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_classification_model():
    """Debug the classification model with test data"""
    
    # Initialize classifier
    config = PipelineConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        classifier = RipenessClassifier(
            model_path=config.CLASSIFICATION_MODEL_PATH,
            device=device
        )
        logger.info("✅ Classification model loaded successfully")
        
        # Print class mappings for verification
        logger.info("📋 Class mappings:")
        for class_id, class_name in classifier.class_names.items():
            ripeness = classifier.ripeness_map.get(class_name, "unknown")
            logger.info(f"  {class_id}: {class_name} -> {ripeness}")
        
        # Test with a simple green image (simulating unripe fruit)
        logger.info("\n🧪 Testing with synthetic green image (should be unripe):")
        green_image = np.full((224, 224, 3), [50, 200, 50], dtype=np.uint8)  # Green image BGR
        logger.info(f"Green image shape: {green_image.shape}, dtype: {green_image.dtype}")
        result = classifier.classify(green_image)
        logger.info(f"Result: {result}")
        
        # Test with a simple orange image (simulating ripe fruit) 
        logger.info("\n🧪 Testing with synthetic orange image (should be ripe):")
        orange_image = np.full((224, 224, 3), [40, 140, 255], dtype=np.uint8)  # Orange image BGR
        logger.info(f"Orange image shape: {orange_image.shape}, dtype: {orange_image.dtype}")
        result = classifier.classify(orange_image)
        logger.info(f"Result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {str(e)}")
        return False

def analyze_model_predictions():
    """Analyze if the model predictions make sense"""
    logger.info("\n📊 Model Analysis Complete")
    logger.info("If unripe fruits are being classified as ripe, possible causes:")
    logger.info("1. Model was trained with incorrect labels")
    logger.info("2. Image preprocessing is different from training")
    logger.info("3. Color space conversion issues")
    logger.info("4. Model weights are corrupted")
    logger.info("5. Class mapping is inverted")

if __name__ == "__main__":
    print("🔍 Debugging Classification Model...")
    success = debug_classification_model()
    if success:
        analyze_model_predictions()
    else:
        print("❌ Debug failed - check logs for details")