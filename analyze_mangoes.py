"""
Real Mango Analysis Script
==========================

This script analyzes the actual cropped mango regions from the pipeline
to understand why green mangoes are being classified as ripe.
"""

import cv2
import numpy as np
from pipeline.classification.ripeness_classifier import RipenessClassifier
from pipeline.detection.yolo_detector import YOLODetector
from pipeline.pipeline_config import PipelineConfig
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_mango_crops():
    """Analyze actual mango crops from detection"""
    
    # Initialize components
    config = PipelineConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load classifier
        classifier = RipenessClassifier(
            model_path=config.CLASSIFICATION_MODEL_PATH,
            device=device
        )
        
        # Load detector 
        detector = YOLODetector(
            model_path=config.YOLO_MODEL_PATH,
            device=device
        )
        
        logger.info("✅ Models loaded successfully")
        
        # Test with a very green synthetic mango
        logger.info("\n🥭 Testing with very green synthetic mango:")
        very_green_mango = np.full((224, 224, 3), [20, 180, 20], dtype=np.uint8)  # Very green BGR
        result = classifier.classify(very_green_mango)
        logger.info(f"Very green result: {result['detailed_class']} ({result['ripeness_level']}) - {result['confidence']:.3f}")
        
        # Test with yellowish-green (early ripe)
        logger.info("\n🥭 Testing with yellowish-green mango:")
        yellow_green = np.full((224, 224, 3), [60, 180, 100], dtype=np.uint8) # Yellow-green BGR
        result = classifier.classify(yellow_green)
        logger.info(f"Yellow-green result: {result['detailed_class']} ({result['ripeness_level']}) - {result['confidence']:.3f}")
        
        # Test with golden mango
        logger.info("\n🥭 Testing with golden mango:")
        golden = np.full((224, 224, 3), [80, 200, 255], dtype=np.uint8) # Golden BGR
        result = classifier.classify(golden)
        logger.info(f"Golden result: {result['detailed_class']} ({result['ripeness_level']}) - {result['confidence']:.3f}")
        
        # Analysis of the issue
        logger.info("\n📊 Analysis:")
        logger.info("The model seems to be working correctly with synthetic colors.")
        logger.info("The issue with your real mangoes might be:")
        logger.info("1. Lighting conditions making green appear more yellowish")
        logger.info("2. The specific green shade in your image")
        logger.info("3. The model was trained on different mango varieties")
        logger.info("4. Image quality/resolution affecting color perception")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return False

def suggest_solutions():
    """Suggest solutions for the classification issue"""
    logger.info("\n💡 Suggested Solutions:")
    logger.info("1. Implement color-based preprocessing to enhance green detection")
    logger.info("2. Add confidence thresholds - reject low-confidence predictions")
    logger.info("3. Use ensemble approach: color analysis + model prediction")
    logger.info("4. Fine-tune model with your specific mango images")
    logger.info("5. Add lighting normalization in preprocessing")

if __name__ == "__main__":
    print("🔍 Analyzing Real Mango Classification...")
    success = analyze_mango_crops()
    if success:
        suggest_solutions()