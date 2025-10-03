"""
Main FRESH ML Predictor
=======================

This is the main predictor class that orchestrates the entire ML pipeline:
1. Object Detection (YOLO) - Detect fruits in images
2. Classification - Classify ripeness of detected fruits
3. Post-processing - Format results with additional analysis
"""

import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from PIL import Image
import time

from .pipeline_config import PipelineConfig
from .detection.yolo_detector import YOLODetector
from .classification.ripeness_classifier import RipenessClassifier
from .utils.image_processor import ImageProcessor
from .utils.postprocessor import ResultPostProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreshMLPredictor:
    """
    Main predictor class for FRESH ML pipeline
    
    Combines YOLO object detection and classification models to provide
    comprehensive fruit analysis including type, ripeness, and quality metrics.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the FRESH ML Predictor
        
        Args:
            config: Pipeline configuration. If None, uses default config.
        """
        self.config = config or PipelineConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.yolo_detector = None
        self.ripeness_classifier = None
        self.image_processor = ImageProcessor(self.config)
        self.postprocessor = ResultPostProcessor(self.config)
        
        # Load models
        self._load_models()
        
        logger.info(f"FRESH ML Predictor initialized on device: {self.device}")
    
    def _load_models(self):
        """Load YOLO and classification models"""
        try:
            # Validate models exist
            if not self.config.validate_models_exist():
                raise FileNotFoundError("Model files not found. Please check model paths in config.")
            
            # Load YOLO detector
            logger.info("Loading YOLO detection model...")
            self.yolo_detector = YOLODetector(
                model_path=self.config.YOLO_MODEL_PATH,
                device=self.device,
                confidence_threshold=self.config.CONFIDENCE_THRESHOLD,
                iou_threshold=self.config.IOU_THRESHOLD
            )
            
            # Load ripeness classifier
            logger.info("Loading ripeness classification model...")
            self.ripeness_classifier = RipenessClassifier(
                model_path=self.config.CLASSIFICATION_MODEL_PATH,
                device=self.device
            )
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(self, 
                image: Union[str, np.ndarray, Image.Image],
                return_visualization: bool = False) -> Dict[str, Any]:
        """
        Predict fruits in image with comprehensive analysis
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            return_visualization: Whether to return annotated image
            
        Returns:
            Dictionary containing detection and classification results
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess image
            logger.info("Processing input image...")
            processed_image = self.image_processor.load_and_preprocess(image)
            
            # Step 2: Run YOLO detection
            logger.info("Running fruit detection...")
            detections = self.yolo_detector.detect(processed_image)
            
            if not detections:
                logger.info("No fruits detected in image")
                return self._format_empty_result(time.time() - start_time)
            
            # Step 3: Classify detected fruits
            logger.info(f"Classifying {len(detections)} detected fruits...")
            classification_results = []
            
            for detection in detections:
                # Crop fruit region
                cropped_fruit = self.image_processor.crop_detection(
                    processed_image, detection
                )
                
                # Classify ripeness
                ripeness_result = self.ripeness_classifier.classify(cropped_fruit)
                
                # Combine detection and classification
                combined_result = {
                    **detection,
                    **ripeness_result
                }
                classification_results.append(combined_result)
            
            # Step 4: Post-process results
            logger.info("Post-processing results...")
            final_results = self.postprocessor.process_results(
                image=processed_image,
                detections=classification_results,
                processing_time=time.time() - start_time
            )
            
            # Step 5: Add visualization if requested
            if return_visualization:
                annotated_image = self.postprocessor.create_visualization(
                    processed_image, classification_results
                )
                final_results['annotated_image'] = annotated_image
            
            logger.info(f"Pipeline completed in {final_results['processing_time']:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _format_empty_result(self, processing_time: float) -> Dict[str, Any]:
        """Format result when no fruits are detected"""
        return {
            'success': True,
            'total_fruits': 0,
            'fruits': [],
            'processing_time': f"{processing_time:.2f}s",
            'message': 'No fruits detected in the image'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'yolo_model': {
                'path': self.config.YOLO_MODEL_PATH,
                'classes': self.config.FRUIT_CLASSES,
                'input_size': self.config.YOLO_INPUT_SIZE
            },
            'classification_model': {
                'path': self.config.CLASSIFICATION_MODEL_PATH,
                'classes': self.config.RIPENESS_CLASSES,
                'input_size': self.config.CLASSIFICATION_INPUT_SIZE
            },
            'device': str(self.device),
            'confidence_threshold': self.config.CONFIDENCE_THRESHOLD,
            'iou_threshold': self.config.IOU_THRESHOLD
        }