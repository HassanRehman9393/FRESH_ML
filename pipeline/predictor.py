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
from .detection.disease_detector import DiseaseDetector
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
        self.disease_detector = None
        self.image_processor = ImageProcessor(self.config)
        self.postprocessor = ResultPostProcessor(self.config)
        
        # Load models
        self._load_models()
        
        logger.info(f"FRESH ML Predictor initialized on device: {self.device}")
    
    def _load_models(self):
        """Load ML models (YOLO and classification are optional)"""
        try:
            # Load YOLO detector (optional)
            try:
                logger.info("Loading YOLO detection model...")
                self.yolo_detector = YOLODetector(
                    model_path=self.config.YOLO_MODEL_PATH,
                    device=self.device,
                    confidence_threshold=self.config.CONFIDENCE_THRESHOLD,
                    iou_threshold=self.config.IOU_THRESHOLD
                )
                logger.info("✅ YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  YOLO model not loaded: {str(e)}")
                logger.warning("   Fruit detection endpoints will not be available")
                self.yolo_detector = None
            
            # Load ripeness classifier (optional)
            try:
                logger.info("Loading ripeness classification model...")
                self.ripeness_classifier = RipenessClassifier(
                    model_path=self.config.CLASSIFICATION_MODEL_PATH,
                    device=self.device
                )
                logger.info("✅ Classification model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Classification model not loaded: {str(e)}")
                logger.warning("   Ripeness classification will not be available")
                self.ripeness_classifier = None
            
            # Load disease detector (optional)
            try:
                logger.info("Loading disease detection models...")
                self.disease_detector = DiseaseDetector(
                    anthracnose_model_path=self.config.ANTHRACNOSE_MODEL_PATH,
                    citrus_canker_model_path=self.config.CITRUS_CANKER_MODEL_PATH,
                    blackspot_model_path=self.config.BLACKSPOT_MODEL_PATH,
                    fruitfly_model_path=self.config.GUAVA_FRUITFLY_MODEL_PATH,
                    device=self.device
                )
                logger.info("✅ Disease detection models loaded")
            except Exception as e:
                logger.warning(f"⚠️  Disease detection models not loaded: {str(e)}")
                self.disease_detector = None
            
            # Check if at least one model loaded
            models_loaded = [
                self.yolo_detector is not None,
                self.ripeness_classifier is not None,
                self.disease_detector is not None
            ]
            
            if not any(models_loaded):
                raise RuntimeError("No models could be loaded! Please check model files.")
            
            logger.info(f"✅ Pipeline initialized with {sum(models_loaded)}/3 core models loaded")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(self, 
                image: Union[str, np.ndarray, Image.Image],
                return_visualization: bool = False,
                confidence_threshold: Optional[float] = None,
                include_disease_detection: bool = True) -> Dict[str, Any]:
        """
        Predict fruits in image with comprehensive analysis
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            return_visualization: Whether to return annotated image
            confidence_threshold: Custom confidence threshold (overrides default)
            include_disease_detection: Whether to include disease detection
            
        Returns:
            Dictionary containing detection, classification, and disease results
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess image
            logger.info("Processing input image...")
            processed_image = self.image_processor.load_and_preprocess(image)
            
            # Step 2: Set custom confidence threshold if provided
            original_confidence = None
            if confidence_threshold is not None:
                original_confidence = self.yolo_detector.confidence_threshold
                self.yolo_detector.confidence_threshold = confidence_threshold
            
            # Step 3: Run YOLO detection
            logger.info("Running fruit detection...")
            detections = self.yolo_detector.detect(processed_image)
            
            # Step 4: Restore original confidence threshold if it was changed
            if original_confidence is not None:
                self.yolo_detector.confidence_threshold = original_confidence
            
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
                
                # Detect disease (optional)
                disease_result = {}
                if include_disease_detection and self.disease_detector is not None:
                    try:
                        disease_result = self.disease_detector.detect_disease(
                            image=cropped_fruit,
                            fruit_type=detection['fruit_type'],
                            return_probabilities=True
                        )
                    except Exception as e:
                        logger.warning(f"Disease detection failed: {e}")
                        disease_result = {
                            'disease': 'unknown',
                            'confidence': 0.0,
                            'is_diseased': False,
                            'error': str(e)
                        }
                
                # Combine detection, classification, and disease results
                combined_result = {
                    **detection,
                    **ripeness_result,
                    'disease_detection': disease_result  # Nest disease results under this key
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
            
            logger.info(f"Pipeline completed in {final_results['processing_time']}")
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
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_fruits': 0,
            'fruits': [],
            'processing_time': f"{processing_time:.2f}s",
            'summary': {
                'analysis_quality': 'no_detection',
                'average_confidence': 0.0,
                'high_quality_detections': 0,
                'fruit_type_distribution': {},
                'ripeness_distribution': {}
            },
            'message': 'No fruits detected in the image'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
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
        
        # Add disease detection info if available
        if self.disease_detector is not None:
            info['disease_detection'] = self.disease_detector.get_model_info()
        
        return info