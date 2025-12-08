"""
Citrus Black Spot Detection Module
===================================

Binary classification for citrus fruits (oranges):
- Healthy oranges
- Black spot diseased oranges

Model: DenseNet-121 with transfer learning
Trained on: 2,922 balanced images (1:1 ratio)
Expected Accuracy: 85-93% on test set
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Union, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BlackspotDetector:
    """
    Citrus Black Spot Detection for Oranges
    
    Binary classifier that detects black spot disease in citrus fruits.
    Works on whole fruit images (not leaves).
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: torch.device = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize blackspot detector
        
        Args:
            model_path: Path to trained model (.pth file)
            device: PyTorch device (cuda or cpu)
            confidence_threshold: Minimum confidence for positive detection (0-1)
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Class mappings (must match training)
        self.class_names = ['blackspot', 'healthy']
        self.class_to_idx = {
            'blackspot': 0,
            'healthy': 1
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Display names for user-facing output
        self.display_names = {
            'blackspot': 'Black Spot Diseased',
            'healthy': 'Healthy'
        }
        
        # Image preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained DenseNet-121 model"""
        try:
            if not self.model_path:
                logger.warning("⚠️  Blackspot model path not provided")
                return
            
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                return
            
            logger.info(f"Loading Citrus Blackspot model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create DenseNet-121 architecture
            self.model = models.densenet121(pretrained=False)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, 2)  # Binary classification
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Log model info
            model_info = {
                'architecture': checkpoint.get('model_name', 'densenet121'),
                'num_classes': checkpoint.get('num_classes', 2),
                'image_size': checkpoint.get('image_size', 224),
                'test_accuracy': checkpoint.get('test_accuracy', 'N/A'),
                'best_val_accuracy': checkpoint.get('best_val_accuracy', 'N/A')
            }
            
            logger.info(f"✅ Blackspot model loaded successfully")
            logger.info(f"   Architecture: {model_info['architecture']}")
            logger.info(f"   Image Size: {model_info['image_size']}x{model_info['image_size']}")
            logger.info(f"   Test Accuracy: {model_info['test_accuracy']}")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load blackspot model: {str(e)}")
            self.model = None
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def detect(self, 
               image: Union[np.ndarray, Image.Image],
               return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Detect citrus black spot disease in an image
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing:
                - disease_detected: bool (True if blackspot detected)
                - prediction: str ('blackspot' or 'healthy')
                - display_name: str (user-friendly name)
                - confidence: float (confidence of prediction, 0-1)
                - probabilities: dict (class probabilities if requested)
                - is_high_confidence: bool (confidence > threshold)
        """
        if self.model is None:
            logger.error("❌ Model not loaded, cannot perform detection")
            return {
                'success': False,
                'error': 'Model not loaded',
                'disease_detected': False,
                'prediction': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Extract results
            predicted_idx = predicted_idx.item()
            confidence_score = confidence.item()
            predicted_class = self.idx_to_class[predicted_idx]
            
            # Build result
            result = {
                'success': True,
                'disease_detected': predicted_class == 'blackspot',
                'prediction': predicted_class,
                'display_name': self.display_names[predicted_class],
                'confidence': round(confidence_score, 4),
                'is_high_confidence': confidence_score >= self.confidence_threshold
            }
            
            # Add class probabilities if requested
            if return_probabilities:
                probs = probabilities[0].cpu().numpy()
                result['probabilities'] = {
                    'blackspot': round(float(probs[0]), 4),
                    'healthy': round(float(probs[1]), 4)
                }
            
            # Add severity assessment (based on confidence)
            if result['disease_detected']:
                if confidence_score >= 0.9:
                    result['severity'] = 'high_confidence'
                    result['severity_description'] = 'Strong indication of black spot disease'
                elif confidence_score >= 0.7:
                    result['severity'] = 'moderate_confidence'
                    result['severity_description'] = 'Moderate indication of black spot disease'
                else:
                    result['severity'] = 'low_confidence'
                    result['severity_description'] = 'Weak indication - requires verification'
            else:
                result['severity'] = 'none'
                result['severity_description'] = 'No disease detected'
            
            logger.info(f"🔍 Blackspot Detection: {predicted_class} (confidence: {confidence_score:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during blackspot detection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'disease_detected': False,
                'prediction': 'error',
                'confidence': 0.0
            }
    
    def detect_batch(self, 
                     images: list,
                     return_probabilities: bool = True) -> list:
        """
        Detect black spot in multiple images
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of detection results
        """
        results = []
        for idx, image in enumerate(images):
            logger.info(f"Processing image {idx + 1}/{len(images)}")
            result = self.detect(image, return_probabilities=return_probabilities)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_name': 'Citrus Black Spot Detector',
            'architecture': 'DenseNet-121',
            'task': 'Binary Classification',
            'classes': self.class_names,
            'display_names': self.display_names,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'model_loaded': self.model is not None,
            'model_path': str(self.model_path) if self.model_path else None,
            'input_size': '224x224',
            'expected_accuracy': '85-93%',
            'training_images': 2922,
            'class_balance': '1:1 (perfect balance)'
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None
