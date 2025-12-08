"""
Guava Fruitfly Disease Detection Module
========================================

Binary classification for guava fruits:
- Healthy guavas
- Fruitfly diseased guavas

Model: DenseNet-121 with transfer learning
Trained on: 9,236 balanced images (1:1 ratio)
Expected Accuracy: 90-95% on test set
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Union, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GuavaFruitflyDetector:
    """
    Guava Fruitfly Disease Detection
    
    Binary classifier that detects fruitfly disease in guava fruits.
    Works on whole fruit images.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: torch.device = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize guava fruitfly detector
        
        Args:
            model_path: Path to trained model (.pth file)
            device: PyTorch device (cuda or cpu)
            confidence_threshold: Minimum confidence for positive detection (0-1)
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Class names (alphabetically ordered as in training)
        self.class_names = ['fruitfly', 'healthy']
        self.num_classes = len(self.class_names)
        
        # Initialize model
        self.model = None
        self.transform = None
        
        if model_path:
            self._load_model()
            self._setup_transforms()
            logger.info("✅ Guava Fruitfly detector initialized successfully")
    
    def _load_model(self) -> None:
        """Load trained DenseNet-121 model from checkpoint"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading Guava Fruitfly model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create DenseNet-121 architecture
            self.model = models.densenet121(pretrained=False)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, self.num_classes)
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Log model info
            logger.info("✅ Guava Fruitfly model loaded successfully")
            logger.info(f"   Architecture: {checkpoint.get('model_name', 'densenet121')}")
            logger.info(f"   Image Size: {checkpoint.get('image_size', 224)}x{checkpoint.get('image_size', 224)}")
            logger.info(f"   Test Accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Guava Fruitfly model: {str(e)}")
            raise
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms (same as training)"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Handle BGR (OpenCV) to RGB conversion
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor
    
    def detect(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, Any]:
        """
        Detect fruitfly disease in a guava fruit image
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Dictionary with detection results:
            {
                'disease_detected': bool,
                'prediction': str ('fruitfly' or 'healthy'),
                'confidence': float (0-1),
                'probabilities': {'fruitfly': float, 'healthy': float},
                'severity': str,
                'severity_description': str
            }
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image)
            
            # Preprocess
            image_tensor = self._preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get predictions
            pred_idx = predicted.item()
            pred_class = self.class_names[pred_idx]
            confidence_score = confidence.item()
            
            # Calculate probabilities for each class
            probs_dict = {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(self.num_classes)
            }
            
            # Determine severity
            disease_detected = (pred_class == 'fruitfly')
            
            if disease_detected:
                if confidence_score >= 0.9:
                    severity = "high_confidence"
                    severity_desc = "High confidence fruitfly infection detected"
                elif confidence_score >= self.confidence_threshold:
                    severity = "moderate_confidence"
                    severity_desc = "Moderate confidence fruitfly infection detected"
                else:
                    severity = "low_confidence"
                    severity_desc = "Low confidence fruitfly infection - manual verification recommended"
            else:
                severity = "none"
                severity_desc = "Guava appears healthy"
            
            return {
                'disease_detected': disease_detected,
                'prediction': pred_class,
                'confidence': confidence_score,
                'probabilities': probs_dict,
                'severity': severity,
                'severity_description': severity_desc
            }
            
        except Exception as e:
            logger.error(f"Error during fruitfly detection: {str(e)}")
            raise
    
    def detect_batch(self, images: List[Union[np.ndarray, Image.Image, str]]) -> List[Dict[str, Any]]:
        """
        Detect fruitfly disease in multiple images
        
        Args:
            images: List of images
            
        Returns:
            List of detection results
        """
        results = []
        for img in images:
            try:
                result = self.detect(img)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                results.append({
                    'error': str(e),
                    'disease_detected': False,
                    'prediction': 'error',
                    'confidence': 0.0
                })
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            return {
                'model_name': checkpoint.get('model_name', 'densenet121'),
                'architecture': 'DenseNet-121',
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'image_size': checkpoint.get('image_size', 224),
                'device': str(self.device),
                'confidence_threshold': self.confidence_threshold,
                'test_accuracy': checkpoint.get('test_accuracy', None),
                'best_val_accuracy': checkpoint.get('best_val_accuracy', None),
                'training_info': {
                    'dataset_size': '9,236 images',
                    'balance': '1:1 (Perfect)',
                    'train_split': '6,464 images (70%)',
                    'val_split': '1,384 images (15%)',
                    'test_split': '1,388 images (15%)'
                }
            }
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            return {
                'model_name': 'densenet121',
                'architecture': 'DenseNet-121',
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'device': str(self.device)
            }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")
