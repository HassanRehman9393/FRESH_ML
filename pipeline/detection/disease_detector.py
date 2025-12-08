"""
Disease Detection Module
========================

Handles disease detection for fruits using trained models:
1. Anthracnose Detection (Mango) - ResNet-50 based
2. Citrus Canker Detection (Orange/Grapefruit) - DenseNet-121 based
3. Citrus Black Spot Detection (Orange) - DenseNet-121 based
4. Guava Fruitfly Detection (Guava) - DenseNet-121 based

All models work on fruit images with automatic routing based on fruit type.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Union, Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DiseaseDetector:
    """
    Unified disease detection class for all fruit diseases
    
    Loads trained disease detection models and provides methods for
    detecting diseases in fruit images with automatic routing based on fruit type.
    
    Disease Models:
    - Anthracnose (Mango) - ResNet-50
    - Citrus Canker (Orange/Grapefruit) - DenseNet-121
    - Black Spot (Orange) - DenseNet-121
    - Fruitfly (Guava) - DenseNet-121
    """
    
    def __init__(self, 
                 anthracnose_model_path: Optional[str] = None,
                 citrus_canker_model_path: Optional[str] = None,
                 blackspot_model_path: Optional[str] = None,
                 fruitfly_model_path: Optional[str] = None,
                 device: torch.device = None):
        """
        Initialize disease detector
        
        Args:
            anthracnose_model_path: Path to anthracnose model (.pth file)
            citrus_canker_model_path: Path to citrus canker model (.pth file)
            blackspot_model_path: Path to citrus blackspot model (.pth file)
            fruitfly_model_path: Path to guava fruitfly model (.pth file)
            device: PyTorch device (cuda or cpu)
        """
        self.anthracnose_model_path = anthracnose_model_path
        self.citrus_canker_model_path = citrus_canker_model_path
        self.blackspot_model_path = blackspot_model_path
        self.fruitfly_model_path = fruitfly_model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class mappings
        self.disease_classes = {
            'anthracnose': {
                0: 'healthy',
                1: 'anthracnose'
            },
            'citrus_canker': {
                0: 'healthy',
                1: 'citrus_canker'
            },
            'blackspot': {
                0: 'blackspot',
                1: 'healthy'
            },
            'fruitfly': {
                0: 'fruitfly',
                1: 'healthy'
            }
        }
        
        # Severity levels (for future enhancement)
        self.severity_levels = {
            0: 'none',
            1: 'mild',
            2: 'moderate',
            3: 'severe'
        }
        
        # Image preprocessing transforms
        self.anthracnose_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.citrus_canker_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.blackspot_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.fruitfly_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load models
        self.anthracnose_model = None
        self.citrus_canker_model = None
        self.blackspot_model = None
        self.fruitfly_model = None
        self._load_models()
    
    def _load_models(self):
        """Load disease detection models"""
        try:
            # Load Anthracnose model (ResNet-50 based)
            if self.anthracnose_model_path:
                logger.info(f"Loading Anthracnose model from {self.anthracnose_model_path}")
                self.anthracnose_model = self._load_anthracnose_model()
                logger.info("✅ Anthracnose model loaded successfully")
            else:
                logger.warning("⚠️  Anthracnose model path not provided")
            
            # Load Citrus Canker model (DenseNet-121 based)
            if self.citrus_canker_model_path:
                logger.info(f"Loading Citrus Canker model from {self.citrus_canker_model_path}")
                self.citrus_canker_model = self._load_citrus_canker_model()
                logger.info("✅ Citrus Canker model loaded successfully")
            else:
                logger.warning("⚠️  Citrus Canker model path not provided")
            
            # Load Blackspot model (DenseNet-121 based)
            if self.blackspot_model_path:
                logger.info(f"Loading Citrus Blackspot model from {self.blackspot_model_path}")
                self.blackspot_model = self._load_blackspot_model()
                logger.info("✅ Citrus Blackspot model loaded successfully")
            else:
                logger.warning("⚠️  Citrus Blackspot model path not provided")
            
            # Load Fruitfly model (DenseNet-121 based)
            if self.fruitfly_model_path:
                logger.info(f"Loading Guava Fruitfly model from {self.fruitfly_model_path}")
                self.fruitfly_model = self._load_fruitfly_model()
                logger.info("✅ Guava Fruitfly model loaded successfully")
            else:
                logger.warning("⚠️  Guava Fruitfly model path not provided")
                
        except Exception as e:
            logger.error(f"Error loading disease models: {str(e)}")
            raise
    
    def _load_anthracnose_model(self):
        """Load Anthracnose detection model (ResNet-50)"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.anthracnose_model_path, map_location=self.device)
            
            # Create ResNet-50 model
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)  # Binary classification
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Anthracnose model: {e}")
            raise
    
    def _load_citrus_canker_model(self):
        """Load Citrus Canker detection model (DenseNet-121)"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.citrus_canker_model_path, map_location=self.device)
            
            # Create DenseNet-121 model
            model = models.densenet121(pretrained=False)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 2)  # Binary classification
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Citrus Canker model: {e}")
            raise
    
    def _load_blackspot_model(self):
        """Load Citrus Blackspot detection model (DenseNet-121)"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.blackspot_model_path, map_location=self.device)
            
            # Create DenseNet-121 model
            model = models.densenet121(pretrained=False)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 2)  # Binary classification
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Blackspot model: {e}")
            raise
    
    def _load_fruitfly_model(self):
        """Load Guava Fruitfly detection model (DenseNet-121)"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.fruitfly_model_path, map_location=self.device)
            
            # Create DenseNet-121 model
            model = models.densenet121(pretrained=False)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 2)  # Binary classification
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Fruitfly model: {e}")
            raise
    
    def detect_anthracnose(self, 
                          image: Union[np.ndarray, Image.Image],
                          return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Detect anthracnose disease in mango fruit/leaf image
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with disease detection results
        """
        if self.anthracnose_model is None:
            logger.error("❌ Anthracnose model not loaded")
            raise RuntimeError("Anthracnose model not loaded")
        
        try:
            logger.debug("🔍 Starting anthracnose detection...")
            
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            logger.debug(f"📐 Image size before transform: {image.size}")
            image_tensor = self.anthracnose_transform(image).unsqueeze(0).to(self.device)
            logger.debug(f"📊 Tensor shape: {image_tensor.shape}")
            
            # Run inference
            with torch.no_grad():
                outputs = self.anthracnose_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get prediction
            predicted_class = predicted.item()
            disease_name = self.disease_classes['anthracnose'][predicted_class]
            confidence_score = confidence.item()
            
            logger.info(f"🦠 Anthracnose result: {disease_name} (confidence: {confidence_score:.3f})")
            logger.debug(f"📊 Raw probabilities - Healthy: {probabilities[0][0].item():.3f}, Diseased: {probabilities[0][1].item():.3f}")
            
            result = {
                'disease': disease_name,
                'confidence': float(confidence_score),
                'is_diseased': disease_name != 'healthy',
                'disease_type': 'anthracnose' if disease_name != 'healthy' else None
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'healthy': float(probabilities[0][0].item()),
                    'anthracnose': float(probabilities[0][1].item())
                }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Anthracnose detection error: {str(e)}")
            raise
    
    def detect_citrus_canker(self, 
                            image: Union[np.ndarray, Image.Image],
                            return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Detect citrus canker disease in orange/grapefruit image
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with disease detection results
        """
        if self.citrus_canker_model is None:
            raise RuntimeError("Citrus Canker model not loaded")
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            image_tensor = self.citrus_canker_transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.citrus_canker_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get prediction
            predicted_class = predicted.item()
            disease_name = self.disease_classes['citrus_canker'][predicted_class]
            confidence_score = confidence.item()
            
            result = {
                'disease': disease_name,
                'confidence': float(confidence_score),
                'is_diseased': disease_name != 'healthy',
                'disease_type': 'citrus_canker' if disease_name != 'healthy' else None
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'healthy': float(probabilities[0][0].item()),
                    'citrus_canker': float(probabilities[0][1].item())
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Citrus Canker detection error: {str(e)}")
            raise
    
    def detect_blackspot(self, 
                        image: Union[np.ndarray, Image.Image],
                        return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Detect citrus blackspot disease in orange fruit image
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with disease detection results
        """
        if self.blackspot_model is None:
            raise RuntimeError("Citrus Blackspot model not loaded")
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            image_tensor = self.blackspot_transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.blackspot_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get prediction
            predicted_class = predicted.item()
            disease_name = self.disease_classes['blackspot'][predicted_class]
            confidence_score = confidence.item()
            
            result = {
                'disease': disease_name,
                'confidence': float(confidence_score),
                'is_diseased': disease_name == 'blackspot',
                'disease_type': 'blackspot' if disease_name == 'blackspot' else None
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'blackspot': float(probabilities[0][0].item()),
                    'healthy': float(probabilities[0][1].item())
                }
            
            logger.info(f"🦠 Blackspot result: {disease_name} (confidence: {confidence_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Blackspot detection error: {str(e)}")
            raise
    
    def detect_fruitfly(self, 
                       image: Union[np.ndarray, Image.Image],
                       return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Detect fruitfly disease in guava fruit image
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with disease detection results
        """
        if self.fruitfly_model is None:
            raise RuntimeError("Guava Fruitfly model not loaded")
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            image_tensor = self.fruitfly_transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.fruitfly_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get prediction
            predicted_class = predicted.item()
            disease_name = self.disease_classes['fruitfly'][predicted_class]
            confidence_score = confidence.item()
            
            result = {
                'disease': disease_name,
                'confidence': float(confidence_score),
                'is_diseased': disease_name == 'fruitfly',
                'disease_type': 'fruitfly' if disease_name == 'fruitfly' else None
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'fruitfly': float(probabilities[0][0].item()),
                    'healthy': float(probabilities[0][1].item())
                }
            
            logger.info(f"🦠 Fruitfly result: {disease_name} (confidence: {confidence_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Fruitfly detection error: {str(e)}")
            raise
    
    def detect_disease(self, 
                      image: Union[np.ndarray, Image.Image],
                      fruit_type: str,
                      return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Detect disease based on fruit type (auto-routes to appropriate model)
        
        Routing Logic:
        - Mango -> Anthracnose detection
        - Orange -> Blackspot detection (prioritized) or Citrus Canker
        - Grapefruit -> Citrus Canker detection
        - Guava -> Fruitfly detection
        
        Args:
            image: Input image (numpy array or PIL Image)
            fruit_type: Type of fruit ('mango', 'orange', 'grapefruit', 'guava')
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with disease detection results
        """
        fruit_type = fruit_type.lower()
        
        # Route to appropriate model based on fruit type
        if fruit_type == 'mango':
            if self.anthracnose_model is None:
                return self._model_not_available_error('anthracnose', fruit_type)
            return self.detect_anthracnose(image, return_probabilities)
        
        elif fruit_type == 'orange':
            # Prioritize blackspot for oranges, fallback to citrus canker
            if self.blackspot_model is not None:
                return self.detect_blackspot(image, return_probabilities)
            elif self.citrus_canker_model is not None:
                return self.detect_citrus_canker(image, return_probabilities)
            else:
                return self._model_not_available_error('blackspot/citrus_canker', fruit_type)
        
        elif fruit_type == 'grapefruit':
            if self.citrus_canker_model is None:
                return self._model_not_available_error('citrus_canker', fruit_type)
            return self.detect_citrus_canker(image, return_probabilities)
        
        elif fruit_type == 'guava':
            if self.fruitfly_model is not None:
                return self.detect_fruitfly(image, return_probabilities)
            elif self.citrus_canker_model is not None:
                # Fallback to citrus canker for guava if fruitfly not available
                return self.detect_citrus_canker(image, return_probabilities)
            else:
                return self._model_not_available_error('fruitfly', fruit_type)
        
        else:
            logger.warning(f"Unknown fruit type: {fruit_type}, skipping disease detection")
            return {
                'disease': 'unknown',
                'confidence': 0.0,
                'is_diseased': False,
                'disease_type': None,
                'error': f'No disease model available for {fruit_type}'
            }
    
    def _model_not_available_error(self, model_name: str, fruit_type: str) -> Dict[str, Any]:
        """Return error dict when model is not available"""
        logger.warning(f"{model_name} model not available for {fruit_type}")
        return {
            'disease': 'unknown',
            'confidence': 0.0,
            'is_diseased': False,
            'disease_type': None,
            'error': f'{model_name} model not loaded for {fruit_type}'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded disease models"""
        return {
            'anthracnose_model': {
                'loaded': self.anthracnose_model is not None,
                'path': self.anthracnose_model_path,
                'architecture': 'ResNet-50',
                'target_fruits': ['mango'],
                'classes': ['healthy', 'anthracnose']
            },
            'citrus_canker_model': {
                'loaded': self.citrus_canker_model is not None,
                'path': self.citrus_canker_model_path,
                'architecture': 'DenseNet-121',
                'target_fruits': ['orange', 'grapefruit'],
                'classes': ['healthy', 'citrus_canker']
            },
            'blackspot_model': {
                'loaded': self.blackspot_model is not None,
                'path': self.blackspot_model_path,
                'architecture': 'DenseNet-121',
                'target_fruits': ['orange'],
                'classes': ['blackspot', 'healthy']
            },
            'fruitfly_model': {
                'loaded': self.fruitfly_model is not None,
                'path': self.fruitfly_model_path,
                'architecture': 'DenseNet-121',
                'target_fruits': ['guava'],
                'classes': ['fruitfly', 'healthy']
            },
            'device': str(self.device)
        }
