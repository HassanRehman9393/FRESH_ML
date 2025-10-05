"""
Ripeness Classification Module
==============================

This module handles fruit ripeness classification using the trained classification_best_fixed.pth model.
Classifies cropped fruit regions into ripeness levels across 16 classes for 4 fruit types.
Fixed model with correct class mappings to resolve green mango misclassification issues.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Union, Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RipenessClassifier:
    """
    Fruit ripeness classification class
    
    Loads the trained classification model and provides methods for classifying
    fruit ripeness levels from cropped fruit images.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device):
        """
        Initialize ripeness classifier
        
        Args:
            model_path: Path to the trained classification model (.pth file)
            device: PyTorch device (cuda or cpu)
        """
        self.model_path = model_path
        self.device = device
        
        # Class mappings based on your training data (16 classes)
        self.class_names = {
            0: "mango_unripe",
            1: "mango_early_ripe", 
            2: "mango_partially_ripe",
            3: "mango_ripe",
            4: "mango_rotten",
            5: "orange_unripe",
            6: "orange_ripe",
            7: "orange_rotten",
            8: "orange_general",
            9: "guava_unripe",
            10: "guava_ripe",
            11: "guava_overripe",
            12: "guava_rotten",
            13: "grapefruit_unripe",
            14: "grapefruit_ripe",
            15: "grapefruit_overripe"
        }
        
        # Simplified ripeness mapping
        self.ripeness_map = {
            "mango_unripe": "unripe",
            "mango_early_ripe": "early_ripe",
            "mango_partially_ripe": "partially_ripe", 
            "mango_ripe": "ripe",
            "mango_rotten": "rotten",
            "orange_unripe": "unripe",
            "orange_ripe": "ripe",
            "orange_rotten": "rotten",
            "orange_general": "general",
            "guava_unripe": "unripe",
            "guava_ripe": "ripe",
            "guava_overripe": "overripe",
            "guava_rotten": "rotten",
            "grapefruit_unripe": "unripe",
            "grapefruit_ripe": "ripe",
            "grapefruit_overripe": "overripe"
        }
        
        # Image preprocessing transforms (standard for classification)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the classification model from file"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Classification model not found at: {self.model_path}")
            
            logger.info(f"Loading classification model from: {self.model_path}")
            
            # Load the model state dict - disable weights_only for model metadata
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create model architecture (assuming ResNet-based model)
            self.model = self._create_model_architecture()
            
            # Load weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Classification model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load classification model: {str(e)}")
            raise
    
    def _create_model_architecture(self):
        """Create the model architecture (ResNet50 with 16 classes)"""
        try:
            import torchvision.models as models
            
            # Create ResNet50 model with 16 output classes
            # Based on the error, the saved model is ResNet50 architecture
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 16)  # 16 classes
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model architecture: {str(e)}")
            raise
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for classification
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Ensure proper data type
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                # Ensure proper shape
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                image = Image.fromarray(image)
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            logger.error(f"Image type: {type(image)}, shape: {getattr(image, 'shape', 'N/A')}, dtype: {getattr(image, 'dtype', 'N/A')}")
            raise
    
    def classify(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Classify fruit ripeness in the input image
        
        Args:
            image: Cropped fruit image (numpy array or PIL Image)
            
        Returns:
            Dictionary containing:
            - ripeness_level: Simplified ripeness level
            - detailed_class: Full class name with fruit type
            - confidence: Classification confidence score
            - all_probabilities: Probabilities for all classes
        """
        if self.model is None:
            raise RuntimeError("Classification model not loaded")
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_class = predicted_class.item()
                confidence = confidence.item()
            
            # Get class names
            detailed_class = self.class_names.get(predicted_class, "unknown")
            ripeness_level = self.ripeness_map.get(detailed_class, "unknown")
            
            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()
            prob_dict = {self.class_names[i]: float(all_probs[i]) for i in range(len(all_probs))}
            
            result = {
                'ripeness_level': ripeness_level,
                'detailed_class': detailed_class,
                'confidence': round(confidence, 4),
                'class_id': predicted_class,
                'all_probabilities': prob_dict
            }
            
            # Debug logging to understand predictions
            top_3_classes = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Classification: {detailed_class} ({ripeness_level}) with confidence {confidence:.3f}")
            logger.debug(f"Top 3 predictions: {top_3_classes}")
            
            # Additional debug for mango predictions specifically
            if "mango" in detailed_class.lower():
                mango_probs = {k: v for k, v in prob_dict.items() if "mango" in k.lower()}
                logger.debug(f"Mango class probabilities: {mango_probs}")
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {
                'ripeness_level': 'unknown',
                'detailed_class': 'unknown',
                'confidence': 0.0,
                'class_id': -1,
                'error': str(e)
            }
    
    def classify_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """
        Classify multiple fruit images in batch
        
        Args:
            images: List of cropped fruit images
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.classify(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch classification failed for image {i}: {str(e)}")
                results.append({
                    'ripeness_level': 'unknown',
                    'detailed_class': 'unknown', 
                    'confidence': 0.0,
                    'class_id': -1,
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'ripeness_mapping': self.ripeness_map,
            'input_size': (224, 224),
            'model_loaded': self.model is not None
        }
    
    def get_fruit_specific_classes(self, fruit_type: str) -> List[str]:
        """
        Get ripeness classes specific to a fruit type
        
        Args:
            fruit_type: Fruit type (mango, orange, guava, grapefruit)
            
        Returns:
            List of ripeness levels for the specified fruit
        """
        fruit_classes = []
        
        for class_name in self.class_names.values():
            if class_name.startswith(fruit_type.lower()):
                ripeness = self.ripeness_map.get(class_name, 'unknown')
                if ripeness not in fruit_classes:
                    fruit_classes.append(ripeness)
        
        return fruit_classes