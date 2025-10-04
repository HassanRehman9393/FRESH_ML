"""
Image Processing Utilities
==========================

This module handles image loading, preprocessing, and cropping operations
for the FRESH ML pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, List, Dict, Any
import logging
from pathlib import Path
import base64
import io

from ..pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processing utilities for the FRESH ML pipeline
    
    Handles image loading, preprocessing, cropping, and format conversions
    for both detection and classification stages.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize image processor
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_and_preprocess(self, 
                          image_input: Union[str, np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """
        Load and preprocess image from various input formats
        
        Args:
            image_input: Image input (file path, numpy array, PIL Image, or bytes)
            
        Returns:
            Preprocessed image as numpy array (BGR format for OpenCV)
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path or base64 string
                if self._is_base64(image_input):
                    image = self._load_from_base64(image_input)
                else:
                    image = self._load_from_path(image_input)
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = image_input.copy()
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = self._pil_to_opencv(image_input)
            elif isinstance(image_input, bytes):
                # Raw bytes
                image = self._load_from_bytes(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Validate image
            if image is None or image.size == 0:
                raise ValueError("Invalid or empty image")
            
            # Ensure 3 channels (BGR)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Validate image size
            height, width = image.shape[:2]
            if height * width * 3 > self.config.MAX_IMAGE_SIZE:
                logger.warning(f"Image too large ({width}x{height}), resizing...")
                image = self._resize_if_needed(image)
            
            logger.info(f"Image loaded and preprocessed: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def crop_detection(self, 
                      image: np.ndarray, 
                      detection: Dict[str, Any],
                      padding: int = 10) -> np.ndarray:
        """
        Crop detected fruit region from image
        
        Args:
            image: Original image
            detection: Detection result with bounding box
            padding: Additional padding around the bounding box
            
        Returns:
            Cropped fruit image
        """
        try:
            # Get bounding box coordinates
            if 'bbox' in detection:
                x, y, w, h = detection['bbox']
            elif 'bbox_xyxy' in detection:
                x1, y1, x2, y2 = detection['bbox_xyxy']
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
            else:
                raise ValueError("No bounding box found in detection")
            
            # Add padding
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(image.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(image.shape[0] - y_pad, h + 2 * padding)
            
            # Crop image
            cropped = image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
            
            # Validate crop
            if cropped.size == 0:
                logger.warning("Empty crop detected, using original detection without padding")
                cropped = image[y:y + h, x:x + w]
            
            logger.debug(f"Cropped fruit region: {cropped.shape}")
            return cropped
            
        except Exception as e:
            logger.error(f"Cropping failed: {str(e)}")
            # Return a small region from center if cropping fails
            h, w = image.shape[:2]
            center_crop = image[h//4:3*h//4, w//4:3*w//4]
            return center_crop if center_crop.size > 0 else image
    
    def crop_multiple_detections(self, 
                               image: np.ndarray, 
                               detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Crop multiple detected fruit regions from image
        
        Args:
            image: Original image
            detections: List of detection results
            
        Returns:
            List of cropped fruit images
        """
        cropped_fruits = []
        
        for i, detection in enumerate(detections):
            try:
                cropped = self.crop_detection(image, detection)
                cropped_fruits.append(cropped)
            except Exception as e:
                logger.error(f"Failed to crop detection {i}: {str(e)}")
                # Add a placeholder or skip
                continue
        
        logger.info(f"Cropped {len(cropped_fruits)} fruit regions")
        return cropped_fruits
    
    def _load_from_path(self, file_path: str) -> np.ndarray:
        """Load image from file path"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        return image
    
    def _load_from_base64(self, base64_string: str) -> np.ndarray:
        """Load image from base64 string"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string and 'data:' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            return self._load_from_bytes(image_bytes)
            
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")
    
    def _load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from raw bytes"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from bytes")
            
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image from bytes: {str(e)}")
    
    def _pil_to_opencv(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        try:
            # Convert PIL to RGB if not already
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_bgr
            
        except Exception as e:
            raise ValueError(f"Failed to convert PIL image: {str(e)}")
    
    def _is_base64(self, s: str) -> bool:
        """Check if string is base64 encoded"""
        try:
            # Check for data URL format
            if s.startswith('data:image'):
                return True
            
            # Try to decode as base64
            if len(s) % 4 == 0:
                base64.b64decode(s, validate=True)
                return True
            
            return False
            
        except Exception:
            return False
    
    def _resize_if_needed(self, image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Resize image if it's too large"""
        height, width = image.shape[:2]
        
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized
        
        return image
    
    def save_image(self, image: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """
        Save image to file
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set compression parameters
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif output_path.lower().endswith('.png'):
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                params = []
            
            success = cv2.imwrite(output_path, image, params)
            
            if success:
                logger.info(f"Image saved to: {output_path}")
            else:
                logger.error(f"Failed to save image to: {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about the image"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'dtype': str(image.dtype),
            'size_bytes': image.nbytes,
            'aspect_ratio': round(width / height, 3)
        }