"""
YOLO Fruit Detection Module
===========================

This module handles YOLO-based fruit detection using the trained yolo_detection_best.pt model.
Detects 4 fruit classes: mango, orange, guava, grapefruit with bounding boxes and confidence scores.
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Dict, Any, Union, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO-based fruit detection class
    
    Loads the trained YOLO model and provides methods for detecting fruits
    in images with configurable confidence and IoU thresholds.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to the trained YOLO model (.pt file)
            device: PyTorch device (cuda or cpu)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Fruit class mapping (YOLOv11s model - trained March 2026)
        # Matches fruit-detection.v2i.yolov11 dataset class IDs
        self.class_names = {
            0: "grapefruit",
            1: "guava",
            2: "mango",
            3: "orange"
        }
        
        # Load model
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model from file"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"YOLO model not found at: {self.model_path}")
            
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to specified device
            self.model.to(self.device)
            
            logger.info(f"YOLO model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fruits in the input image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries containing:
            - fruit_type: Detected fruit class name
            - confidence: Detection confidence score
            - bbox: Bounding box coordinates [x, y, width, height]
            - bbox_xyxy: Bounding box in xyxy format [x1, y1, x2, y2]
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Process results
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract detection data
                        bbox_xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Convert to xywh format
                        x1, y1, x2, y2 = bbox_xyxy
                        x = int(x1)
                        y = int(y1)
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        
                        # Create detection dictionary
                        detection = {
                            'fruit_type': self.class_names.get(class_id, 'unknown'),
                            'confidence': round(confidence, 4),
                            'bbox': [x, y, width, height],  # [x, y, w, h]
                            'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],  # [x1, y1, x2, y2]
                            'class_id': class_id
                        }
                        
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} fruits")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return []
    
    def detect_and_visualize(self, 
                           image: np.ndarray, 
                           save_path: str = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect fruits and create visualization
        
        Args:
            image: Input image as numpy array
            save_path: Optional path to save annotated image
            
        Returns:
            Tuple of (detections_list, annotated_image)
        """
        # Get detections
        detections = self.detect(image)
        
        # Create annotated image
        annotated_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            fruit_type = detection['fruit_type']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"{fruit_type}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            logger.info(f"Annotated image saved to: {save_path}")
        
        return detections, annotated_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'model_loaded': self.model is not None
        }
    
    def update_thresholds(self, confidence: float = None, iou: float = None):
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
            logger.info(f"Updated confidence threshold to: {confidence}")
            
        if iou is not None:
            self.iou_threshold = iou
            logger.info(f"Updated IoU threshold to: {iou}")