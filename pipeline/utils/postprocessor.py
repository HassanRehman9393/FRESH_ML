"""
Result Post-Processing Module
=============================

This module handles post-processing of detection and classification results,
formatting them into the final output structure with additional analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from datetime import datetime

from ..pipeline_config import PipelineConfig
from ..classification.classification_utils import ClassificationUtils

logger = logging.getLogger(__name__)

class ResultPostProcessor:
    """
    Post-processor for FRESH ML pipeline results
    
    Combines detection and classification results, adds analysis metrics,
    and formats the final output structure.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize result post-processor
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.utils = ClassificationUtils()
    
    def process_results(self, 
                       image: np.ndarray,
                       detections: List[Dict[str, Any]], 
                       processing_time: float) -> Dict[str, Any]:
        """
        Process and format the complete pipeline results
        
        Args:
            image: Original input image
            detections: Combined detection and classification results
            processing_time: Total processing time in seconds
            
        Returns:
            Formatted final results
        """
        try:
            # Process individual fruit results
            processed_fruits = []
            
            for i, detection in enumerate(detections):
                try:
                    # Format individual fruit result
                    fruit_result = self._format_fruit_result(detection, i + 1)
                    processed_fruits.append(fruit_result)
                    
                except Exception as e:
                    logger.error(f"Error processing fruit {i + 1}: {str(e)}")
                    continue
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(processed_fruits)
            
            # Get image metadata
            image_info = self._get_image_metadata(image)
            
            # Format final result
            final_result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'processing_time': f"{processing_time:.2f}s",
                'image_metadata': image_info,
                'total_fruits': len(processed_fruits),
                'fruits': processed_fruits,
                'summary': summary_stats,
                'pipeline_info': {
                    'yolo_model': self.config.YOLO_MODEL_PATH,
                    'classification_model': self.config.CLASSIFICATION_MODEL_PATH,
                    'confidence_threshold': self.config.CONFIDENCE_THRESHOLD,
                    'iou_threshold': self.config.IOU_THRESHOLD
                }
            }
            
            logger.info(f"Results processed: {len(processed_fruits)} fruits analyzed")
            return final_result
            
        except Exception as e:
            logger.error(f"Result processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time': f"{processing_time:.2f}s"
            }
    
    def _format_fruit_result(self, detection: Dict[str, Any], fruit_id: int) -> Dict[str, Any]:
        """Format individual fruit analysis result"""
        
        # Extract detection data
        fruit_type = detection.get('fruit_type', 'unknown')
        detection_confidence = detection.get('confidence', 0.0)
        bbox = detection.get('bbox', [0, 0, 0, 0])
        bbox_xyxy = detection.get('bbox_xyxy', [0, 0, 0, 0])
        
        # Extract classification data
        ripeness_level = detection.get('ripeness_level', 'unknown')
        detailed_class = detection.get('detailed_class', 'unknown')
        classification_confidence = detection.get('confidence', 0.0)
        
        # Calculate additional metrics
        size_category = self.utils.estimate_size_category(bbox)
        color_info = self.utils.get_color_analysis(ripeness_level, fruit_type)
        
        # Check type consistency
        is_consistent = detailed_class.startswith(fruit_type.lower()) if detailed_class != 'unknown' else True
        
        # Calculate quality score
        quality_score = self.utils.calculate_quality_score(
            classification_confidence, ripeness_level, is_consistent
        )
        
        # Format result
        fruit_result = {
            'fruit_id': fruit_id,
            'fruit_type': fruit_type,
            'ripeness_level': ripeness_level,
            'detailed_classification': detailed_class,
            'confidence_scores': {
                'detection': round(detection_confidence, 4),
                'classification': round(classification_confidence, 4),
                'overall': round((detection_confidence + classification_confidence) / 2, 4)
            },
            'quality_score': round(quality_score, 4),
            'location': {
                'bounding_box': bbox,  # [x, y, width, height]
                'bbox_xyxy': bbox_xyxy,  # [x1, y1, x2, y2]
                'center_point': [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2],
                'size_category': size_category
            },
            'appearance': {
                'expected_colors': color_info,
                'size_pixels': bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            },
            'analysis_flags': {
                'type_consistency': is_consistent,
                'high_confidence': classification_confidence > 0.8,
                'good_quality': quality_score > 0.7
            }
        }
        
        return fruit_result
    
    def _calculate_summary_stats(self, fruits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for all detected fruits"""
        
        if not fruits:
            return {
                'fruit_counts': {},
                'ripeness_distribution': {},
                'average_confidence': 0.0,
                'average_quality_score': 0.0,
                'total_area': 0,
                'analysis_quality': 'no_fruits'
            }
        
        # Count fruits by type
        fruit_counts = {}
        ripeness_counts = {}
        confidences = []
        quality_scores = []
        total_area = 0
        
        for fruit in fruits:
            # Fruit type counts
            fruit_type = fruit['fruit_type']
            fruit_counts[fruit_type] = fruit_counts.get(fruit_type, 0) + 1
            
            # Ripeness distribution
            ripeness = fruit['ripeness_level']
            ripeness_counts[ripeness] = ripeness_counts.get(ripeness, 0) + 1
            
            # Confidence scores
            confidences.append(fruit['confidence_scores']['overall'])
            quality_scores.append(fruit['quality_score'])
            
            # Total area
            bbox = fruit['location']['bounding_box']
            if len(bbox) >= 4:
                total_area += bbox[2] * bbox[3]
        
        # Calculate averages
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Determine overall analysis quality
        if avg_quality > 0.8:
            analysis_quality = 'excellent'
        elif avg_quality > 0.6:
            analysis_quality = 'good'
        elif avg_quality > 0.4:
            analysis_quality = 'fair'
        else:
            analysis_quality = 'poor'
        
        return {
            'fruit_counts': fruit_counts,
            'ripeness_distribution': ripeness_counts,
            'average_confidence': round(avg_confidence, 4),
            'average_quality_score': round(avg_quality, 4),
            'total_area_pixels': total_area,
            'analysis_quality': analysis_quality,
            'high_quality_detections': sum(1 for q in quality_scores if q > 0.7),
            'consistent_classifications': sum(1 for f in fruits if f['analysis_flags']['type_consistency'])
        }
    
    def _get_image_metadata(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract metadata from the input image"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'aspect_ratio': round(width / height, 3),
            'total_pixels': width * height,
            'size_category': self._categorize_image_size(width, height)
        }
    
    def _categorize_image_size(self, width: int, height: int) -> str:
        """Categorize image size"""
        total_pixels = width * height
        
        if total_pixels < 300000:  # Less than ~550x550
            return 'small'
        elif total_pixels < 1000000:  # Less than ~1000x1000
            return 'medium'
        elif total_pixels < 2000000:  # Less than ~1400x1400
            return 'large'
        else:
            return 'very_large'
    
    def create_visualization(self, 
                           image: np.ndarray, 
                           results: List[Dict[str, Any]],
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization of detection and classification results
        
        Args:
            image: Original image
            results: Combined detection and classification results
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image with bounding boxes and labels
        """
        try:
            # Create copy for annotation
            annotated = image.copy()
            
            # Color mapping for different fruit types
            colors = {
                'mango': (0, 255, 255),      # Yellow
                'orange': (0, 165, 255),     # Orange
                'guava': (0, 255, 0),        # Green
                'grapefruit': (255, 192, 203), # Pink
                'unknown': (128, 128, 128)    # Gray
            }
            
            for result in results:
                try:
                    # Get data
                    fruit_type = result.get('fruit_type', 'unknown')
                    ripeness = result.get('ripeness_level', 'unknown')
                    confidence = result.get('confidence', 0.0)
                    bbox = result.get('bbox', [0, 0, 0, 0])
                    
                    if len(bbox) < 4:
                        continue
                    
                    x, y, w, h = bbox
                    color = colors.get(fruit_type, colors['unknown'])
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    
                    # Create label
                    label = f"{fruit_type}: {ripeness}"
                    conf_label = f"Conf: {confidence:.2f}"
                    
                    # Calculate text size
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    (conf_w, conf_h), _ = cv2.getTextSize(conf_label, font, font_scale, thickness)
                    
                    # Draw label background
                    label_bg_h = label_h + conf_h + 15
                    label_bg_w = max(label_w, conf_w) + 10
                    
                    cv2.rectangle(annotated, 
                                (x, y - label_bg_h - 5), 
                                (x + label_bg_w, y), 
                                color, -1)
                    
                    # Draw text
                    cv2.putText(annotated, label, 
                              (x + 5, y - conf_h - 10), 
                              font, font_scale, (0, 0, 0), thickness)
                    
                    cv2.putText(annotated, conf_label, 
                              (x + 5, y - 5), 
                              font, font_scale, (0, 0, 0), thickness)
                    
                except Exception as e:
                    logger.error(f"Error annotating result: {str(e)}")
                    continue
            
            # Add summary text
            summary_text = f"Detected: {len(results)} fruits"
            cv2.putText(annotated, summary_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, annotated)
                logger.info(f"Visualization saved to: {save_path}")
            
            return annotated
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            return image