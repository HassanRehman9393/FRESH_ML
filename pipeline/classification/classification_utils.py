"""
Classification Utilities
========================

Utility functions for fruit classification and result matching.
"""

from typing import Dict, List, Any, Optional

class ClassificationUtils:
    """Utility functions for classification pipeline"""
    
    @staticmethod
    def match_fruit_to_classes(fruit_type: str, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match detected fruit type with classification result
        
        Args:
            fruit_type: Detected fruit type from YOLO (mango, orange, guava, grapefruit)
            classification_result: Result from ripeness classifier
            
        Returns:
            Enhanced result with fruit type validation
        """
        detailed_class = classification_result.get('detailed_class', '')
        detected_fruit = fruit_type.lower()
        
        # Check if classification matches detection
        is_consistent = detailed_class.startswith(detected_fruit)
        
        # Extract fruit type from detailed class
        classified_fruit = detailed_class.split('_')[0] if '_' in detailed_class else 'unknown'
        
        enhanced_result = {
            **classification_result,
            'detected_fruit_type': detected_fruit,
            'classified_fruit_type': classified_fruit,
            'type_consistency': is_consistent,
            'confidence_adjusted': classification_result.get('confidence', 0.0) * (1.0 if is_consistent else 0.8)
        }
        
        return enhanced_result
    
    @staticmethod
    def get_color_analysis(ripeness_level: str, fruit_type: str) -> Dict[str, str]:
        """
        Get expected color characteristics based on fruit type and ripeness
        
        Args:
            ripeness_level: Ripeness level (unripe, ripe, overripe, rotten)
            fruit_type: Fruit type (mango, orange, guava, grapefruit)
            
        Returns:
            Dictionary with color information
        """
        color_map = {
            'mango': {
                'unripe': {'primary': 'green', 'secondary': 'light_green'},
                'early_ripe': {'primary': 'yellow_green', 'secondary': 'yellow'},
                'partially_ripe': {'primary': 'yellow', 'secondary': 'orange'},
                'ripe': {'primary': 'orange', 'secondary': 'red'},
                'rotten': {'primary': 'brown', 'secondary': 'black'}
            },
            'orange': {
                'unripe': {'primary': 'green', 'secondary': 'light_green'},
                'ripe': {'primary': 'orange', 'secondary': 'deep_orange'},
                'rotten': {'primary': 'brown', 'secondary': 'black'},
                'general': {'primary': 'orange', 'secondary': 'yellow'}
            },
            'guava': {
                'unripe': {'primary': 'green', 'secondary': 'dark_green'},
                'ripe': {'primary': 'yellow', 'secondary': 'light_yellow'},
                'overripe': {'primary': 'yellow_brown', 'secondary': 'brown'},
                'rotten': {'primary': 'brown', 'secondary': 'black'}
            },
            'grapefruit': {
                'unripe': {'primary': 'green', 'secondary': 'light_green'},
                'ripe': {'primary': 'pink', 'secondary': 'light_pink'},
                'overripe': {'primary': 'yellow', 'secondary': 'pale_yellow'},
                'rotten': {'primary': 'brown', 'secondary': 'black'}
            }
        }
        
        fruit_colors = color_map.get(fruit_type.lower(), {})
        ripeness_colors = fruit_colors.get(ripeness_level, {'primary': 'unknown', 'secondary': 'unknown'})
        
        return ripeness_colors
    
    @staticmethod
    def estimate_size_category(bbox: List[int]) -> str:
        """
        Estimate fruit size category based on bounding box dimensions
        
        Args:
            bbox: Bounding box [x, y, width, height]
            
        Returns:
            Size category (small, medium, large)
        """
        if len(bbox) < 4:
            return 'unknown'
        
        width, height = bbox[2], bbox[3]
        area = width * height
        
        # Size thresholds (can be adjusted based on image resolution)
        if area < 5000:  # Less than ~70x70 pixels
            return 'small'
        elif area < 15000:  # Less than ~120x120 pixels
            return 'medium'
        else:
            return 'large'
    
    @staticmethod
    def calculate_quality_score(confidence: float, 
                              ripeness_level: str, 
                              type_consistency: bool) -> float:
        """
        Calculate overall quality score for the fruit analysis
        
        Args:
            confidence: Classification confidence
            ripeness_level: Detected ripeness level
            type_consistency: Whether detection and classification agree on fruit type
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score from confidence
        base_score = confidence
        
        # Ripeness bonus/penalty
        ripeness_multiplier = {
            'ripe': 1.0,
            'early_ripe': 0.95,
            'partially_ripe': 0.9,
            'unripe': 0.85,
            'overripe': 0.8,
            'rotten': 0.6,
            'general': 0.9,
            'unknown': 0.5
        }
        
        ripeness_factor = ripeness_multiplier.get(ripeness_level, 0.5)
        
        # Type consistency bonus
        consistency_factor = 1.0 if type_consistency else 0.9
        
        # Calculate final score
        quality_score = base_score * ripeness_factor * consistency_factor
        
        return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1
    
    @staticmethod
    def format_analysis_result(detection: Dict[str, Any], 
                             classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format combined detection and classification result
        
        Args:
            detection: YOLO detection result
            classification: Ripeness classification result
            
        Returns:
            Formatted analysis result
        """
        # Match fruit types
        matched_result = ClassificationUtils.match_fruit_to_classes(
            detection['fruit_type'], classification
        )
        
        # Get color analysis
        color_info = ClassificationUtils.get_color_analysis(
            matched_result['ripeness_level'], 
            detection['fruit_type']
        )
        
        # Estimate size
        size_category = ClassificationUtils.estimate_size_category(detection['bbox'])
        
        # Calculate quality score
        quality_score = ClassificationUtils.calculate_quality_score(
            matched_result['confidence'],
            matched_result['ripeness_level'],
            matched_result['type_consistency']
        )
        
        # Format final result
        result = {
            'fruit_id': hash(f"{detection['bbox']}{detection['fruit_type']}") % 10000,
            'fruit_type': detection['fruit_type'],
            'ripeness_level': matched_result['ripeness_level'],
            'detailed_class': matched_result['detailed_class'],
            'confidence': matched_result['confidence'],
            'adjusted_confidence': matched_result['confidence_adjusted'],
            'quality_score': quality_score,
            'bounding_box': detection['bbox'],
            'bbox_xyxy': detection['bbox_xyxy'],
            'size_category': size_category,
            'expected_colors': color_info,
            'type_consistency': matched_result['type_consistency'],
            'analysis_metadata': {
                'detection_confidence': detection['confidence'],
                'classification_confidence': classification['confidence'],
                'detected_fruit': matched_result['detected_fruit_type'],
                'classified_fruit': matched_result['classified_fruit_type']
            }
        }
        
        return result