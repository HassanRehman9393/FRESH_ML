"""
Color Analysis Module
====================

This module provides computer vision-based color analysis for fruit images.
It extracts dominant colors from fruit regions to improve color estimation accuracy.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
import webcolors

class ColorAnalyzer:
    """
    Analyzes colors in fruit images using computer vision techniques
    """
    
    def __init__(self):
        """Initialize color analyzer"""
        self.color_names = {
            'green': [(0, 100, 0), (50, 255, 50)],
            'yellow': [(0, 200, 200), (60, 255, 255)],
            'orange': [(100, 140, 0), (255, 200, 100)],
            'red': [(0, 0, 100), (100, 100, 255)],
            'brown': [(40, 40, 20), (100, 100, 80)]
        }
    
    def extract_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from image using K-means clustering
        
        Args:
            image: Input image (BGR format)
            k: Number of dominant colors to extract
            
        Returns:
            List of dominant colors as (B, G, R) tuples
        """
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers back to uint8
        centers = np.uint8(centers)
        
        return [tuple(center) for center in centers]
    
    def bgr_to_color_name(self, bgr_color: Tuple[int, int, int]) -> str:
        """
        Convert BGR color to closest color name
        
        Args:
            bgr_color: Color in BGR format
            
        Returns:
            Color name string
        """
        b, g, r = bgr_color
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv
        
        # Analyze color based on HSV values
        if s < 50:  # Low saturation - grayish colors
            if v > 200:
                return 'white'
            elif v < 50:
                return 'black'
            else:
                return 'gray'
        
        # Determine color based on hue
        if h < 10 or h > 170:
            return 'red'
        elif h < 25:
            return 'orange'  
        elif h < 35:
            return 'yellow'
        elif h < 85:
            return 'green'
        elif h < 125:
            return 'cyan'
        elif h < 145:
            return 'blue'
        else:
            return 'purple'
    
    def analyze_fruit_color(self, fruit_image: np.ndarray) -> Dict[str, any]:
        """
        Analyze color of a fruit image region
        
        Args:
            fruit_image: Cropped fruit image
            
        Returns:
            Dictionary with color analysis results
        """
        if fruit_image is None or fruit_image.size == 0:
            return {'primary_color': 'unknown', 'confidence': 0.0}
        
        # Extract dominant colors
        dominant_colors = self.extract_dominant_colors(fruit_image, k=3)
        
        # Analyze each dominant color
        color_analysis = []
        for color in dominant_colors:
            color_name = self.bgr_to_color_name(color)
            color_analysis.append({
                'color': color_name,
                'bgr_value': color
            })
        
        # Determine primary color (most relevant for fruit)
        primary_color = color_analysis[0]['color'] if color_analysis else 'unknown'
        
        # Calculate confidence based on color consistency
        color_counts = {}
        for analysis in color_analysis:
            color = analysis['color']
            color_counts[color] = color_counts.get(color, 0) + 1
        
        max_count = max(color_counts.values()) if color_counts else 0
        confidence = max_count / len(color_analysis) if color_analysis else 0.0
        
        return {
            'primary_color': primary_color,
            'secondary_colors': [analysis['color'] for analysis in color_analysis[1:]],
            'confidence': confidence,
            'dominant_colors': dominant_colors,
            'color_distribution': color_counts
        }
    
    def estimate_ripeness_from_color(self, color_analysis: Dict[str, any]) -> str:
        """
        Estimate ripeness based on color analysis
        
        Args:
            color_analysis: Result from analyze_fruit_color
            
        Returns:
            Estimated ripeness level
        """
        primary_color = color_analysis.get('primary_color', 'unknown')
        
        if primary_color == 'green':
            return 'unripe'
        elif primary_color in ['yellow', 'orange']:
            return 'ripe'
        elif primary_color in ['red', 'brown']:
            return 'overripe'
        else:
            return 'unknown'