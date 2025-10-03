"""
FRESH ML Pipeline Package
========================

This package contains the ML inference pipeline for fruit detection and classification.
Includes YOLO object detection and classification models for comprehensive fruit analysis.
"""

__version__ = "1.0.0"
__author__ = "FRESH ML Team"

from .predictor import FreshMLPredictor
from .pipeline_config import PipelineConfig

__all__ = [
    "FreshMLPredictor",
    "PipelineConfig"
]