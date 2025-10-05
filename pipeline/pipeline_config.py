"""
Pipeline Configuration
======================

Configuration settings for the FRESH ML pipeline including model paths,
image processing parameters, and output formatting options.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

@dataclass
class PipelineConfig:
    """Configuration class for FRESH ML pipeline"""
    
    # Model paths
    YOLO_MODEL_PATH: str = "models/yolo_detection_best.pt"
    CLASSIFICATION_MODEL_PATH: str = "models/classification_best.pth"
    
    # Image processing
    YOLO_INPUT_SIZE: Tuple[int, int] = (640, 640)
    CLASSIFICATION_INPUT_SIZE: Tuple[int, int] = (224, 224)
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    # Fruit classes - using default_factory to avoid mutable default
    FRUIT_CLASSES: List[str] = field(default_factory=lambda: ["mango", "orange", "guava", "grapefruit"])
    RIPENESS_CLASSES: List[str] = field(default_factory=lambda: [
        "mango_unripe", "mango_early_ripe", "mango_partially_ripe", "mango_ripe", "mango_rotten",
        "orange_unripe", "orange_ripe", "orange_rotten", "orange_general",
        "guava_unripe", "guava_ripe", "guava_overripe", "guava_rotten",
        "grapefruit_unripe", "grapefruit_ripe", "grapefruit_overripe", "grapefruit_rotten"
    ])
    
    # Processing parameters
    MAX_IMAGE_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
    
    # Output configuration
    INCLUDE_VISUALIZATION: bool = False
    RETURN_CROPPED_FRUITS: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Convert relative paths to absolute paths
        base_path = Path(__file__).parent.parent
        
        if not os.path.isabs(self.YOLO_MODEL_PATH):
            self.YOLO_MODEL_PATH = str(base_path / self.YOLO_MODEL_PATH)
            
        if not os.path.isabs(self.CLASSIFICATION_MODEL_PATH):
            self.CLASSIFICATION_MODEL_PATH = str(base_path / self.CLASSIFICATION_MODEL_PATH)
    
    @classmethod
    def get_ripeness_for_fruit(cls, fruit_type: str) -> List[str]:
        """Get ripeness levels for a specific fruit type"""
        ripeness_map = {
            "mango": ["unripe", "early_ripe", "partially_ripe", "ripe", "rotten"],
            "orange": ["unripe", "ripe", "rotten", "general"],
            "guava": ["unripe", "ripe", "overripe", "rotten"],
            "grapefruit": ["unripe", "ripe", "overripe", "rotten"]
        }
        return ripeness_map.get(fruit_type, ["unknown"])
    
    def validate_models_exist(self) -> bool:
        """Check if model files exist"""
        yolo_exists = os.path.exists(self.YOLO_MODEL_PATH)
        classification_exists = os.path.exists(self.CLASSIFICATION_MODEL_PATH)
        return yolo_exists and classification_exists