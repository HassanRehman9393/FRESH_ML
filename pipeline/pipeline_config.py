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
    
    # Model paths - loaded from local models/ directory
    @property
    def YOLO_MODEL_PATH(self) -> str:
        local_path = "models/yolov11s_best.pt"
        return local_path if os.path.exists(local_path) else None

    @property
    def CLASSIFICATION_MODEL_PATH(self) -> str:
        local_path = "models/classification_best_fixed.pth"
        return local_path if os.path.exists(local_path) else None

    @property
    def ANTHRACNOSE_MODEL_PATH(self) -> str:
        local_path = "models/anthracnose_detection_model.pth"
        return local_path if os.path.exists(local_path) else None

    @property
    def CITRUS_CANKER_MODEL_PATH(self) -> str:
        local_path = "models/citrus_canker_detection_model.pth"
        return local_path if os.path.exists(local_path) else None

    @property
    def BLACKSPOT_MODEL_PATH(self) -> str:
        local_path = "models/citrus_blackspot_detection_model.pth"
        return local_path if os.path.exists(local_path) else None

    @property
    def GUAVA_FRUITFLY_MODEL_PATH(self) -> str:
        local_path = "models/guava_fruitfly_detection_model.pth"
        return local_path if os.path.exists(local_path) else None
    
    # Image processing
    YOLO_INPUT_SIZE: Tuple[int, int] = (640, 640)
    CLASSIFICATION_INPUT_SIZE: Tuple[int, int] = (224, 224)
    DISEASE_DETECTION_INPUT_SIZE: Tuple[int, int] = (224, 224)
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    DISEASE_CONFIDENCE_THRESHOLD: float = 0.7  # Higher threshold for disease detection
    
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
        # Model paths are now properties that handle their own path resolution
        # No need to modify them in __post_init__ anymore
        pass
    
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
        """Check if at least one model file exists (all are optional now)"""
        yolo_exists = self.YOLO_MODEL_PATH and os.path.exists(self.YOLO_MODEL_PATH)
        classification_exists = self.CLASSIFICATION_MODEL_PATH and os.path.exists(self.CLASSIFICATION_MODEL_PATH)
        anthracnose_exists = self.ANTHRACNOSE_MODEL_PATH and os.path.exists(self.ANTHRACNOSE_MODEL_PATH)
        citrus_canker_exists = self.CITRUS_CANKER_MODEL_PATH and os.path.exists(self.CITRUS_CANKER_MODEL_PATH)
        blackspot_exists = self.BLACKSPOT_MODEL_PATH and os.path.exists(self.BLACKSPOT_MODEL_PATH)
        fruitfly_exists = self.GUAVA_FRUITFLY_MODEL_PATH and os.path.exists(self.GUAVA_FRUITFLY_MODEL_PATH)
        
        # Log model status
        if not yolo_exists:
            print("⚠️  YOLO detection model not found (optional)")
        else:
            print("✅ YOLO detection model loaded successfully")
            
        if not classification_exists:
            print("⚠️  Classification model not found (optional)")
        else:
            print("✅ Classification model loaded successfully")
            
        if not anthracnose_exists:
            print("⚠️  Anthracnose detection model not found (optional)")
        else:
            print("✅ Anthracnose detection model loaded successfully")
            
        if not citrus_canker_exists:
            print("⚠️  Citrus Canker detection model not found (optional)")
        else:
            print("✅ Citrus Canker detection model loaded successfully")
            
        if not blackspot_exists:
            print("⚠️  Citrus Blackspot detection model not found (optional)")
        else:
            print("✅ Citrus Blackspot detection model loaded successfully")
        
        if not fruitfly_exists:
            print("⚠️  Guava Fruitfly detection model not found (optional)")
        else:
            print("✅ Guava Fruitfly detection model loaded successfully")
        
        # At least one model must exist
        models_found = [yolo_exists, classification_exists, anthracnose_exists, citrus_canker_exists, blackspot_exists, fruitfly_exists]
        return any(models_found)
    
    # ==================== YIELD PREDICTION CONFIGURATION ====================
    
    @property
    def YIELD_MODEL_PATH(self) -> str:
        local_path = "models/yield_model.joblib"
        return local_path if os.path.exists(local_path) else None
    
    # Yield prediction defaults
    YIELD_DEFAULT_SAMPLING_PATTERN: str = "w-shaped"  # or "zigzag"
    YIELD_MIN_CONFIDENCE: float = 0.6
    YIELD_CACHE_DURATION_HOURS: int = 24
    YIELD_MIN_DETECTIONS_FOR_PREDICTION: int = 5
    
    # Regional yield averages (kg/hectare) - Pakistan only
    REGIONAL_YIELD_AVERAGES: Dict[str, float] = field(default_factory=lambda: {
        "mango": 8500,
        "orange": 32000,
        "guava": 22000,
        "grapefruit": 28000,
    })
    
    # Default fruit weights (kg) for yield conversion
    FRUIT_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "mango": 0.25,
        "orange": 0.18,
        "guava": 0.18,
        "grapefruit": 0.35,
    })