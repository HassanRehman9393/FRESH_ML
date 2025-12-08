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
    
    # Model paths - now dynamically loaded from Digital Ocean Spaces
    @property
    def YOLO_MODEL_PATH(self) -> str:
        """Get YOLO model path from DO Spaces"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.utils.do_spaces_model_manager import get_model_path
            
            model_path = get_model_path("yolo_detection_best.pt")
            if model_path and os.path.exists(model_path):
                return model_path
        except Exception as e:
            pass  # Silently fail, will try local
        
        # Fallback to local
        local_path = "models/yolo_detection_best.pt"
        if os.path.exists(local_path):
            return local_path
        
        # Return None instead of raising error (optional model)
        return None
    
    @property
    def CLASSIFICATION_MODEL_PATH(self) -> str:
        """Get classification model path from DO Spaces"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.utils.do_spaces_model_manager import get_model_path
            
            model_path = get_model_path("classification_best_fixed.pth")
            if model_path and os.path.exists(model_path):
                return model_path
        except Exception as e:
            pass  # Silently fail, will try local
        
        # Fallback to local
        local_path = "models/classification_best_fixed.pth"
        if os.path.exists(local_path):
            return local_path
        
        # Return None instead of raising error (optional model)
        return None
    
    @property
    def ANTHRACNOSE_MODEL_PATH(self) -> str:
        """Get Anthracnose disease detection model path from DO Spaces"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.utils.do_spaces_model_manager import get_model_path
            
            model_path = get_model_path("anthracnose_detection_model.pth")
            if model_path and os.path.exists(model_path):
                return model_path
        except Exception as e:
            print(f"⚠️  Failed to get Anthracnose model from DO Spaces: {e}")
        
        # Fallback to local
        local_path = "models/anthracnose_detection_model.pth"
        if os.path.exists(local_path):
            return local_path
        
        return None  # Optional model
    
    @property
    def CITRUS_CANKER_MODEL_PATH(self) -> str:
        """Get Citrus Canker disease detection model path from DO Spaces"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.utils.do_spaces_model_manager import get_model_path
            
            model_path = get_model_path("citrus_canker_detection_model.pth")
            if model_path and os.path.exists(model_path):
                return model_path
        except Exception as e:
            print(f"⚠️  Failed to get Citrus Canker model from DO Spaces: {e}")
        
        # Fallback to local
        local_path = "models/citrus_canker_detection_model.pth"
        if os.path.exists(local_path):
            return local_path
        
        return None  # Optional model
    
    @property
    def BLACKSPOT_MODEL_PATH(self) -> str:
        """Get Citrus Blackspot disease detection model path from DO Spaces"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.utils.do_spaces_model_manager import get_model_path
            
            model_path = get_model_path("citrus_blackspot_detection_model.pth")
            if model_path and os.path.exists(model_path):
                return model_path
        except Exception as e:
            print(f"⚠️  Failed to get Citrus Blackspot model from DO Spaces: {e}")
        
        # Fallback to local
        local_path = "models/citrus_blackspot_detection_model.pth"
        if os.path.exists(local_path):
            return local_path
        
        return None  # Optional model
    
    @property
    def GUAVA_FRUITFLY_MODEL_PATH(self) -> str:
        """Get guava fruitfly model path from DO Spaces"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.utils.do_spaces_model_manager import get_model_path
            
            model_path = get_model_path("guava_fruitfly_detection_model.pth")
            if model_path and os.path.exists(model_path):
                return model_path
        except Exception as e:
            print(f"⚠️  Failed to get Guava Fruitfly model from DO Spaces: {e}")
        
        # Fallback to local
        local_path = "models/guava_fruitfly_detection_model.pth"
        if os.path.exists(local_path):
            return local_path
        
        return None  # Optional model
    
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