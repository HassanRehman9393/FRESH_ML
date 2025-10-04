"""
API Request/Response Schemas
============================

Pydantic models for API request and response validation.
Updated to align with Supabase database schema for backend integration.
"""

# Import database-aligned models as primary models
from .database_models import (
    # Request Models
    FruitDetectionRequest,
    SingleImageDetectionRequest,
    ImageData,
    DetectionOptions,
    
    # Response Models  
    FruitDetectionResponse,
    ImageProcessingResult,
    DetectionResult,
    BoundingBoxData,
    DatabaseRecords,
    
    # Database Models
    ImageRecord,
    DetectionRecord,
    ClassificationRecord,
    RipenessLevel,
    
    # Batch Processing
    BatchStatusResponse,
    BatchProcessingStatus,
    
    # Health and Info
    HealthStatus,
    ModelInfo,
    
    # Error Models
    APIError
)

# Legacy aliases for backward compatibility (if needed)
BoundingBox = BoundingBoxData
FruitDetection = DetectionResult
ImageResult = ImageProcessingResult
FruitDetectionSingleRequest = SingleImageDetectionRequest

# Re-export all models for easy importing
__all__ = [
    # Request Models
    "FruitDetectionRequest",
    "SingleImageDetectionRequest", 
    "ImageData",
    "DetectionOptions",
    
    # Response Models
    "FruitDetectionResponse",
    "ImageProcessingResult",
    "DetectionResult", 
    "BoundingBoxData",
    "DatabaseRecords",
    
    # Database Models
    "ImageRecord",
    "DetectionRecord",
    "ClassificationRecord",
    "RipenessLevel",
    
    # Batch Processing
    "BatchStatusResponse",
    "BatchProcessingStatus",
    
    # Health and Info  
    "HealthStatus",
    "ModelInfo",
    
    # Error Models
    "APIError",
    
    # Legacy aliases
    "BoundingBox",
    "FruitDetection", 
    "ImageResult",
    "FruitDetectionSingleRequest"
]