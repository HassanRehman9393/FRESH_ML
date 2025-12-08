"""
Database-Aligned Schemas
========================

Pydantic models that align with the Supabase database structure.
These models ensure the API responses can be directly stored in the database.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid

# Enum for ripeness levels (matches database constraint)
class RipenessLevel(str, Enum):
    """Ripeness levels matching database constraints"""
    RIPE = "ripe"
    UNRIPE = "unripe" 
    OVERRIPE = "overripe"
    ROTTEN = "rotten"

# Enum for disease types
class DiseaseType(str, Enum):
    """Disease types for disease detection"""
    HEALTHY = "healthy"
    ANTHRACNOSE = "anthracnose"
    CITRUS_CANKER = "citrus_canker"
    BLACKSPOT = "blackspot"
    FRUITFLY = "fruitfly"
    UNKNOWN = "unknown"

# Severity levels for disease
class DiseaseSeverity(str, Enum):
    """Disease severity levels"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

# Database Models (matching Supabase schema)
class ImageRecord(BaseModel):
    """Image record matching the images table"""
    id: Optional[str] = Field(None, description="Image UUID (auto-generated)")
    user_id: str = Field(..., description="User UUID")
    file_path: str = Field(..., description="Image file path")
    file_name: str = Field(..., description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Image metadata")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

class DetectionRecord(BaseModel):
    """Detection result matching the detection_results table"""
    detection_id: Optional[str] = Field(None, description="Detection UUID (auto-generated)")
    user_id: str = Field(..., description="User UUID")
    image_id: str = Field(..., description="Image UUID")
    fruit_type: Optional[str] = Field(None, description="Detected fruit type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bounding_box: Dict[str, Any] = Field(..., description="Bounding box coordinates")
    created_at: Optional[datetime] = Field(None, description="Detection timestamp")

class ClassificationRecord(BaseModel):
    """Classification result matching the classification_results table"""
    classification_id: Optional[str] = Field(None, description="Classification UUID (auto-generated)")
    detection_id: str = Field(..., description="Detection UUID")
    ripeness_level: RipenessLevel = Field(..., description="Ripeness classification")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    estimated_color: Optional[str] = Field(None, description="Estimated fruit color")
    estimated_size: Optional[str] = Field(None, description="Estimated fruit size")
    created_at: Optional[datetime] = Field(None, description="Classification timestamp")

class DiseaseDetectionRecord(BaseModel):
    """Disease detection result matching the disease_detections table"""
    disease_detection_id: Optional[str] = Field(None, description="Disease detection UUID (auto-generated)")
    detection_id: str = Field(..., description="Detection UUID")
    disease_type: DiseaseType = Field(..., description="Detected disease type")
    is_diseased: bool = Field(..., description="Whether fruit is diseased")
    disease_confidence: float = Field(..., ge=0.0, le=1.0, description="Disease detection confidence")
    severity_level: Optional[str] = Field(None, description="Disease severity (mild, moderate, severe, critical)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    created_at: Optional[datetime] = Field(None, description="Disease detection timestamp")

# API Request Models
class ImageData(BaseModel):
    """Single image data for processing"""
    id: str = Field(..., description="Unique identifier for the image")
    data: str = Field(..., description="Base64 encoded image data")
    format: str = Field("jpg", description="Image format")

class DetectionOptions(BaseModel):
    """Detection processing options"""
    return_visualization: bool = Field(False, description="Return annotated visualization")
    confidence_threshold: Optional[float] = Field(None, ge=0.1, le=1.0, description="Custom confidence threshold")
    save_to_database: bool = Field(True, description="Whether to prepare data for database storage")

class FruitDetectionRequest(BaseModel):
    """Main detection request with user context"""
    user_id: str = Field(..., description="User UUID from authentication")
    images: List[ImageData] = Field(..., min_items=1, max_items=10, description="Images to process")
    options: DetectionOptions = Field(default_factory=DetectionOptions, description="Processing options")

class SingleImageDetectionRequest(BaseModel):
    """Single image detection request"""
    user_id: str = Field(..., description="User UUID from authentication")
    image_base64: str = Field(..., description="Base64 encoded image data")
    image_name: str = Field(..., description="Original image filename")
    return_visualization: bool = Field(False, description="Return annotated visualization")
    confidence_threshold: Optional[float] = Field(None, ge=0.1, le=1.0, description="Custom confidence threshold")

# API Response Models (Database-Ready)
class BoundingBoxData(BaseModel):
    """Bounding box data for database storage"""
    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate") 
    y2: int = Field(..., description="Bottom-right y coordinate")
    center_x: int = Field(..., description="Center x coordinate")
    center_y: int = Field(..., description="Center y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")

class DetectionResult(BaseModel):
    """Single fruit detection result (database-ready)"""
    fruit_type: str = Field(..., description="Detected fruit type (mango, orange, guava, grapefruit)")
    detection_confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBoxData = Field(..., description="Fruit location in image")
    
    # Classification results
    ripeness_level: RipenessLevel = Field(..., description="Classified ripeness level")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    
    # Disease detection results
    disease_type: Optional[DiseaseType] = Field(None, description="Detected disease type")
    disease_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Disease detection confidence")
    is_diseased: Optional[bool] = Field(None, description="Whether fruit is diseased")
    
    # Additional metadata
    estimated_color: Optional[str] = Field(None, description="Estimated fruit color")
    estimated_size: Optional[str] = Field(None, description="Estimated fruit size category")

class ImageProcessingResult(BaseModel):
    """Complete result for a single processed image"""
    # Image metadata (for images table)
    image_metadata: Dict[str, Any] = Field(..., description="Image metadata for database")
    
    # Processing results
    total_fruits_detected: int = Field(..., ge=0, description="Total number of fruits detected")
    detection_results: List[DetectionResult] = Field(default_factory=list, description="Individual fruit results")
    
    # Summary statistics
    processing_summary: Dict[str, Any] = Field(..., description="Processing summary statistics")
    
    # Visualization (optional)
    visualization_available: bool = Field(False, description="Whether visualization image is available")
    visualization_path: Optional[str] = Field(None, description="Path to visualization image if saved")
    visualization_base64: Optional[str] = Field(None, description="Base64 encoded annotated image with bounding boxes")

class FruitDetectionResponse(BaseModel):
    """Main API response (backend-friendly)"""
    success: bool = Field(..., description="Processing success status")
    timestamp: datetime = Field(..., description="Processing timestamp")
    processing_time: str = Field(..., description="Total processing time")
    
    # User context
    user_id: str = Field(..., description="User UUID")
    
    # Results for each processed image
    results: List["ImageProcessingResult"] = Field(..., description="Results for each image")
    
    # Database records (ready for insertion)
    database_records: "DatabaseRecords" = Field(..., description="Formatted records for database insertion")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Any processing errors")
    message: Optional[str] = Field(None, description="Additional message")

class DatabaseRecords(BaseModel):
    """Database-ready records for backend insertion"""
    images: List[ImageRecord] = Field(..., description="Image records for images table")
    detections: List[DetectionRecord] = Field(..., description="Detection records for detection_results table")
    classifications: List[ClassificationRecord] = Field(..., description="Classification records for classification_results table")
    disease_detections: List[DiseaseDetectionRecord] = Field(default_factory=list, description="Disease detection records for disease_detections table")

# Batch Processing Models
class BatchProcessingStatus(str, Enum):
    """Batch processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BatchStatusResponse(BaseModel):
    """Response for batch processing status"""
    task_id: str = Field(..., description="Task identifier")
    status: BatchProcessingStatus = Field(..., description="Current processing status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    user_id: str = Field(..., description="User UUID")
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    total_images: int = Field(..., ge=0, description="Total number of images")
    processed_images: int = Field(..., ge=0, description="Number of processed images")
    results: Optional[List[ImageProcessingResult]] = Field(None, description="Results if completed")
    database_records: Optional[DatabaseRecords] = Field(None, description="Database records if completed")
    error_message: Optional[str] = Field(None, description="Error message if failed")

# Health and Info Models
class HealthStatus(BaseModel):
    """API health status"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Current timestamp")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    version: str = Field(..., description="API version")
    database_compatible: bool = Field(True, description="Database schema compatibility")

class ModelInfo(BaseModel):
    """Information about loaded models"""
    yolo_model: Dict[str, Any] = Field(..., description="YOLO model information")
    classification_model: Dict[str, Any] = Field(..., description="Classification model information")
    disease_detection_model: Optional[Dict[str, Any]] = Field(None, description="Disease detection model information")
    device: str = Field(..., description="Processing device")
    confidence_threshold: float = Field(..., description="Default confidence threshold")
    iou_threshold: float = Field(..., description="IOU threshold for detection")
    supported_fruits: List[str] = Field(..., description="List of supported fruit types")
    supported_ripeness_levels: List[str] = Field(..., description="List of supported ripeness levels")
    supported_disease_types: List[str] = Field(default_factory=lambda: ["healthy", "anthracnose", "citrus_canker", "blackspot"], description="List of supported disease types")

# Citrus Blackspot Detection Models
class BlackspotDetectionRequest(BaseModel):
    """Request for citrus blackspot detection"""
    user_id: Optional[str] = Field(None, description="User UUID (optional for testing)")
    return_probabilities: bool = Field(True, description="Return class probabilities")
    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Confidence threshold for positive detection")

class BlackspotDetectionResult(BaseModel):
    """Citrus blackspot detection result"""
    success: bool = Field(..., description="Detection success status")
    disease_detected: bool = Field(..., description="Whether black spot disease is detected")
    prediction: str = Field(..., description="Prediction class (blackspot or healthy)")
    display_name: str = Field(..., description="User-friendly prediction name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    is_high_confidence: bool = Field(..., description="Whether confidence exceeds threshold")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    severity: str = Field(..., description="Disease severity level")
    severity_description: str = Field(..., description="Severity description")

class BlackspotDetectionResponse(BaseModel):
    """Response for citrus blackspot detection"""
    success: bool = Field(..., description="Overall processing success")
    timestamp: datetime = Field(..., description="Processing timestamp")
    processing_time: str = Field(..., description="Processing duration")
    user_id: str = Field(..., description="User UUID")
    filename: str = Field(..., description="Image filename")
    
    # Detection result
    result: BlackspotDetectionResult = Field(..., description="Detection result")
    
    # Model info
    model_info: Dict[str, Any] = Field(..., description="Model information")
    
    # Optional fields
    error: Optional[str] = Field(None, description="Error message if failed")
    recommendations: Optional[List[str]] = Field(None, description="Treatment recommendations")

# Disease Detection Response Models
class DiseaseDetectionResult(BaseModel):
    """Single disease detection result"""
    disease_type: DiseaseType = Field(..., description="Detected disease type")
    is_diseased: bool = Field(..., description="Whether disease is present")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Disease detection confidence")
    severity: Optional[DiseaseSeverity] = Field(None, description="Disease severity level")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    affected_area_percentage: Optional[float] = Field(None, description="Percentage of fruit affected")
    recommendations: Optional[List[str]] = Field(None, description="Treatment recommendations")

class DiseaseAnalysisResponse(BaseModel):
    """Response for disease detection analysis"""
    success: bool = Field(..., description="Processing success status")
    timestamp: datetime = Field(..., description="Processing timestamp")
    processing_time: str = Field(..., description="Total processing time")
    user_id: str = Field(..., description="User UUID")
    
    # Image info
    image_id: str = Field(..., description="Image UUID")
    filename: str = Field(..., description="Image filename")
    
    # Disease detection results
    disease_detected: bool = Field(..., description="Whether any disease was detected")
    disease_results: List[DiseaseDetectionResult] = Field(..., description="Disease detection results")
    
    # Summary statistics
    total_fruits_analyzed: int = Field(..., ge=0, description="Total number of fruits analyzed")
    total_diseased: int = Field(..., ge=0, description="Number of diseased fruits")
    total_healthy: int = Field(..., ge=0, description="Number of healthy fruits")
    
    # Disease distribution
    disease_distribution: Dict[str, int] = Field(..., description="Distribution of diseases detected")
    
    # Optional fields
    errors: List[str] = Field(default_factory=list, description="Any processing errors")
    message: Optional[str] = Field(None, description="Additional message")

# Error Models
class APIError(BaseModel):
    """Standardized API error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")

# Guava Fruitfly Detection Schemas
class GuavaFruitflyDetectionRequest(BaseModel):
    """Request model for guava fruitfly disease detection"""
    user_id: Optional[str] = Field(None, description="User UUID (optional)")
    return_probabilities: bool = Field(default=True, description="Return class probabilities")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")

class GuavaFruitflyDetectionResult(BaseModel):
    """Detection result for a single guava"""
    success: bool = Field(..., description="Detection success")
    disease_detected: bool = Field(..., description="Whether fruitfly disease was detected")
    prediction: str = Field(..., description="Predicted class (fruitfly or healthy)")
    display_name: str = Field(..., description="Display name for prediction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    is_high_confidence: bool = Field(..., description="Whether confidence is high (>= 0.9)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    severity: str = Field(..., description="Disease severity level")
    severity_description: str = Field(..., description="Description of severity")

class GuavaFruitflyDetectionResponse(BaseModel):
    """Complete response for guava fruitfly detection"""
    success: bool = Field(..., description="Processing success status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    processing_time: Optional[str] = Field(None, description="Total processing time")
    user_id: str = Field(..., description="User UUID")
    filename: str = Field(..., description="Image filename")
    
    # Detection result
    result: GuavaFruitflyDetectionResult = Field(..., description="Detection result")
    
    # Model information
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    
    # Recommendations
    recommendations: Optional[List[str]] = Field(None, description="Treatment/action recommendations")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if any")

    user_id: Optional[str] = Field(None, description="User UUID if available")