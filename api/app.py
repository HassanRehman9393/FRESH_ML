"""
FRESH ML API Application
========================

FastAPI application for fruit detection and classification.
Clean, structured API with async batch processing support.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import base64
import cv2
import numpy as np
import time
import asyncio
import uuid
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

# Import pipeline
from pipeline.predictor import FreshMLPredictor
from pipeline.pipeline_config import PipelineConfig

# Import schemas
from api.schemas.models import (
    FruitDetectionRequest,
    SingleImageDetectionRequest, 
    FruitDetectionResponse,
    BatchStatusResponse,
    BatchProcessingStatus,
    HealthStatus,
    ModelInfo,
    ImageProcessingResult,
    DetectionResult,
    BoundingBoxData,
    DatabaseRecords,
    ImageRecord,
    DetectionRecord,
    ClassificationRecord,
    RipenessLevel
)
from datetime import datetime
import uuid as uuid_lib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
predictor = None
batch_tasks: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global predictor
    try:
        logger.info("🚀 Starting FRESH ML API...")
        logger.info("📋 Loading ML models...")
        
        config = PipelineConfig()
        if not config.validate_models_exist():
            logger.error("❌ Model files not found!")
            raise RuntimeError("Model files not found")
        
        predictor = FreshMLPredictor(config)
        logger.info("✅ FRESH ML Pipeline initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FRESH ML API...")

# Initialize FastAPI app
app = FastAPI(
    title="FRESH ML API",
    description="Fruit Detection and Ripeness Classification API",
    version="1.0.0",
    lifespan=lifespan
)

# Health check endpoint
@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    return HealthStatus(
        status="healthy" if predictor is not None else "unhealthy",
        timestamp=datetime.now(),
        models_loaded=predictor is not None,
        version="1.0.0",
        database_compatible=True
    )

# Model information endpoint
@app.get("/api/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    model_info = predictor.get_model_info()
    
    return ModelInfo(
        yolo_model=model_info['yolo_model'],
        classification_model=model_info['classification_model'],
        device=model_info['device'],
        confidence_threshold=model_info['confidence_threshold'],
        iou_threshold=model_info['iou_threshold'],
        supported_fruits=model_info['yolo_model']['classes'],
        supported_ripeness_levels=[level.value for level in RipenessLevel]
    )

# Main fruit detection endpoint - File upload
@app.post("/api/detection/fruits", response_model=FruitDetectionResponse)
async def detect_fruits_upload(
    file: UploadFile = File(...),
    user_id: str = None,
    return_visualization: bool = False,
    confidence_threshold: float = None
):
    """
    Detect and classify fruits from uploaded image file
    
    Args:
        file: Uploaded image file (JPEG, PNG, BMP)
        user_id: User UUID for database association
        return_visualization: Whether to return annotated visualization
        confidence_threshold: Custom confidence threshold (optional)
    
    Returns:
        Fruit detection and classification results (database-ready)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Default user_id for testing if not provided
    if not user_id:
        user_id = "00000000-0000-0000-0000-000000000000"  # Default test user
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image file."
            )
        
        # Read and validate file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > 10:  # 10MB limit
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.1f}MB. Maximum size is 10MB."
            )
        
        logger.info(f"🖼️  Processing uploaded image: {file.filename} ({file_size_mb:.1f}MB) for user: {user_id}")
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")
        
        # Process through pipeline
        start_time = time.time()
        pipeline_result = predictor.predict(
            image=image,
            return_visualization=return_visualization,
            confidence_threshold=confidence_threshold
        )
        processing_time = time.time() - start_time
        
        # Convert to database-ready format
        image_result, database_records = convert_pipeline_result_to_database_format(
            pipeline_result, user_id, file.filename or "uploaded_image.jpg"
        )
        
        response = FruitDetectionResponse(
            success=True,
            timestamp=datetime.now(),
            processing_time=f"{processing_time:.2f}s",
            user_id=user_id,
            results=[image_result],
            database_records=database_records,
            errors=[]
        )
        
        logger.info(f"✅ Processing completed: {image_result.total_fruits_detected} fruits detected")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Base64 image detection endpoint
@app.post("/api/detection/fruits/base64", response_model=FruitDetectionResponse)
async def detect_fruits_base64(request: SingleImageDetectionRequest):
    """
    Detect and classify fruits from base64 encoded image
    
    Args:
        request: Request containing base64 encoded image
    
    Returns:
        Fruit detection and classification results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.info(f"🖼️  Processing base64 image ({len(image_bytes)/1024:.1f}KB) for user: {request.user_id}")
        
        # Process through pipeline
        start_time = time.time()
        pipeline_result = predictor.predict(
            image=image,
            return_visualization=request.return_visualization,
            confidence_threshold=request.confidence_threshold
        )
        processing_time = time.time() - start_time
        
        # Convert to database-ready format
        image_result, database_records = convert_pipeline_result_to_database_format(
            pipeline_result, request.user_id, request.image_name
        )
        
        response = FruitDetectionResponse(
            success=True,
            timestamp=datetime.now(),
            processing_time=f"{processing_time:.2f}s",
            user_id=request.user_id,
            results=[image_result],
            database_records=database_records,
            errors=[]
        )
        
        logger.info(f"✅ Processing completed: {image_result.total_fruits_detected} fruits detected")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Base64 processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Base64 processing failed: {str(e)}")

# Batch processing endpoint
@app.post("/api/detection/fruits/batch")
async def detect_fruits_batch(
    request: FruitDetectionRequest,
    background_tasks: BackgroundTasks
):
    """
    Process multiple images in batch (asynchronous)
    
    Args:
        request: Request containing multiple images
        background_tasks: FastAPI background tasks
    
    Returns:
        Task ID for tracking batch processing
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        batch_tasks[task_id] = {
            "status": BatchProcessingStatus.PENDING,
            "progress": 0.0,
            "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "completed_at": None,
            "total_images": len(request.images),
            "processed_images": 0,
            "results": [],
            "error_message": None
        }
        
        # Start background processing
        background_tasks.add_task(process_batch_images, task_id, request)
        
        logger.info(f"🚀 Started batch processing: {task_id} ({len(request.images)} images)")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"Batch processing started for {len(request.images)} images",
            "status_endpoint": f"/api/detection/batch/status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"❌ Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Batch status endpoint
@app.get("/api/detection/batch/status/{task_id}", response_model=BatchStatusResponse)
async def get_batch_status(task_id: str):
    """
    Get status of batch processing task
    
    Args:
        task_id: Task identifier
    
    Returns:
        Current status of batch processing
    """
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = batch_tasks[task_id]
    
    return BatchStatusResponse(
        task_id=task_id,
        status=task_data["status"],
        progress=task_data["progress"],
        started_at=task_data["started_at"],
        completed_at=task_data["completed_at"],
        total_images=task_data["total_images"],
        processed_images=task_data["processed_images"],
        results=task_data["results"] if task_data["status"] == BatchProcessingStatus.COMPLETED else None,
        error_message=task_data["error_message"]
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FRESH ML API",
        "version": "1.0.0",
        "description": "Fruit Detection and Ripeness Classification API",
        "endpoints": {
            "/api/health": "Health check",
            "/api/models/info": "Model information",
            "/api/detection/fruits": "Single image detection (file upload)",
            "/api/detection/fruits/base64": "Single image detection (base64)",
            "/api/detection/fruits/batch": "Batch image processing",
            "/docs": "API documentation"
        }
    }

# Helper functions
def convert_pipeline_result_to_database_format(
    pipeline_result: Dict[str, Any], 
    user_id: str,
    image_filename: str = "uploaded_image.jpg"
) -> tuple[ImageProcessingResult, DatabaseRecords]:
    """Convert pipeline result to database-ready format"""
    
    # Generate UUIDs
    image_id = str(uuid_lib.uuid4())
    
    # Convert detection results
    detection_results = []
    image_records = []
    detection_records = []
    classification_records = []
    
    # Create image record
    image_record = ImageRecord(
        id=image_id,
        user_id=user_id,
        file_path=f"uploads/{image_filename}",
        file_name=image_filename,
        metadata={
            "processing_info": pipeline_result.get('processing_info', {}),
            "image_dimensions": pipeline_result.get('image_info', {})
        }
    )
    image_records.append(image_record)
    
    # Process each detected fruit
    for fruit in pipeline_result.get('fruits', []):
        detection_id = str(uuid_lib.uuid4())
        
        # Extract bounding box data
        bbox_data = fruit.get('location', {}).get('bbox', [0, 0, 0, 0])
        center_data = fruit.get('location', {}).get('center', [0, 0])
        
        bounding_box = BoundingBoxData(
            x1=bbox_data[0],
            y1=bbox_data[1], 
            x2=bbox_data[2],
            y2=bbox_data[3],
            center_x=center_data[0],
            center_y=center_data[1],
            width=bbox_data[2] - bbox_data[0],
            height=bbox_data[3] - bbox_data[1]
        )
        
        # Map ripeness level to enum
        ripeness_str = fruit.get('ripeness_level', 'ripe').lower()
        ripeness_level = RipenessLevel.RIPE  # default
        if ripeness_str in ['unripe', 'green']:
            ripeness_level = RipenessLevel.UNRIPE
        elif ripeness_str in ['overripe', 'very_ripe']:
            ripeness_level = RipenessLevel.OVERRIPE
        elif ripeness_str in ['rotten', 'bad']:
            ripeness_level = RipenessLevel.ROTTEN
        
        # Create detection result
        detection_result = DetectionResult(
            fruit_type=fruit.get('fruit_type', ''),
            detection_confidence=fruit.get('confidence_scores', {}).get('detection', 0.0),
            bounding_box=bounding_box,
            ripeness_level=ripeness_level,
            classification_confidence=fruit.get('confidence_scores', {}).get('classification', 0.0),
            estimated_color=fruit.get('estimated_color'),
            estimated_size=fruit.get('estimated_size'),
            quality_score=fruit.get('quality_score')
        )
        detection_results.append(detection_result)
        
        # Create database records
        detection_record = DetectionRecord(
            detection_id=detection_id,
            user_id=user_id,
            image_id=image_id,
            fruit_type=fruit.get('fruit_type'),
            confidence=fruit.get('confidence_scores', {}).get('detection', 0.0),
            bounding_box={
                "x1": bbox_data[0], "y1": bbox_data[1],
                "x2": bbox_data[2], "y2": bbox_data[3],
                "center_x": center_data[0], "center_y": center_data[1],
                "width": bbox_data[2] - bbox_data[0],
                "height": bbox_data[3] - bbox_data[1]
            }
        )
        detection_records.append(detection_record)
        
        classification_record = ClassificationRecord(
            detection_id=detection_id,
            ripeness_level=ripeness_level,
            confidence_score=fruit.get('confidence_scores', {}).get('classification', 0.0),
            estimated_color=fruit.get('estimated_color'),
            estimated_size=fruit.get('estimated_size')
        )
        classification_records.append(classification_record)
    
    # Create image processing result
    processing_result = ImageProcessingResult(
        image_metadata={
            "image_id": image_id,
            "filename": image_filename,
            "processing_info": pipeline_result.get('processing_info', {})
        },
        total_fruits_detected=len(detection_results),
        detection_results=detection_results,
        processing_summary={
            "analysis_quality": pipeline_result.get('summary', {}).get('analysis_quality', 'good'),
            "average_confidence": pipeline_result.get('summary', {}).get('average_confidence', 0.0),
            "fruit_type_distribution": pipeline_result.get('summary', {}).get('fruit_type_distribution', {}),
            "ripeness_distribution": pipeline_result.get('summary', {}).get('ripeness_distribution', {})
        },
        visualization_available='annotated_image' in pipeline_result,
        visualization_path=pipeline_result.get('visualization_path')
    )
    
    # Create database records container
    database_records = DatabaseRecords(
        images=image_records,
        detections=detection_records,
        classifications=classification_records
    )
    
    return processing_result, database_records

async def process_batch_images(task_id: str, request: FruitDetectionRequest):
    """Background task to process batch images"""
    try:
        batch_tasks[task_id]["status"] = BatchProcessingStatus.PROCESSING
        
        results = []
        total_images = len(request.images)
        
        for i, image_data in enumerate(request.images):
            try:
                # Decode image
                image_bytes = base64.b64decode(image_data.data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Process through pipeline
                    pipeline_result = predictor.predict(
                        image=image,
                        return_visualization=request.options.return_visualization,
                        confidence_threshold=request.options.confidence_threshold
                    )
                    
                    # Convert to database-ready format
                    api_result, _ = convert_pipeline_result_to_database_format(
                        pipeline_result, request.user_id, f"batch_image_{image_data.id}.jpg"
                    )
                    results.append(api_result)
                
                # Update progress
                batch_tasks[task_id]["processed_images"] = i + 1
                batch_tasks[task_id]["progress"] = ((i + 1) / total_images) * 100
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error processing image {image_data.id}: {str(e)}")
                continue
        
        # Mark as completed
        batch_tasks[task_id]["status"] = BatchProcessingStatus.COMPLETED
        batch_tasks[task_id]["completed_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
        batch_tasks[task_id]["results"] = results
        batch_tasks[task_id]["progress"] = 100.0
        
        logger.info(f"✅ Batch processing completed: {task_id} ({len(results)} results)")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {task_id} - {str(e)}")
        batch_tasks[task_id]["status"] = BatchProcessingStatus.FAILED
        batch_tasks[task_id]["error_message"] = str(e)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FRESH ML API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🚀 Starting FRESH ML API Server...")
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )