# FRESH ML - Fruit Detection and Classification API

Professional REST API for fruit detection and ripeness classification using YOLOv11s and ResNet50 models.

## Overview

FRESH ML provides automated fruit detection and ripeness classification capabilities through a production-ready FastAPI server. The system processes images to identify fruits and classify their ripeness levels with database-ready response formats.

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB RAM minimum
- 2GB free disk space

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HassanRehman9393/FRESH_ML.git
   cd FRESH_ML
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   Ensure the following models exist in the `models/` directory:
   - `yolov11s_best.pt` (19.2MB) - YOLOv11s fruit detection model
   - `classification_best_fixed.pth` (94MB) - Updated model with correct class mappings

## Running the API

### Start the server
```bash
python main.py --host 127.0.0.1 --port 8000
```

### Available options
```bash
python main.py --help
```
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 8000)
- `--log-level`: Logging level (default: info)

### Health check
Once running, verify the API is working:
```
GET http://127.0.0.1:8000/api/health
```

## API Endpoints

### Fruit Detection and Classification
- **POST** `/api/detection/fruits` - Upload image file
- **POST** `/api/detection/fruits/base64` - Base64 encoded image
- **POST** `/api/detection/fruits/batch` - Multiple images (async)

### System Information
- **GET** `/api/health` - Health status
- **GET** `/api/models/info` - Model information
- **GET** `/docs` - Interactive API documentation

## Model Performance

### Object Detection Model (YOLOv11s)
- **Model Size**: 19.2MB
- **Parameters**: 9.4M
- **Classes**: 4 (grapefruit, guava, mango, orange)
- **mAP@0.5**: 79.1%
- **mAP@0.5:0.95**: 55.4%
- **Per-Class mAP@0.5**:
  - Mango: 96.7%
  - Grapefruit: 78.0%
  - Orange: 76.4%
  - Guava: 65.2%
- **Training**: 147 epochs planned, early stopped at 122
- **Processing Speed**: ~50ms per image
- **Input Size**: 512×512 pixels

### Classification Model (ResNet50)
- **Model Size**: 94MB
- **Classes**: 16 ripeness levels across fruit types
- **Accuracy**: 96.8%
- **Processing Speed**: ~100ms per image
- **Input Size**: 224×224 pixels

### Combined System Performance
- **Total Processing Time**: ~8-10 seconds for 11 fruits
- **Memory Usage**: ~1.5GB GPU memory
- **Throughput**: 6-8 images per minute
- **Confidence Threshold**: 0.35 (configurable)

## Response Format

The API returns database-ready responses with the following structure:

```json
{
  "success": true,
  "timestamp": "2025-10-04T12:00:00",
  "processing_time": "8.88s",
  "user_id": "user-uuid",
  "results": [{
    "total_fruits_detected": 11,
    "detection_results": [{
      "fruit_type": "mango",
      "detection_confidence": 0.87,
      "ripeness_level": "ripe",
      "classification_confidence": 0.92,
      "bounding_box": {
        "x1": 100, "y1": 150,
        "x2": 200, "y2": 250
      }
    }]
  }],
  "database_records": {
    "images": [...],
    "detections": [...],
    "classifications": [...]
  }
}
```

## Architecture

```
FRESH_ML/
├── api/                    # FastAPI application
│   ├── app.py             # Main API server
│   └── schemas/           # Request/response models
├── pipeline/              # ML processing pipeline
│   ├── detection/         # YOLO fruit detection
│   ├── classification/    # ResNet ripeness classification
│   └── utils/            # Image processing utilities
├── models/               # Trained model files
├── main.py              # Server entry point
└── requirements.txt     # Dependencies
```

## Usage Example

```python
import requests
import base64

# Read image file
with open("fruit_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# API request
response = requests.post("http://127.0.0.1:8000/api/detection/fruits/base64", 
    json={
        "user_id": "your-user-id",
        "image_base64": image_data,
        "image_name": "fruit_image.jpg",
        "return_visualization": False,
        "confidence_threshold": 0.35
    }
)

result = response.json()
print(f"Detected {result['results'][0]['total_fruits_detected']} fruits")
```

## Production Deployment

For production environments:
1. Use WSGI server (gunicorn/uvicorn)
2. Configure reverse proxy (nginx)
3. Set up SSL certificates
4. Implement proper logging and monitoring
5. Use database connection pooling

## License

This project is licensed under the MIT License.