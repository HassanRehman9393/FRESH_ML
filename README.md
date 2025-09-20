# FRESH ML - Mango Dataset

**Unified Mango Dataset for Object Detection and Ripeness Classification**

This repository contains organized mango datasets for the FRESH ML project - a drone-based agricultural AI system for fruit detection, quality assessment, and ripeness classification.

## 📊 Dataset Overview

### **Total Dataset Size**
- **Object Detection:** 4,941 images with YOLO annotations
- **Classification:** 1,303 images across 5 ripeness classes
- **Total Images:** ~6,244 images

### **Dataset Splits**
| Dataset Type | Train | Validation | Test | Total |
|-------------|-------|------------|------|-------|
| **Object Detection** | 3,458 | 741 | 742 | 4,941 |
| **Classification** | 909 | 196 | 198 | 1,303 |

## 📁 Dataset Structure

```
data/unified/
├── mango_detection/           # YOLO Object Detection Dataset
│   ├── train/
│   │   ├── images/           # 3,458 training images
│   │   └── labels/           # YOLO format annotations (.txt)
│   ├── validation/
│   │   ├── images/           # 741 validation images
│   │   └── labels/           # YOLO format annotations (.txt)
│   ├── test/
│   │   ├── images/           # 742 test images
│   │   └── labels/           # YOLO format annotations (.txt)
│   └── data.yaml            # YOLO configuration file
│
└── mango_classification/      # Ripeness Classification Dataset
    ├── train/                # 909 training images (70%)
    │   ├── early_ripe/       # 166 images
    │   ├── partially_ripe/   # 166 images
    │   ├── ripe/             # 164 images
    │   ├── rotten/           # 81 images
    │   └── unripe/           # 166 images
    ├── validation/           # 196 validation images (15%)
    │   ├── early_ripe/       # 36 images
    │   ├── partially_ripe/   # 36 images
    │   ├── ripe/             # 35 images
    │   ├── rotten/           # 17 images
    │   └── unripe/           # 36 images
    ├── test/                 # 198 test images (15%)
    │   ├── early_ripe/       # 36 images
    │   ├── partially_ripe/   # 36 images
    │   ├── ripe/             # 36 images
    │   ├── rotten/           # 18 images
    │   └── unripe/           # 36 images
    └── config.yaml          # Classification configuration
```

## 🎯 Ripeness Classes

The classification dataset includes 5 distinct ripeness stages:

1. **Early Ripe** - Very early stage, mostly green
2. **Partially Ripe** - Starting to develop color, mixed green/yellow
3. **Ripe** - Fully ripe, ready for consumption
4. **Rotten** - Overripe/damaged fruit
5. **Unripe** - Completely unripe, hard and green

## 📋 Data Specifications

### **Object Detection Format**
- **Format:** YOLO v8 compatible
- **Image Size:** Variable (will be resized to 640x640 for training)
- **Annotation Format:** `.txt` files with normalized coordinates
- **Classes:** 1 class (mango)
- **Label Format:** `class_id center_x center_y width height` (normalized 0-1)

### **Classification Format**
- **Format:** Organized folder structure
- **Image Size:** Variable (will be resized to 224x224 for training)
- **Classes:** 5 ripeness classes
- **Naming Convention:** `{class}_{split}_{index:04d}.jpg`

## 🚀 Usage

### **For Object Detection Training**
```python
# YOLO v8 training
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data/unified/mango_detection/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### **For Classification Training**
```python
# TensorFlow/Keras training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/unified/mango_classification/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

## 📈 Performance Targets

Based on the PRD requirements:

### **Object Detection**
- **Target mAP:** >95%
- **Inference Time:** <2 seconds per image
- **Model Size:** <100MB

### **Classification**
- **Target Accuracy:** >95%
- **Inference Time:** <1 second per image
- **Classes:** 5 ripeness stages

## 📊 Dataset Quality

### **Strengths**
✅ **Large Volume:** Nearly 5,000 detection images, 1,300+ classification images  
✅ **Balanced Classes:** Good distribution across ripeness stages  
✅ **Real Field Data:** Authentic mango orchard conditions  
✅ **Multiple Sources:** Combined from 5 different YOLO datasets  
✅ **Proper Splits:** 70/15/15 train/validation/test distribution  

## 🔧 Data Preprocessing

Recommended preprocessing for training:

### **Object Detection**
```python
# Image augmentation for YOLO
augmentations = [
    {'mosaic': 1.0},
    {'mixup': 0.1},
    {'hsv_h': 0.015},
    {'hsv_s': 0.7},
    {'hsv_v': 0.4},
    {'degrees': 0.0},
    {'translate': 0.1},
    {'scale': 0.5},
    {'shear': 0.0},
    {'perspective': 0.0},
    {'flipud': 0.0},
    {'fliplr': 0.5}
]
```

### **Classification**
```python
# Image augmentation for classification
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

## 🎯 Next Steps

1. **Model Training:** Begin with object detection model training
2. **Classification Training:** Train ripeness classification model
3. **Integration:** Combine detection + classification pipeline
4. **Validation:** Test on real drone-captured images
5. **Optimization:** Model quantization and optimization for deployment

## 📝 Dataset Statistics

### **Image Distribution by Source**
- **mangoes.v1i.yolov8-obb:** Contributing to detection dataset
- **mangoes.v2i.yolov8-obb:** Contributing to detection dataset  
- **mango.v1i.yolov8-obb:** Contributing to detection dataset
- **Mango.v9i.yolov8-obb:** Contributing to detection dataset
- **deepfruits-mango-xs1as-goqi.v2i.yolov8-obb:** Contributing to detection dataset
- **Archive Dataset:** Source for classification data

### **Class Balance (Classification)**
| Class | Train | Val | Test | Total | Percentage |
|-------|-------|-----|------|-------|------------|
| Early Ripe | 166 | 36 | 36 | 238 | 18.3% |
| Partially Ripe | 166 | 36 | 36 | 238 | 18.3% |
| Ripe | 164 | 35 | 36 | 235 | 18.0% |
| Unripe | 166 | 36 | 36 | 238 | 18.3% |
| Rotten | 81 | 17 | 18 | 116 | 8.9% |

**Note:** Rotten class has fewer samples - consider data augmentation or synthetic generation for better balance.

## 🏷️ License and Usage

This dataset is organized for the FRESH ML agricultural AI project. Ensure compliance with original dataset licenses when using for research or commercial purposes.

---

**Project:** FRESH ML - Agricultural AI System  
**Version:** 1.0  
**Last Updated:** September 2025