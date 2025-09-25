# FRESH ML - Multi-Fruit Dataset

**Unified Multi-Fruit Dataset for Object Detection and Classification**

This repository contains organized datasets for the FRESH ML project - a drone-based agricultural AI system for fruit detection, quality assessment, and classification across multiple fruit types.

## 📊 Dataset Overview

### **Total Dataset Size**
- **Mango Object Detection:** 4,941 images with YOLO annotations
- **Mango Classification:** 1,303 images across 5 ripeness classes
- **Grapefruit Detection:** 1,520 images with YOLO annotations  
- **Guava Detection:** 5,842 images (converted from classification)
- **Orange Detection:** 7,224 images (converted from classification)
- **Total Images: ~20,830 images across 4 fruit types**

### **Dataset Splits**
| Fruit Type | Train | Validation | Test | Total |
|------------|-------|------------|------|-------|
| **Mango Detection** | 3,458 | 741 | 742 | 4,941 |
| **Mango Classification** | 909 | 196 | 198 | 1,303 |
| **Grapefruit Detection** | 1,064 | 228 | 228 | 1,520 |
| **Guava Detection** | 4,089 | 876 | 877 | 5,842 |
| **Orange Detection** | 4,356 | 934 | 934 | 7,224 |

## 📁 Dataset Structure

```
data/unified/
├── mango/
│   ├── object_detection/          # YOLO Object Detection Dataset
│   │   ├── train/
│   │   │   ├── images/           # 3,458 training images
│   │   │   └── labels/           # YOLO format annotations (.txt)
│   │   ├── val/
│   │   │   ├── images/           # 741 validation images  
│   │   │   └── labels/           # YOLO format annotations (.txt)
│   │   ├── test/
│   │   │   ├── images/           # 742 test images
│   │   │   └── labels/           # YOLO format annotations (.txt)
│   │   └── yolo_config.yaml     # YOLO configuration file
│   └── classification/           # Ripeness Classification Dataset
│       ├── train/               # 909 training images (70%)
│       │   ├── ripe/            # 164 images
│       │   ├── overripe/        # 81 images  
│       │   ├── underripe/       # 166 images
│       │   ├── partially_ripe/  # 166 images
│       │   └── unripe/          # 166 images
│       ├── val/                 # 196 validation images (15%)
│       │   ├── ripe/            # 35 images
│       │   ├── overripe/        # 17 images
│       │   ├── underripe/       # 36 images
│       │   ├── partially_ripe/  # 36 images
│       │   └── unripe/          # 36 images
│       └── test/                # 198 test images (15%)
│           ├── ripe/            # 36 images
│           ├── overripe/        # 18 images
│           ├── underripe/       # 36 images
│           ├── partially_ripe/  # 36 images
│           └── unripe/          # 36 images
├── grapefruit/
│   └── object_detection/
│       ├── train/
│       │   ├── images/          # 1,064 training images
│       │   └── labels/          # YOLO format annotations
│       ├── val/
│       │   ├── images/          # 228 validation images
│       │   └── labels/          # YOLO format annotations
│       ├── test/
│       │   ├── images/          # 228 test images
│       │   └── labels/          # YOLO format annotations
│       └── yolo_config.yaml
├── guava/
│   └── object_detection/
│       ├── train/
│       │   ├── images/          # 4,089 training images
│       │   └── labels/          # YOLO format annotations
│       ├── val/
│       │   ├── images/          # 876 validation images
│       │   └── labels/          # YOLO format annotations
│       ├── test/
│       │   ├── images/          # 877 test images
│       │   └── labels/          # YOLO format annotations
│       └── yolo_config.yaml
└── orange/
    └── object_detection/
        ├── train/
        │   ├── images/          # 4,356 training images
        │   └── labels/          # YOLO format annotations
        ├── val/
        │   ├── images/          # 934 validation images
        │   └── labels/          # YOLO format annotations
        ├── test/
        │   ├── images/          # 934 test images
        │   └── labels/          # YOLO format annotations
        └── yolo_config.yaml
```

## 🎯 Object Detection Classes

### **Mango Ripeness Classes (Classification)**
1. **Ripe** - Perfect eating condition, golden-yellow color
2. **Overripe** - Past optimal ripeness, very soft texture
3. **Underripe** - Early ripening stage, mixed green-yellow
4. **Partially Ripe** - Mid-ripening stage, yellow-orange color
5. **Unripe** - Green mangoes, firm texture

### **Fruit Detection Classes (Object Detection)**
- **Mango** - General mango fruit detection
- **Grapefruit** - Citrus fruit detection  
- **Guava** - Tropical guava fruit detection
- **Orange** - Orange citrus fruit detection

The classification dataset includes 5 distinct ripeness stages:
## 📋 Data Specifications

### **Object Detection Format**
- **Format:** YOLO v8 compatible
- **Image Size:** Variable (will be resized to 640x640 for training)
- **Annotation Format:** `.txt` files with normalized coordinates
- **Classes:** 
  - Mango: 1 class (mango)
  - Grapefruit: 1 class (grapefruit)
  - Guava: 1 class (guava) 
  - Orange: 1 class (orange)
- **Label Format:** `class_id center_x center_y width height` (normalized 0-1)

### **Classification Format (Mango Only)**
- **Format:** Organized folder structure
- **Image Size:** Variable (will be resized to 224x224 for training)
- **Classes:** 5 ripeness classes
- **Naming Convention:** `{class}_{split}_{index:04d}.jpg`

## 🚀 Usage Examples

### **Multi-Fruit Object Detection Training**
```python
from ultralytics import YOLO

# Train on all fruits
fruits = ['mango', 'grapefruit', 'guava', 'orange']
models = {}

for fruit in fruits:
    print(f"Training {fruit} detection model...")
    model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt
    model.train(
        data=f'data/unified/{fruit}/object_detection/yolo_config.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name=f'{fruit}_detection'
    )
    models[fruit] = model

# Validate models
for fruit, model in models.items():
    results = model.val(data=f'data/unified/{fruit}/object_detection/yolo_config.yaml')
    print(f"{fruit} mAP@0.5: {results.box.map50}")
```

### **Mango Classification Training**
```python
# YOLO v8 training
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data/unified/mango_detection/data.yaml',
### **Mango Classification Training**
```python
# TensorFlow/Keras training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/unified/mango/classification/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_directory(
    'data/unified/mango/classification/val', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

## 📈 Performance Targets

Based on the PRD requirements:

### **Multi-Fruit Object Detection**
- **Target mAP:** >95% per fruit type
- **Inference Time:** <2 seconds per image
- **Model Size:** <100MB per fruit model
- **Combined Model:** Multi-class detection across all 4 fruit types

### **Mango Classification**
- **Target Accuracy:** >95%
- **Inference Time:** <1 second per image
- **Classes:** 5 ripeness stages

## 📊 Dataset Quality

### **Strengths**
✅ **Large Volume:** 20,830+ total images across all fruit types  
✅ **Multi-Fruit Coverage:** Comprehensive datasets for 4 different fruits  
✅ **Balanced Distribution:** Proper train/validation/test splits (70/15/15)  
✅ **Real Field Data:** Authentic orchard and agricultural conditions  
✅ **Multiple Sources:** Combined from 8+ different datasets (YOLO & classification)  
✅ **Format Consistency:** All datasets standardized to YOLO format  
✅ **Rich Annotations:** Quality bounding box labels and class information  

## 🔧 Data Preprocessing

Recommended preprocessing for training:

### **Multi-Fruit Object Detection**
```python
# Image augmentation for YOLO training
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

# Individual fruit model training
fruits = ['mango', 'grapefruit', 'guava', 'orange']
for fruit in fruits:
    # Each fruit has its own YOLO config file
    config_path = f'data/unified/{fruit}/object_detection/yolo_config.yaml'
```

### **Mango Classification**
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

1. **Multi-Fruit Detection:** Train separate YOLO models for each fruit type
2. **Combined Detection:** Explore multi-class detection across all fruits  
3. **Mango Classification:** Train ripeness classification model
4. **Pipeline Integration:** Combine detection + classification for mangoes
5. **Validation:** Test on real drone-captured images across all fruit types
6. **Model Optimization:** Quantization and optimization for edge deployment
7. **Performance Benchmarking:** Validate against PRD requirements

## 📝 Dataset Statistics Summary

### **Image Distribution by Fruit Type**
| Fruit Type | Detection Images | Classification Images | Total Images |
|------------|------------------|----------------------|--------------|
| **Mango** | 4,941 | 1,303 | **6,244** |
| **Grapefruit** | 1,520 | - | **1,520** |
| **Guava** | 5,842 | - | **5,842** |
| **Orange** | 7,224 | - | **7,224** |
| **TOTAL** | **19,527** | **1,303** | **20,830** |

### **Source Dataset Breakdown**
#### Mango Detection Sources:
- mangoes.v1i.yolov8-obb
- mangoes.v2i.yolov8-obb  
- mango.v1i.yolov8-obb
- Mango.v9i.yolov8-obb
- deepfruits-mango-xs1as-goqi.v2i.yolov8-obb

#### Other Fruit Sources:
- Grapefruit: Fresh and Rotten Fruits dataset (converted from classification)
- Guava: Fresh and Rotten Fruits dataset (converted from classification)  
- Orange: Fruits dataset (converted from classification)

### **Mango Classification Balance**
| Class | Train | Val | Test | Total | Percentage |
|-------|-------|-----|------|-------|------------|
| Ripe | 164 | 35 | 36 | 235 | 18.0% |
| Overripe | 81 | 17 | 18 | 116 | 8.9% |
| Underripe | 166 | 36 | 36 | 238 | 18.3% |
| Partially Ripe | 166 | 36 | 36 | 238 | 18.3% |
| Unripe | 166 | 36 | 36 | 238 | 18.3% |
| **TOTAL** | **909** | **196** | **198** | **1,303** | **100%** |

---

**Dataset Ready for Training!** 🚀  
*All datasets are properly organized, labeled, and split for immediate use in YOLO v8 training and TensorFlow/Keras classification.*

For questions or support, please refer to the FRESH ML PRD document or contact the development team.
| Unripe | 166 | 36 | 36 | 238 | 18.3% |
| Rotten | 81 | 17 | 18 | 116 | 8.9% |

**Note:** Rotten class has fewer samples - consider data augmentation or synthetic generation for better balance.

## 🏷️ License and Usage

This dataset is organized for the FRESH ML agricultural AI project. Ensure compliance with original dataset licenses when using for research or commercial purposes.

---

**Project:** FRESH ML - Agricultural AI System  
**Version:** 1.0  
**Last Updated:** September 2025