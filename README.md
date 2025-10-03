# FRESH ML - Unified Fruit Analysis System

**Comprehensive Multi-Fruit Dataset for Object Detection and Classification**

This repository contains organized datasets, trained models, and notebooks for the FRESH ML project - a drone-based agricultural AI system for fruit detection, quality assessment, and ripeness classification across multiple fruit types.

## 🏗️ **Repository Structure**

```
FRESH_ML/
├── 📁 models/                       # Production-ready trained models
│   ├── classification_best.pth     # Best fruit classification model (94MB)
│   └── yolo_detection_best.pt      # Best YOLO object detection model (6MB)
├── 📁 notebooks/                   # Jupyter notebooks for training & analysis
│   ├── classification_training.ipynb
│   ├── yolo_detection_training.ipynb
│   └── yolo_detection_training_kaggle.ipynb
├── 📁 scripts/                     # Python training scripts
│   ├── train_fruit_classification.py
│   └── train_multi_fruit_yolo.py
├── 📁 data/                        # Raw datasets (preserved structure)
│   └── unified/                    # Multi-fruit datasets
├── 📁 archive/                     # Old models, results, and backups
│   ├── fruit_classification_final.pth
│   ├── yolo_training_results/
│   └── yolo_training_results.zip
└── 📄 Documentation files          # README, PRD, guides
```

## 📊 Dataset Overview

### **Unified Datasets**
- **Multi-Fruit Detection:** 18,527 images with YOLO annotations (4 fruit classes)
- **Multi-Fruit Classification:** 8,460 images across 16 ripeness/variety classes
- **Total Images: 26,987 images across 4 fruit types**

### **Dataset Statistics**
| Dataset Type | Train | Validation | Test | Total |
|-------------|-------|------------|------|-------|
| **Object Detection** | 12,967 | 2,779 | 2,781 | 18,527 |
| **Classification** | 5,916 | 1,263 | 1,281 | 8,460 |
| **Combined Total** | 18,883 | 4,042 | 4,062 | **26,987** |

## 📁 Unified Dataset Structure

```
data/unified/
├── multi_fruit_detection/           # 4-Class Object Detection (18,527 images)
│   ├── train/                       # 12,967 training images
│   │   ├── images/                  # Training images
│   │   └── labels/                  # YOLO format annotations (.txt)
│   ├── val/                         # 2,779 validation images
│   │   ├── images/                  # Validation images
│   │   └── labels/                  # YOLO format annotations (.txt)
│   ├── test/                        # 2,781 test images
│   │   ├── images/                  # Test images
│   │   └── labels/                  # YOLO format annotations (.txt)
│   └── dataset.yaml                 # YOLO configuration file
├── fruit_classification/            # 16-Class Classification (8,460 images)
│   ├── train/                       # 5,916 training images
│   │   ├── mango_unripe/            # 166 images (class 0)
│   │   ├── mango_early_ripe/        # 166 images (class 1)
│   │   ├── mango_partially_ripe/    # 166 images (class 2)
│   │   ├── mango_ripe/              # 164 images (class 3)
│   │   ├── mango_rotten/            # 81 images (class 4)
│   │   ├── orange_unripe/           # 560 images (class 5)
│   │   ├── orange_ripe/             # 561 images (class 6)
│   │   ├── orange_rotten/           # 560 images (class 7)
│   │   ├── orange_general/          # 335 images (class 8)
│   │   ├── guava_unripe/            # 641 images (class 9)
│   │   ├── guava_ripe/              # 713 images (class 10)
│   │   ├── guava_overripe/          # 209 images (class 11)
│   │   ├── guava_rotten/            # 564 images (class 12)
│   │   ├── guava_general/           # 343 images (class 13)
│   │   ├── grapefruit_pink/         # 343 images (class 14)
│   │   └── grapefruit_white/        # 344 images (class 15)
│   ├── val/                         # 1,263 validation images (15%)
│   ├── test/                        # 1,281 test images (15%)
│   └── dataset.yaml                 # Classification configuration
└── mango_classification/            # Original Mango-Only Classification
    ├── train/                       # Original mango classification data
    ├── validation/                  # (kept for reference)
    ├── test/
    └── config.yaml
```

## 🎯 Class Definitions

### **Object Detection Classes (4 Classes)**
| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | mango | Mango fruit detection |
| 1 | grapefruit | Grapefruit detection |
| 2 | guava | Guava fruit detection |
| 3 | orange | Orange fruit detection |

### **Classification Classes (16 Classes)**
| Class ID | Class Name | Category | Description |
|----------|------------|----------|-------------|
| 0 | mango_unripe | Mango Ripeness | Green mangoes, firm texture |
| 1 | mango_early_ripe | Mango Ripeness | Early ripening stage, mixed green-yellow |
| 2 | mango_partially_ripe | Mango Ripeness | Mid-ripening stage, yellow-orange |
| 3 | mango_ripe | Mango Ripeness | Perfect eating condition, golden-yellow |
| 4 | mango_rotten | Mango Ripeness | Overripe/rotten condition |
| 5 | orange_unripe | Orange Ripeness | Unripe oranges |
| 6 | orange_ripe | Orange Ripeness | Ripe oranges |
| 7 | orange_rotten | Orange Ripeness | Rotten oranges |
| 8 | orange_general | Orange Ripeness | General oranges (no ripeness info) |
| 9 | guava_unripe | Guava Ripeness | Unripe guavas |
| 10 | guava_ripe | Guava Ripeness | Ripe guavas |
| 11 | guava_overripe | Guava Ripeness | Overripe guavas |
| 12 | guava_rotten | Guava Ripeness | Rotten guavas |
| 13 | guava_general | Guava Ripeness | General guavas (no ripeness info) |
| 14 | grapefruit_pink | Grapefruit Variety | Pink grapefruit variety |
| 15 | grapefruit_white | Grapefruit Variety | White grapefruit variety |
## 📋 Data Specifications

### **Object Detection Format**
- **Format:** YOLO v8 compatible
- **Image Size:** Variable (resized to 640x640 for training)
- **Annotation Format:** `.txt` files with normalized coordinates
- **Label Format:** `class_id center_x center_y width height` (normalized 0-1)
- **Classes:** 4 fruit types (mango=0, grapefruit=1, guava=2, orange=3)

### **Classification Format**
- **Format:** Organized folder structure by class
- **Image Size:** Variable (resized to 224x224 for training)
- **Classes:** 16 classes (ripeness levels and varieties)
- **Naming Convention:** `{class}_{split}_{index:05d}.{ext}`
- **Splits:** 70% train / 15% validation / 15% test

## 🚀 Quick Start

### **Object Detection Training**
```python
from ultralytics import YOLO

# Train unified 4-class detection model
model = YOLO('yolov8n.pt')
model.train(
    data='data/unified/multi_fruit_detection/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### **Classification Training**
```python
# Use PyTorch/TensorFlow for 16-class classification
# Dataset ready at: data/unified/fruit_classification/
# Training script: train_multi_fruit_yolo.py
```

## 📈 Performance Targets

- **Detection mAP:** >95% (4 fruit classes)
- **Classification Accuracy:** >95% (16 classes)  
- **Inference Time:** <2 seconds per image

## � Files

- `train_multi_fruit_yolo.py` - YOLO detection training script
- `create_unified_classification_dataset.py` - Dataset organization script
- `TRAINING_GUIDE.md` - Training instructions
- `requirements.txt` - Python dependencies

---

**FRESH ML - Agricultural AI System**  
*Unified datasets ready for immediate training*