# FRESH ML - Training Guide & Dataset Summary

## 📊 **Dataset Overview**

### **Multi-Class Detection Dataset**
- **Location:** `data/unified/multi_fruit_detection/`
- **Total Images:** 18,527 images
- **Classes:** 4 fruit types

| Fruit Type | Class ID | Train | Val | Test | Total |
|------------|----------|--------|-----|------|-------|
| 🥭 **Mango** | 0 | 3,458 | 741 | 742 | **4,941** |
| 🍊 **Grapefruit** | 1 | 1,064 | 228 | 228 | **1,520** |
| 🍈 **Guava** | 2 | 4,089 | 876 | 877 | **5,842** |
| 🍊 **Orange** | 3 | 4,356 | 934 | 934 | **6,224** |
| **TOTALS** | - | **12,967** | **2,779** | **2,781** | **18,527** |

### **Multi-Class Classification Dataset**
- **Location:** `data/unified/fruit_classification/`
- **Total Images:** 8,460 images
- **Classes:** 16 fruit types and ripeness levels

| Fruit | Classes | Train | Val | Test | Total |
|-------|---------|-------|-----|------|-------|
| 🥭 **Mango** | 5 classes (0-4) | 743 | 160 | 162 | **1,065** |
| 🍊 **Orange** | 4 classes (5-8) | 2,016 | 431 | 434 | **2,881** |
| 🍈 **Guava** | 5 classes (9-13) | 2,470 | 527 | 535 | **3,532** |
| 🍊 **Grapefruit** | 2 classes (14-15) | 687 | 146 | 149 | **982** |
| **TOTALS** | **16 classes** | **5,916** | **1,264** | **1,280** | **8,460** |

---

## 🛠️ **Setup Instructions**

### **1. Environment Setup**
```bash
# Create virtual environment
python -m venv fresh_ml_env

# Activate environment (Windows)
fresh_ml_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### **2. GPU Setup (Recommended)**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 **Training Process**

### **1. Train Multi-Class YOLO Detection**
```bash
python train_multi_fruit_yolo.py
```

**Configuration:**
- **Model:** YOLO v8 Nano
- **Image Size:** 640x640
- **Batch Size:** 16 (GPU) / 8 (CPU)  
- **Epochs:** 100 (early stopping)
- **Target:** >95% mAP@0.5
- **Results:** `runs/detect/multi_fruit_detection_v1/`

**⏱️ Training Time:**
- **Local GPU:** ~2-3 hours (RTX 3060/4060)
- **Google Colab (T4):** ~1.5-2 hours (faster GPU)
- **Google Colab Pro (V100/A100):** ~45-60 minutes (premium GPU)
- **Without GPU:** ~12-15 hours (CPU only)

### **2. Train Multi-Class Classification**
```bash
python train_fruit_classification.py
```

**Configuration:**
- **Model:** ResNet50 with transfer learning
- **Image Size:** 224x224
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Learning Rate:** 0.001 with StepLR scheduler
- **Data Augmentation:** Rotation, flip, color jitter, affine transforms
- **Target:** >95% accuracy
- **Results:** `best_fruit_classification_model.pth` and `fruit_classification_final.pth`

**⏱️ Training Time:**
- **Local GPU:** ~1-2 hours (RTX 3060/4060)
- **Google Colab (T4):** ~45-60 minutes (faster GPU)
- **Google Colab Pro (V100/A100):** ~25-35 minutes (premium GPU)
- **Without GPU:** ~8-10 hours (CPU only)

---

## 📋 **Performance Targets**

**YOLO Detection:** >95% mAP@0.5, <2s inference  
**Classification:** >95% accuracy, <1s inference  
**Model Size:** <100MB each

---

## 📁 **File Structure**

```
FRESH_ML/
├── data/unified/
│   ├── multi_fruit_detection/         # 18,527 YOLO detection images
│   ├── fruit_classification/          # 8,460 classification images
│   └── mango_classification/          # Original mango data (reference)
├── train_multi_fruit_yolo.py         # YOLO detection training (single file)
├── train_fruit_classification.py     # CNN classification training (single file)
├── create_unified_classification_dataset.py  # Dataset creation script
├── requirements.txt                   # Dependencies
├── README.md                          # Project overview
└── TRAINING_GUIDE.md                 # This guide
```

---

## 📦 **Model Outputs**

### **Detection Model Outputs:**
- `runs/detect/multi_fruit_detection_v1/weights/best.pt` - Best YOLO model
- `runs/detect/multi_fruit_detection_v1/weights/last.pt` - Final YOLO model
- Training metrics and validation plots

### **Classification Model Outputs:**
- `best_fruit_classification_model.pth` - Best classification model
- `fruit_classification_final.pth` - Final model with metadata
- `confusion_matrix.png` - Classification performance visualization
- `training_history.png` - Training/validation curves

---

## 🔗 **Next Steps: Model Integration**

After both models are trained:

1. **Create Combined Pipeline**
   - YOLO detects fruit locations
   - CNN classifies each detected fruit
   - Combine results for complete analysis

2. **Model Deployment**
   - Package both models together
   - Create inference pipeline
   - Optimize for real-time performance

3. **Performance Testing**
   - End-to-end accuracy validation
   - Speed benchmarking
   - Mobile deployment preparation

---

## ✅ **Success Criteria**

**Training Complete When:**
- [ ] YOLO Detection mAP@0.5 > 95%
- [ ] Classification Accuracy > 95%
- [ ] Models ready for pipeline integration

**Total Training Data: 26,987 images**