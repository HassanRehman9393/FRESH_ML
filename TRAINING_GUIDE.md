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

*Note: Mango classification dataset (1,065 images) available for future ripeness analysis after color pipeline is built.*

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

### **Train Multi-Class YOLO Model**
```bash
python train_multi_fruit_yolo.py
```

**Configuration:**
- **Model:** YOLO v8 Nano
- **Image Size:** 640x640
- **Batch Size:** 16 (GPU) / 8 (CPU)  
- **Epochs:** 100 (early stopping)
- **Target:** >95% mAP@0.5
- **Time:** 2-4 hours (GPU) / 8-12 hours (CPU)

**Results saved in:** `runs/detect/multi_fruit_detection_v1/`

---

## 📋 **Performance Target**

**YOLO Detection:**
- **Overall mAP@0.5:** >95%
- **Inference Time:** <2 seconds
- **Model Size:** <100MB

---

## 📁 **File Structure**

```
FRESH_ML/
├── data/unified/
│   ├── multi_fruit_detection/     # 18,527 YOLO images
│   └── mango_classification/      # 1,065 images (for future use)
├── train_multi_fruit_yolo.py     # YOLO training script
├── requirements.txt              # Dependencies
└── TRAINING_GUIDE.md            # This guide
```

---

## ✅ **Success Criteria**

**Training Complete When:**
- [ ] YOLO mAP@0.5 > 95%
- [ ] Model trained successfully
- [ ] Ready for color analysis pipeline

**Training Data: 18,527 detection images**