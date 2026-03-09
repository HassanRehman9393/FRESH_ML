# YOLOv11 Fruit Detection Training Plan
**Date:** March 7, 2026  
**Dataset:** fruit-detection.v2i.yolov11 (3,950 images)  
**Model:** YOLOv11s (Small) - Recommended for production

---

## 📋 Executive Summary

**Objective:** Train a robust YOLOv11s model for 4-class fruit detection (grapefruit, guava, mango, orange) to replace the underperforming previous model.

**Key Challenges:**
- Moderate class imbalance (2.57:1 ratio)
- Mango underrepresented (14.07%)
- Orange overrepresented (36.16%)

**Strategy:** Use YOLOv11s with class weights, aggressive augmentation, and early stopping for balanced, production-ready performance.

---

## 🎯 Model Selection: YOLOv11s

### Why YOLOv11s over YOLOv11n?

| Aspect | YOLOv11n (Nano) | YOLOv11s (Small) | Decision |
|--------|-----------------|------------------|----------|
| **Parameters** | ~2.6M | ~9.4M | ✅ Small still lightweight |
| **Speed** | ~50 FPS | ~40 FPS | ✅ Small fast enough |
| **mAP (typical)** | ~37% | ~45% | ✅ **+8% accuracy** |
| **Small object detection** | Moderate | Good | ✅ Better for distant fruits |
| **Training time** | ~2-3 hrs | ~4-5 hrs | ✅ Acceptable for quality |
| **Production readiness** | Good | **Excellent** | ✅ **WINNER** |

**Verdict:** YOLOv11s provides significantly better accuracy for minimal speed trade-off. Critical for production where accuracy issues caused this retraining.

---

## 📊 Dataset Configuration

```yaml
Dataset: fruit-detection.v2i.yolov11
Classes: ['grapefruit', 'guava', 'mango', 'orange']
Class IDs: 0=grapefruit, 1=guava, 2=mango, 3=orange

Split Ratio:
  - Train: 2,765 images (70%)
  - Valid: 593 images (15%)
  - Test: 592 images (15%)

Image Size: 512x512 (preprocessed)
Format: YOLOv11 normalized bounding boxes
```

---

## ⚖️ Class Imbalance Handling

### Training Set Distribution
```
Orange:     4,058 instances (36.16%) ⚠️ OVERREPRESENTED
Guava:      3,151 instances (28.08%) ✅
Grapefruit: 2,435 instances (21.70%) ✅
Mango:      1,579 instances (14.07%) ⚠️ UNDERREPRESENTED

Imbalance Ratio: 2.57:1
```

### Solution: Computed Class Weights
```python
# Inverse frequency weighting
class_weights = {
    0: 4.61,  # grapefruit
    1: 3.56,  # guava
    2: 7.11,  # mango (BOOSTED - lowest class)
    3: 2.77   # orange (REDUCED - highest class)
}
```

**Impact:** Forces model to pay 2.5x more attention to mango detection, preventing orange bias.

---

## 🎨 Augmentation Strategy

### Critical Augmentations (Address Imbalance)
```python
mosaic = 1.0          # Mix 4 images - creates synthetic mango samples
mixup = 0.15          # Blend images - regularization
copy_paste = 0.1      # Copy-paste fruits between images
```

### Geometric Augmentations
```python
degrees = 10.0        # Rotation ±10° (fruits can be rotated)
translate = 0.1       # Translation 10% (camera angle variance)
scale = 0.5           # Scale 0.5-1.5x (distance variance)
shear = 2.0           # Shear transformation
fliplr = 0.5          # Horizontal flip (50% chance)
flipud = 0.0          # NO vertical flip (fruits don't grow upside down)
perspective = 0.0001  # Minimal perspective distortion
```

### Color Augmentations (Ripeness Variation)
```python
hsv_h = 0.015         # Hue shift (green ↔ yellow ripeness)
hsv_s = 0.7           # Saturation variation
hsv_v = 0.4           # Brightness (lighting conditions)
```

**Result:** ~3-4x effective dataset size, balanced mango representation.

---

## 🏋️ Training Configuration

### Hyperparameters
```python
Model: yolov11s.pt (pretrained COCO weights)
Epochs: 150
Patience: 25 (early stopping)
Batch Size: 16 (adjust based on GPU)
Image Size: 512 (matches preprocessing)
Device: CUDA (GPU required)

Optimizer: AdamW
Learning Rate: 0.001 (initial)
LR Scheduler: Cosine annealing with warmup
Warmup Epochs: 3
Weight Decay: 0.0005

Momentum: 0.937
```

### Loss Function Weights
```python
box_loss_gain = 0.05      # Bounding box regression
cls_loss_gain = 0.5       # Classification (increased for hard classes)
dfl_loss_gain = 1.5       # Distribution focal loss
```

### Performance Targets
```
Target mAP@0.5: ≥85%
Target mAP@0.5:0.95: ≥60%
Target Precision: ≥80% (all classes)
Target Recall: ≥75% (especially mango)
```

---

## 📈 Training Strategy

### Phase 1: Warmup (Epochs 1-3)
- Linear LR warmup from 0 to 0.001
- Frozen backbone layers (optional)
- Stabilize gradients

### Phase 2: Main Training (Epochs 4-100)
- Full model training
- Cosine LR decay
- Heavy augmentation
- Monitor validation mAP

### Phase 3: Fine-tuning (Epochs 100-150)
- Reduced augmentation
- Lower learning rate
- Focus on hard examples
- Early stopping active

### Early Stopping Criteria
- Patience: 25 epochs without improvement
- Metric: mAP@0.5:0.95 (more strict than mAP@0.5)
- Restore: Best weights from validation peak

---

## 🔍 Validation & Monitoring

### Real-time Metrics
```python
Metrics to Track:
  - mAP@0.5 (primary metric)
  - mAP@0.5:0.95 (strict metric)
  - Precision (per class)
  - Recall (per class)
  - Box loss
  - Class loss
  - DFL loss

Per-Class Monitoring:
  - Mango metrics (ensure not ignored)
  - Orange metrics (ensure not dominating)
  - Confusion matrix analysis
```

### Validation Frequency
- Every epoch
- Save best model (highest mAP@0.5:0.95)
- Generate plots every 10 epochs

---

## 💾 Model Export & Delivery

### Training Outputs
```
/kaggle/working/runs/detect/train/
  ├── weights/
  │   ├── best.pt          # Best validation mAP model ✅ PRODUCTION
  │   └── last.pt          # Last epoch checkpoint
  ├── results.csv          # Training metrics log
  ├── confusion_matrix.png # Per-class performance
  ├── results.png          # Loss/mAP curves
  ├── PR_curve.png         # Precision-Recall
  └── labels.jpg           # Label distribution visualization
```

### Production Model
```python
File: yolo_detection_best_v2.pt
Expected Size: ~20-25 MB
Format: PyTorch (.pt)
Classes: ['grapefruit', 'guava', 'mango', 'orange']
Input Size: 512x512 RGB
Output: Bounding boxes + confidence + class
```

---

## 🔄 Pipeline Integration Plan

### After Training Complete:

**Step 1:** Update class mapping in `yolo_detector.py`
```python
# OLD (current pipeline)
self.class_names = {
    0: "mango",
    1: "orange", 
    2: "guava",
    3: "grapefruit"
}

# NEW (match dataset)
self.class_names = {
    0: "grapefruit",
    1: "guava",
    2: "mango",
    3: "orange"
}
```

**Step 2:** Replace model file
```bash
cp yolo_detection_best_v2.pt FRESH_ML/models/yolo_detection_best.pt
```

**Step 3:** Test pipeline
```python
# Run test suite
pytest tests/test_yolo_detector.py
```

**Step 4:** Validation on real images
- Test with orchard images
- Verify all 4 classes detected correctly
- Check confidence thresholds

---

## 📊 Success Criteria

### Must-Have (Training)
- ✅ Training completes without errors
- ✅ mAP@0.5 ≥ 85% on validation set
- ✅ All 4 classes achieve ≥70% recall
- ✅ Mango class not ignored (≥70% precision)
- ✅ Model converges (loss plateaus)

### Must-Have (Production)
- ✅ Model loads in pipeline without errors
- ✅ Inference speed ≥10 FPS on CPU, ≥40 FPS on GPU
- ✅ No class mapping errors
- ✅ Accurate detection on test images
- ✅ Better performance than old model

### Nice-to-Have
- 🎯 mAP@0.5:0.95 ≥ 65%
- 🎯 Balanced precision across all classes (±5%)
- 🎯 <5% false positive rate
- 🎯 Robust to lighting variations

---

## ⏱️ Timeline

```
Total Estimated Time: 6-8 hours

Phase 1: Setup & Verification (30 min)
  - Upload dataset to Kaggle
  - Verify data integrity
  - Install dependencies

Phase 2: Training (4-6 hours)
  - Main training loop
  - Validation monitoring
  - Automatic checkpointing

Phase 3: Evaluation (30 min)
  - Analyze results
  - Export best model
  - Generate report

Phase 4: Download & Testing (1 hour)
  - Download trained model
  - Local inference testing
  - Pipeline integration
```

---

## 🚨 Risk Mitigation

### Risk 1: GPU Memory Overflow
**Mitigation:** Start with batch_size=16, reduce to 8 if OOM

### Risk 2: Mango Class Still Ignored
**Mitigation:** Class weights + mosaic augmentation specifically boost mango

### Risk 3: Overfitting (Small Dataset)
**Mitigation:** Heavy augmentation + early stopping + dropout

### Risk 4: Training Divergence
**Mitigation:** Warmup + gradient clipping + learning rate scheduling

### Risk 5: Class Mapping Confusion
**Mitigation:** Clear documentation + pipeline update checklist

---

## 📝 Pre-Training Checklist

- [ ] Dataset uploaded to Kaggle
- [ ] Dataset added as notebook input
- [ ] GPU accelerator enabled (P100 or T4)
- [ ] Internet enabled (for model download)
- [ ] Notebook runtime set to "Always save output"
- [ ] Class weights calculated correctly
- [ ] Augmentation configuration validated
- [ ] Training parameters reviewed

---

## 🎯 Next Steps

1. ✅ **Create Kaggle notebook** with all optimizations
2. ⏳ Upload dataset to Kaggle (if not already)
3. ⏳ Run training (4-6 hours)
4. ⏳ Download best.pt model
5. ⏳ Update pipeline class mapping
6. ⏳ Test on production data
7. ⏳ Deploy to FRESH ML API

---

**Ready for Training:** ✅ All parameters optimized for your specific dataset and production requirements.
