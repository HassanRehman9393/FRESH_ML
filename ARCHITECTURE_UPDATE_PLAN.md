# 🏗️ FRESH ML Architecture Update Plan
## Adding Leaf Detection for Complete Analytics

**Date:** October 23, 2025  
**Status:** Planning Phase  
**Priority:** HIGH

---

## 📊 Current State Analysis

### ✅ Models You Already Have:
1. **Fruit Object Detection (YOLO)** - 4 classes: mango, orange, guava, grapefruit
2. **Fruit Classification** - 3 classes: green, ripe, overripe
3. **Anthracnose Detection** - Binary: healthy vs diseased (handles fruits AND leaves)
4. **Citrus Canker Detection** - Binary: healthy vs diseased (handles fruits AND leaves)

### ❌ What's Missing:
- **Leaf Object Detection** - Cannot separate leaves from fruits in drone images
- **Analytics Gap** - Cannot report separate fruit vs leaf counts/health

### 🎯 PRD Requirements:
- ✅ Fruit detection & classification
- ✅ Disease detection (both fruits and leaves)
- ❌ **Separate analytics for fruits vs leaves** (MISSING)
- ❌ **Leaf object detection** (NEEDED for analytics)

---

## 💡 Recommended Solution: Unified YOLO + Smart Routing

### Architecture Overview:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRONE IMAGE INPUT                             │
│              (Can contain fruits + leaves together)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               STEP 1: UNIFIED OBJECT DETECTION                   │
│                      (Enhanced YOLO Model)                       │
│                                                                  │
│  Classes: ["mango", "orange", "guava", "grapefruit", "leaf"]   │
│                                                                  │
│  Output: Bounding boxes with labels                             │
│    Example: [                                                    │
│      {bbox: [10,20,50,60], class: "mango", conf: 0.92},        │
│      {bbox: [100,30,40,55], class: "leaf", conf: 0.88},        │
│      {bbox: [200,150,60,70], class: "orange", conf: 0.95}      │
│    ]                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
    ┌───────────────────────┐  ┌────────────────────┐
    │   FRUIT DETECTIONS    │  │  LEAF DETECTIONS   │
    │  (mango/orange/guava/ │  │     (leaf only)    │
    │      grapefruit)      │  │                    │
    └───────────────────────┘  └────────────────────┘
              ↓                          ↓
    ┌───────────────────────┐  ┌────────────────────┐
    │ STEP 2a: CLASSIFY     │  │ STEP 2b: SKIP      │
    │   FRUIT RIPENESS      │  │ (No leaf           │
    │                       │  │  classification    │
    │ Input: Fruit crop     │  │  needed)           │
    │ Model: Ripeness CNN   │  │                    │
    │ Output: green/ripe/   │  └────────────────────┘
    │         overripe      │            ↓
    └───────────────────────┘            │
              ↓                          │
              │                          │
    ┌─────────┴──────────────────────────┴─────────┐
    │         STEP 3: DISEASE DETECTION             │
    │                                               │
    │  Route based on fruit type:                   │
    │  ├─ Mango → Anthracnose model                │
    │  ├─ Orange/Grapefruit → Citrus Canker model │
    │  └─ Leaf → Both models (check both diseases) │
    │                                               │
    │  Models already handle fruits AND leaves!     │
    └───────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────┐
    │       STEP 4: AGGREGATION & ANALYTICS           │
    │                                                 │
    │  Combined Results:                              │
    │  {                                              │
    │    "summary": {                                 │
    │      "total_fruits": 12,                        │
    │      "total_leaves": 45,                        │
    │      "fruit_health_score": 85.3,               │
    │      "leaf_health_score": 72.1                 │
    │    },                                           │
    │    "fruits": [                                  │
    │      {                                          │
    │        "type": "mango",                         │
    │        "ripeness": "ripe",                      │
    │        "disease": "anthracnose",               │
    │        "disease_confidence": 0.91,             │
    │        "bbox": [10, 20, 50, 60]                │
    │      },                                         │
    │      ...                                        │
    │    ],                                           │
    │    "leaves": [                                  │
    │      {                                          │
    │        "disease": "healthy",                   │
    │        "disease_confidence": 0.88,             │
    │        "bbox": [100, 30, 40, 55]               │
    │      },                                         │
    │      ...                                        │
    │    ]                                            │
    │  }                                              │
    └─────────────────────────────────────────────────┘
```

---

## 🎯 Implementation Plan

### Phase 1: Prepare Unified YOLO Dataset (Week 1)

**Goal:** Create a dataset with 5 classes: 4 fruits + 1 leaf

**Tasks:**
1. **Gather leaf images with annotations**
   - Use existing leaf images from Anthracnose/Citrus Canker datasets
   - Annotate with bounding boxes (class: "leaf")
   - Aim for ~2000-3000 leaf images

2. **Combine with existing fruit dataset**
   - Your current fruit YOLO dataset (4 classes)
   - Add leaf class as class_id: 4

3. **Dataset structure:**
   ```
   unified_fruit_leaf_dataset/
   ├── train/
   │   ├── images/
   │   │   ├── fruit_img_001.jpg
   │   │   ├── leaf_img_001.jpg
   │   │   └── mixed_img_001.jpg  # Has both fruits and leaves
   │   └── labels/
   │       ├── fruit_img_001.txt  # 0 0.5 0.3 0.2 0.15  (mango)
   │       ├── leaf_img_001.txt   # 4 0.7 0.6 0.1 0.2  (leaf)
   │       └── mixed_img_001.txt  # Multiple lines (fruits + leaves)
   ├── val/
   └── test/
   ```

4. **Class mapping:**
   ```yaml
   names:
     0: mango
     1: orange
     2: guava
     3: grapefruit
     4: leaf
   ```

### Phase 2: Train Unified YOLO Model (Week 1-2)

**Goal:** Train YOLO with 5 classes

**Script:** `scripts/train_unified_fruit_leaf_yolo.py` (see below)

**Expected Performance:**
- Fruit detection: mAP@50 > 0.85 (maintain current performance)
- Leaf detection: mAP@50 > 0.75 (acceptable for analytics)

### Phase 3: Update Pipeline Code (Week 2)

**Files to Update:**

1. **`pipeline/detection/yolo_detector.py`**
   - Add "leaf" to class_names
   - Update detection logic to return object type

2. **`pipeline/predictor.py`**
   - Add leaf processing branch
   - Skip classification for leaves
   - Route leaves to disease detection
   - Aggregate separate fruit/leaf results

3. **`pipeline/utils/postprocessor.py`**
   - Update analytics to show separate fruit/leaf counts
   - Calculate separate health scores
   - Format results for reporting

### Phase 4: Testing & Validation (Week 3)

**Test Cases:**
1. ✅ Image with only fruits → Same as current behavior
2. ✅ Image with only leaves → New leaf detection + disease check
3. ✅ Image with mixed fruit+leaf → Separate detection + proper routing
4. ✅ Analytics output → Shows separate counts and health scores

---

## 📋 Detailed Implementation

### 1. Training Script for Unified YOLO

Create: `scripts/train_unified_fruit_leaf_yolo.py`

```python
"""
Train Unified YOLO Model with Fruits + Leaves
==============================================

Trains YOLO model to detect:
- 4 fruit types: mango, orange, guava, grapefruit
- 1 object type: leaf

This enables separate analytics for fruits vs leaves in drone imagery.
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

# Configuration
CONFIG = {
    'model': 'yolov8m.pt',  # Medium model (good balance)
    'data': 'data/unified_fruit_leaf/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'workers': 4,
    'device': 0,  # GPU
    'name': 'unified_fruit_leaf_detection',
    'patience': 20,  # Early stopping
    'save': True,
    'save_period': 10,
}

# Data YAML content
DATA_YAML = """
path: ../data/unified_fruit_leaf  # Dataset root
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: mango
  1: orange
  2: guava
  3: grapefruit
  4: leaf
"""

def train_unified_yolo():
    # Create data.yaml
    data_yaml_path = Path('data/unified_fruit_leaf/data.yaml')
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(DATA_YAML)
    
    # Initialize YOLO
    model = YOLO(CONFIG['model'])
    
    # Train
    results = model.train(
        data=str(data_yaml_path),
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        workers=CONFIG['workers'],
        device=CONFIG['device'],
        name=CONFIG['name'],
        patience=CONFIG['patience'],
        save=CONFIG['save'],
        save_period=CONFIG['save_period'],
        verbose=True
    )
    
    # Validate
    metrics = model.val()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"mAP@50: {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")
    print(f"Best model: {model.trainer.best}")
    print("="*70)

if __name__ == "__main__":
    train_unified_yolo()
```

### 2. Updated Predictor Logic

Key changes in `pipeline/predictor.py`:

```python
def predict(self, image):
    # Step 1: Detect all objects (fruits + leaves)
    detections = self.yolo_detector.detect(image)
    
    # Separate fruits and leaves
    fruits = []
    leaves = []
    
    for detection in detections:
        if detection['class_name'] in ['mango', 'orange', 'guava', 'grapefruit']:
            fruits.append(detection)
        elif detection['class_name'] == 'leaf':
            leaves.append(detection)
    
    # Step 2: Process fruits (classification + disease)
    fruit_results = []
    for fruit in fruits:
        crop = self._extract_crop(image, fruit['bbox'])
        
        # Classify ripeness
        ripeness = self.ripeness_classifier.predict(crop)
        
        # Detect disease
        disease = self._detect_disease(crop, fruit['class_name'])
        
        fruit_results.append({
            'type': fruit['class_name'],
            'ripeness': ripeness,
            'disease': disease,
            'bbox': fruit['bbox'],
            'confidence': fruit['confidence']
        })
    
    # Step 3: Process leaves (skip classification, only disease)
    leaf_results = []
    for leaf in leaves:
        crop = self._extract_crop(image, leaf['bbox'])
        
        # Detect disease (check both models)
        disease = self._detect_leaf_disease(crop)
        
        leaf_results.append({
            'disease': disease,
            'bbox': leaf['bbox'],
            'confidence': leaf['confidence']
        })
    
    # Step 4: Aggregate results
    return {
        'summary': {
            'total_fruits': len(fruits),
            'total_leaves': len(leaves),
            'fruit_health_score': self._calculate_health_score(fruit_results),
            'leaf_health_score': self._calculate_health_score(leaf_results)
        },
        'fruits': fruit_results,
        'leaves': leaf_results,
        'visualization': self._create_visualization(image, fruit_results, leaf_results)
    }

def _detect_leaf_disease(self, crop):
    """Check leaf for both Anthracnose and Citrus Canker"""
    # Check both disease models
    anthracnose_result = self.anthracnose_detector.predict(crop)
    canker_result = self.citrus_canker_detector.predict(crop)
    
    # Return the disease with higher confidence
    if anthracnose_result['confidence'] > canker_result['confidence']:
        return anthracnose_result
    else:
        return canker_result
```

---

## ✅ Benefits of This Approach

### 1. **Minimal Training Required**
- Only need to train 1 unified YOLO model
- Reuse existing disease models as-is
- No new classification models needed

### 2. **Complete Analytics**
```json
{
  "summary": {
    "total_fruits": 15,
    "total_leaves": 42,
    "fruit_health_score": 87.3,
    "leaf_health_score": 68.5,
    "diseased_fruits": 2,
    "diseased_leaves": 13
  },
  "breakdown": {
    "fruits": {
      "mango": 8,
      "orange": 5,
      "guava": 2
    },
    "diseases": {
      "anthracnose_fruits": 1,
      "anthracnose_leaves": 8,
      "citrus_canker_fruits": 1,
      "citrus_canker_leaves": 5
    }
  }
}
```

### 3. **Drone Image Support**
- Handles mixed content (fruits + leaves in one image)
- Proper object separation for accurate counting
- Spatial relationships preserved

### 4. **Scalable Architecture**
- Easy to add more fruit types
- Easy to add more disease models
- Clean separation of concerns

---

## 🚧 Alternative: Don't Add Leaf Detection (NOT RECOMMENDED)

**If you choose NOT to add leaf detection:**

❌ **Limitations:**
- Cannot report leaf counts separately
- Cannot show leaf health score
- Analytics will be incomplete
- Cannot distinguish "20 leaves diseased" vs "20 fruits diseased"
- Drone images with leaves will be ignored/misclassified

❌ **Impact on Reporting:**
```
Current (without leaf detection):
"15 objects detected, 3 diseased"  ← What objects? Fruits? Leaves? Both?

Desired (with leaf detection):
"12 fruits detected (2 diseased), 45 leaves detected (13 diseased)"  ← Clear!
```

**Verdict:** NOT recommended. Analytics would be incomplete.

---

## 📊 Recommended Timeline

| Week | Task | Status |
|------|------|--------|
| Week 1 | Prepare unified dataset (fruits + leaves) | 🔴 TODO |
| Week 1-2 | Train unified YOLO model | 🔴 TODO |
| Week 2 | Update pipeline code (detection + predictor) | 🔴 TODO |
| Week 3 | Testing & validation | 🔴 TODO |
| Week 3 | Deploy updated model | 🔴 TODO |

**Total Time:** ~3 weeks for complete implementation

---

## 🎯 Next Steps

### Immediate Actions:
1. ✅ Review this architecture plan
2. 🔴 Gather/annotate leaf images for YOLO training
3. 🔴 Create `train_unified_fruit_leaf_yolo.py` script
4. 🔴 Start training unified YOLO model

### Questions to Answer:
- [ ] Do you have enough leaf images for YOLO training? (Need ~2000-3000)
- [ ] Do you have annotation tool for bounding boxes? (LabelImg, CVAT, Roboflow)
- [ ] What's your dataset hosting plan? (Local, Kaggle, Google Drive)

---

## 💬 Final Recommendation

**YES, add leaf detection using the Unified YOLO approach.**

**Why:**
✅ Enables complete analytics (separate fruit/leaf counts)  
✅ Handles drone images with mixed content  
✅ Minimal training effort (only 1 YOLO model)  
✅ Reuses existing disease models  
✅ Scalable and maintainable architecture  
✅ Meets PRD requirements for comprehensive reporting  

**Next:** Start with leaf image annotation → Train unified YOLO → Update pipeline code

---

**Questions? Need help with implementation? Let me know!** 🚀
