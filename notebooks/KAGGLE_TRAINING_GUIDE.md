# 🥭 Anthracnose Detection Model - Kaggle Training Guide

## 📋 Quick Reference

This guide will help you train the anthracnose detection model on Kaggle.

## 🚀 Steps to Run on Kaggle

### 1. **Prepare Your Dataset**

Upload your dataset to Kaggle as a dataset:

1. Compress your unified dataset:
   ```bash
   # On your local machine
   cd d:\FYP\FRESH_ML\data\unified
   # Compress anthracnose folder (and healthy folder if you have it)
   ```

2. Upload to Kaggle:
   - Go to kaggle.com/datasets
   - Click "New Dataset"
   - Upload the compressed file
   - Title: "Anthracnose Mango Disease Dataset"
   - Make it public or private

### 2. **Create a New Kaggle Notebook**

1. Go to kaggle.com/code
2. Click "New Notebook"
3. Choose "Notebook" (not Script)
4. Enable GPU:
   - Click "⚙️ Settings" (right sidebar)
   - Under "Accelerator", select "GPU T4 x2" or "GPU P100"
   - Click "Save"

### 3. **Add Your Dataset**

1. Click "+ Add Data" (right sidebar)
2. Search for your uploaded dataset
3. Click "Add"
4. Your dataset will be available at `/kaggle/input/your-dataset-name/`

### 4. **Upload the Notebook**

1. Click "File" → "Upload Notebook"
2. Select `anthracnose_detection_training.ipynb`
3. Wait for upload to complete

### 5. **Update Paths in Cell 3**

In the configuration cell, update:

```python
class Config:
    # Update this path to match your Kaggle dataset
    DATA_DIR = Path('/kaggle/input/anthracnose-mango-disease-dataset')
    # ... rest remains same
```

### 6. **Run All Cells**

1. Click "Run All" or press Shift+Enter for each cell
2. Monitor the training progress
3. Training will take ~2-4 hours depending on GPU

### 7. **Download Results**

After training completes:

1. Check `/kaggle/working/` for outputs
2. Download these files:
   - `anthracnose_detection_model.pth` (final model)
   - `best_model.pth` (best checkpoint)
   - `training_history.png`
   - `confusion_matrix.png`
   - `test_results.csv`

---

## 📊 Expected Results

With the current dataset (2,960 anthracnose images):

| Metric | Expected Value |
|--------|---------------|
| **Training Time** | 2-4 hours (GPU) |
| **Test Accuracy** | 85-95%* |
| **Precision** | 85-93% |
| **Recall** | 85-93% |
| **F1-Score** | 85-93% |

*Note: Accuracy depends on whether you have healthy images. Current dataset contains only anthracnose images.

---

## ⚠️ Important Notes

### 1. **Binary Classification Requires Healthy Images**

The current dataset only has anthracnose images. For binary classification (Healthy vs Anthracnose):

- You need to add healthy mango images
- Recommended: ~2,500-3,000 healthy images
- Structure them as:
  ```
  healthy/
  ├── train/images/healthy/
  ├── val/images/healthy/
  └── test/images/healthy/
  ```

### 2. **Single Class Classification (Anthracnose Only)**

If you want to train on anthracnose detection only:

In Cell 3, change:
```python
NUM_CLASSES = 1  # Instead of 2
```

And modify the model architecture accordingly.

### 3. **Dataset Imbalance**

Current dataset:
- **Leaves:** 86.6% (2,563 images)
- **Fruits:** 13.4% (397 images)

Consider:
- Training separate models for fruits vs leaves
- Using weighted loss function
- Collecting more fruit images

---

## 🔧 Troubleshooting

### Problem: "CUDA out of memory"

**Solution:**
```python
# Reduce batch size in Config
BATCH_SIZE = 16  # or even 8
```

### Problem: "Dataset path not found"

**Solution:**
```python
# Check your dataset path
!ls /kaggle/input/
# Update DATA_DIR in Config
```

### Problem: "Training too slow"

**Solution:**
- Ensure GPU is enabled (Settings → Accelerator → GPU)
- Reduce image size: `IMAGE_SIZE = 192` (instead of 224)
- Reduce epochs: `NUM_EPOCHS = 30`

### Problem: "Model overfitting"

**Solution:**
```python
# Increase dropout, weight decay
WEIGHT_DECAY = 1e-3  # Instead of 1e-4

# Add more augmentation in Cell 5
# Enable early stopping (already included)
```

---

## 🎯 Optimization Tips

### 1. **Use Mixed Precision Training**

Add to training loop:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

### 2. **Try Different Architectures**

In Cell 3, change:
```python
MODEL_NAME = 'efficientnet_b0'  # Faster, smaller
# or
MODEL_NAME = 'resnet34'  # Lighter than ResNet50
```

### 3. **Use Learning Rate Finder**

Add before training:
```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

### 4. **Enable Kaggle Persistence**

To save your work:
- Click "Save Version"
- Choose "Save & Run All"
- Your outputs will be preserved

---

## 📦 After Training - Integration

### 1. **Download Model to Local**

```bash
# Copy from Kaggle downloads to your project
move anthracnose_detection_model.pth d:\FYP\FRESH_ML\models\
```

### 2. **Load Model in Your API**

```python
import torch
from torchvision import models

# Load checkpoint
checkpoint = torch.load('models/anthracnose_detection_model.pth')

# Recreate model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
```

### 3. **Create Inference Pipeline**

See Cell 17 in the notebook for the `predict_image()` function.

---

## 📚 Additional Resources

### Kaggle-Specific Tips

1. **Internet Access:** Enable in Settings for downloading packages
2. **Session Time:** 12 hours max per session
3. **GPU Quota:** 30 hours/week (T4) or 30 hours/week (P100)
4. **Save Frequently:** Use "Save Version" to preserve progress

### Recommended Kaggle Datasets to Add

Search and add these for healthy images:
- "Mango Dataset"
- "Fruit Images"
- "Plant Disease Dataset"

### Alternative Models to Try

1. **EfficientNet-B3:** Better accuracy, slower
2. **MobileNet-V2:** Faster inference, mobile-friendly
3. **Vision Transformer:** State-of-the-art, needs more data

---

## 🎓 Training Checklist

- [ ] Dataset uploaded to Kaggle
- [ ] Notebook uploaded and paths updated
- [ ] GPU enabled in settings
- [ ] Healthy images added (if binary classification)
- [ ] Run all cells successfully
- [ ] Training completed without errors
- [ ] Test accuracy > 85%
- [ ] Downloaded all output files
- [ ] Model integrated into API
- [ ] Tested with real images

---

## 💡 Pro Tips

1. **Use Kaggle Datasets:** Find existing mango disease datasets
2. **Version Control:** Save notebook versions regularly
3. **Monitor Training:** Watch the loss curves in real-time
4. **Test Quickly:** Run with `NUM_EPOCHS=5` first to test
5. **Use TensorBoard:** Add TensorBoard logging for better visualization
6. **Ensemble Models:** Train multiple models and average predictions

---

## 🤝 Support

If you encounter issues:

1. Check Kaggle discussion forums
2. Review notebook outputs carefully
3. Check GPU quota remaining
4. Verify dataset structure matches expected format

---

**Happy Training! 🚀**

---

**Notebook:** `notebooks/anthracnose_detection_training.ipynb`  
**Created:** October 2025  
**For:** FRESH ML - Module 2 (Disease Detection)
