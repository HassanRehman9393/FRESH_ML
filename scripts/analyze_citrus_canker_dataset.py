"""
Comprehensive analysis of Citrus Canker dataset
Analyzes all subdirectories and provides detailed statistics
"""

import os
from pathlib import Path
from collections import defaultdict

# Base directory
BASE_DIR = Path(r'd:\FYP\FRESH_ML\Citrus Canker')

def count_images(directory):
    """Count all image files in a directory"""
    if not directory.exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = 0
    for file in directory.rglob('*'):
        if file.is_file() and file.suffix in image_extensions:
            count += 1
    return count

def analyze_citrus_canker_dataset():
    """Analyze complete Citrus Canker dataset structure"""
    
    print("=" * 80)
    print("🍊 CITRUS CANKER DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    total_stats = {
        'canker_fruits': 0,
        'healthy_fruits': 0,
        'canker_leaves': 0,
        'healthy_leaves': 0,
        'mixed_or_unknown': 0
    }
    
    # ==================== CANKER FRUIT ANALYSIS ====================
    print("\n" + "=" * 80)
    print("🍎 CANKER FRUIT FOLDER ANALYSIS")
    print("=" * 80)
    
    # 1. canker fruit.v5i.yolov11 (YOLO format - object detection)
    print("\n📦 1. canker fruit.v5i.yolov11 (YOLO Object Detection)")
    print("    Purpose: Object detection dataset (blackspot, citrus-canker, greening)")
    yolo_fruit_dir = BASE_DIR / 'Canker fruit' / 'canker fruit.v5i.yolov11'
    yolo_fruit_train = count_images(yolo_fruit_dir / 'train' / 'images')
    yolo_fruit_valid = count_images(yolo_fruit_dir / 'valid' / 'images')
    yolo_fruit_test = count_images(yolo_fruit_dir / 'test' / 'images')
    yolo_fruit_total = yolo_fruit_train + yolo_fruit_valid + yolo_fruit_test
    
    print(f"    Train:      {yolo_fruit_train:,} images")
    print(f"    Valid:      {yolo_fruit_valid:,} images")
    print(f"    Test:       {yolo_fruit_test:,} images")
    print(f"    Total:      {yolo_fruit_total:,} images")
    print(f"    Note:       Mixed diseased fruits (NOT binary classification)")
    
    # 2. Fruits folder (Binary classification ready)
    print("\n📦 2. Fruits folder (Classification Ready)")
    fruits_dir = BASE_DIR / 'Canker fruit' / 'Fruits'
    canker_fruits = count_images(fruits_dir / 'Canker')
    healthy_fruits = count_images(fruits_dir / 'healthy')
    
    print(f"    Canker:     {canker_fruits:,} images ✅")
    print(f"    Healthy:    {healthy_fruits:,} images ✅")
    print(f"    Total:      {canker_fruits + healthy_fruits:,} images")
    print(f"    Note:       Ready for binary classification!")
    
    total_stats['canker_fruits'] += canker_fruits
    total_stats['healthy_fruits'] += healthy_fruits
    total_stats['mixed_or_unknown'] += yolo_fruit_total
    
    # ==================== CANKER LEAVES ANALYSIS ====================
    print("\n" + "=" * 80)
    print("🍃 CANKER LEAVES FOLDER ANALYSIS")
    print("=" * 80)
    
    # 1. canker.v4i.yolov11 (YOLO format - object detection)
    print("\n📦 1. canker.v4i.yolov11 (YOLO Object Detection)")
    print("    Purpose: Canker lesion detection on leaves")
    yolo_leaves_dir = BASE_DIR / 'canker leaves' / 'canker.v4i.yolov11'
    yolo_leaves_train = count_images(yolo_leaves_dir / 'train' / 'images')
    yolo_leaves_valid = count_images(yolo_leaves_dir / 'valid' / 'images')
    yolo_leaves_test = count_images(yolo_leaves_dir / 'test' / 'images')
    yolo_leaves_total = yolo_leaves_train + yolo_leaves_valid + yolo_leaves_test
    
    print(f"    Train:      {yolo_leaves_train:,} images")
    print(f"    Valid:      {yolo_leaves_valid:,} images")
    print(f"    Test:       {yolo_leaves_test:,} images")
    print(f"    Total:      {yolo_leaves_total:,} images")
    print(f"    Note:       Only canker class (no healthy for binary classification)")
    
    # 2. Leaves folder (Binary classification ready)
    print("\n📦 2. Leaves folder (Classification Ready)")
    leaves_dir = BASE_DIR / 'canker leaves' / 'Leaves'
    canker_leaves_1 = count_images(leaves_dir / 'canker')
    healthy_leaves_1 = count_images(leaves_dir / 'healthy')
    
    print(f"    Canker:     {canker_leaves_1:,} images ✅")
    print(f"    Healthy:    {healthy_leaves_1:,} images ✅")
    print(f"    Total:      {canker_leaves_1 + healthy_leaves_1:,} images")
    print(f"    Note:       Ready for binary classification!")
    
    total_stats['canker_leaves'] += canker_leaves_1
    total_stats['healthy_leaves'] += healthy_leaves_1
    
    # 3. Sweetorange folder
    print("\n📦 3. Sweetorange folder (Large Dataset)")
    sweetorange_dir = BASE_DIR / 'canker leaves' / 'Sweetorange' / 'Sweetorange' / 'Original Image'
    canker_sweetorange = count_images(sweetorange_dir / 'Citrus canker')
    healthy_sweetorange = count_images(sweetorange_dir / 'Healthy leaf')
    
    print(f"    Canker:     {canker_sweetorange:,} images ✅")
    print(f"    Healthy:    {healthy_sweetorange:,} images ✅")
    print(f"    Total:      {canker_sweetorange + healthy_sweetorange:,} images")
    print(f"    Note:       Largest binary classification dataset!")
    
    total_stats['canker_leaves'] += canker_sweetorange
    total_stats['healthy_leaves'] += healthy_sweetorange
    
    # 4. Dataset of Citrus Canker Growth Rate
    print("\n📦 4. Dataset of Citrus Canker Growth Rate")
    print("    Purpose: Time-series disease progression (6 stages)")
    growth_rate_dir = BASE_DIR / 'canker leaves' / 'Dataset of Citrus Canker Growth Rate' / 'Dataset of Citrus Canker Growth Rate'
    growth_rate_total = count_images(growth_rate_dir)
    
    for stage in ['Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5', 'Stage6']:
        stage_count = count_images(growth_rate_dir / stage)
        print(f"    {stage}:     {stage_count:,} images")
    
    print(f"    Total:      {growth_rate_total:,} images")
    print(f"    Note:       ALL canker leaves (progression stages, no healthy)")
    
    total_stats['canker_leaves'] += growth_rate_total
    
    # 5. Dataset of Citrus Canker Growth Rate through Detached Method
    print("\n📦 5. Dataset of Citrus Canker Growth Rate (Detached Method)")
    print("    Purpose: Time-series disease progression with detached leaves (6 stages)")
    detached_dir = BASE_DIR / 'canker leaves' / 'Dataset of Citrus Canker Growth Rate through Detached Method'
    detached_total = count_images(detached_dir)
    
    for stage in ['Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5', 'Stage6']:
        stage_count = count_images(detached_dir / stage)
        print(f"    {stage}:     {stage_count:,} images")
    
    print(f"    Total:      {detached_total:,} images")
    print(f"    Note:       ALL canker leaves (progression stages, no healthy)")
    
    total_stats['canker_leaves'] += detached_total
    total_stats['mixed_or_unknown'] += yolo_leaves_total
    
    # ==================== MIXED FOLDER ANALYSIS ====================
    print("\n" + "=" * 80)
    print("🔀 MIXED FOLDER ANALYSIS (Fruits + Leaves)")
    print("=" * 80)
    
    print("\n📦 Citrus Canker.v6i.yolov11 (YOLO Object Detection)")
    print("    Purpose: Resistance level classification (11 classes)")
    print("    Classes: Fruit (Highly-Resistant, Highly-Susceptible, Moderately-Resistant,")
    print("             Moderately-Susceptible, Susceptible)")
    print("             Leaf (Highly-Resistant, Highly-Susceptible, Moderately-Resistant,")
    print("             Moderately-Susceptible, Resistant, Susceptible)")
    
    mixed_dir = BASE_DIR / 'Mixed' / 'Citrus Canker.v6i.yolov11'
    mixed_train = count_images(mixed_dir / 'train' / 'images')
    mixed_valid = count_images(mixed_dir / 'valid' / 'images')
    mixed_test = count_images(mixed_dir / 'test' / 'images')
    mixed_total = mixed_train + mixed_valid + mixed_test
    
    print(f"    Train:      {mixed_train:,} images")
    print(f"    Valid:      {mixed_valid:,} images")
    print(f"    Test:       {mixed_test:,} images")
    print(f"    Total:      {mixed_total:,} images")
    print(f"    Note:       Multi-class resistance levels (NOT binary classification)")
    
    total_stats['mixed_or_unknown'] += mixed_total
    
    # ==================== OVERALL SUMMARY ====================
    print("\n" + "=" * 80)
    print("📊 OVERALL DATASET SUMMARY")
    print("=" * 80)
    
    print("\n🎯 USABLE FOR BINARY CLASSIFICATION (Healthy vs Canker):")
    print(f"    Canker Fruits:    {total_stats['canker_fruits']:,} images ✅")
    print(f"    Healthy Fruits:   {total_stats['healthy_fruits']:,} images ✅")
    print(f"    Canker Leaves:    {total_stats['canker_leaves']:,} images ✅")
    print(f"    Healthy Leaves:   {total_stats['healthy_leaves']:,} images ✅")
    print(f"    ─────────────────────────────────────────")
    print(f"    Total Canker:     {total_stats['canker_fruits'] + total_stats['canker_leaves']:,} images")
    print(f"    Total Healthy:    {total_stats['healthy_fruits'] + total_stats['healthy_leaves']:,} images")
    print(f"    Grand Total:      {total_stats['canker_fruits'] + total_stats['healthy_fruits'] + total_stats['canker_leaves'] + total_stats['healthy_leaves']:,} images")
    
    print("\n⚠️  YOLO DATASETS (Object Detection - Not for Binary Classification):")
    print(f"    Mixed/Multi-class: {total_stats['mixed_or_unknown']:,} images")
    print(f"    Note: These are for object detection/multi-class, not binary classification")
    
    # ==================== BREAKDOWN BY CATEGORY ====================
    print("\n" + "=" * 80)
    print("📈 DETAILED BREAKDOWN")
    print("=" * 80)
    
    print("\n🍎 FRUIT IMAGES:")
    print(f"    Canker:           {total_stats['canker_fruits']:,} images")
    print(f"    Healthy:          {total_stats['healthy_fruits']:,} images")
    print(f"    Total Fruits:     {total_stats['canker_fruits'] + total_stats['healthy_fruits']:,} images")
    print(f"    Balance:          {total_stats['canker_fruits']/(total_stats['canker_fruits'] + total_stats['healthy_fruits'])*100:.1f}% canker, {total_stats['healthy_fruits']/(total_stats['canker_fruits'] + total_stats['healthy_fruits'])*100:.1f}% healthy")
    
    print("\n🍃 LEAF IMAGES:")
    print(f"    Canker:           {total_stats['canker_leaves']:,} images")
    print(f"    Healthy:          {total_stats['healthy_leaves']:,} images")
    print(f"    Total Leaves:     {total_stats['canker_leaves'] + total_stats['healthy_leaves']:,} images")
    print(f"    Balance:          {total_stats['canker_leaves']/(total_stats['canker_leaves'] + total_stats['healthy_leaves'])*100:.1f}% canker, {total_stats['healthy_leaves']/(total_stats['canker_leaves'] + total_stats['healthy_leaves'])*100:.1f}% healthy")
    
    # ==================== RECOMMENDATIONS ====================
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    total_usable = total_stats['canker_fruits'] + total_stats['healthy_fruits'] + total_stats['canker_leaves'] + total_stats['healthy_leaves']
    
    print(f"\n✅ Dataset Size: {total_usable:,} total images")
    
    if total_usable >= 5000:
        print("   Status: EXCELLENT - More than enough for training! ✅")
    elif total_usable >= 3000:
        print("   Status: VERY GOOD - Sufficient for training ✅")
    elif total_usable >= 1500:
        print("   Status: GOOD - Adequate for training with augmentation ⚠️")
    else:
        print("   Status: MODERATE - May need heavy augmentation ⚠️")
    
    print(f"\n🔍 Class Balance:")
    total_canker = total_stats['canker_fruits'] + total_stats['canker_leaves']
    total_healthy = total_stats['healthy_fruits'] + total_stats['healthy_leaves']
    canker_percent = total_canker / (total_canker + total_healthy) * 100
    healthy_percent = total_healthy / (total_canker + total_healthy) * 100
    
    print(f"   Canker:  {total_canker:,} ({canker_percent:.1f}%)")
    print(f"   Healthy: {total_healthy:,} ({healthy_percent:.1f}%)")
    
    if 40 <= canker_percent <= 60:
        print("   Balance: EXCELLENT - Nearly balanced ✅")
    elif 30 <= canker_percent <= 70:
        print("   Balance: GOOD - Acceptable imbalance ✅")
    else:
        print("   Balance: IMBALANCED - May need class weighting ⚠️")
    
    print(f"\n📋 Next Steps:")
    print("   1. ✅ Use 'Fruits' folder for fruit binary classification")
    print("   2. ✅ Combine 'Leaves' + 'Sweetorange' folders for leaf classification")
    print("   3. ⚠️  Optionally include Growth Rate datasets (all canker, no healthy)")
    print("   4. ⚠️  YOLO datasets are for object detection, not binary classification")
    print("   5. 🔧 Create unified dataset with deduplication")
    print("   6. 🔧 Apply 70/15/15 train/val/test split")
    print("   7. 🚀 Train ResNet-50 or DenseNet-121 model")
    
    print("\n" + "=" * 80)
    print("✨ ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return total_stats

if __name__ == '__main__':
    stats = analyze_citrus_canker_dataset()
