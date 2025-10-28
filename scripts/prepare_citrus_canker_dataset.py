"""
Complete Citrus Canker Dataset Preparation Script
Combines all sources, deduplicates, and creates train/val/test splits

Similar to prepare_complete_anthracnose_dataset.py but for Citrus Canker
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from collections import defaultdict
from datetime import datetime
import random

# Set random seed for reproducibility
random.seed(42)

# Base directories
BASE_DIR = Path(r'd:\FYP\FRESH_ML\Citrus Canker')
OUTPUT_DIR = Path(r'd:\FYP\FRESH_ML\data\unified\citrus_canker')

# Dataset sources
SOURCES = {
    'fruits': {
        'filtered_canker_only_train': BASE_DIR / 'Canker fruit' / 'Filtered Canker Only' / 'train',
        'filtered_canker_only_valid': BASE_DIR / 'Canker fruit' / 'Filtered Canker Only' / 'valid',
        'filtered_canker_only_test': BASE_DIR / 'Canker fruit' / 'Filtered Canker Only' / 'test',
        'fruits_canker': BASE_DIR / 'Canker fruit' / 'Fruits' / 'Canker',
        'fruits_healthy': BASE_DIR / 'Canker fruit' / 'Fruits' / 'healthy',
    },
    'leaves': {
        'leaves_canker': BASE_DIR / 'canker leaves' / 'Leaves' / 'canker',
        'leaves_healthy': BASE_DIR / 'canker leaves' / 'Leaves' / 'healthy',
        'sweetorange_canker': BASE_DIR / 'canker leaves' / 'Sweetorange' / 'Sweetorange' / 'Original Image' / 'Citrus canker',
        'sweetorange_healthy': BASE_DIR / 'canker leaves' / 'Sweetorange' / 'Sweetorange' / 'Original Image' / 'Healthy leaf',
        'growth_rate': BASE_DIR / 'canker leaves' / 'Dataset of Citrus Canker Growth Rate' / 'Dataset of Citrus Canker Growth Rate',
        'detached_method': BASE_DIR / 'canker leaves' / 'Dataset of Citrus Canker Growth Rate through Detached Method',
    }
}

def compute_image_hash(image_path, hash_size=8):
    """
    Compute perceptual hash for duplicate detection
    Uses average hashing algorithm
    """
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
        return bits
    except Exception as e:
        print(f"⚠️  Error hashing {image_path}: {e}")
        return None

def collect_fruit_images():
    """
    Collect all fruit images from multiple sources
    Returns dict: {category: [(path, source, hash), ...]}
    """
    print("\n" + "=" * 80)
    print("🍎 COLLECTING FRUIT IMAGES")
    print("=" * 80)
    
    images = {
        'canker': [],
        'healthy': []
    }
    
    # 1. Filtered Canker Only (from YOLO dataset)
    print("\n📦 1. Filtered Canker Only (YOLO - canker only)")
    for split in ['train', 'valid', 'test']:
        source_dir = SOURCES['fruits'][f'filtered_canker_only_{split}']
        if source_dir.exists():
            count = 0
            for img_path in source_dir.glob('*.jpg'):
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['canker'].append((img_path, f'filtered_yolo_{split}', img_hash))
                    count += 1
            print(f"   {split}: {count:,} canker images")
    
    # 2. Fruits/Canker
    print("\n📦 2. Fruits/Canker")
    source_dir = SOURCES['fruits']['fruits_canker']
    if source_dir.exists():
        count = 0
        for img_path in source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['canker'].append((img_path, 'fruits_canker', img_hash))
                    count += 1
        print(f"   Canker: {count:,} images")
    
    # 3. Fruits/healthy
    print("\n📦 3. Fruits/healthy")
    source_dir = SOURCES['fruits']['fruits_healthy']
    if source_dir.exists():
        count = 0
        for img_path in source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['healthy'].append((img_path, 'fruits_healthy', img_hash))
                    count += 1
        print(f"   Healthy: {count:,} images")
    
    total_canker = len(images['canker'])
    total_healthy = len(images['healthy'])
    print(f"\n📊 Fruit Collection Summary:")
    print(f"   Canker:  {total_canker:,} images")
    print(f"   Healthy: {total_healthy:,} images")
    print(f"   Total:   {total_canker + total_healthy:,} images")
    
    return images

def collect_leaf_images():
    """
    Collect all leaf images from multiple sources
    Returns dict: {category: [(path, source, hash), ...]}
    """
    print("\n" + "=" * 80)
    print("🍃 COLLECTING LEAF IMAGES")
    print("=" * 80)
    
    images = {
        'canker': [],
        'healthy': []
    }
    
    # 1. Leaves/canker
    print("\n📦 1. Leaves/canker")
    source_dir = SOURCES['leaves']['leaves_canker']
    if source_dir.exists():
        count = 0
        for img_path in source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['canker'].append((img_path, 'leaves_canker', img_hash))
                    count += 1
        print(f"   Canker: {count:,} images")
    
    # 2. Leaves/healthy
    print("\n📦 2. Leaves/healthy")
    source_dir = SOURCES['leaves']['leaves_healthy']
    if source_dir.exists():
        count = 0
        for img_path in source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['healthy'].append((img_path, 'leaves_healthy', img_hash))
                    count += 1
        print(f"   Healthy: {count:,} images")
    
    # 3. Sweetorange
    print("\n📦 3. Sweetorange (Large Dataset)")
    
    # Canker
    source_dir = SOURCES['leaves']['sweetorange_canker']
    if source_dir.exists():
        count = 0
        for img_path in source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['canker'].append((img_path, 'sweetorange_canker', img_hash))
                    count += 1
        print(f"   Canker: {count:,} images")
    
    # Healthy
    source_dir = SOURCES['leaves']['sweetorange_healthy']
    if source_dir.exists():
        count = 0
        for img_path in source_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_hash = compute_image_hash(img_path)
                if img_hash:
                    images['healthy'].append((img_path, 'sweetorange_healthy', img_hash))
                    count += 1
        print(f"   Healthy: {count:,} images")
    
    # 4. Growth Rate Dataset (all canker, no healthy)
    print("\n📦 4. Growth Rate Dataset (6 stages - all canker)")
    source_dir = SOURCES['leaves']['growth_rate']
    if source_dir.exists():
        total_growth = 0
        for stage in ['Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5', 'Stage6']:
            stage_dir = source_dir / stage
            if stage_dir.exists():
                count = 0
                for img_path in stage_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_hash = compute_image_hash(img_path)
                        if img_hash:
                            images['canker'].append((img_path, f'growth_rate_{stage.lower()}', img_hash))
                            count += 1
                            total_growth += 1
                print(f"   {stage}: {count:,} images")
        print(f"   Total Growth Rate: {total_growth:,} images")
    
    # 5. Detached Method Dataset (all canker, no healthy)
    print("\n📦 5. Detached Method Dataset (6 stages - all canker)")
    source_dir = SOURCES['leaves']['detached_method']
    if source_dir.exists():
        total_detached = 0
        for stage in ['Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5', 'Stage6']:
            stage_dir = source_dir / stage
            if stage_dir.exists():
                count = 0
                for img_path in stage_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_hash = compute_image_hash(img_path)
                        if img_hash:
                            images['canker'].append((img_path, f'detached_{stage.lower()}', img_hash))
                            count += 1
                            total_detached += 1
                print(f"   {stage}: {count:,} images")
        print(f"   Total Detached Method: {total_detached:,} images")
    
    total_canker = len(images['canker'])
    total_healthy = len(images['healthy'])
    print(f"\n📊 Leaf Collection Summary:")
    print(f"   Canker:  {total_canker:,} images")
    print(f"   Healthy: {total_healthy:,} images")
    print(f"   Total:   {total_canker + total_healthy:,} images")
    
    return images

def deduplicate_images(images_dict):
    """
    Remove duplicate images based on perceptual hash
    Returns deduplicated dict and duplicates info
    """
    print("\n" + "=" * 80)
    print("🔍 DEDUPLICATING IMAGES")
    print("=" * 80)
    
    deduplicated = {
        'canker': [],
        'healthy': []
    }
    
    duplicates_info = []
    
    for category in ['canker', 'healthy']:
        print(f"\n📊 Processing {category.upper()} images...")
        
        hash_dict = {}
        original_count = len(images_dict[category])
        
        for img_path, source, img_hash in images_dict[category]:
            if img_hash not in hash_dict:
                # First occurrence - keep it
                hash_dict[img_hash] = (img_path, source)
                deduplicated[category].append((img_path, source, img_hash))
            else:
                # Duplicate found
                original_path, original_source = hash_dict[img_hash]
                duplicates_info.append({
                    'category': category,
                    'original': str(original_path),
                    'original_source': original_source,
                    'duplicate': str(img_path),
                    'duplicate_source': source,
                    'hash': img_hash
                })
        
        unique_count = len(deduplicated[category])
        duplicate_count = original_count - unique_count
        dedup_rate = (duplicate_count / original_count * 100) if original_count > 0 else 0
        
        print(f"   Original:   {original_count:,} images")
        print(f"   Unique:     {unique_count:,} images")
        print(f"   Duplicates: {duplicate_count:,} images ({dedup_rate:.1f}%)")
    
    total_original = len(images_dict['canker']) + len(images_dict['healthy'])
    total_unique = len(deduplicated['canker']) + len(deduplicated['healthy'])
    total_duplicates = total_original - total_unique
    overall_dedup_rate = (total_duplicates / total_original * 100) if total_original > 0 else 0
    
    print(f"\n📊 Overall Deduplication Summary:")
    print(f"   Total Original: {total_original:,} images")
    print(f"   Total Unique:   {total_unique:,} images")
    print(f"   Total Removed:  {total_duplicates:,} duplicates ({overall_dedup_rate:.1f}%)")
    
    return deduplicated, duplicates_info

def split_dataset(images_dict, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test with stratification
    """
    print("\n" + "=" * 80)
    print("📊 SPLITTING DATASET (70/15/15)")
    print("=" * 80)
    
    splits = {
        'train': {'canker': [], 'healthy': []},
        'val': {'canker': [], 'healthy': []},
        'test': {'canker': [], 'healthy': []}
    }
    
    for category in ['canker', 'healthy']:
        images = images_dict[category].copy()
        random.shuffle(images)
        
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        splits['train'][category] = images[:train_size]
        splits['val'][category] = images[train_size:train_size + val_size]
        splits['test'][category] = images[train_size + val_size:]
        
        print(f"\n{category.upper()}:")
        print(f"   Total:  {total:,} images")
        print(f"   Train:  {len(splits['train'][category]):,} images ({len(splits['train'][category])/total*100:.1f}%)")
        print(f"   Val:    {len(splits['val'][category]):,} images ({len(splits['val'][category])/total*100:.1f}%)")
        print(f"   Test:   {len(splits['test'][category]):,} images ({len(splits['test'][category])/total*100:.1f}%)")
    
    # Overall statistics
    total_train = len(splits['train']['canker']) + len(splits['train']['healthy'])
    total_val = len(splits['val']['canker']) + len(splits['val']['healthy'])
    total_test = len(splits['test']['canker']) + len(splits['test']['healthy'])
    
    print(f"\n📊 Overall Split Summary:")
    print(f"   Train:  {total_train:,} images ({len(splits['train']['canker']):,} canker + {len(splits['train']['healthy']):,} healthy)")
    print(f"   Val:    {total_val:,} images ({len(splits['val']['canker']):,} canker + {len(splits['val']['healthy']):,} healthy)")
    print(f"   Test:   {total_test:,} images ({len(splits['test']['canker']):,} canker + {len(splits['test']['healthy']):,} healthy)")
    
    return splits

def copy_images_to_output(splits):
    """
    Copy images to output directory with proper structure
    """
    print("\n" + "=" * 80)
    print("📁 COPYING IMAGES TO OUTPUT DIRECTORY")
    print("=" * 80)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for category in ['canker', 'healthy']:
            output_path = OUTPUT_DIR / split / category
            output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    
    manifest = []
    
    for split in ['train', 'val', 'test']:
        print(f"\n📦 Copying {split.upper()} split...")
        
        for category in ['canker', 'healthy']:
            images = splits[split][category]
            output_dir = OUTPUT_DIR / split / category
            
            for idx, (img_path, source, img_hash) in enumerate(images, 1):
                # Create unique filename: category_source_index_originalname.ext
                original_name = img_path.stem
                extension = img_path.suffix
                new_name = f"{category}_{source}_{idx:04d}_{original_name}{extension}"
                output_path = output_dir / new_name
                
                # Copy image
                shutil.copy2(img_path, output_path)
                
                # Add to manifest
                manifest.append({
                    'split': split,
                    'category': category,
                    'output_path': str(output_path.relative_to(OUTPUT_DIR)),
                    'original_path': str(img_path),
                    'source': source,
                    'hash': img_hash,
                    'index': idx
                })
            
            print(f"   {category}: {len(images):,} images copied")
    
    print(f"\n✅ All images copied successfully!")
    
    return manifest

def generate_statistics(splits, duplicates_info, manifest):
    """
    Generate comprehensive statistics and save reports
    """
    print("\n" + "=" * 80)
    print("📊 GENERATING STATISTICS AND REPORTS")
    print("=" * 80)
    
    # Calculate statistics
    stats = {
        'creation_date': datetime.now().isoformat(),
        'total_images': sum(len(splits[s]['canker']) + len(splits[s]['healthy']) for s in ['train', 'val', 'test']),
        'by_split': {},
        'by_class': {
            'canker': sum(len(splits[s]['canker']) for s in ['train', 'val', 'test']),
            'healthy': sum(len(splits[s]['healthy']) for s in ['train', 'val', 'test'])
        },
        'by_category': defaultdict(int),
        'by_source': defaultdict(int),
        'class_balance': {},
        'deduplication': {
            'total_duplicates_removed': len(duplicates_info),
            'deduplication_rate': 0
        }
    }
    
    # Split statistics
    for split in ['train', 'val', 'test']:
        stats['by_split'][split] = {
            'canker': len(splits[split]['canker']),
            'healthy': len(splits[split]['healthy']),
            'total': len(splits[split]['canker']) + len(splits[split]['healthy'])
        }
    
    # Source statistics
    for entry in manifest:
        stats['by_source'][entry['source']] += 1
        
        # Determine category (fruit vs leaf)
        if 'fruit' in entry['source'].lower() or 'filtered_yolo' in entry['source']:
            stats['by_category']['fruits'] += 1
        else:
            stats['by_category']['leaves'] += 1
    
    # Class balance
    total = stats['total_images']
    stats['class_balance'] = {
        'canker': {
            'count': stats['by_class']['canker'],
            'percentage': (stats['by_class']['canker'] / total * 100) if total > 0 else 0
        },
        'healthy': {
            'count': stats['by_class']['healthy'],
            'percentage': (stats['by_class']['healthy'] / total * 100) if total > 0 else 0
        }
    }
    
    # Save manifest
    manifest_file = OUTPUT_DIR / 'dataset_manifest.json'
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Manifest saved: {manifest_file}")
    
    # Save duplicates report
    duplicates_file = OUTPUT_DIR / 'duplicates_report.json'
    with open(duplicates_file, 'w', encoding='utf-8') as f:
        json.dump(duplicates_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Duplicates report saved: {duplicates_file}")
    
    # Save statistics
    stats_file = OUTPUT_DIR / 'dataset_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Statistics saved: {stats_file}")
    
    # Generate README
    readme_content = f"""# Citrus Canker Dataset - Unified and Deduplicated

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Images:** {stats['total_images']:,}  
**Deduplication Rate:** {len(duplicates_info) / (stats['total_images'] + len(duplicates_info)) * 100:.1f}%

## Dataset Statistics

### Overall Composition
- **Total Images:** {stats['total_images']:,}
- **Canker:** {stats['by_class']['canker']:,} ({stats['class_balance']['canker']['percentage']:.1f}%)
- **Healthy:** {stats['by_class']['healthy']:,} ({stats['class_balance']['healthy']['percentage']:.1f}%)

### By Split
- **Train:** {stats['by_split']['train']['total']:,} images ({stats['by_split']['train']['canker']:,} canker + {stats['by_split']['train']['healthy']:,} healthy)
- **Val:** {stats['by_split']['val']['total']:,} images ({stats['by_split']['val']['canker']:,} canker + {stats['by_split']['val']['healthy']:,} healthy)
- **Test:** {stats['by_split']['test']['total']:,} images ({stats['by_split']['test']['canker']:,} canker + {stats['by_split']['test']['healthy']:,} healthy)

### By Category
- **Fruits:** {stats['by_category']['fruits']:,} images
- **Leaves:** {stats['by_category']['leaves']:,} images

### By Source
"""
    
    for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
        readme_content += f"- **{source}:** {count:,} images\n"
    
    readme_content += f"""

## Class Weighting Recommendation

Due to class imbalance ({stats['class_balance']['canker']['percentage']:.1f}% canker vs {stats['class_balance']['healthy']['percentage']:.1f}% healthy), use class weights:

```python
class_weight = {{
    0: 1.0,  # healthy (baseline)
    1: {stats['class_balance']['healthy']['count'] / stats['class_balance']['canker']['count']:.4f}  # canker (weighted down)
}}
```

Or in PyTorch:
```python
from torch.nn import CrossEntropyLoss

class_weights = torch.tensor([1.0, {stats['class_balance']['healthy']['count'] / stats['class_balance']['canker']['count']:.4f}])
criterion = CrossEntropyLoss(weight=class_weights)
```

## Usage Example

### PyTorch ImageFolder
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    root='data/unified/citrus_canker/train',
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### TensorFlow
```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/unified/citrus_canker/train',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)
```

## Files Generated

- `dataset_manifest.json` - Complete file tracking with sources
- `duplicates_report.json` - List of removed duplicates
- `dataset_statistics.json` - Detailed statistics
- `README.md` - This file

## Sources Included

1. **Filtered Canker Only** (from YOLO dataset) - Pure canker images
2. **Fruits/Canker** - Original canker fruit images
3. **Fruits/healthy** - Healthy fruit images
4. **Leaves/canker** - Canker leaf images
5. **Leaves/healthy** - Healthy leaf images
6. **Sweetorange** - Large balanced dataset (canker + healthy leaves)
7. **Growth Rate Dataset** - Disease progression stages (all canker)
8. **Detached Method Dataset** - Disease progression on detached leaves (all canker)

## Training Recommendations

- **Model:** DenseNet-121 (PRD recommendation) or ResNet-50
- **Input Size:** 224x224
- **Batch Size:** 32
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss with class weights
- **Data Augmentation:** Heavy augmentation for healthy class
- **Early Stopping:** Patience of 10 epochs
- **Target Metrics:** >88% detection rate, <10% false positives (from PRD)

---

**Status:** Ready for training! 🚀
"""
    
    readme_file = OUTPUT_DIR / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ README saved: {readme_file}")
    
    return stats

def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("🍊 CITRUS CANKER DATASET PREPARATION")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Step 1: Collect fruit images
    fruit_images = collect_fruit_images()
    
    # Step 2: Collect leaf images
    leaf_images = collect_leaf_images()
    
    # Step 3: Combine all images
    print("\n" + "=" * 80)
    print("🔗 COMBINING ALL IMAGES")
    print("=" * 80)
    
    all_images = {
        'canker': fruit_images['canker'] + leaf_images['canker'],
        'healthy': fruit_images['healthy'] + leaf_images['healthy']
    }
    
    total_collected = len(all_images['canker']) + len(all_images['healthy'])
    print(f"\n📊 Total Collected:")
    print(f"   Canker:  {len(all_images['canker']):,} images")
    print(f"   Healthy: {len(all_images['healthy']):,} images")
    print(f"   Total:   {total_collected:,} images")
    
    # Step 4: Deduplicate
    deduplicated, duplicates_info = deduplicate_images(all_images)
    
    # Step 5: Split dataset
    splits = split_dataset(deduplicated)
    
    # Step 6: Copy images to output
    manifest = copy_images_to_output(splits)
    
    # Step 7: Generate statistics
    stats = generate_statistics(splits, duplicates_info, manifest)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✨ DATASET PREPARATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 Final Dataset:")
    print(f"   Total Unique Images: {stats['total_images']:,}")
    print(f"   Canker:  {stats['by_class']['canker']:,} ({stats['class_balance']['canker']['percentage']:.1f}%)")
    print(f"   Healthy: {stats['by_class']['healthy']:,} ({stats['class_balance']['healthy']['percentage']:.1f}%)")
    print(f"\n   Train: {stats['by_split']['train']['total']:,} images")
    print(f"   Val:   {stats['by_split']['val']['total']:,} images")
    print(f"   Test:  {stats['by_split']['test']['total']:,} images")
    
    print(f"\n📁 Output Location:")
    print(f"   {OUTPUT_DIR}")
    
    print(f"\n📄 Files Generated:")
    print(f"   ✅ dataset_manifest.json")
    print(f"   ✅ duplicates_report.json")
    print(f"   ✅ dataset_statistics.json")
    print(f"   ✅ README.md")
    
    print(f"\n🎯 Class Weight Recommendation:")
    weight = stats['class_balance']['healthy']['count'] / stats['class_balance']['canker']['count']
    print(f"   class_weight = {{0: 1.0, 1: {weight:.4f}}}")
    
    print("\n" + "=" * 80)
    print("🚀 READY FOR TRAINING!")
    print("=" * 80)

if __name__ == '__main__':
    main()
