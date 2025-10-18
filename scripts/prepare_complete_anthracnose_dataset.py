"""
Complete Anthracnose Dataset Preparation Script
================================================

This script creates a unified binary classification dataset (Healthy vs Anthracnose)
by combining ALL available fruit and leaf images from multiple sources.

FRUIT SOURCES (Total: 249 healthy, 170 anthracnose from Mango.v4):
- Mango.v4-v4.yolov11 (YOLO format with Healthy class)
  - train: 175 healthy, 170 anthracnose
  - valid: 48 healthy, 56 anthracnose  
  - test: 26 healthy, 26 anthracnose

LEAF SOURCES (Total: 2,646 healthy, 2,264 anthracnose):
- Original Mango Dataset: 246 healthy, 283 anthracnose
- Raw Images Five Class: 600 healthy, 600 anthracnose
- MLD24: 800 healthy, 800 anthracnose
- Augmented Dataset: 1,000 healthy, 1,000 anthracnose
- Mango Leafs YOLO: 0 healthy, 581 anthracnose

GRAND TOTAL EXPECTED: 2,895 healthy, 2,434 anthracnose = 5,329 images
After deduplication: ~4,000-4,500 unique images (estimated 15-25% duplicates)

Output Structure:
data/unified/anthracnose/
├── train/
│   ├── healthy/
│   └── anthracnose/
├── val/
│   ├── healthy/
│   └── anthracnose/
└── test/
    ├── healthy/
    └── anthracnose/

Split Ratio: 70% train, 15% val, 15% test
"""

import os
import shutil
import random
import hashlib
import json
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
from tqdm import tqdm

# Configuration
SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'unified' / 'anthracnose'
ANTHRACNOSE_DIR = BASE_DIR / 'Anthracnose'

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Source configurations
FRUIT_SOURCES = {
    'Mango.v4-v4.yolov11': {
        'path': ANTHRACNOSE_DIR / 'Mango Fruit' / 'Mango.v4-v4.yolov11',
        'type': 'yolo_split',  # Already has train/valid/test splits
        'has_healthy': True,
        'healthy_pattern': 'He*.jpg',
        'anthracnose_pattern': 'An*.jpg'
    }
}

LEAF_SOURCES = {
    'Original_Mango_Dataset': {
        'path': ANTHRACNOSE_DIR / 'Mango Leaves' / 'Mango Dataset' / 'Original Mango Dataset' / 'Original Mango Dataset',
        'type': 'class_folders',
        'has_healthy': True
    },
    'Raw_Images_Five_Class': {
        'path': ANTHRACNOSE_DIR / 'Mango Leaves' / 'Raw Images of Five Class (Resized)' / 'Raw Images of Five Class (Resized)' / 'Resize mango leaf disease',
        'type': 'class_folders',
        'has_healthy': True
    },
    'MLD24': {
        'path': ANTHRACNOSE_DIR / 'Mango Leaves' / 'MLD24' / 'MLD24' / 'MLD24',
        'type': 'class_folders',
        'has_healthy': True
    },
    'Augmented_Dataset': {
        'path': ANTHRACNOSE_DIR / 'Mango Leaves' / 'Mango Dataset' / 'Augmented Mango dataset' / 'Augmented Mango dataset',
        'type': 'class_folders',
        'has_healthy': True
    },
    'Mango_Leafs_YOLO': {
        'path': ANTHRACNOSE_DIR / 'Mango Leaves' / 'Mango' / 'Mango' / 'Leafs' / 'Anthracnose.v4i.folder',
        'type': 'yolo_anthracnose_only',
        'has_healthy': False
    }
}


def compute_image_hash(image_path, hash_size=8):
    """
    Compute perceptual hash of an image for duplicate detection.
    Uses 8x8 grayscale hash for fast comparison.
    """
    try:
        img = Image.open(image_path).convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return int(bits, 2)
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None


def collect_fruit_images():
    """Collect all fruit images (healthy + anthracnose) from Mango.v4 dataset."""
    print("\n" + "="*70)
    print("📦 COLLECTING FRUIT IMAGES")
    print("="*70)
    
    images = {'healthy': [], 'anthracnose': []}
    
    source = FRUIT_SOURCES['Mango.v4-v4.yolov11']
    base_path = source['path']
    
    if not base_path.exists():
        print(f"⚠️  Fruit source not found: {base_path}")
        return images
    
    # Collect from train/valid/test splits
    for split in ['train', 'valid', 'test']:
        img_dir = base_path / split / 'images'
        if not img_dir.exists():
            continue
        
        # Healthy images (He*.jpg)
        healthy_imgs = list(img_dir.glob(source['healthy_pattern']))
        for img_path in healthy_imgs:
            images['healthy'].append({
                'path': img_path,
                'source': f'Mango.v4_{split}',
                'category': 'fruit'
            })
        
        # Anthracnose images (An*.jpg)
        anth_imgs = list(img_dir.glob(source['anthracnose_pattern']))
        for img_path in anth_imgs:
            images['anthracnose'].append({
                'path': img_path,
                'source': f'Mango.v4_{split}',
                'category': 'fruit'
            })
        
        print(f"  ✓ {split:6s}: {len(healthy_imgs):3d} healthy, {len(anth_imgs):3d} anthracnose")
    
    print(f"\n📊 Fruit Total: {len(images['healthy'])} healthy, {len(images['anthracnose'])} anthracnose")
    return images


def collect_leaf_images():
    """Collect all leaf images from multiple sources."""
    print("\n" + "="*70)
    print("🍃 COLLECTING LEAF IMAGES")
    print("="*70)
    
    images = {'healthy': [], 'anthracnose': []}
    
    for source_name, source_config in LEAF_SOURCES.items():
        base_path = source_config['path']
        
        if not base_path.exists():
            print(f"⚠️  {source_name}: Not found - {base_path}")
            continue
        
        source_images = {'healthy': 0, 'anthracnose': 0}
        
        if source_config['type'] == 'class_folders':
            # Anthracnose folder
            anth_dir = base_path / 'Anthracnose'
            if anth_dir.exists():
                for img_path in anth_dir.glob('*.jpg'):
                    images['anthracnose'].append({
                        'path': img_path,
                        'source': source_name,
                        'category': 'leaf'
                    })
                    source_images['anthracnose'] += 1
            
            # Healthy folder
            if source_config['has_healthy']:
                healthy_dir = base_path / 'Healthy'
                if healthy_dir.exists():
                    for img_path in healthy_dir.glob('*.jpg'):
                        images['healthy'].append({
                            'path': img_path,
                            'source': source_name,
                            'category': 'leaf'
                        })
                        source_images['healthy'] += 1
        
        elif source_config['type'] == 'yolo_anthracnose_only':
            # YOLO format with only anthracnose
            for split in ['train', 'valid', 'test']:
                split_dir = base_path / split
                if split_dir.exists():
                    for img_path in split_dir.glob('**/*.jpg'):
                        images['anthracnose'].append({
                            'path': img_path,
                            'source': f'{source_name}_{split}',
                            'category': 'leaf'
                        })
                        source_images['anthracnose'] += 1
        
        print(f"  ✓ {source_name:30s}: {source_images['healthy']:4d} healthy, {source_images['anthracnose']:4d} anthracnose")
    
    print(f"\n📊 Leaf Total: {len(images['healthy'])} healthy, {len(images['anthracnose'])} anthracnose")
    return images


def deduplicate_images(images_dict):
    """
    Remove duplicate images using perceptual hashing.
    Returns deduplicated images and duplicate report.
    """
    print("\n" + "="*70)
    print("🔍 DEDUPLICATING IMAGES")
    print("="*70)
    
    deduplicated = {'healthy': [], 'anthracnose': []}
    duplicates = {'healthy': [], 'anthracnose': []}
    
    for class_name in ['healthy', 'anthracnose']:
        print(f"\n  Processing {class_name}...")
        images = images_dict[class_name]
        
        hash_map = {}
        
        for img_info in tqdm(images, desc=f"  Hashing {class_name}"):
            img_hash = compute_image_hash(img_info['path'])
            
            if img_hash is None:
                continue
            
            if img_hash in hash_map:
                # Duplicate found
                duplicates[class_name].append({
                    'duplicate': str(img_info['path']),
                    'original': str(hash_map[img_hash]['path']),
                    'source_dup': img_info['source'],
                    'source_orig': hash_map[img_hash]['source']
                })
            else:
                # Unique image
                hash_map[img_hash] = img_info
                deduplicated[class_name].append(img_info)
        
        print(f"  ✓ {class_name:12s}: {len(deduplicated[class_name]):4d} unique, {len(duplicates[class_name]):4d} duplicates removed")
    
    total_unique = len(deduplicated['healthy']) + len(deduplicated['anthracnose'])
    total_duplicates = len(duplicates['healthy']) + len(duplicates['anthracnose'])
    
    print(f"\n📊 Deduplication Summary:")
    print(f"   Total Unique: {total_unique}")
    print(f"   Total Duplicates: {total_duplicates}")
    print(f"   Deduplication Rate: {(total_duplicates/(total_unique+total_duplicates)*100):.1f}%")
    
    return deduplicated, duplicates


def split_dataset(images_dict, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split images into train/val/test sets while maintaining class balance.
    """
    print("\n" + "="*70)
    print("✂️  SPLITTING DATASET")
    print("="*70)
    
    splits = {
        'train': {'healthy': [], 'anthracnose': []},
        'val': {'healthy': [], 'anthracnose': []},
        'test': {'healthy': [], 'anthracnose': []}
    }
    
    for class_name in ['healthy', 'anthracnose']:
        images = images_dict[class_name].copy()
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits['train'][class_name] = images[:train_end]
        splits['val'][class_name] = images[train_end:val_end]
        splits['test'][class_name] = images[val_end:]
        
        print(f"  {class_name:12s}: {len(splits['train'][class_name]):4d} train, "
              f"{len(splits['val'][class_name]):4d} val, {len(splits['test'][class_name]):4d} test")
    
    return splits


def copy_images_to_output(splits, output_dir):
    """
    Copy images to final output directory structure.
    """
    print("\n" + "="*70)
    print("📁 CREATING OUTPUT DIRECTORY STRUCTURE")
    print("="*70)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['healthy', 'anthracnose']:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    manifest = {'train': {}, 'val': {}, 'test': {}}
    
    for split in ['train', 'val', 'test']:
        print(f"\n  Copying {split} images...")
        
        for class_name in ['healthy', 'anthracnose']:
            images = splits[split][class_name]
            target_dir = output_dir / split / class_name
            
            for idx, img_info in enumerate(tqdm(images, desc=f"    {class_name:12s}")):
                src_path = img_info['path']
                
                # Create unique filename: category_source_originalname
                category = img_info['category']
                source = img_info['source']
                orig_name = src_path.name
                
                new_name = f"{category}_{source}_{idx:04d}_{orig_name}"
                dst_path = target_dir / new_name
                
                try:
                    shutil.copy2(src_path, dst_path)
                    
                    # Add to manifest
                    if class_name not in manifest[split]:
                        manifest[split][class_name] = []
                    
                    manifest[split][class_name].append({
                        'filename': new_name,
                        'original_path': str(src_path),
                        'source': source,
                        'category': category
                    })
                except Exception as e:
                    print(f"\n    ⚠️  Error copying {src_path}: {e}")
    
    return manifest


def generate_statistics(splits, manifest, duplicates, output_dir):
    """
    Generate comprehensive statistics and reports.
    """
    print("\n" + "="*70)
    print("📊 GENERATING STATISTICS")
    print("="*70)
    
    # Calculate statistics
    stats = {
        'total_images': 0,
        'by_split': {},
        'by_class': {'healthy': 0, 'anthracnose': 0},
        'by_category': {'fruit': 0, 'leaf': 0},
        'by_source': Counter(),
        'duplicates_removed': len(duplicates['healthy']) + len(duplicates['anthracnose'])
    }
    
    for split in ['train', 'val', 'test']:
        stats['by_split'][split] = {
            'healthy': len(splits[split]['healthy']),
            'anthracnose': len(splits[split]['anthracnose']),
            'total': len(splits[split]['healthy']) + len(splits[split]['anthracnose'])
        }
        stats['total_images'] += stats['by_split'][split]['total']
        stats['by_class']['healthy'] += len(splits[split]['healthy'])
        stats['by_class']['anthracnose'] += len(splits[split]['anthracnose'])
        
        # Count by category and source
        for class_name in ['healthy', 'anthracnose']:
            for img_info in splits[split][class_name]:
                stats['by_category'][img_info['category']] += 1
                stats['by_source'][img_info['source']] += 1
    
    # Save manifest
    manifest_path = output_dir / 'dataset_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Manifest saved: {manifest_path}")
    
    # Save duplicate report
    duplicates_path = output_dir / 'duplicates_report.json'
    with open(duplicates_path, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Duplicates report saved: {duplicates_path}")
    
    # Save statistics
    stats_path = output_dir / 'dataset_statistics.json'
    stats['by_source'] = dict(stats['by_source'])  # Convert Counter to dict
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Statistics saved: {stats_path}")
    
    # Generate README
    readme_content = f"""# Anthracnose Detection Dataset (Binary Classification)

## 📊 Dataset Statistics

**Total Images:** {stats['total_images']}
- **Healthy:** {stats['by_class']['healthy']} ({stats['by_class']['healthy']/stats['total_images']*100:.1f}%)
- **Anthracnose:** {stats['by_class']['anthracnose']} ({stats['by_class']['anthracnose']/stats['total_images']*100:.1f}%)

**Category Distribution:**
- **Fruits:** {stats['by_category']['fruit']} ({stats['by_category']['fruit']/stats['total_images']*100:.1f}%)
- **Leaves:** {stats['by_category']['leaf']} ({stats['by_category']['leaf']/stats['total_images']*100:.1f}%)

**Split Distribution:**
- **Train:** {stats['by_split']['train']['total']} images ({stats['by_split']['train']['healthy']} healthy, {stats['by_split']['train']['anthracnose']} anthracnose)
- **Val:** {stats['by_split']['val']['total']} images ({stats['by_split']['val']['healthy']} healthy, {stats['by_split']['val']['anthracnose']} anthracnose)
- **Test:** {stats['by_split']['test']['total']} images ({stats['by_split']['test']['total']} healthy, {stats['by_split']['test']['anthracnose']} anthracnose)

**Duplicates Removed:** {stats['duplicates_removed']}

## 📂 Directory Structure

```
anthracnose/
├── train/
│   ├── healthy/      ({stats['by_split']['train']['healthy']} images)
│   └── anthracnose/  ({stats['by_split']['train']['anthracnose']} images)
├── val/
│   ├── healthy/      ({stats['by_split']['val']['healthy']} images)
│   └── anthracnose/  ({stats['by_split']['val']['anthracnose']} images)
└── test/
    ├── healthy/      ({stats['by_split']['test']['healthy']} images)
    └── anthracnose/  ({stats['by_split']['test']['anthracnose']} images)
```

## 🎯 Usage

### PyTorch ImageFolder

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('train', transform=transform)
val_dataset = datasets.ImageFolder('val', transform=transform)
test_dataset = datasets.ImageFolder('test', transform=transform)
```

### TensorFlow

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    'train',
    image_size=(224, 224),
    batch_size=32
)
```

## 📝 Source Attribution

This dataset combines images from:
{chr(10).join(f'- {source}: {count} images' for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True))}

## 📄 Files

- `dataset_manifest.json` - Complete file tracking
- `duplicates_report.json` - Removed duplicates log
- `dataset_statistics.json` - Detailed statistics
- `README.md` - This file

---

**Created:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Script:** prepare_complete_anthracnose_dataset.py
**Project:** FRESH ML - Module 2 (Disease Detection)
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ README saved: {readme_path}")
    
    return stats


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("🥭 ANTHRACNOSE DATASET PREPARATION")
    print("="*70)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Random Seed: {SEED}")
    
    # Step 1: Collect fruit images
    fruit_images = collect_fruit_images()
    
    # Step 2: Collect leaf images
    leaf_images = collect_leaf_images()
    
    # Step 3: Combine all images
    all_images = {
        'healthy': fruit_images['healthy'] + leaf_images['healthy'],
        'anthracnose': fruit_images['anthracnose'] + leaf_images['anthracnose']
    }
    
    total_collected = len(all_images['healthy']) + len(all_images['anthracnose'])
    print(f"\n📦 Total Collected: {total_collected} images")
    print(f"   Healthy: {len(all_images['healthy'])}")
    print(f"   Anthracnose: {len(all_images['anthracnose'])}")
    
    # Step 4: Deduplicate
    unique_images, duplicates = deduplicate_images(all_images)
    
    # Step 5: Split dataset
    splits = split_dataset(unique_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Step 6: Create output directory
    if OUTPUT_DIR.exists():
        print(f"\n⚠️  Output directory exists: {OUTPUT_DIR}")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print("  ✓ Deleted old directory")
        else:
            print("  ✗ Aborted")
            return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 7: Copy images
    manifest = copy_images_to_output(splits, OUTPUT_DIR)
    
    # Step 8: Generate statistics
    stats = generate_statistics(splits, manifest, duplicates, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Statistics:")
    print(f"   Total Images: {stats['total_images']}")
    print(f"   Healthy: {stats['by_class']['healthy']} ({stats['by_class']['healthy']/stats['total_images']*100:.1f}%)")
    print(f"   Anthracnose: {stats['by_class']['anthracnose']} ({stats['by_class']['anthracnose']/stats['total_images']*100:.1f}%)")
    print(f"   Fruits: {stats['by_category']['fruit']}, Leaves: {stats['by_category']['leaf']}")
    print(f"   Duplicates Removed: {stats['duplicates_removed']}")
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print("\n🎉 Ready for model training!")


if __name__ == '__main__':
    main()
