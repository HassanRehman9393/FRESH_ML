"""
Anthracnose Dataset Preparation Script
======================================

This script organizes all anthracnose images from multiple sources into a unified dataset
with proper train/validation/test splitting for disease detection model training.

Features:
- Deduplication using image hashing
- Proper train/val/test splitting (70/15/15)
- Handles both fruits and leaves
- Preserves YOLO labels where available
- Creates manifest files for tracking
"""

import os
import shutil
import hashlib
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
ANTHRACNOSE_DIR = BASE_DIR / "Anthracnose"
OUTPUT_DIR = BASE_DIR / "data" / "unified" / "anthracnose"

# Dataset sources configuration
FRUIT_SOURCES = {
    "anthracnose_simple": {
        "path": "Mango Fruit/Anthracnose",
        "type": "classification",
        "has_labels": False
    },
    "anthracnose_bg_removed": {
        "path": "Mango Fruit/Background-Removed/Anthracnose",
        "type": "classification",
        "has_labels": False,
        "skip": True  # Duplicates of anthracnose_simple
    },
    "anthracnose_disease": {
        "path": "Mango Fruit/Anthracnose-disease/Anthracnose/Anthracnose",
        "type": "classification",
        "has_labels": False
    },
    "archive_bg_removed": {
        "path": "Mango Fruit/archive/MangoFruitDDS/SenMangoFruitDDS_bgremoved/Anthracnose",
        "type": "classification",
        "has_labels": False,
        "skip": True  # Duplicates
    },
    "archive_original": {
        "path": "Mango Fruit/archive/MangoFruitDDS/SenMangoFruitDDS_original/Anthracnose",
        "type": "classification",
        "has_labels": False,
        "skip": True  # Duplicates
    },
    "yolo_mango_disease_v3": {
        "path": "Mango Fruit/MANGO DISEASE.v3i.yolov11",
        "type": "yolo",
        "has_labels": True,
        "class_name": "anthracnose",
        "skip": True  # Unclear labels in data.yaml
    },
    "yolo_mango_v4": {
        "path": "Mango Fruit/Mango.v4-v4.yolov11",
        "type": "yolo",
        "has_labels": True,
        "class_id": 0,  # Anthracnose is class 0
        "class_name": "Anthracnose"
    }
}

LEAF_SOURCES = {
    "original_mango": {
        "path": "Mango Leaves/Mango Dataset/Original Mango Dataset/Original Mango Dataset/Anthracnose",
        "type": "classification",
        "has_labels": False
    },
    "anthracnose_v4i": {
        "path": "Mango Leaves/Mango/Mango/Leafs/Anthracnose.v4i.folder",
        "type": "classification_split",
        "has_labels": False,
        "train_path": "train/Anthracnose",
        "valid_path": "valid/Anthracnose",
        "test_path": "test/Anthracnose"
    },
    "mld24": {
        "path": "Mango Leaves/MLD24/MLD24/MLD24/Anthracnose",
        "type": "classification",
        "has_labels": False
    },
    "resized": {
        "path": "Mango Leaves/Raw Images of Five Class (Resized)/Raw Images of Five Class (Resized)/Resize mango leaf disease/Anthracnose",
        "type": "classification",
        "has_labels": False
    },
    "augmented": {
        "path": "Mango Leaves/Mango Dataset/Augmented Mango dataset/Augmented Mango dataset/Anthracnose",
        "type": "classification",
        "has_labels": False,
        "skip": False  # Keep augmented data
    }
}


def compute_image_hash(image_path):
    """Compute perceptual hash of image for duplicate detection"""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to 8x8 for hashing
            img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = ''.join(['1' if px > avg else '0' for px in pixels])
            return int(bits, 2)
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None


def is_valid_image(image_path):
    """Check if image is valid and readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False


def collect_images_from_source(source_name, source_config, base_path, category):
    """Collect images from a single source"""
    images = []
    source_path = base_path / source_config["path"]
    
    if not source_path.exists():
        print(f"⚠️  Warning: {source_path} does not exist")
        return images
    
    source_type = source_config["type"]
    
    if source_type == "classification":
        # Simple directory with images
        for img_file in source_path.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                if is_valid_image(img_file):
                    images.append({
                        'path': img_file,
                        'source': source_name,
                        'category': category,
                        'type': 'classification',
                        'label': None
                    })
    
    elif source_type == "classification_split":
        # Pre-split dataset
        for split in ['train', 'valid', 'test']:
            split_key = f"{split}_path"
            if split_key in source_config:
                split_path = source_path / source_config[split_key]
                if split_path.exists():
                    for img_file in split_path.glob("*"):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            if is_valid_image(img_file):
                                images.append({
                                    'path': img_file,
                                    'source': source_name,
                                    'category': category,
                                    'type': 'classification',
                                    'label': None,
                                    'original_split': split
                                })
    
    elif source_type == "yolo":
        # YOLO dataset with labels
        class_id = source_config.get('class_id', 0)
        
        for split in ['train', 'valid', 'test']:
            img_dir = source_path / split / 'images'
            label_dir = source_path / split / 'labels'
            
            if img_dir.exists() and label_dir.exists():
                for img_file in img_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        label_file = label_dir / (img_file.stem + '.txt')
                        
                        # Check if this image has anthracnose label
                        has_anthracnose = False
                        if label_file.exists():
                            with open(label_file, 'r') as f:
                                for line in f:
                                    if line.strip().startswith(str(class_id) + ' '):
                                        has_anthracnose = True
                                        break
                        
                        if has_anthracnose and is_valid_image(img_file):
                            images.append({
                                'path': img_file,
                                'source': source_name,
                                'category': category,
                                'type': 'yolo',
                                'label': label_file if label_file.exists() else None,
                                'original_split': split
                            })
    
    return images


def deduplicate_images(images):
    """Remove duplicate images based on hash"""
    print("\n🔍 Deduplicating images...")
    
    hash_to_images = defaultdict(list)
    unique_images = []
    duplicates = []
    
    for img_data in tqdm(images, desc="Computing hashes"):
        img_hash = compute_image_hash(img_data['path'])
        if img_hash:
            img_data['hash'] = img_hash
            hash_to_images[img_hash].append(img_data)
    
    for img_hash, img_list in hash_to_images.items():
        if len(img_list) == 1:
            unique_images.append(img_list[0])
        else:
            # Keep the first one, mark others as duplicates
            unique_images.append(img_list[0])
            duplicates.extend(img_list[1:])
    
    print(f"✅ Found {len(unique_images)} unique images")
    print(f"🗑️  Removed {len(duplicates)} duplicates")
    
    return unique_images, duplicates


def split_dataset(images, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test"""
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = images[:train_end]
    val_set = images[train_end:val_end]
    test_set = images[val_end:]
    
    return {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }


def copy_images_to_unified(splits, output_dir):
    """Copy images to unified directory structure"""
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images' / 'anthracnose').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels' / 'anthracnose').mkdir(parents=True, exist_ok=True)
    
    stats = {
        'train': {'fruits': 0, 'leaves': 0, 'total': 0},
        'val': {'fruits': 0, 'leaves': 0, 'total': 0},
        'test': {'fruits': 0, 'leaves': 0, 'total': 0}
    }
    
    manifest = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split_name, images in splits.items():
        print(f"\n📁 Processing {split_name} set ({len(images)} images)...")
        
        for idx, img_data in enumerate(tqdm(images, desc=f"Copying {split_name}")):
            category = img_data['category']
            ext = img_data['path'].suffix
            
            # Create unique filename
            new_filename = f"{category}_{split_name}_{idx:05d}{ext}"
            dest_img = output_dir / split_name / 'images' / 'anthracnose' / new_filename
            
            # Copy image
            shutil.copy2(img_data['path'], dest_img)
            
            # Copy label if exists (YOLO)
            if img_data.get('label') and img_data['label'].exists():
                dest_label = output_dir / split_name / 'labels' / 'anthracnose' / (new_filename.replace(ext, '.txt'))
                shutil.copy2(img_data['label'], dest_label)
            
            # Update stats
            stats[split_name][category] += 1
            stats[split_name]['total'] += 1
            
            # Add to manifest
            manifest[split_name].append({
                'filename': new_filename,
                'original_path': str(img_data['path']),
                'source': img_data['source'],
                'category': category,
                'has_label': img_data.get('label') is not None
            })
    
    return stats, manifest


def create_dataset_yaml(output_dir, stats):
    """Create YOLO-compatible data.yaml"""
    yaml_content = f"""# Anthracnose Disease Detection Dataset
# Created: {Path(__file__).name}

path: {output_dir.absolute()}
train: train/images/anthracnose
val: val/images/anthracnose
test: test/images/anthracnose

# Classes
nc: 1
names: ['anthracnose']

# Dataset Statistics
# Train: {stats['train']['total']} images (Fruits: {stats['train']['fruits']}, Leaves: {stats['train']['leaves']})
# Val:   {stats['val']['total']} images (Fruits: {stats['val']['fruits']}, Leaves: {stats['val']['leaves']})
# Test:  {stats['test']['total']} images (Fruits: {stats['test']['fruits']}, Leaves: {stats['test']['leaves']})
# Total: {stats['train']['total'] + stats['val']['total'] + stats['test']['total']} images
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created {yaml_path}")


def main():
    print("=" * 70)
    print("🥭 ANTHRACNOSE UNIFIED DATASET PREPARATION")
    print("=" * 70)
    
    # Collect all images
    print("\n📂 Collecting images from sources...")
    
    all_images = []
    
    print("\n🍋 Processing FRUIT sources...")
    for source_name, config in FRUIT_SOURCES.items():
        if config.get('skip', False):
            print(f"⏭️  Skipping {source_name} (duplicates/unclear labels)")
            continue
        
        images = collect_images_from_source(source_name, config, ANTHRACNOSE_DIR, 'fruits')
        all_images.extend(images)
        print(f"  ✓ {source_name}: {len(images)} images")
    
    print("\n🍃 Processing LEAF sources...")
    for source_name, config in LEAF_SOURCES.items():
        if config.get('skip', False):
            print(f"⏭️  Skipping {source_name}")
            continue
        
        images = collect_images_from_source(source_name, config, ANTHRACNOSE_DIR, 'leaves')
        all_images.extend(images)
        print(f"  ✓ {source_name}: {len(images)} images")
    
    print(f"\n📊 Total collected: {len(all_images)} images")
    
    # Deduplicate
    unique_images, duplicates = deduplicate_images(all_images)
    
    # Split dataset
    print("\n✂️  Splitting dataset (70% train / 15% val / 15% test)...")
    splits = split_dataset(unique_images)
    
    print(f"  📈 Train: {len(splits['train'])} images")
    print(f"  📊 Val:   {len(splits['val'])} images")
    print(f"  📉 Test:  {len(splits['test'])} images")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy images to unified structure
    print("\n📦 Creating unified dataset structure...")
    stats, manifest = copy_images_to_unified(splits, OUTPUT_DIR)
    
    # Create data.yaml
    create_dataset_yaml(OUTPUT_DIR, stats)
    
    # Save manifest
    manifest_path = OUTPUT_DIR / 'dataset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Created {manifest_path}")
    
    # Save duplicate report
    duplicate_report = OUTPUT_DIR / 'duplicates_report.json'
    with open(duplicate_report, 'w') as f:
        json.dump([{'path': str(d['path']), 'source': d['source']} for d in duplicates], f, indent=2)
    print(f"✅ Created {duplicate_report}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("✨ DATASET CREATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\n📊 Final Statistics:")
    print(f"  Train: {stats['train']['total']:4d} images (Fruits: {stats['train']['fruits']:4d}, Leaves: {stats['train']['leaves']:4d})")
    print(f"  Val:   {stats['val']['total']:4d} images (Fruits: {stats['val']['fruits']:4d}, Leaves: {stats['val']['leaves']:4d})")
    print(f"  Test:  {stats['test']['total']:4d} images (Fruits: {stats['test']['fruits']:4d}, Leaves: {stats['test']['leaves']:4d})")
    print(f"  " + "-" * 50)
    print(f"  TOTAL: {stats['train']['total'] + stats['val']['total'] + stats['test']['total']:4d} images")
    
    print("\n📂 Directory structure created:")
    print(f"""
{OUTPUT_DIR}/
├── data.yaml                    # YOLO dataset configuration
├── dataset_manifest.json        # Detailed file tracking
├── duplicates_report.json       # Duplicate images removed
├── train/
│   ├── images/anthracnose/     # Training images
│   └── labels/anthracnose/     # Training labels (if available)
├── val/
│   ├── images/anthracnose/     # Validation images
│   └── labels/anthracnose/     # Validation labels (if available)
└── test/
    ├── images/anthracnose/     # Test images
    └── labels/anthracnose/     # Test labels (if available)
    """)
    
    print("\n🚀 Next steps:")
    print("  1. Review the dataset in: data/unified/anthracnose/")
    print("  2. Check dataset_manifest.json for image sources")
    print("  3. Verify duplicates_report.json for removed images")
    print("  4. Start training your anthracnose detection model!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
