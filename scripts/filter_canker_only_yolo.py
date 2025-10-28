"""
Extract ONLY Citrus Canker images from YOLO object detection dataset
Filters out blackspot and greening, keeping only pure canker images
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# Paths
YOLO_DATASET = Path(r'd:\FYP\FRESH_ML\Citrus Canker\Canker fruit\canker fruit.v5i.yolov11')
OUTPUT_DIR = Path(r'd:\FYP\FRESH_ML\Citrus Canker\Canker fruit\Filtered Canker Only')

# Class mapping from data.yaml
CLASS_MAP = {
    0: 'blackspot',
    1: 'citrus-canker',  # WE WANT THIS ONE
    2: 'greening'
}

def read_yolo_label(label_file):
    """
    Read YOLO label file and return list of class IDs
    Format: class_id x_center y_center width height
    """
    classes = []
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    classes.append(class_id)
    return classes

def filter_canker_only_images():
    """
    Extract images that ONLY contain citrus-canker (class 1)
    Exclude any images with blackspot (class 0) or greening (class 2)
    """
    
    print("=" * 80)
    print("🍊 FILTERING CITRUS CANKER IMAGES FROM YOLO DATASET")
    print("=" * 80)
    
    # Statistics
    stats = {
        'train': defaultdict(int),
        'valid': defaultdict(int),
        'test': defaultdict(int)
    }
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        output_split_dir = OUTPUT_DIR / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    print(f"✅ Created train/, valid/, test/ folders")
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n" + "=" * 80)
        print(f"📦 Processing {split.upper()} split")
        print("=" * 80)
        
        images_dir = YOLO_DATASET / split / 'images'
        labels_dir = YOLO_DATASET / split / 'labels'
        
        if not images_dir.exists():
            print(f"⚠️  {split} images directory not found, skipping")
            continue
        
        # Get all images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"\n📊 Total images in {split}: {len(image_files)}")
        
        canker_only_count = 0
        mixed_disease_count = 0
        blackspot_only_count = 0
        greening_only_count = 0
        no_label_count = 0
        
        # Process each image
        for image_file in image_files:
            # Find corresponding label file
            label_file = labels_dir / (image_file.stem + '.txt')
            
            if not label_file.exists():
                stats[split]['no_label'] += 1
                no_label_count += 1
                continue
            
            # Read classes in this image
            classes = read_yolo_label(label_file)
            
            if not classes:
                stats[split]['no_label'] += 1
                no_label_count += 1
                continue
            
            # Get unique classes
            unique_classes = set(classes)
            
            # Check if ONLY citrus-canker (class 1)
            if unique_classes == {1}:
                # ✅ PURE CANKER - Copy this image
                output_path = OUTPUT_DIR / split / image_file.name
                shutil.copy2(image_file, output_path)
                stats[split]['canker_only'] += 1
                canker_only_count += 1
            
            # Check if mixed diseases
            elif 1 in unique_classes and len(unique_classes) > 1:
                # ⚠️ MIXED - Has canker + other diseases
                stats[split]['mixed_canker'] += 1
                mixed_disease_count += 1
            
            # Check if only blackspot
            elif unique_classes == {0}:
                stats[split]['blackspot_only'] += 1
                blackspot_only_count += 1
            
            # Check if only greening
            elif unique_classes == {2}:
                stats[split]['greening_only'] += 1
                greening_only_count += 1
            
            # Other combinations
            else:
                stats[split]['other'] += 1
        
        # Print split summary
        print(f"\n📊 {split.upper()} Summary:")
        print(f"   ✅ Canker ONLY (pure):        {canker_only_count:,} images (COPIED)")
        print(f"   ⚠️  Mixed (canker + others):  {mixed_disease_count:,} images (EXCLUDED)")
        print(f"   ❌ Blackspot only:            {blackspot_only_count:,} images (EXCLUDED)")
        print(f"   ❌ Greening only:             {greening_only_count:,} images (EXCLUDED)")
        print(f"   ⚠️  No label:                 {no_label_count:,} images (EXCLUDED)")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("📊 OVERALL SUMMARY")
    print("=" * 80)
    
    total_canker_only = stats['train']['canker_only'] + stats['valid']['canker_only'] + stats['test']['canker_only']
    total_mixed = stats['train']['mixed_canker'] + stats['valid']['mixed_canker'] + stats['test']['mixed_canker']
    total_blackspot = stats['train']['blackspot_only'] + stats['valid']['blackspot_only'] + stats['test']['blackspot_only']
    total_greening = stats['train']['greening_only'] + stats['valid']['greening_only'] + stats['test']['greening_only']
    
    print(f"\n✅ EXTRACTED (Pure Canker Only):")
    print(f"   Train:  {stats['train']['canker_only']:,} images")
    print(f"   Valid:  {stats['valid']['canker_only']:,} images")
    print(f"   Test:   {stats['test']['canker_only']:,} images")
    print(f"   Total:  {total_canker_only:,} images ✅")
    
    print(f"\n❌ EXCLUDED:")
    print(f"   Mixed diseases (canker + others): {total_mixed:,} images")
    print(f"   Blackspot only:                   {total_blackspot:,} images")
    print(f"   Greening only:                    {total_greening:,} images")
    print(f"   Total excluded:                   {total_mixed + total_blackspot + total_greening:,} images")
    
    # Create summary file
    summary_file = OUTPUT_DIR / 'FILTERING_SUMMARY.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CITRUS CANKER FILTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Source: canker fruit.v5i.yolov11/\n")
        f.write("Filter: ONLY images with pure citrus-canker (class 1)\n")
        f.write("Excluded: Mixed diseases, blackspot, greening\n\n")
        
        f.write("EXTRACTED IMAGES:\n")
        f.write(f"  Train:  {stats['train']['canker_only']:,} images\n")
        f.write(f"  Valid:  {stats['valid']['canker_only']:,} images\n")
        f.write(f"  Test:   {stats['test']['canker_only']:,} images\n")
        f.write(f"  TOTAL:  {total_canker_only:,} images\n\n")
        
        f.write("EXCLUDED:\n")
        f.write(f"  Mixed diseases: {total_mixed:,} images\n")
        f.write(f"  Blackspot only: {total_blackspot:,} images\n")
        f.write(f"  Greening only:  {total_greening:,} images\n\n")
        
        f.write("BREAKDOWN BY SPLIT:\n")
        for split in ['train', 'valid', 'test']:
            f.write(f"\n{split.upper()}:\n")
            f.write(f"  Canker only:    {stats[split]['canker_only']:,}\n")
            f.write(f"  Mixed:          {stats[split]['mixed_canker']:,}\n")
            f.write(f"  Blackspot only: {stats[split]['blackspot_only']:,}\n")
            f.write(f"  Greening only:  {stats[split]['greening_only']:,}\n")
    
    print(f"\n📄 Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("✨ FILTERING COMPLETE!")
    print("=" * 80)
    
    print(f"\n📁 Filtered images location:")
    print(f"   {OUTPUT_DIR}")
    print(f"\n💡 Next steps:")
    print(f"   1. You now have {total_canker_only:,} PURE canker images")
    print(f"   2. Combine with Fruits/Canker (78 images) = {total_canker_only + 78:,} total canker")
    print(f"   3. Still need healthy images from Fruits/healthy (22 images)")
    print(f"   4. Or combine with Sweetorange dataset (588 canker + 547 healthy)")
    
    return stats

if __name__ == '__main__':
    stats = filter_canker_only_images()
