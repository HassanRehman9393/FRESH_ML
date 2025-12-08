"""
Quick helper script to check if you're ready to run the dataset preparation
"""

from pathlib import Path

BASE_DIR = Path(r'c:\Users\hp\Desktop\FYP\FRESH_ML')
HEALTHY_DIR = BASE_DIR / 'healthy_citrus'

print("\n" + "="*80)
print("🔍 CITRUS BLACK SPOT DATASET - READINESS CHECK")
print("="*80)

# Check blackspot sources
blackspot_found = False
yolo_dir = BASE_DIR / 'citrus_blackspot_clean' / 'train' / 'images'
if yolo_dir.exists():
    blackspot_count = len(list(yolo_dir.glob('*.jpg')))
    print(f"\n✅ Blackspot images: {blackspot_count} found")
    blackspot_found = True
else:
    print("\n❌ Blackspot images: NOT FOUND")

# Check healthy images
healthy_found = False
if HEALTHY_DIR.exists():
    healthy_count = len(list(HEALTHY_DIR.glob('*.jpg')) + list(HEALTHY_DIR.glob('*.jpeg')) + list(HEALTHY_DIR.glob('*.png')))
    if healthy_count > 0:
        print(f"✅ Healthy images: {healthy_count} found in {HEALTHY_DIR}")
        healthy_found = True
    else:
        print(f"❌ Healthy images: Folder exists but is EMPTY")
        print(f"   Location: {HEALTHY_DIR}")
else:
    print(f"❌ Healthy images: Folder NOT FOUND")
    print(f"   Expected location: {HEALTHY_DIR}")

# Overall status
print("\n" + "="*80)
if blackspot_found and healthy_found:
    print("✅ READY TO RUN!")
    print("="*80)
    print("\nNext step:")
    print("C:/Users/hp/Desktop/FYP/FRESH_ML/.venv/Scripts/python.exe scripts\\prepare_blackspot_classification_dataset.py")
else:
    print("❌ NOT READY YET")
    print("="*80)
    
    if not healthy_found:
        print("\n📝 TO DO:")
        print(f"1. Create folder: mkdir {HEALTHY_DIR}")
        print("2. Add 2,000-3,000 healthy citrus/orange fruit images")
        print("3. Run this check again: python scripts\\check_readiness.py")
        print("\n💡 TIP: Download from Kaggle.com - search 'orange fruit dataset'")

print("\n" + "="*80)
