"""
Organize training results into model and archive folders
- model/ : Contains the main production model file
- archive/ : Contains all other files (checkpoints, visualizations, logs)
"""

import shutil
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent / 'results'
MODEL_DIR = RESULTS_DIR / 'model'
ARCHIVE_DIR = RESULTS_DIR / 'archive'

# File organization
MODEL_FILES = [
    'anthracnose_detection_model.pth',  # Final production model
]

ARCHIVE_FILES = [
    'best_model.pth',                    # Best checkpoint during training
    'checkpoint_epoch_10.pth',           # Intermediate checkpoint
    'checkpoint_epoch_20.pth',           # Intermediate checkpoint
    'confusion_matrix.png',              # Confusion matrix visualization
    'sample_images.png',                 # Sample images visualization
    'test_results.csv',                  # Test metrics
    'training_history.csv',              # Training progression
    'training_history.png',              # Training curves
]

def organize_results():
    """Organize results into model and archive folders"""
    
    print("=" * 70)
    print("📁 ORGANIZING TRAINING RESULTS")
    print("=" * 70)
    
    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"\n❌ Error: Results directory not found: {RESULTS_DIR}")
        return
    
    # Create directories
    MODEL_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    print(f"\n✅ Created directories:")
    print(f"   📂 {MODEL_DIR}")
    print(f"   📂 {ARCHIVE_DIR}")
    
    # Move model files
    print(f"\n📦 Moving MODEL files:")
    for filename in MODEL_FILES:
        source = RESULTS_DIR / filename
        if source.exists():
            dest = MODEL_DIR / filename
            shutil.move(str(source), str(dest))
            print(f"   ✅ {filename} → model/")
        else:
            print(f"   ⚠️  {filename} not found, skipping")
    
    # Move archive files
    print(f"\n📚 Moving ARCHIVE files:")
    for filename in ARCHIVE_FILES:
        source = RESULTS_DIR / filename
        if source.exists():
            dest = ARCHIVE_DIR / filename
            shutil.move(str(source), str(dest))
            print(f"   ✅ {filename} → archive/")
        else:
            print(f"   ⚠️  {filename} not found, skipping")
    
    # Summary
    print("\n" + "=" * 70)
    print("✨ ORGANIZATION COMPLETE!")
    print("=" * 70)
    
    model_files = list(MODEL_DIR.glob('*'))
    archive_files = list(ARCHIVE_DIR.glob('*'))
    
    print(f"\n📂 model/ ({len(model_files)} files):")
    for file in sorted(model_files):
        print(f"   - {file.name}")
    
    print(f"\n📂 archive/ ({len(archive_files)} files):")
    for file in sorted(archive_files):
        print(f"   - {file.name}")
    
    print("\n" + "=" * 70)
    print("📝 USAGE:")
    print("=" * 70)
    print("\n🚀 For Production Deployment:")
    print(f"   Use: {MODEL_DIR / 'anthracnose_detection_model.pth'}")
    print("\n📚 For Analysis/Reference:")
    print(f"   Check: {ARCHIVE_DIR}")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    organize_results()
