import torch

models = {
    'anthracnose': 'models_cache/anthracnose_detection_model.pth',
    'citrus_canker': 'models_cache/citrus_canker_detection_model.pth',
    'blackspot': 'models_cache/citrus_blackspot_detection_model.pth',
    'fruitfly': 'models_cache/guava_fruitfly_detection_model.pth'
}

print("="*70)
print("DISEASE MODEL CLASS ORDER CHECK")
print("="*70)

for name, path in models.items():
    print(f"\n{name.upper()}:")
    checkpoint = torch.load(path, map_location='cpu')
    class_names = checkpoint.get('class_names', checkpoint.get('classes', 'NOT FOUND'))
    print(f"  Classes: {class_names}")
    if isinstance(class_names, list):
        for idx, cls in enumerate(class_names):
            print(f"    Index {idx}: {cls}")
