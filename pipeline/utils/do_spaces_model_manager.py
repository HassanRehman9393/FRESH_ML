"""
Local Model Manager
===================

Serves ML models directly from the local models/ folder.
Digital Ocean Spaces integration has been replaced with local file serving.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple


LOCAL_MODELS_DIR = Path("models")

MODEL_FILES = {
    'yolo_detection_best.pt':           'yolo_detection_best.pt',
    'yolov11s_best.pt':                 'yolov11s_best.pt',
    'classification_best_fixed.pth':    'classification_best_fixed.pth',
    'anthracnose_detection_model.pth':  'anthracnose_detection_model.pth',
    'citrus_canker_detection_model.pth': 'citrus_canker_detection_model.pth',
    'citrus_blackspot_detection_model.pth': 'citrus_blackspot_detection_model.pth',
    'guava_fruitfly_detection_model.pth': 'guava_fruitfly_detection_model.pth',
    'yield_model.joblib':               'yield_model.joblib',
}


class DOSpacesModelManager:
    """
    Drop-in replacement for the Digital Ocean Spaces model manager.
    Reads models directly from the local models/ directory.
    """

    def __init__(self):
        self.models_dir = LOCAL_MODELS_DIR
        self.models = MODEL_FILES
        # Keep cache_dir attribute for any code that references it
        self.cache_dir = LOCAL_MODELS_DIR

    def is_model_cached(self, model_name: str) -> Tuple[bool, Optional[Path]]:
        model_path = self.models_dir / model_name
        if model_path.exists():
            return True, model_path
        return False, None

    def get_model_path(self, model_name: str) -> Optional[str]:
        print(f"🔍 Getting model: {model_name}")
        model_path = self.models_dir / model_name
        if model_path.exists():
            print(f"✅ Using local model: {model_name}")
            return str(model_path)
        print(f"❌ Model not found in models/ folder: {model_name}")
        return None

    def list_available_models(self) -> Dict[str, str]:
        return {
            name: str(self.models_dir / filename)
            for name, filename in self.models.items()
            if (self.models_dir / filename).exists()
        }

    def get_cache_info(self) -> Dict:
        info = {
            "cache_dir": str(self.models_dir),
            "cache_duration_hours": "∞ (local)",
            "cached_models": [],
            "total_cache_size_mb": 0,
        }
        total = 0
        for f in self.models_dir.glob("*.p*"):
            size_mb = f.stat().st_size / (1024 * 1024)
            total += size_mb
            info["cached_models"].append({
                "name": f.name,
                "size_mb": round(size_mb, 2),
                "cached_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        for f in self.models_dir.glob("*.joblib"):
            size_mb = f.stat().st_size / (1024 * 1024)
            total += size_mb
            info["cached_models"].append({
                "name": f.name,
                "size_mb": round(size_mb, 2),
                "cached_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        info["total_cache_size_mb"] = round(total, 2)
        return info

    def clear_cache(self) -> bool:
        print("ℹ️  Local model manager: clear_cache is a no-op (models are not cached, they live in models/)")
        return True


# ---------------------------------------------------------------------------
# Global instance + convenience functions (same API as the old DO manager)
# ---------------------------------------------------------------------------

_model_manager: Optional[DOSpacesModelManager] = None


def get_model_manager() -> DOSpacesModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = DOSpacesModelManager()
    return _model_manager


def get_model_path(model_name: str) -> Optional[str]:
    return get_model_manager().get_model_path(model_name)


def list_available_models() -> Dict[str, str]:
    return get_model_manager().list_available_models()


def clear_model_cache() -> bool:
    return get_model_manager().clear_cache()


def main():
    print("🧪 Local Model Manager")
    print("=" * 50)
    manager = DOSpacesModelManager()

    print(f"📁 Models directory: {manager.models_dir.resolve()}")
    info = manager.get_cache_info()
    print(f"   Found {len(info['cached_models'])} model file(s) ({info['total_cache_size_mb']} MB total)")

    print("\n📋 Available models:")
    for name, path in manager.list_available_models().items():
        print(f"   ✅ {name}")

    missing = [n for n in MODEL_FILES if not (manager.models_dir / n).exists()]
    if missing:
        print("\n⚠️  Missing models (not in models/ folder):")
        for name in missing:
            print(f"   ❌ {name}")


if __name__ == "__main__":
    main()
