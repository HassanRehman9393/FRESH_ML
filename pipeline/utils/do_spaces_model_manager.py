"""
Digital Ocean Spaces Model Manager
==================================

This module handles downloading and caching ML models from Digital Ocean Spaces.
It provides seamless integration with the existing pipeline while using 
remote model storage.
"""

import os
import hashlib
import requests
import boto3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables
load_dotenv()

class DOSpacesModelManager:
    """Manages downloading and caching models from Digital Ocean Spaces"""
    
    def __init__(self):
        self.access_key = os.getenv('DO_SPACES_ACCESS_KEY')
        self.secret_key = os.getenv('DO_SPACES_SECRET_KEY')
        self.endpoint = os.getenv('DO_SPACES_ENDPOINT')
        self.bucket = os.getenv('DO_SPACES_BUCKET')
        self.region = os.getenv('DO_SPACES_REGION')
        
        if not all([self.access_key, self.secret_key, self.endpoint, self.bucket]):
            raise ValueError("Missing Digital Ocean Spaces configuration in .env file")
        
        # Initialize S3 client for signed URLs
        self.s3_client = boto3.client('s3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
        
        # Configuration
        self.cache_dir = Path("models_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache settings
        self.cache_duration = timedelta(hours=24)  # Cache models for 24 hours
        
        # Model mapping
        self.models = {
            'yolo_detection_best.pt': 'models/yolo_detection_best.pt',
            'classification_best_fixed.pth': 'models/classification_best_fixed.pth',
            'anthracnose_detection_model.pth': 'models/anthracnose_detection_model.pth',
            'citrus_canker_detection_model.pth': 'models/citrus_canker_detection_model.pth'
        }
    
    def is_model_cached(self, model_name: str) -> Tuple[bool, Optional[Path]]:
        """Check if model is cached locally and valid"""
        model_path = self.cache_dir / model_name
        
        if not model_path.exists():
            return False, None
        
        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
        if cache_age > self.cache_duration:
            print(f"📅 Cached model {model_name} expired")
            return False, None
        
        print(f"✅ Using cached model: {model_name}")
        return True, model_path
    
    def create_signed_url(self, model_name: str, expires_in: int = 3600) -> Optional[str]:
        """Create signed URL for model download"""
        try:
            if model_name not in self.models:
                print(f"❌ Unknown model: {model_name}")
                return None
            
            storage_path = self.models[model_name]
            
            signed_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': storage_path},
                ExpiresIn=expires_in
            )
            
            return signed_url
            
        except Exception as e:
            print(f"❌ Error creating signed URL for {model_name}: {e}")
            return None
    
    def download_model(self, model_name: str) -> Optional[Path]:
        """Download model from Digital Ocean Spaces"""
        try:
            print(f"📥 Downloading {model_name} from Digital Ocean Spaces...")
            
            # Create signed URL
            signed_url = self.create_signed_url(model_name)
            if not signed_url:
                return None
            
            # Download using requests for better progress tracking
            response = requests.get(signed_url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            
            # Download to temporary file first
            temp_path = self.cache_dir / f"{model_name}.tmp"
            downloaded_size = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"   Progress: {progress:.1f}% ({downloaded_size // 1024 // 1024} MB)", end='\r')
            
            print()  # New line after progress
            
            # Move to final location
            final_path = self.cache_dir / model_name
            shutil.move(temp_path, final_path)
            
            print(f"✅ Downloaded: {model_name} ({downloaded_size // 1024 // 1024} MB)")
            return final_path
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            
            # Clean up temp file if exists
            temp_path = self.cache_dir / f"{model_name}.tmp"
            if temp_path.exists():
                temp_path.unlink()
            
            return None
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get local path to model (download if necessary)"""
        print(f"🔍 Getting model: {model_name}")
        
        # Check if cached
        is_cached, cached_path = self.is_model_cached(model_name)
        if is_cached and cached_path:
            return str(cached_path)
        
        # Download model
        downloaded_path = self.download_model(model_name)
        if downloaded_path:
            return str(downloaded_path)
        
        print(f"❌ Failed to get model: {model_name}")
        return None
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available models"""
        return self.models
    
    def clear_cache(self) -> bool:
        """Clear model cache"""
        try:
            print("🧹 Clearing model cache...")
            
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            
            print("✅ Cache cleared")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict:
        """Get information about cached models"""
        cache_info = {
            "cache_dir": str(self.cache_dir),
            "cache_duration_hours": self.cache_duration.total_seconds() / 3600,
            "cached_models": [],
            "total_cache_size_mb": 0
        }
        
        if not self.cache_dir.exists():
            return cache_info
        
        total_size = 0
        for model_file in self.cache_dir.glob("*.p*"):
            stats = model_file.stat()
            size_mb = stats.st_size / (1024 * 1024)
            total_size += size_mb
            
            cache_info["cached_models"].append({
                "name": model_file.name,
                "size_mb": round(size_mb, 2),
                "cached_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "age_hours": round((datetime.now() - datetime.fromtimestamp(stats.st_mtime)).total_seconds() / 3600, 1)
            })
        
        cache_info["total_cache_size_mb"] = round(total_size, 2)
        return cache_info


# Global instance for easy access
_model_manager = None

def get_model_manager() -> DOSpacesModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = DOSpacesModelManager()
    return _model_manager

def get_model_path(model_name: str) -> Optional[str]:
    """Convenience function to get model path"""
    manager = get_model_manager()
    return manager.get_model_path(model_name)

def list_available_models() -> Dict[str, str]:
    """Convenience function to list available models"""
    manager = get_model_manager()
    return manager.list_available_models()

def clear_model_cache() -> bool:
    """Convenience function to clear cache"""
    manager = get_model_manager()
    return manager.clear_cache()


def main():
    """Test the model manager"""
    print("🧪 Testing Digital Ocean Spaces Model Manager")
    print("=" * 60)
    
    try:
        manager = DOSpacesModelManager()
        
        # List available models
        print("📋 Available models:")
        models = manager.list_available_models()
        for name, path in models.items():
            print(f"   • {name} → {path}")
        
        # Show cache info
        print(f"\n📁 Cache info:")
        cache_info = manager.get_cache_info()
        print(f"   Directory: {cache_info['cache_dir']}")
        print(f"   Duration: {cache_info['cache_duration_hours']} hours")
        print(f"   Cached models: {len(cache_info['cached_models'])}")
        print(f"   Total size: {cache_info['total_cache_size_mb']} MB")
        
        # Test downloading a model (YOLO is smaller)
        print(f"\n🧪 Testing download: yolo_detection_best.pt")
        model_path = manager.get_model_path('yolo_detection_best.pt')
        if model_path:
            print(f"✅ Model available at: {model_path}")
        else:
            print(f"❌ Failed to get model")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())