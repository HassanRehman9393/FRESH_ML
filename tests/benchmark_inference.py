"""
Benchmark: Inference Speed Test
================================
Measure inference performance of the new YOLOv11s model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import cv2
import torch
import numpy as np
from pipeline.detection.yolo_detector import YOLODetector

def benchmark_inference():
    """Benchmark YOLOv11s inference speed"""
    print("⏱️ Benchmarking YOLO Inference Speed...")
    print("=" * 70)
    
    try:
        # Initialize detector
        print("\n📦 Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector = YOLODetector(
            model_path="models_cache/yolo_detection_best.pt",
            device=device
        )
        print(f"✅ Model loaded on {device}")
        
        # Create a dummy test image (512x512 RGB - matches training size)
        print("\n🖼️ Creating test image (512x512 RGB)...")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Warmup run (first inference is always slower)
        print("🔥 Warmup run...")
        _ = detector.detect(test_image)
        
        # Benchmark multiple runs
        num_runs = 100
        print(f"\n⏱️ Running {num_runs} inferences...")
        
        start_time = time.time()
        for i in range(num_runs):
            _ = detector.detect(test_image)
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i + 1}/{num_runs}", end='\r')
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = num_runs / total_time
        
        print("\n" + "=" * 70)
        print("📊 Benchmark Results:")
        print("-" * 70)
        print(f"   Device: {device}")
        print(f"   Total runs: {num_runs}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average inference time: {avg_time*1000:.2f} ms")
        print(f"   Throughput: {fps:.1f} FPS")
        print("-" * 70)
        
        # Performance assessment
        print("\n📈 Performance Assessment:")
        if device.type == "cuda":
            if fps >= 40:
                print(f"   ✅ EXCELLENT - {fps:.1f} FPS (target: ≥40 FPS on GPU)")
            elif fps >= 30:
                print(f"   ✅ GOOD - {fps:.1f} FPS (target: ≥30 FPS on GPU)")
            elif fps >= 20:
                print(f"   ⚠️ ACCEPTABLE - {fps:.1f} FPS (slightly below target)")
            else:
                print(f"   ❌ SLOW - {fps:.1f} FPS (below 20 FPS)")
        else:  # CPU
            if fps >= 10:
                print(f"   ✅ GOOD - {fps:.1f} FPS (target: ≥10 FPS on CPU)")
            elif fps >= 5:
                print(f"   ⚠️ ACCEPTABLE - {fps:.1f} FPS (target: ≥5 FPS on CPU)")
            else:
                print(f"   ❌ SLOW - {fps:.1f} FPS (below 5 FPS)")
        
        print("\n💡 Model Comparison (estimated):")
        print(f"   Old model (~6 MB): ~50 FPS on GPU")
        print(f"   New YOLOv11s (19.2 MB): ~{fps:.0f} FPS on {device}")
        print(f"   Trade-off: {'Better accuracy' if fps >= 30 else 'Accuracy gain with slight speed decrease'}")
        
        print("\n" + "=" * 70)
        print("✅ BENCHMARK COMPLETE")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ BENCHMARK FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = benchmark_inference()
    sys.exit(0 if success else 1)
