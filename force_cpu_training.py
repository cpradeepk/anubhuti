#!/usr/bin/env python3
"""
Force CPU Training Configuration
Workaround for GPU compatibility issues with compute capability 12.0
"""

import os
import tensorflow as tf

def configure_cpu_only():
    """Configure TensorFlow to use CPU only"""
    
    # Hide GPU from TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Set CPU optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    # Configure TensorFlow for optimal CPU performance
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
    
    print("✅ Configured TensorFlow for CPU-only training")
    print("✅ CPU optimizations enabled")
    
    # Test CPU performance
    import time
    import numpy as np
    
    print("🧪 Testing CPU performance...")
    
    # CPU matrix multiplication test
    a = tf.random.normal([2000, 2000])
    b = tf.random.normal([2000, 2000])
    
    start_time = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start_time
    
    print(f"✅ CPU computation test passed ({cpu_time*1000:.1f}ms)")
    print(f"✅ Using {tf.config.threading.get_intra_op_parallelism_threads()} CPU threads")
    
    return True

def test_yamnet_cpu():
    """Test YAMNet loading with CPU"""
    try:
        import tensorflow_hub as hub
        import numpy as np
        import time
        
        print("🧪 Testing YAMNet with CPU...")
        
        # Load YAMNet
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("✅ YAMNet loaded successfully")
        
        # Test with dummy audio
        dummy_audio = np.random.randn(16000 * 2)  # 2 seconds
        
        start_time = time.time()
        embeddings = yamnet_model(dummy_audio)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"✅ YAMNet CPU inference: {inference_time:.1f}ms")
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ YAMNet CPU test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuring CPU-only training for RTX 5060 compatibility")
    print("=" * 60)
    
    configure_cpu_only()
    success = test_yamnet_cpu()
    
    print("=" * 60)
    if success:
        print("🎉 CPU training configuration successful!")
        print("💡 Your training will use CPU instead of GPU")
        print("⏱️  Expected training time: 8-12 minutes (still quite fast)")
        print("")
        print("Next steps:")
        print("1. Run: python3 force_cpu_training.py")
        print("2. Then: ./run_enhanced_training.sh --dataset yamnet_implementation/../")
    else:
        print("❌ CPU configuration failed")
