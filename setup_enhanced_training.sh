#!/bin/bash

# Enhanced YAMNet Training Environment Setup
# Compatible with latest TensorFlow versions

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Enhanced YAMNet Training Environment Setup         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    warning "Not in a virtual environment. Creating one..."
    
    # Create virtual environment
    python3 -m venv enhanced_yamnet_env
    source enhanced_yamnet_env/bin/activate
    log "✅ Created and activated virtual environment: enhanced_yamnet_env"
else
    log "✅ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Upgrade pip
log "📦 Upgrading pip..."
pip install --upgrade pip

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

log "🐍 Python version: $PYTHON_VERSION"

# Check if Python version is 3.8 or higher
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    error "Python 3.8+ required. Current version: $PYTHON_VERSION"
else
    log "✅ Python version check passed"
fi

# Install TensorFlow with GPU support
log "🚀 Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda] || {
    warning "GPU TensorFlow installation failed, trying CPU version..."
    pip install tensorflow
}

# Install other dependencies
log "📦 Installing other dependencies..."
pip install tensorflow-hub
pip install librosa soundfile
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install pygame  # For audio playback in labeling interface
pip install numpy  # Ensure compatible version

# Verify TensorFlow installation
log "🧪 Verifying TensorFlow installation..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
if gpus:
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.name}')
    
    # Test GPU functionality
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print('✅ GPU computation test passed')
    except Exception as e:
        print(f'⚠️  GPU computation test failed: {e}')
        print('Will use CPU for training')
else:
    print('ℹ️  No GPUs found, will use CPU')

# Check TensorFlow Hub
try:
    import tensorflow_hub as hub
    print('✅ TensorFlow Hub available')
except ImportError as e:
    print(f'❌ TensorFlow Hub error: {e}')

# Check audio libraries
try:
    import librosa
    import soundfile as sf
    print('✅ Audio libraries available')
except ImportError as e:
    print(f'❌ Audio libraries error: {e}')

print('🎉 Installation verification completed!')
"

# Create requirements file for future reference
log "📋 Creating requirements.txt..."
cat > requirements_enhanced.txt << EOF
# Enhanced YAMNet Training Requirements
# Compatible with latest TensorFlow versions

# Core ML libraries
tensorflow[and-cuda]>=2.15.0
tensorflow-hub>=0.15.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# Machine learning utilities
scikit-learn>=1.3.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
pygame>=2.5.0  # For audio playback in labeling

# Optional: Jupyter for analysis
jupyter>=1.0.0
ipykernel>=6.25.0
EOF

log "✅ Requirements saved to requirements_enhanced.txt"

# Test YAMNet loading
log "🧪 Testing YAMNet model loading..."
python3 -c "
import tensorflow_hub as hub
import tensorflow as tf
import time

print('Loading YAMNet model...')
start_time = time.time()
try:
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    load_time = time.time() - start_time
    print(f'✅ YAMNet loaded successfully in {load_time:.1f} seconds')
    
    # Test with dummy audio
    import numpy as np
    dummy_audio = np.random.randn(16000 * 3)  # 3 seconds of audio
    
    start_time = time.time()
    embeddings = yamnet_model(dummy_audio)
    inference_time = (time.time() - start_time) * 1000
    
    print(f'✅ YAMNet inference test passed ({inference_time:.1f}ms)')
    print(f'   Embeddings shape: {embeddings.shape}')
    
except Exception as e:
    print(f'❌ YAMNet test failed: {e}')
"

# Create optimized training configuration
log "⚙️  Creating optimized training configuration..."
cat > training_config.py << 'EOF'
"""
Optimized Training Configuration for Enhanced YAMNet
"""

import os
import tensorflow as tf

def setup_tensorflow_optimizations():
    """Setup TensorFlow for optimal performance"""
    
    # Environment variables
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"✅ GPU optimization enabled for {len(gpus)} GPU(s)")
            print("✅ Mixed precision enabled")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Enable XLA JIT compilation
    tf.config.optimizer.set_jit(True)
    print("✅ XLA JIT compilation enabled")

def get_optimal_batch_size():
    """Get optimal batch size based on available memory"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # GPU available - can use larger batch sizes
        return 32
    else:
        # CPU only - use smaller batch size
        return 16

def get_training_callbacks(model_path):
    """Get optimized training callbacks"""
    return [
        tf.keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=7,
            min_lr=1e-7,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_accuracy',
            save_weights_only=False
        )
    ]

if __name__ == "__main__":
    setup_tensorflow_optimizations()
    print(f"Optimal batch size: {get_optimal_batch_size()}")
EOF

log "✅ Training configuration created: training_config.py"

# Create quick test script
log "🧪 Creating quick test script..."
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of enhanced training environment"""

import sys
import time
import numpy as np

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow: {e}")
        return False
    
    try:
        import tensorflow_hub as hub
        print("✅ TensorFlow Hub")
    except ImportError as e:
        print(f"❌ TensorFlow Hub: {e}")
        return False
    
    try:
        import librosa
        import soundfile as sf
        print("✅ Audio libraries")
    except ImportError as e:
        print(f"❌ Audio libraries: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        print("✅ Scikit-learn")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU functionality"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("ℹ️  No GPUs found - will use CPU")
        return True
    
    print(f"Found {len(gpus)} GPU(s)")
    
    try:
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start_time = time.time()
            c = tf.matmul(a, b)
            gpu_time = time.time() - start_time
        
        print(f"✅ GPU computation test passed ({gpu_time*1000:.1f}ms)")
        return True
        
    except Exception as e:
        print(f"⚠️  GPU test failed: {e}")
        return False

def test_yamnet():
    """Test YAMNet loading"""
    import tensorflow_hub as hub
    import tensorflow as tf
    
    print("Testing YAMNet loading...")
    try:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Test with dummy audio
        dummy_audio = np.random.randn(16000 * 2)  # 2 seconds
        embeddings = yamnet_model(dummy_audio)
        
        print(f"✅ YAMNet test passed - embeddings shape: {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ YAMNet test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Enhanced YAMNet Training Environment Test")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_gpu()
    success &= test_yamnet()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! Environment is ready for enhanced training.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)
EOF

chmod +x quick_test.py

# Run quick test
log "🧪 Running environment test..."
python3 quick_test.py

echo ""
echo -e "${GREEN}🎉 Enhanced YAMNet training environment setup completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Test environment: python3 quick_test.py"
echo "2. Run enhanced training: ./run_enhanced_training.sh --dataset /path/to/dataset"
echo "3. Process real-world audio: ./run_enhanced_training.sh --real-world-audio audio.wav"
echo ""
echo -e "${GREEN}Your environment is ready for enhanced YAMNet training! 🎵🤖✨${NC}"
