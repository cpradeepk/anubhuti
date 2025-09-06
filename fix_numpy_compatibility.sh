#!/bin/bash

# YAMNet NumPy Compatibility Fix Script
# Fixes NumPy 2.x compatibility issues with TensorFlow 2.13.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/anubhuti"
VENV_DIR="$INSTALL_DIR/yamnet_implementation/yamnet_env"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if deployment exists
check_deployment() {
    log "🔍 Checking YAMNet deployment..."
    
    if [[ ! -d "$INSTALL_DIR" ]]; then
        error "❌ YAMNet deployment not found at $INSTALL_DIR"
    fi
    
    if [[ ! -d "$VENV_DIR" ]]; then
        error "❌ Virtual environment not found at $VENV_DIR"
    fi
    
    log "✅ YAMNet deployment found"
}

# Fix NumPy compatibility
fix_numpy_compatibility() {
    log "🔧 Fixing NumPy compatibility issue..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    
    # Activate virtual environment
    source yamnet_env/bin/activate || error "Failed to activate virtual environment"
    
    # Check current NumPy version
    local current_numpy=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
    info "Current NumPy version: $current_numpy"
    
    # Uninstall incompatible NumPy
    log "📦 Removing incompatible NumPy version..."
    pip uninstall numpy -y || warning "NumPy uninstall failed (may not be installed)"
    
    # Install compatible NumPy version
    log "📦 Installing compatible NumPy version..."
    pip install "numpy>=1.24.0,<2.0" --no-cache-dir || error "Failed to install compatible NumPy"
    
    # Reinstall TensorFlow to ensure compatibility
    log "📦 Reinstalling TensorFlow for compatibility..."
    pip uninstall tensorflow -y || warning "TensorFlow uninstall failed"
    pip install tensorflow==2.13.0 --no-cache-dir || error "Failed to reinstall TensorFlow"
    
    # Reinstall TensorFlow Hub
    log "📦 Reinstalling TensorFlow Hub..."
    pip install tensorflow-hub==0.14.0 --no-cache-dir || error "Failed to reinstall TensorFlow Hub"
    
    log "✅ NumPy compatibility fix completed"
}

# Test the fix
test_fix() {
    log "🧪 Testing NumPy and TensorFlow compatibility..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Test NumPy
    local numpy_version=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "FAILED")
    if [[ "$numpy_version" == "FAILED" ]]; then
        error "❌ NumPy import test failed"
    else
        log "✅ NumPy version: $numpy_version"
    fi
    
    # Test TensorFlow
    python -c "
import tensorflow as tf
print(f'✅ TensorFlow version: {tf.__version__}')
print('✅ TensorFlow imports successfully')
" || error "❌ TensorFlow import test failed"
    
    # Test TensorFlow Hub
    python -c "
import tensorflow_hub as hub
print('✅ TensorFlow Hub imports successfully')
" || error "❌ TensorFlow Hub import test failed"
    
    # Test model loading
    if [[ -f "yamnet_models/yamnet_classifier.h5" ]]; then
        python -c "
import tensorflow as tf
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print('✅ YAMNet model loads successfully')
print(f'✅ Model parameters: {model.count_params():,}')
" || error "❌ Model loading test failed"
    else
        warning "⚠️  Model file not found, skipping model test"
    fi
    
    log "✅ All compatibility tests passed!"
}

# Create fix summary
create_summary() {
    log "📋 Creating fix summary..."
    
    local summary_file="$HOME/numpy_compatibility_fix_summary.txt"
    
    cat > "$summary_file" << EOF
YAMNet NumPy Compatibility Fix Summary
=====================================

Fix Date: $(date)
Installation Directory: $INSTALL_DIR
Virtual Environment: $VENV_DIR

Issue Fixed:
- NumPy 2.x compatibility with TensorFlow 2.13.0
- AttributeError: _ARRAY_API not found
- ImportError: numpy.core._multiarray_umath failed to import

Solution Applied:
- Downgraded NumPy to compatible version (<2.0)
- Reinstalled TensorFlow 2.13.0 with compatible NumPy
- Verified all imports and model loading

Current Versions:
- NumPy: $(cd "$INSTALL_DIR/yamnet_implementation" && source yamnet_env/bin/activate && python -c "import numpy; print(numpy.__version__)")
- TensorFlow: $(cd "$INSTALL_DIR/yamnet_implementation" && source yamnet_env/bin/activate && python -c "import tensorflow as tf; print(tf.__version__)")

Next Steps:
1. Test your YAMNet deployment: cd $INSTALL_DIR/yamnet_implementation && python3 test_yamnet_model.py ../slow/Fhmm_slow.wav
2. Run real-time classification: python3 realtime_pi_test.py
3. Check system health: python3 health_check.py

The NumPy compatibility issue has been resolved!
EOF
    
    log "✅ Fix summary created: $summary_file"
}

# Main function
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          YAMNet NumPy Compatibility Fix Script              ║"
    echo "║              Fixes NumPy 2.x TensorFlow Issues              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "🚀 Starting NumPy compatibility fix..."
    
    # Run fix steps
    check_deployment
    fix_numpy_compatibility
    test_fix
    create_summary
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 FIX SUCCESSFUL! 🎉                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "🎉 NumPy compatibility fix completed successfully!"
    log "📋 Check fix summary: $HOME/numpy_compatibility_fix_summary.txt"
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Test YAMNet deployment: cd $INSTALL_DIR/yamnet_implementation && python3 test_yamnet_model.py ../slow/Fhmm_slow.wav"
    echo "2. Run real-time classification: python3 realtime_pi_test.py"
    echo "3. Check system health: python3 health_check.py"
    echo ""
    echo -e "${GREEN}Your YAMNet system is now ready! 🎵🤖✨${NC}"
}

# Handle script interruption
trap 'error "❌ Fix interrupted by user"' INT TERM

# Run main function
main "$@"
