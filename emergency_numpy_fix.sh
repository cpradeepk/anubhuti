#!/bin/bash

# Emergency NumPy Fix Script
# Quick fix for NumPy 2.x compatibility issues

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
INSTALL_DIR="$HOME/anubhuti"
VENV_DIR="$INSTALL_DIR/yamnet_implementation/yamnet_env"

echo -e "${GREEN}🚨 Emergency NumPy Fix for YAMNet Deployment${NC}"
echo "=================================================="

# Check deployment
if [[ ! -d "$INSTALL_DIR/yamnet_implementation" ]]; then
    error "YAMNet deployment not found at $INSTALL_DIR"
fi

cd "$INSTALL_DIR/yamnet_implementation"

# Activate virtual environment
log "🔧 Activating virtual environment..."
source yamnet_env/bin/activate || error "Failed to activate virtual environment"

# Create constraints file
log "📋 Creating version constraints..."
cat > constraints.txt << 'EOF'
numpy==1.24.3
tensorflow==2.13.0
tensorflow-hub==0.14.0
keras==2.13.1
scipy<1.12.0
EOF

# Emergency fix - force correct NumPy version
log "🚨 Emergency NumPy version fix..."
pip uninstall numpy -y || true
pip install --force-reinstall --no-deps numpy==1.24.3 || error "Failed to install NumPy 1.24.3"

# Reinstall TensorFlow with constraints
log "🔧 Reinstalling TensorFlow with version lock..."
pip uninstall tensorflow keras -y || true
pip install --constraint constraints.txt --no-cache-dir tensorflow==2.13.0 || error "Failed to install TensorFlow"

# Test the fix
log "🧪 Testing the emergency fix..."

# Test NumPy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "FAILED")
if [[ "$NUMPY_VERSION" == "1.24.3" ]]; then
    log "✅ NumPy version: $NUMPY_VERSION"
else
    error "❌ NumPy version incorrect: $NUMPY_VERSION (expected 1.24.3)"
fi

# Test TensorFlow import
if python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null; then
    log "✅ TensorFlow imports successfully"
else
    error "❌ TensorFlow import failed"
fi

# Test TensorFlow Hub
if python -c "import tensorflow_hub as hub; print('TensorFlow Hub: OK')" 2>/dev/null; then
    log "✅ TensorFlow Hub imports successfully"
else
    warning "⚠️  TensorFlow Hub import failed - reinstalling..."
    pip install --constraint constraints.txt tensorflow-hub==0.14.0
fi

# Test model loading if available
if [[ -f "yamnet_models/yamnet_classifier.h5" ]]; then
    log "🤖 Testing model loading..."
    if python -c "
import tensorflow as tf
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print('✅ Model loads successfully')
print(f'Model parameters: {model.count_params():,}')
" 2>/dev/null; then
        log "✅ Model loading test passed"
    else
        warning "⚠️  Model loading failed - may need retraining"
        echo "   Run: python3 train_yamnet_model.py --dataset ../"
    fi
else
    warning "⚠️  Model file not found - will need training"
fi

# Create backup of constraints file
cp constraints.txt constraints_backup.txt

echo ""
echo -e "${GREEN}🎉 Emergency fix completed!${NC}"
echo "================================"
echo ""
echo "✅ NumPy locked to version 1.24.3"
echo "✅ TensorFlow 2.13.0 working"
echo "✅ Version constraints file created"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Test audio classification: python3 test_yamnet_model.py ../slow/Fhmm_slow.wav"
echo "2. If model loading failed, retrain: python3 train_yamnet_model.py --dataset ../"
echo "3. Test real-time: python3 realtime_pi_test.py"
echo ""
echo -e "${GREEN}Your YAMNet system should now work! 🎵🤖✨${NC}"

# Save fix summary
cat > emergency_fix_summary.txt << EOF
Emergency NumPy Fix Summary
==========================
Fix Date: $(date)
NumPy Version: $NUMPY_VERSION
TensorFlow Version: $(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "Unknown")

Fix Applied:
- Forced NumPy downgrade to 1.24.3
- Reinstalled TensorFlow 2.13.0 with version constraints
- Created constraints.txt to prevent future upgrades

Status: $(if [[ "$NUMPY_VERSION" == "1.24.3" ]]; then echo "SUCCESS"; else echo "NEEDS ATTENTION"; fi)

Next Steps:
1. Test: python3 test_yamnet_model.py ../slow/Fhmm_slow.wav
2. If needed: python3 train_yamnet_model.py --dataset ../
3. Real-time: python3 realtime_pi_test.py
EOF

log "📋 Fix summary saved to: emergency_fix_summary.txt"
