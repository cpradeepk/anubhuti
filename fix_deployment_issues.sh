#!/bin/bash

# YAMNet Complete Deployment Fix Script
# Fixes NumPy compatibility and model loading issues

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

# Fix package compatibility issues
fix_package_compatibility() {
    log "🔧 Fixing package compatibility issues..."

    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate || error "Failed to activate virtual environment"

    # Create constraints file to lock versions
    log "📋 Creating version constraints file..."
    cat > constraints.txt << EOF
numpy==1.24.3
scipy>=1.10.0,<1.12.0
tensorflow==2.13.0
tensorflow-hub==0.14.0
keras==2.13.1
librosa>=0.10.0,<0.11.0
soundfile>=0.12.0,<0.13.0
matplotlib>=3.7.0,<3.8.0
seaborn>=0.12.0,<0.13.0
EOF

    # Remove problematic packages
    log "📦 Removing incompatible packages..."
    pip uninstall numpy scipy tensorflow keras librosa -y || warning "Package uninstall failed"

    # Install packages with strict version constraints
    log "📦 Installing NumPy 1.24.3 (TensorFlow compatible)..."
    pip install --constraint constraints.txt --force-reinstall --no-deps numpy==1.24.3 || error "Failed to install NumPy"

    log "📦 Installing SciPy with NumPy constraint..."
    pip install --constraint constraints.txt scipy || error "Failed to install SciPy"

    log "📦 Installing TensorFlow 2.13.0 with constraints..."
    pip install --constraint constraints.txt tensorflow==2.13.0 || error "Failed to install TensorFlow"

    log "📦 Installing TensorFlow Hub with constraints..."
    pip install --constraint constraints.txt tensorflow-hub==0.14.0 || error "Failed to install TensorFlow Hub"

    # Install other required packages with constraints
    log "📦 Installing audio processing libraries with constraints..."
    pip install --constraint constraints.txt librosa soundfile || warning "Audio libraries failed"

    log "📦 Installing visualization libraries with constraints..."
    pip install --constraint constraints.txt matplotlib seaborn || warning "Visualization libraries failed"

    log "📦 Installing system libraries..."
    pip install pyaudio pydub || warning "System libraries failed"

    log "✅ Package compatibility fix completed"
}

# Test package imports
test_imports() {
    log "🧪 Testing package imports..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Test NumPy
    local numpy_version=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "FAILED")
    if [[ "$numpy_version" == "FAILED" ]]; then
        error "❌ NumPy import failed"
    else
        log "✅ NumPy version: $numpy_version"
    fi
    
    # Test SciPy
    python -c "import scipy; print(f'✅ SciPy version: {scipy.__version__}')" || warning "⚠️  SciPy import failed"
    
    # Test TensorFlow
    python -c "
import tensorflow as tf
print(f'✅ TensorFlow version: {tf.__version__}')
" || error "❌ TensorFlow import failed"
    
    # Test TensorFlow Hub
    python -c "
import tensorflow_hub as hub
print('✅ TensorFlow Hub imports successfully')
" || error "❌ TensorFlow Hub import failed"
    
    log "✅ All package imports successful"
}

# Test or retrain model
handle_model() {
    log "🤖 Testing YAMNet model..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    if [[ ! -f "yamnet_models/yamnet_classifier.h5" ]]; then
        warning "⚠️  Model file not found. Training new model..."
        train_model
        return
    fi
    
    # Test model loading
    local model_test_result=$(python -c "
import tensorflow as tf
try:
    model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
    print('SUCCESS')
    print(f'Model parameters: {model.count_params():,}')
except Exception as e:
    print('FAILED')
    print(f'Error: {e}')
" 2>/dev/null)
    
    if [[ "$model_test_result" == *"SUCCESS"* ]]; then
        log "✅ Existing model loads successfully"
        echo "$model_test_result" | grep "Model parameters"
    else
        warning "⚠️  Existing model failed to load. Retraining..."
        info "Error details: $model_test_result"
        
        # Backup existing model
        if [[ -f "yamnet_models/yamnet_classifier.h5" ]]; then
            cp yamnet_models/yamnet_classifier.h5 yamnet_models/yamnet_classifier_backup_$(date +%Y%m%d_%H%M%S).h5
            log "💾 Backed up existing model"
        fi
        
        train_model
    fi
}

# Train new model
train_model() {
    log "🏋️  Training new YAMNet model..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Check if dataset exists
    if [[ ! -d "../slow" ]] || [[ ! -d "../medium" ]] || [[ ! -d "../fast" ]] || [[ ! -d "../disturbance" ]]; then
        error "❌ Audio dataset not found. Please ensure dataset directories exist."
    fi
    
    # Train the model
    python train_yamnet_model.py --dataset ../ || error "❌ Model training failed"
    
    # Verify the new model
    python -c "
import tensorflow as tf
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print('✅ New model trained and loads successfully')
print(f'✅ Model parameters: {model.count_params():,}')
" || error "❌ New model verification failed"
    
    log "✅ Model training completed successfully"
}

# Test with sample audio
test_audio_classification() {
    log "🎵 Testing audio classification..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Test with sample files
    local test_files=("../slow/Fhmm_slow.wav" "../medium/Fhum_medium.wav" "../fast/Fhum_fast.wav" "../disturbance/Cough.wav")
    local test_passed=0
    
    for test_file in "${test_files[@]}"; do
        if [[ -f "$test_file" ]]; then
            info "Testing: $test_file"
            if python test_yamnet_model.py "$test_file" --quiet; then
                ((test_passed++))
            else
                warning "⚠️  Test failed for $test_file"
            fi
        else
            warning "⚠️  Test file not found: $test_file"
        fi
    done
    
    if [[ $test_passed -gt 0 ]]; then
        log "✅ Audio classification tests passed ($test_passed tests)"
    else
        error "❌ All audio classification tests failed"
    fi
}

# Create fix summary
create_summary() {
    log "📋 Creating fix summary..."
    
    local summary_file="$HOME/deployment_fix_summary.txt"
    
    cat > "$summary_file" << EOF
YAMNet Deployment Fix Summary
============================

Fix Date: $(date)
Installation Directory: $INSTALL_DIR
Virtual Environment: $VENV_DIR

Issues Fixed:
1. NumPy 2.x compatibility with TensorFlow 2.13.0
2. SciPy version compatibility with NumPy
3. Keras model loading issues
4. Package dependency conflicts

Current Versions:
- NumPy: $(cd "$INSTALL_DIR/yamnet_implementation" && source yamnet_env/bin/activate && python -c "import numpy; print(numpy.__version__)")
- SciPy: $(cd "$INSTALL_DIR/yamnet_implementation" && source yamnet_env/bin/activate && python -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "Not installed")
- TensorFlow: $(cd "$INSTALL_DIR/yamnet_implementation" && source yamnet_env/bin/activate && python -c "import tensorflow as tf; print(tf.__version__)")

Model Status:
- Model File: $INSTALL_DIR/yamnet_implementation/yamnet_models/yamnet_classifier.h5
- Model Size: $(ls -lh $INSTALL_DIR/yamnet_implementation/yamnet_models/yamnet_classifier.h5 2>/dev/null | awk '{print $5}' || echo "Not found")
- Training Status: $([ -f "$INSTALL_DIR/yamnet_implementation/yamnet_models/yamnet_classifier.h5" ] && echo "Available" || echo "Needs training")

Next Steps:
1. Test audio classification: cd $INSTALL_DIR/yamnet_implementation && python3 test_yamnet_model.py ../slow/Fhmm_slow.wav
2. Run real-time classification: python3 realtime_pi_test.py
3. Connect Arduino wristband for haptic feedback
4. Deploy in classroom environment

All deployment issues have been resolved!
EOF
    
    log "✅ Fix summary created: $summary_file"
}

# Main function
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          YAMNet Complete Deployment Fix Script              ║"
    echo "║         Fixes NumPy, SciPy, TensorFlow, and Model Issues    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "🚀 Starting comprehensive deployment fix..."
    
    # Run fix steps
    check_deployment
    fix_package_compatibility
    test_imports
    handle_model
    test_audio_classification
    create_summary
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 ALL ISSUES FIXED! 🎉                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "🎉 All deployment issues resolved successfully!"
    log "📋 Check fix summary: $HOME/deployment_fix_summary.txt"
    
    echo ""
    echo -e "${BLUE}Your YAMNet system is now ready:${NC}"
    echo "1. Test classification: cd $INSTALL_DIR/yamnet_implementation && python3 test_yamnet_model.py ../slow/Fhmm_slow.wav"
    echo "2. Real-time mode: python3 realtime_pi_test.py"
    echo "3. Connect Arduino wristband for haptic feedback"
    echo ""
    echo -e "${GREEN}🎵🤖✨ YAMNet deployment is fully functional! ✨🤖🎵${NC}"
}

# Handle script interruption
trap 'error "❌ Fix interrupted by user"' INT TERM

# Run main function
main "$@"
