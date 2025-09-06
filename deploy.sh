#!/bin/bash

# YAMNet Speech Classification Pipeline - Automated Deployment Script for Raspberry Pi
# This script automates the complete deployment process on Raspberry Pi 4

set -e  # Exit on any error

# Configuration
REPO_URL="https://github.com/cpradeepk/anubhuti.git"
INSTALL_DIR="$HOME/anubhuti"
VENV_DIR="$INSTALL_DIR/yamnet_implementation/yamnet_env"
LOG_FILE="$HOME/yamnet_deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Progress indicator
show_progress() {
    local duration=$1
    local message=$2
    echo -n "$message"
    for ((i=0; i<duration; i++)); do
        echo -n "."
        sleep 1
    done
    echo " Done!"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    log "🔍 Checking if running on Raspberry Pi..."
    
    if [[ -f /proc/device-tree/model ]]; then
        local model=$(cat /proc/device-tree/model)
        if [[ $model == *"Raspberry Pi"* ]]; then
            log "✅ Detected: $model"
            return 0
        fi
    fi
    
    warning "⚠️  Not running on Raspberry Pi. Continuing anyway..."
    return 0
}

# Check system requirements
check_requirements() {
    log "🔍 Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "❌ Python3 is not installed"
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log "✅ Python version: $python_version"
    
    # Check available memory
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ $available_memory -lt 500 ]]; then
        warning "⚠️  Low available memory: ${available_memory}MB. Consider increasing swap space."
    else
        log "✅ Available memory: ${available_memory}MB"
    fi
    
    # Check disk space
    local available_space=$(df -h / | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ ${available_space%.*} -lt 2 ]]; then
        error "❌ Insufficient disk space. Need at least 2GB free."
    else
        log "✅ Available disk space: ${available_space}"
    fi
}

# Update system packages
update_system() {
    log "📦 Updating system packages..."
    
    sudo apt update -y || error "Failed to update package list"
    
    # Install essential packages
    local packages=(
        "python3-pip"
        "python3-venv" 
        "python3-dev"
        "git"
        "portaudio19-dev"
        "libasound2-dev"
        "alsa-utils"
        "pulseaudio"
        "pulseaudio-utils"
        "libsndfile1-dev"
        "libflac-dev"
        "libvorbis-dev"
        "wget"
        "curl"
    )
    
    for package in "${packages[@]}"; do
        info "Installing $package..."
        sudo apt install -y "$package" || warning "Failed to install $package"
    done
    
    log "✅ System packages updated successfully"
}

# Clone or update repository
setup_repository() {
    log "📁 Setting up repository..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        log "📂 Repository already exists. Updating..."
        cd "$INSTALL_DIR"
        
        # Create backup of current state
        local backup_dir="$HOME/anubhuti_backup_$(date +%Y%m%d_%H%M%S)"
        cp -r "$INSTALL_DIR" "$backup_dir" || warning "Failed to create backup"
        log "💾 Backup created at: $backup_dir"
        
        # Pull latest changes
        git pull origin main || error "Failed to update repository"
        log "✅ Repository updated successfully"
    else
        log "📥 Cloning repository..."
        git clone "$REPO_URL" "$INSTALL_DIR" || error "Failed to clone repository"
        log "✅ Repository cloned successfully"
    fi
    
    # Verify repository structure
    cd "$INSTALL_DIR"
    if [[ ! -d "yamnet_implementation" ]]; then
        error "❌ Invalid repository structure: yamnet_implementation directory not found"
    fi
    
    if [[ ! -f "yamnet_implementation/yamnet_models/yamnet_classifier.h5" ]]; then
        error "❌ Model file not found: yamnet_classifier.h5"
    fi
    
    log "✅ Repository structure verified"
}

# Setup Python virtual environment
setup_python_environment() {
    log "🐍 Setting up Python virtual environment..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "yamnet_env" ]]; then
        python3 -m venv yamnet_env || error "Failed to create virtual environment"
        log "✅ Virtual environment created"
    else
        log "📂 Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source yamnet_env/bin/activate || error "Failed to activate virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip || error "Failed to upgrade pip"
    
    log "✅ Python environment ready"
}

# Install Python dependencies
install_dependencies() {
    log "📚 Installing Python dependencies..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Install TensorFlow (this takes the longest)
    info "Installing TensorFlow (this may take 10-15 minutes)..."
    show_progress 5 "Preparing TensorFlow installation"
    pip install tensorflow==2.13.0 --no-cache-dir || error "Failed to install TensorFlow"
    
    # Install other dependencies
    local dependencies=(
        "tensorflow-hub==0.14.0"
        "librosa==0.10.1"
        "soundfile==0.12.1"
        "numpy==1.24.3"
        "scikit-learn==1.3.0"
        "matplotlib==3.7.2"
        "seaborn==0.12.2"
        "pyaudio"
        "pydub"
    )
    
    for dep in "${dependencies[@]}"; do
        info "Installing $dep..."
        pip install "$dep" || warning "Failed to install $dep"
    done
    
    # Verify installations
    python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || error "TensorFlow verification failed"
    python -c "import tensorflow_hub as hub; print('TensorFlow Hub: OK')" || error "TensorFlow Hub verification failed"
    python -c "import librosa; print('Librosa: OK')" || error "Librosa verification failed"
    
    log "✅ Python dependencies installed successfully"
}

# Configure audio system
configure_audio() {
    log "🎵 Configuring audio system..."
    
    # Create ALSA configuration
    cat > "$HOME/.asoundrc" << 'EOF'
pcm.!default {
    type asym
    playback.pcm "plughw:0,0"
    capture.pcm "plughw:1,0"
}
ctl.!default {
    type hw
    card 1
}
EOF
    
    log "✅ Audio configuration created"
    
    # Test audio devices
    if command -v arecord &> /dev/null; then
        info "Available audio input devices:"
        arecord -l | tee -a "$LOG_FILE" || warning "Failed to list audio devices"
    fi
}

# Test model deployment
test_deployment() {
    log "🧪 Testing model deployment..."
    
    cd "$INSTALL_DIR/yamnet_implementation"
    source yamnet_env/bin/activate
    
    # Test model loading
    python3 -c "
import tensorflow as tf
print('Loading YAMNet classifier...')
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print(f'✅ Model loaded successfully!')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')
print(f'Total parameters: {model.count_params():,}')
" || error "Model loading test failed"
    
    # Test with sample audio if available
    if [[ -f "../slow/Fhmm_slow.wav" ]]; then
        info "Testing with sample audio file..."
        python3 test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet || warning "Sample audio test failed"
    fi
    
    log "✅ Model deployment test passed"
}

# Create deployment summary
create_summary() {
    log "📋 Creating deployment summary..."
    
    local summary_file="$HOME/yamnet_deployment_summary.txt"
    
    cat > "$summary_file" << EOF
YAMNet Speech Classification Pipeline - Deployment Summary
=========================================================

Deployment Date: $(date)
Installation Directory: $INSTALL_DIR
Virtual Environment: $VENV_DIR
Log File: $LOG_FILE

System Information:
- OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
- Python: $(python3 --version)
- Available Memory: $(free -h | awk 'NR==2{print $7}')
- Available Disk: $(df -h / | awk 'NR==2{print $4}')

Model Information:
- Model File: $INSTALL_DIR/yamnet_implementation/yamnet_models/yamnet_classifier.h5
- Model Size: $(ls -lh $INSTALL_DIR/yamnet_implementation/yamnet_models/yamnet_classifier.h5 | awk '{print $5}')

Quick Start Commands:
1. Activate environment: source $VENV_DIR/bin/activate
2. Test single file: python3 test_yamnet_model.py audio_file.wav
3. Run health check: python3 health_check.py
4. Update system: cd $INSTALL_DIR && git pull origin main

Next Steps:
- Connect USB microphone for real-time testing
- Run real-time classification with: python3 realtime_pi_test.py
- Connect Arduino wristband for haptic feedback
- Set up as system service for automatic startup

For troubleshooting, check the log file: $LOG_FILE
EOF
    
    log "✅ Deployment summary created: $summary_file"
}

# Main deployment function
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          YAMNet Speech Classification Pipeline               ║"
    echo "║              Automated Deployment Script                    ║"
    echo "║                  for Raspberry Pi 4                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "🚀 Starting YAMNet deployment process..."
    
    # Create log file
    touch "$LOG_FILE"
    log "📝 Logging to: $LOG_FILE"
    
    # Run deployment steps
    check_raspberry_pi
    check_requirements
    update_system
    setup_repository
    setup_python_environment
    install_dependencies
    configure_audio
    test_deployment
    create_summary
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 DEPLOYMENT SUCCESSFUL! 🎉              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "🎉 YAMNet deployment completed successfully!"
    log "📋 Check deployment summary: $HOME/yamnet_deployment_summary.txt"
    log "🧪 Run health check: cd $INSTALL_DIR/yamnet_implementation && python3 health_check.py"
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Connect USB microphone for audio input"
    echo "2. Test real-time classification: cd $INSTALL_DIR/yamnet_implementation && python3 realtime_pi_test.py"
    echo "3. Connect Arduino wristband for haptic feedback"
    echo "4. Deploy in classroom environment"
    echo ""
    echo -e "${GREEN}Your YAMNet speech classification system is ready! 🎵🤖✨${NC}"
}

# Handle script interruption
trap 'error "❌ Deployment interrupted by user"' INT TERM

# Run main function
main "$@"
