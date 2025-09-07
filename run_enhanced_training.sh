#!/bin/bash

# Optimized Enhanced YAMNet Training Script
# Handles GPU setup and environment optimization

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
echo "║          Enhanced YAMNet Training with GPU Optimization     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Set optimal environment variables
export TF_ENABLE_ONEDNN_OPTS=1
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_CACHE_PATH=/tmp/cuda_cache
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Create CUDA cache directory
mkdir -p /tmp/cuda_cache

# Check GPU availability
log "🔍 Checking GPU setup..."
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', len(gpus))
if gpus:
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.name}')
    # Configure GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✅ GPU memory growth configured')
    except RuntimeError as e:
        print(f'GPU configuration error: {e}')
else:
    print('⚠️  No GPUs found - will use CPU')
"

# Parse command line arguments
DATASET_DIR=""
REAL_WORLD_AUDIO=""
LABELED_SEGMENTS=""
OUTPUT_DIR="enhanced_models"
EPOCHS=50
MODE="full"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        --real-world-audio)
            REAL_WORLD_AUDIO="$2"
            shift 2
            ;;
        --labeled-segments)
            LABELED_SEGMENTS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Enhanced YAMNet Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DIR              Path to original dataset directory"
            echo "  --real-world-audio FILE    Path to 45-minute classroom audio"
            echo "  --labeled-segments FILE    Path to labeled segments JSON"
            echo "  --output-dir DIR           Output directory (default: enhanced_models)"
            echo "  --epochs N                 Training epochs (default: 50)"
            echo "  --mode MODE                Mode: segment|label|train|full (default: full)"
            echo "  --help                     Show this help"
            echo ""
            echo "Examples:"
            echo "  # Full pipeline with real-world audio"
            echo "  $0 --dataset ../dataset --real-world-audio classroom.wav"
            echo ""
            echo "  # Just segmentation"
            echo "  $0 --dataset ../dataset --real-world-audio classroom.wav --mode segment"
            echo ""
            echo "  # Training with labeled segments"
            echo "  $0 --dataset ../dataset --labeled-segments labels.json --mode train"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET_DIR" ]]; then
    error "Dataset directory is required. Use --dataset /path/to/dataset"
fi

if [[ ! -d "$DATASET_DIR" ]]; then
    error "Dataset directory not found: $DATASET_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

log "🚀 Starting enhanced YAMNet training..."
info "Dataset: $DATASET_DIR"
info "Output: $OUTPUT_DIR"
info "Mode: $MODE"

# Check if enhanced training pipeline exists
if [[ ! -f "enhanced_training_pipeline.py" ]]; then
    error "enhanced_training_pipeline.py not found in current directory"
fi

# Handle different modes
case $MODE in
    "segment")
        if [[ -z "$REAL_WORLD_AUDIO" ]]; then
            error "Real-world audio file required for segmentation mode"
        fi
        log "🔪 Running segmentation mode..."
        python3 enhanced_training_pipeline.py \
            --dataset "$DATASET_DIR" \
            --real-world-audio "$REAL_WORLD_AUDIO" \
            --output-dir "$OUTPUT_DIR"
        ;;
    
    "label")
        log "🏷️  Running labeling interface..."
        if [[ -d "$OUTPUT_DIR/labeling" ]]; then
            cd "$OUTPUT_DIR/labeling"
            python3 label_segments.py
            cd - > /dev/null
        else
            error "Labeling directory not found. Run segmentation first."
        fi
        ;;
    
    "train")
        if [[ -z "$LABELED_SEGMENTS" ]]; then
            error "Labeled segments file required for training mode"
        fi
        log "🏋️  Running training mode..."
        
        # Warn about potential JIT compilation delay
        if nvidia-smi > /dev/null 2>&1; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
            warning "GPU detected: $GPU_NAME"
            warning "First run may take 10-30 minutes for CUDA kernel compilation"
            warning "Subsequent runs will be much faster (kernels are cached)"
            echo ""
            read -p "Continue with training? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Training cancelled by user"
                exit 0
            fi
        fi
        
        python3 enhanced_training_pipeline.py \
            --dataset "$DATASET_DIR" \
            --labeled-segments "$LABELED_SEGMENTS" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS"
        ;;
    
    "full")
        if [[ -n "$REAL_WORLD_AUDIO" ]]; then
            log "🔄 Running full pipeline with real-world audio..."
            
            # Step 1: Segmentation
            log "Step 1: Segmenting real-world audio..."
            python3 enhanced_training_pipeline.py \
                --dataset "$DATASET_DIR" \
                --real-world-audio "$REAL_WORLD_AUDIO" \
                --output-dir "$OUTPUT_DIR"
            
            # Step 2: Interactive labeling
            log "Step 2: Interactive labeling required..."
            echo ""
            echo -e "${YELLOW}Next steps:${NC}"
            echo "1. Label the segments: cd $OUTPUT_DIR/labeling && python3 label_segments.py"
            echo "2. Resume training: $0 --dataset $DATASET_DIR --labeled-segments $OUTPUT_DIR/labeling/segment_labels.json --mode train"
            echo ""
            log "Pipeline paused for manual labeling step"
            
        elif [[ -n "$LABELED_SEGMENTS" ]]; then
            log "🔄 Running training with labeled segments..."
            python3 enhanced_training_pipeline.py \
                --dataset "$DATASET_DIR" \
                --labeled-segments "$LABELED_SEGMENTS" \
                --output-dir "$OUTPUT_DIR" \
                --epochs "$EPOCHS"
        else
            log "🔄 Running standard training (no real-world audio)..."
            python3 enhanced_training_pipeline.py \
                --dataset "$DATASET_DIR" \
                --output-dir "$OUTPUT_DIR" \
                --epochs "$EPOCHS"
        fi
        ;;
    
    *)
        error "Invalid mode: $MODE. Use: segment, label, train, or full"
        ;;
esac

# Show results
if [[ -f "$OUTPUT_DIR/yamnet_classifier_enhanced.h5" ]]; then
    log "🎉 Enhanced model training completed!"
    
    # Show model info
    MODEL_SIZE=$(ls -lh "$OUTPUT_DIR/yamnet_classifier_enhanced.h5" | awk '{print $5}')
    log "📊 Model file: $OUTPUT_DIR/yamnet_classifier_enhanced.h5 ($MODEL_SIZE)"
    
    if [[ -f "$OUTPUT_DIR/yamnet_model_metadata_enhanced.json" ]]; then
        log "📋 Metadata: $OUTPUT_DIR/yamnet_model_metadata_enhanced.json"
    fi
    
    echo ""
    echo -e "${GREEN}🎯 Next steps:${NC}"
    echo "1. Test model: python3 test_enhanced_model.py $OUTPUT_DIR/yamnet_classifier_enhanced.h5"
    echo "2. Deploy to Pi: Copy model to Raspberry Pi and use deploy_pretrained.sh"
    echo "3. Compare performance with original model"
    echo ""
    echo -e "${GREEN}🚀 Enhanced YAMNet model ready for deployment! 🎵🤖✨${NC}"
fi

log "✅ Enhanced training pipeline completed!"
