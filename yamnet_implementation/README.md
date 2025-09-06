# YAMNet Speech Classification Pipeline

A comprehensive audio classification system using Google's pre-trained YAMNet model for vocal sound and disturbance detection. This system classifies audio into four categories for vibration-based feedback control.

## 🎯 Overview

This pipeline uses YAMNet (Yet Another Mobile Network) as a frozen feature extractor to generate 1024-dimensional embeddings from audio, then trains a lightweight classifier on top for specific vocal sound classification.

### Classes and Arduino Integration
- **slow**: Slow vocal sounds ("sooo", long "hummm") → Arduino Command 1 (Top motor)
- **medium**: Medium-paced vocal sounds ("soo", "hum") → Arduino Command 2 (Bottom motor)  
- **fast**: Fast vocal sounds ("so-so-so", rapid "hm-hm-hm") → Arduino Command 3 (Both motors)
- **disturbance**: Non-vocal sounds (claps, coughs, background noise) → Arduino Command 0 (No vibration)

## 📁 File Structure

```
yamnet_implementation/
├── requirements.txt              # Python dependencies
├── yamnet_utils.py              # Core utility functions
├── train_yamnet_model.py        # Model training script
├── test_yamnet_model.py         # Single file testing
├── process_long_audio.py        # Long audio processing
├── README.md                    # This file
└── yamnet_models/               # Output directory (created during training)
    ├── yamnet_classifier.h5     # Trained model
    ├── yamnet_model_metadata.json # Model metadata
    ├── training_history.png     # Training plots
    └── confusion_matrix.png     # Confusion matrix
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv yamnet_env
source yamnet_env/bin/activate  # On Windows: yamnet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your audio files in this structure:
```
dataset/
├── slow/          # Slow vocal sounds
├── medium/        # Medium-paced vocal sounds
├── fast/          # Fast vocal sounds
└── disturbance/   # Non-vocal sounds
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`

### 3. Train Model

```bash
python train_yamnet_model.py --dataset dataset/ --output yamnet_models/
```

Training options:
- `--dataset`: Path to dataset directory (required)
- `--output`: Output directory for models (default: yamnet_models)
- `--test-split`: Test set ratio (default: 0.15)
- `--val-split`: Validation set ratio (default: 0.15)

### 4. Test Single Files

```bash
python test_yamnet_model.py audio_file.wav
```

Options:
- `--model`: Path to trained model (default: yamnet_models/yamnet_classifier.h5)
- `--metadata`: Path to metadata file (optional)
- `--quiet`: Suppress detailed output
- `--json-output`: Save results to JSON file

### 5. Process Long Audio

```bash
python process_long_audio.py long_audio.wav --chunk-duration 5.0 --overlap 0.5
```

Options:
- `--chunk-duration`: Duration of each chunk in seconds (default: 5.0)
- `--overlap`: Overlap ratio between chunks (default: 0.5)
- `--chunk-details`: Show detailed results for each chunk
- `--json-output`: Save results to JSON file

## 🏗️ Architecture Details

### YAMNet Feature Extraction
- **Input**: 16kHz mono audio (automatically resampled)
- **Output**: 1024-dimensional embeddings per audio frame
- **Aggregation**: Mean pooling across time frames for fixed-size representation

### Classifier Architecture
```
Input(1024) → Dense(512, ReLU) → Dropout(0.3) → 
Dense(256, ReLU) → Dropout(0.4) → Dense(4, Softmax)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical crossentropy
- **Batch Size**: 32
- **Max Epochs**: 50 with early stopping
- **Class Weights**: Automatically balanced for imbalanced datasets

## 📊 Performance Features

### Single File Processing
- Automatic audio format detection and conversion
- YAMNet embedding extraction
- Real-time classification with confidence scores
- Arduino command mapping

### Long Audio Processing
- **Sliding Window**: 5-second chunks with 50% overlap (configurable)
- **Aggregation Methods**: Majority voting, confidence weighting
- **Performance**: Processes audio faster than real-time
- **Memory Efficient**: Processes chunks sequentially

### Example Output
```
🎯 YAMNET CLASSIFIER PREDICTION RESULTS
================================================================================
🎵 File: test_audio.wav
⏱️  Duration: 3.45 seconds
🎯 Predicted Class: medium
📊 Confidence: 0.847 (84.7%)
🤖 Arduino Command: 2
🎮 Motor Action: Bottom motor vibrates ('medium' sound)

📊 All Class Probabilities:
👉 medium      : 0.847 (84.7%)
   slow        : 0.098 (9.8%)
   fast        : 0.032 (3.2%)
   disturbance : 0.023 (2.3%)
```

## 🔧 Advanced Usage

### Custom Class Mapping
Modify the class mapping in `train_yamnet_model.py`:
```python
self.class_mapping = {
    'slow': 0,
    'medium': 1, 
    'fast': 2,
    'disturbance': 3
}
```

### Batch Processing
```python
from test_yamnet_model import YAMNetModelTester

tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
results = tester.batch_predict(["file1.wav", "file2.wav", "file3.wav"])
```

### Integration with Existing Systems
The YAMNet classifier maintains compatibility with existing DS-CNN systems:
- Same class indices and Arduino command mapping
- Compatible output format for `realtime_audio_processor.py`
- Drop-in replacement for existing model files

## 🚨 Troubleshooting

### Common Issues

**1. TensorFlow Hub Download Issues**
```bash
# Set proxy if needed
export TFHUB_CACHE_DIR=/path/to/cache
export HTTP_PROXY=http://proxy:port
```

**2. Audio Format Errors**
- Ensure audio files are not corrupted
- Install additional codecs: `pip install pydub[mp3]`

**3. Memory Issues with Long Audio**
- Reduce chunk duration: `--chunk-duration 3.0`
- Reduce overlap: `--overlap 0.25`

**4. Low Accuracy**
- Increase dataset size (recommended: 100+ samples per class)
- Balance dataset across classes
- Check audio quality and consistency

### Performance Optimization

**For Raspberry Pi Deployment:**
```bash
# Convert to TensorFlow Lite for faster inference
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('yamnet_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

## 📈 Expected Performance

### Typical Results
- **Training Time**: 5-15 minutes (depending on dataset size)
- **Inference Speed**: <50ms per file on modern hardware
- **Memory Usage**: ~500MB during training, ~200MB during inference
- **Accuracy**: 70-90% (depends on dataset quality and size)

### Scalability
- **Small Dataset**: 20-100 samples per class → 60-75% accuracy
- **Medium Dataset**: 100-500 samples per class → 75-85% accuracy  
- **Large Dataset**: 500+ samples per class → 85-95% accuracy

## 🔗 Integration Notes

This YAMNet implementation is designed to be a drop-in replacement for the existing DS-CNN system while providing:
- **Better Generalization**: Pre-trained on AudioSet (2M+ audio clips)
- **Faster Training**: No need to train from scratch
- **Robust Features**: YAMNet embeddings capture rich audio representations
- **Scalability**: Handles larger datasets more effectively

The Arduino integration remains identical - same command mapping and communication protocol.

## 📚 References

- [YAMNet Paper](https://arxiv.org/abs/1912.06670)
- [TensorFlow Hub YAMNet](https://tfhub.dev/google/yamnet/1)
- [AudioSet Dataset](https://research.google.com/audioset/)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your dataset structure matches the requirements
3. Ensure all dependencies are correctly installed
4. Test with the provided example scripts first
