# YAMNet Speech Classification Pipeline - API Documentation

## 📚 API Overview

This document provides comprehensive API documentation for the YAMNet speech classification pipeline, including all classes, methods, and usage examples.

## 🏗️ Core Components

### 1. YAMNetProcessor Class

The main utility class for YAMNet model operations and audio processing.

```python
from yamnet_utils import YAMNetProcessor

processor = YAMNetProcessor()
```

#### Methods

##### `__init__(self, yamnet_model_url: str = None)`
Initialize YAMNet processor with optional custom model URL.

**Parameters:**
- `yamnet_model_url` (str, optional): Custom YAMNet model URL. Defaults to Google's YAMNet.

**Example:**
```python
# Use default YAMNet model
processor = YAMNetProcessor()

# Use custom model URL
processor = YAMNetProcessor("https://custom-yamnet-url")
```

##### `preprocess_audio(self, audio_path: str) -> np.ndarray`
Preprocess audio file for YAMNet input requirements.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
- `np.ndarray`: Preprocessed audio array (16kHz mono)

**Example:**
```python
audio = processor.preprocess_audio("audio_file.wav")
print(f"Audio shape: {audio.shape}, Sample rate: 16000Hz")
```

##### `extract_embeddings(self, audio: np.ndarray) -> np.ndarray`
Extract YAMNet embeddings from preprocessed audio.

**Parameters:**
- `audio` (np.ndarray): Preprocessed audio array

**Returns:**
- `np.ndarray`: YAMNet embeddings (shape: [num_frames, 1024])

**Example:**
```python
embeddings = processor.extract_embeddings(audio)
print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: (N, 1024) where N is number of frames
```

##### `process_audio_file(self, audio_path: str) -> Tuple[np.ndarray, Dict]`
Complete audio processing pipeline from file to embeddings.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
- `Tuple[np.ndarray, Dict]`: (embeddings, metadata)

**Example:**
```python
embeddings, metadata = processor.process_audio_file("test.wav")
print(f"Duration: {metadata['duration_seconds']:.2f}s")
print(f"Frames: {metadata['num_frames']}")
```

### 2. YAMNetModelTester Class

High-level interface for audio classification and testing.

```python
from test_yamnet_model import YAMNetModelTester

tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
```

#### Methods

##### `__init__(self, model_path: str, metadata_path: str = None)`
Initialize model tester with trained classifier.

**Parameters:**
- `model_path` (str): Path to trained model (.h5 file)
- `metadata_path` (str, optional): Path to metadata JSON file

**Example:**
```python
# Basic initialization
tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")

# With custom metadata
tester = YAMNetModelTester(
    "yamnet_models/yamnet_classifier.h5",
    "yamnet_models/yamnet_model_metadata.json"
)
```

##### `predict_single_file(self, audio_path: str) -> Dict`
Classify a single audio file and return detailed results.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
- `Dict`: Comprehensive prediction results

**Return Dictionary Structure:**
```python
{
    'file_path': str,                    # Original file path
    'file_name': str,                    # File name only
    'predicted_class_idx': int,          # Class index (0-3)
    'predicted_class_name': str,         # Class name
    'confidence': float,                 # Prediction confidence (0-1)
    'arduino_command': int,              # Arduino command (0-4)
    'all_probabilities': {               # All class probabilities
        'slow': float,
        'medium': float,
        'fast': float,
        'disturbance': float
    },
    'audio_metadata': {                  # Audio file metadata
        'duration_seconds': float,
        'sample_rate': int,
        'num_frames': int
    }
}
```

**Example:**
```python
result = tester.predict_single_file("test_audio.wav")

print(f"Predicted: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Arduino Command: {result['arduino_command']}")

# Access all probabilities
for class_name, prob in result['all_probabilities'].items():
    print(f"{class_name}: {prob:.3f}")
```

##### `batch_predict(self, audio_files: List[str]) -> Dict`
Process multiple audio files in batch.

**Parameters:**
- `audio_files` (List[str]): List of audio file paths

**Returns:**
- `Dict`: Batch processing results

**Return Dictionary Structure:**
```python
{
    'total_files': int,
    'successful_predictions': int,
    'failed_predictions': int,
    'results': List[Dict],              # List of individual results
    'failed_files': List[Dict]          # List of failed files with errors
}
```

**Example:**
```python
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
batch_results = tester.batch_predict(audio_files)

print(f"Processed: {batch_results['successful_predictions']}/{batch_results['total_files']}")

for result in batch_results['results']:
    print(f"{result['file_name']}: {result['predicted_class_name']} ({result['confidence']:.3f})")
```

### 3. LongAudioProcessor Class

Specialized processor for long audio files using sliding window approach.

```python
from process_long_audio import LongAudioProcessor

processor = LongAudioProcessor(
    "yamnet_models/yamnet_classifier.h5",
    chunk_duration=5.0,
    overlap=0.5
)
```

#### Methods

##### `__init__(self, model_path: str, metadata_path: str = None, chunk_duration: float = 5.0, overlap: float = 0.5)`
Initialize long audio processor.

**Parameters:**
- `model_path` (str): Path to trained model
- `metadata_path` (str, optional): Path to metadata file
- `chunk_duration` (float): Duration of each chunk in seconds
- `overlap` (float): Overlap ratio between chunks (0.0 to 1.0)

##### `process_long_audio(self, audio_path: str) -> Dict`
Process long audio file with sliding window approach.

**Parameters:**
- `audio_path` (str): Path to long audio file

**Returns:**
- `Dict`: Long audio processing results

**Return Dictionary Structure:**
```python
{
    'file_path': str,
    'file_name': str,
    'total_duration': float,
    'processing_time': float,
    'processing_method': str,           # 'sliding_window' or 'single_chunk'
    'chunk_duration': float,
    'overlap_ratio': float,
    'num_chunks': int,
    'successful_chunks': int,
    'dominant_class': str,              # Most frequent class
    'dominant_confidence': float,       # Average confidence for dominant class
    'dominant_percentage': float,       # Percentage of chunks with dominant class
    'overall_confidence': float,        # Overall average confidence
    'arduino_command': int,             # Command for dominant class
    'class_distribution': {             # Distribution of classes across chunks
        'slow': float,
        'medium': float,
        'fast': float,
        'disturbance': float
    },
    'chunk_results': List[Dict],        # Individual chunk results
    'all_probabilities': Dict           # Average probabilities across all chunks
}
```

**Example:**
```python
result = processor.process_long_audio("long_recording.wav")

print(f"Duration: {result['total_duration']:.1f}s")
print(f"Processing time: {result['processing_time']:.1f}s")
print(f"Dominant class: {result['dominant_class']} ({result['dominant_percentage']*100:.1f}%)")

# Show class distribution
for class_name, percentage in result['class_distribution'].items():
    chunk_count = int(percentage * result['num_chunks'])
    print(f"{class_name}: {chunk_count} chunks ({percentage*100:.1f}%)")
```

## 🛠️ Utility Functions

### Audio Processing Utilities

##### `aggregate_embeddings(embeddings: np.ndarray, method: str = 'mean') -> np.ndarray`
Aggregate temporal embeddings into fixed-size representation.

**Parameters:**
- `embeddings` (np.ndarray): YAMNet embeddings (shape: [num_frames, 1024])
- `method` (str): Aggregation method ('mean', 'max', 'median')

**Returns:**
- `np.ndarray`: Aggregated embedding (shape: [1024])

**Example:**
```python
from yamnet_utils import aggregate_embeddings

# Mean aggregation (default)
agg_embedding = aggregate_embeddings(embeddings, method='mean')

# Max pooling
agg_embedding = aggregate_embeddings(embeddings, method='max')

# Median aggregation
agg_embedding = aggregate_embeddings(embeddings, method='median')
```

##### `chunk_audio(audio: np.ndarray, chunk_duration: float, overlap: float) -> List[np.ndarray]`
Split audio into overlapping chunks.

**Parameters:**
- `audio` (np.ndarray): Input audio array
- `chunk_duration` (float): Duration of each chunk in seconds
- `overlap` (float): Overlap ratio (0.0 to 1.0)

**Returns:**
- `List[np.ndarray]`: List of audio chunks

**Example:**
```python
from yamnet_utils import chunk_audio

chunks = chunk_audio(audio, chunk_duration=5.0, overlap=0.5)
print(f"Created {len(chunks)} chunks from {len(audio)/16000:.1f}s audio")
```

### Dataset Utilities

##### `load_dataset(dataset_path: str, class_mapping: Dict = None) -> Tuple`
Load and preprocess dataset for training.

**Parameters:**
- `dataset_path` (str): Path to dataset directory
- `class_mapping` (Dict, optional): Custom class mapping

**Returns:**
- `Tuple`: (embeddings, labels, file_paths, class_mapping)

**Example:**
```python
from yamnet_utils import load_dataset

embeddings, labels, file_paths, class_mapping = load_dataset("../dataset/")
print(f"Loaded {len(embeddings)} samples")
print(f"Classes: {list(class_mapping.keys())}")
```

##### `create_balanced_splits(indices: List, labels: List, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict`
Create balanced train/validation/test splits.

**Parameters:**
- `indices` (List): Sample indices
- `labels` (List): Sample labels
- `train_ratio` (float): Training set ratio
- `val_ratio` (float): Validation set ratio
- `test_ratio` (float): Test set ratio

**Returns:**
- `Dict`: Split information with indices and labels

**Example:**
```python
from yamnet_utils import create_balanced_splits

splits = create_balanced_splits(
    list(range(len(labels))), 
    labels,
    train_ratio=0.7,
    val_ratio=0.15, 
    test_ratio=0.15
)

print(f"Train: {len(splits['train']['indices'])} samples")
print(f"Val: {len(splits['val']['indices'])} samples") 
print(f"Test: {len(splits['test']['indices'])} samples")
```

## 🔧 Configuration and Metadata

### Model Metadata Structure

The model metadata JSON file contains important configuration information:

```json
{
    "model_info": {
        "model_type": "yamnet_classifier",
        "version": "1.0",
        "creation_date": "2024-01-01T12:00:00",
        "training_duration_minutes": 10.5,
        "total_parameters": 657156,
        "trainable_parameters": 657156
    },
    "class_mapping": {
        "slow": 0,
        "medium": 1,
        "fast": 2,
        "disturbance": 3
    },
    "arduino_mapping": {
        "slow": 1,
        "medium": 2,
        "fast": 3,
        "disturbance": 0
    },
    "training_config": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "early_stopping_patience": 10
    },
    "performance_metrics": {
        "test_accuracy": 0.90,
        "test_precision": 0.947,
        "test_recall": 0.90,
        "test_f1_score": 0.923
    },
    "dataset_info": {
        "total_samples": 132,
        "train_samples": 112,
        "test_samples": 20,
        "class_distribution": {
            "slow": 36,
            "medium": 36,
            "fast": 36,
            "disturbance": 24
        }
    }
}
```

### Loading Metadata

```python
from yamnet_utils import load_model_metadata

metadata = load_model_metadata("yamnet_models/yamnet_model_metadata.json")

# Access configuration
class_mapping = metadata['class_mapping']
arduino_mapping = metadata['arduino_mapping']
performance = metadata['performance_metrics']

print(f"Test accuracy: {performance['test_accuracy']:.1%}")
```

## 🚨 Error Handling

### Common Exceptions

#### `AudioProcessingError`
Raised when audio file cannot be processed.

```python
try:
    result = tester.predict_single_file("corrupted_audio.wav")
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
```

#### `ModelLoadingError`
Raised when model files cannot be loaded.

```python
try:
    tester = YAMNetModelTester("nonexistent_model.h5")
except ModelLoadingError as e:
    print(f"Model loading failed: {e}")
```

#### `InvalidAudioFormatError`
Raised when audio format is not supported.

```python
try:
    audio = processor.preprocess_audio("unsupported_format.xyz")
except InvalidAudioFormatError as e:
    print(f"Unsupported audio format: {e}")
```

## 📊 Performance Monitoring

### Built-in Performance Metrics

```python
# Enable performance monitoring
tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5", enable_monitoring=True)

# Process files
for audio_file in audio_files:
    result = tester.predict_single_file(audio_file)

# Get performance statistics
stats = tester.get_performance_stats()
print(f"Average inference time: {stats['avg_inference_time']:.1f}ms")
print(f"Peak memory usage: {stats['peak_memory_mb']:.1f}MB")
```

## 🔌 Integration Examples

### Arduino Integration

```python
import serial
from test_yamnet_model import YAMNetModelTester

# Initialize components
tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
arduino = serial.Serial('/dev/ttyUSB0', 9600)

# Process audio and send command
result = tester.predict_single_file("audio.wav")
command = result['arduino_command']

# Send to Arduino
arduino.write(str(command).encode())
print(f"Sent command {command} to Arduino")

arduino.close()
```

### Real-time Processing

```python
import pyaudio
import numpy as np
from test_yamnet_model import YAMNetModelTester

# Initialize audio stream
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")

# Real-time processing loop
try:
    while True:
        # Capture audio chunk
        data = stream.read(CHUNK * 48)  # ~3 seconds at 16kHz
        audio_array = np.frombuffer(data, dtype=np.float32)
        
        # Save temporary file and process
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, RATE)
            
            # Classify
            result = tester.predict_single_file(tmp.name)
            print(f"Class: {result['predicted_class_name']}, Confidence: {result['confidence']:.3f}")
            
        os.unlink(tmp.name)
        
except KeyboardInterrupt:
    print("Stopping real-time processing...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
```

This API documentation provides comprehensive coverage of all available classes, methods, and integration patterns for the YAMNet speech classification pipeline.
