# Audio Classification System for Sound-to-Vibration Project

A complete audio classification system that classifies human vocal commands and controls motor vibrations via Arduino.

## Project Overview

- **Input:** Human vocal commands (primarily "soo" and "humm" sounds)
- **Output:** Classification into 4 rhythm categories (disturbance, slow, medium, fast)
- **End Goal:** Real-time audio classification that sends commands to Arduino for motor vibration control

## Dataset Structure

The system expects audio files organized in the following folder structure:
```
Anubhuti/
├── disturbance/    # Class 0 - Disturbance sounds
├── slow/          # Class 1 - Slow rhythm sounds
├── medium/        # Class 2 - Medium rhythm sounds
├── fast/          # Class 3 - Fast rhythm sounds
└── ...
```

## Technical Specifications

- **Audio Format:** .wav files (recommended)
- **Sampling Rate:** 22050 Hz
- **Duration:** Normalized to exactly 3 seconds
- **Features:** MFCC with 13 coefficients
- **Model:** DS-CNN with Dense layers
- **Classes:** 4 (0=disturbance, 1=slow, 2=medium, 3=fast)

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
```bash
pip install librosa numpy soundfile scikit-learn tensorflow keras matplotlib tqdm
```

## Usage

### Step 1: Preprocess Dataset
```bash
python dataset_preprocess.py
```
This script will:
- Scan dataset folders for .wav files
- Normalize all audio to 3 seconds
- Extract MFCC features
- Create `data.json` with structured data

### Step 2: Train Model
```bash
python train_model.py
```
This script will:
- Load preprocessed data from `data.json`
- Split data into train/test sets (80/20)
- Train DS-CNN model
- Save trained model as `model.h5`
- Generate training history plot

### Step 3: Test Model
```bash
python test_model.py <path_to_audio_file.wav>
```
Example:
```bash
python test_model.py sample_audio.wav
```

This script will:
- Load the trained model
- Preprocess the input audio file
- Make predictions with confidence scores
- Display results for all 4 classes

## File Descriptions

### Core Scripts

1. **`dataset_preprocess.py`**
   - Recursively scans dataset folders for .wav files
   - Normalizes audio duration to 3 seconds (pad/truncate)
   - Extracts MFCC features with 13 coefficients
   - Creates structured `data.json` file
   - Includes error handling and progress tracking

2. **`train_model.py`**
   - Loads preprocessed data from `data.json`
   - Implements DS-CNN architecture with Dense layers
   - Trains model with validation split and early stopping
   - Saves model as `model.h5` and generates training plots
   - Displays comprehensive training metrics

3. **`test_model.py`**
   - Loads trained model and performs inference
   - Accepts command-line audio file input
   - Preprocesses audio identically to training pipeline
   - Displays prediction results with confidence scores
   - Provides Arduino integration guidance

### Output Files

- **`data.json`** - Preprocessed dataset with MFCC features and labels
- **`model.h5`** - Trained classification model
- **`best_model.h5`** - Best model checkpoint during training
- **`training_history.png`** - Training accuracy and loss plots
- **`progress_log.md`** - Detailed progress tracking

## Model Architecture

The DS-CNN model consists of:
- Input layer (flattened MFCC features)
- Dense layer (512 units, ReLU activation)
- Batch normalization + Dropout (0.3)
- Dense layer (256 units, ReLU activation)
- Batch normalization + Dropout (0.4)
- Dense layer (128 units, ReLU activation)
- Batch normalization + Dropout (0.5)
- Output layer (4 units, softmax activation)

## Arduino Integration

The system outputs class predictions as integers:
- `0` → disturbance
- `1` → slow
- `2` → medium
- `3` → fast

These can be easily mapped to Arduino motor control commands.

## Troubleshooting

### Common Issues

1. **No audio files found**
   - Ensure .wav files are placed in the correct folders
   - Check that folder names match exactly: `disturbance`, `slow`, `medium`, `fast`

2. **Model training fails**
   - Ensure `data.json` exists (run preprocessing first)
   - Check that you have sufficient training data (at least 10 samples per class)

3. **Prediction errors**
   - Ensure both `model.h5` and `data.json` exist
   - Check that input audio file is accessible and valid

### Performance Tips

- Use high-quality .wav files for better accuracy
- Ensure balanced dataset across all 4 classes
- Consider data augmentation if you have limited samples
- Monitor training plots for overfitting

## Future Enhancements

- Real-time audio classification
- Support for additional audio formats
- Web interface for easy testing
- Direct Arduino serial communication
- Model optimization for embedded systems

## Notes

- Original specification mentioned 19 classes, but current implementation supports 4 classes based on existing dataset structure
- All MFCC parameters are consistent across preprocessing, training, and testing
- System designed with Arduino integration in mind for motor vibration control