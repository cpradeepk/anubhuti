#!/usr/bin/env python3
"""
Audio Classification Model Testing Script

This script loads a trained model and performs inference on new audio files.
It handles:
- Loading trained model from model.h5
- Loading class mapping from data.json
- Command-line argument handling for input audio files
- Audio preprocessing identical to training pipeline
- Prediction with confidence scores for all classes
- Error handling for invalid files and formats

Usage:
    python3 test_model.py <path_to_audio_file.wav>

Examples:
    python3 test_model.py audio.wav
    python3 test_model.py /path/to/audio/file.wav
    python3 test_model.py disturbance/test_audio.wav

Author: Audio Classification System
Date: 2025-08-30
"""

import sys
import json
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
import os
import traceback
from pathlib import Path
import argparse

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
tf.get_logger().setLevel('ERROR')

# Configuration constants - CRITICAL: Must match preprocessing and training scripts
SAMPLE_RATE = 22050  # Hz - consistent with preprocessing
DURATION = 3.0       # seconds - normalize all audio to this duration
N_MFCC = 13         # number of MFCC coefficients to extract
HOP_LENGTH = 512    # MFCC hop length - must match preprocessing
N_FFT = 2048        # FFT window size - must match preprocessing

def load_model_and_mapping(model_path="model.h5", data_path="data.json", metadata_path="model_metadata.json"):
    """
    Load the trained model, class mapping, and normalization parameters.

    Args:
        model_path (str): Path to the trained model file
        data_path (str): Path to the data.json file containing class mapping
        metadata_path (str): Path to the model metadata file

    Returns:
        tuple: (model, class_mapping, normalization_params) or (None, None, None) if loading fails
    """
    try:
        print("🔄 Loading model and configuration...")

        # Check if required files exist
        required_files = [model_path, data_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            print(f"❌ Error: Required files not found: {missing_files}")
            print("\nTo fix this issue:")
            if "data.json" in missing_files:
                print("  1. Run: python3 dataset_preprocess.py")
            if "model.h5" in missing_files:
                print("  2. Run: python3 train_model.py")
            return None, None, None

        # Load trained model
        print(f"🤖 Loading trained model from {model_path}...")
        try:
            model = keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully")

            # Print model info
            total_params = model.count_params()
            print(f"   Model parameters: {total_params:,}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None, None

        # Load class mapping from data.json
        print(f"📋 Loading class mapping from {data_path}...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "mapping" not in data:
                print("❌ Error: No 'mapping' key found in data.json")
                return None, None, None

            class_mapping = data["mapping"]
            print(f"✅ Class mapping loaded: {class_mapping}")

        except Exception as e:
            print(f"❌ Error loading class mapping: {e}")
            return None, None, None

        # Try to load normalization parameters from metadata
        normalization_params = None
        if os.path.exists(metadata_path):
            try:
                print(f"📄 Loading normalization parameters from {metadata_path}...")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if "normalization_params" in metadata:
                    normalization_params = metadata["normalization_params"]
                    print("✅ Normalization parameters loaded from metadata")
                else:
                    print("⚠️  No normalization parameters found in metadata")

            except Exception as e:
                print(f"⚠️  Warning: Could not load metadata: {e}")
        else:
            print("⚠️  No metadata file found - will use basic normalization")

        # Validate model and mapping compatibility
        expected_classes = len(class_mapping)
        model_output_shape = model.output_shape[-1]

        if model_output_shape != expected_classes:
            print(f"❌ Error: Model output shape ({model_output_shape}) doesn't match class mapping ({expected_classes})")
            return None, None, None

        print(f"✅ Model and mapping are compatible ({expected_classes} classes)")

        return model, class_mapping, normalization_params

    except Exception as e:
        print(f"❌ Unexpected error loading model and mapping: {e}")
        traceback.print_exc()
        return None, None, None

def normalize_audio_duration(audio_data, target_duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Normalize audio to exactly target_duration seconds.
    CRITICAL: This function must be IDENTICAL to the one used in preprocessing.

    Args:
        audio_data (np.array): Input audio signal
        target_duration (float): Target duration in seconds
        sample_rate (int): Audio sampling rate

    Returns:
        np.array: Normalized audio signal of exact target duration
    """
    if len(audio_data) == 0:
        # Handle empty audio
        target_length = int(target_duration * sample_rate)
        return np.zeros(target_length, dtype=np.float32)

    target_length = int(target_duration * sample_rate)
    current_length = len(audio_data)

    if current_length == target_length:
        return audio_data.astype(np.float32)
    elif current_length < target_length:
        # Pad with zeros if audio is shorter
        padding = target_length - current_length
        padded_audio = np.pad(audio_data, (0, padding), mode='constant', constant_values=0.0)
        return padded_audio.astype(np.float32)
    else:
        # Truncate from center if audio is longer
        start_idx = (current_length - target_length) // 2
        end_idx = start_idx + target_length
        truncated_audio = audio_data[start_idx:end_idx]
        return truncated_audio.astype(np.float32)

def extract_mfcc_features(audio_data, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extract MFCC features from audio data.
    CRITICAL: This function must be IDENTICAL to the one used in preprocessing.

    Args:
        audio_data (np.array): Input audio signal
        sample_rate (int): Audio sampling rate
        n_mfcc (int): Number of MFCC coefficients to extract

    Returns:
        np.array: MFCC features with shape (n_mfcc, time_frames) or None if failed
    """
    try:
        if len(audio_data) == 0:
            print("❌ Error: Empty audio data for MFCC extraction")
            return None

        # Ensure audio is float32 and normalized
        audio_data = audio_data.astype(np.float32)

        # Extract MFCC features with IDENTICAL parameters to preprocessing
        mfcc_features = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            window='hann',
            center=True,
            pad_mode='constant'
        )

        # Validate output shape
        if mfcc_features.shape[0] != n_mfcc:
            print(f"❌ Error: Expected {n_mfcc} MFCC coefficients, got {mfcc_features.shape[0]}")
            return None

        return mfcc_features.astype(np.float32)

    except Exception as e:
        print(f"❌ Error extracting MFCC features: {e}")
        traceback.print_exc()
        return None

def preprocess_audio_file(file_path, normalization_params=None):
    """
    Preprocess an audio file for inference.
    CRITICAL: This must use IDENTICAL preprocessing steps as the training pipeline.

    Args:
        file_path (str): Path to the audio file
        normalization_params (dict): Normalization parameters from training (if available)

    Returns:
        np.array: Preprocessed features ready for model input, or None if failed
    """
    try:
        print(f"🔄 Preprocessing audio file: {file_path}")

        # Validate file exists and is accessible
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found - {file_path}")
            return None

        if not os.path.isfile(file_path):
            print(f"❌ Error: Not a file - {file_path}")
            return None

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"❌ Error: Empty file - {file_path}")
            return None

        print(f"📄 File size: {file_size / 1024:.1f} KB")

        # Check file extension (warn but don't fail)
        supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in supported_extensions:
            print(f"⚠️  Warning: '{file_ext}' files may not be fully supported")
            print(f"   Recommended format: .wav")
            print(f"   Attempting to load anyway...")

        # Load audio file with librosa
        print("🎵 Loading audio...")
        try:
            audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
        except Exception as load_error:
            print(f"❌ Error loading audio: {load_error}")
            return None

        # Validate loaded audio
        if audio_data is None or len(audio_data) == 0:
            print(f"❌ Error: Empty or invalid audio data")
            return None

        # Check for invalid values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            print(f"❌ Error: Audio contains invalid values (NaN/Inf)")
            return None

        original_duration = len(audio_data) / SAMPLE_RATE
        print(f"✅ Audio loaded: {len(audio_data):,} samples, {original_duration:.2f} seconds")

        if original_duration < 0.1:
            print(f"⚠️  Warning: Very short audio ({original_duration:.2f}s)")
        elif original_duration > 10.0:
            print(f"ℹ️  Long audio detected ({original_duration:.2f}s) - will be truncated to {DURATION}s")

        # Normalize duration to exactly 3 seconds (IDENTICAL to preprocessing)
        print(f"⏱️  Normalizing audio duration to {DURATION} seconds...")
        normalized_audio = normalize_audio_duration(audio_data)
        final_duration = len(normalized_audio) / SAMPLE_RATE
        print(f"✅ Audio normalized: {len(normalized_audio):,} samples, {final_duration:.2f} seconds")

        # Extract MFCC features (IDENTICAL to preprocessing)
        print(f"🔊 Extracting {N_MFCC} MFCC features...")
        mfcc_features = extract_mfcc_features(normalized_audio)

        if mfcc_features is None:
            print("❌ Failed to extract MFCC features")
            return None

        print(f"✅ MFCC features extracted: shape {mfcc_features.shape}")

        # Validate MFCC features
        if np.any(np.isnan(mfcc_features)) or np.any(np.isinf(mfcc_features)):
            print("❌ Error: MFCC features contain invalid values (NaN/Inf)")
            return None

        # Flatten features for model input (IDENTICAL to training)
        flattened_features = mfcc_features.flatten()
        print(f"📐 Features flattened: {flattened_features.shape}")

        # Apply normalization (IDENTICAL to training)
        if normalization_params is not None:
            print("🔧 Applying training normalization parameters...")
            try:
                feature_min = np.array(normalization_params['min'])
                feature_max = np.array(normalization_params['max'])

                # Ensure shapes match
                if len(feature_min) != len(flattened_features):
                    print(f"⚠️  Warning: Normalization parameter shape mismatch")
                    print(f"   Expected: {len(feature_min)}, Got: {len(flattened_features)}")
                    print("   Using basic normalization instead")
                    raise ValueError("Shape mismatch")

                # Apply saved normalization
                feature_range = feature_max - feature_min
                feature_range = np.where(feature_range == 0, 1, feature_range)
                normalized_features = (flattened_features - feature_min) / feature_range

                print("✅ Applied training normalization parameters")

            except Exception as e:
                print(f"⚠️  Warning: Could not apply training normalization: {e}")
                print("   Falling back to basic normalization")
                normalization_params = None

        if normalization_params is None:
            print("🔧 Applying basic min-max normalization...")
            feature_min = flattened_features.min()
            feature_max = flattened_features.max()

            if feature_max > feature_min:
                normalized_features = (flattened_features - feature_min) / (feature_max - feature_min)
            else:
                normalized_features = np.zeros_like(flattened_features)

            print("✅ Applied basic normalization")

        # Validate normalized features
        if np.any(np.isnan(normalized_features)) or np.any(np.isinf(normalized_features)):
            print("❌ Error: Normalization produced invalid values")
            return None

        # Reshape for model input (add batch dimension)
        model_input = normalized_features.reshape(1, -1).astype(np.float32)

        print(f"✅ Features prepared for model: shape {model_input.shape}")
        print(f"   Min: {model_input.min():.6f}, Max: {model_input.max():.6f}")
        print(f"   Mean: {model_input.mean():.6f}, Std: {model_input.std():.6f}")

        return model_input

    except Exception as e:
        print(f"❌ Unexpected error preprocessing audio file: {e}")
        traceback.print_exc()
        return None

def make_prediction(model, features, class_mapping):
    """
    Make prediction on preprocessed features and display comprehensive results.

    Args:
        model: Trained Keras model
        features (np.array): Preprocessed features
        class_mapping (list): List of class names

    Returns:
        dict: Dictionary containing prediction results
    """
    try:
        print("\n🔮 Making prediction...")

        # Get prediction probabilities
        predictions = model.predict(features, verbose=0)
        confidence_scores = predictions[0]  # Remove batch dimension

        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = class_mapping[predicted_class_idx]
        max_confidence = confidence_scores[predicted_class_idx]

        # Calculate additional metrics
        entropy = -np.sum(confidence_scores * np.log(confidence_scores + 1e-10))
        max_entropy = np.log(len(class_mapping))
        normalized_entropy = entropy / max_entropy

        # Display results
        print("\n" + "=" * 70)
        print("🎯 PREDICTION RESULTS")
        print("=" * 70)
        print(f"🏆 Predicted Class: {predicted_class}")
        print(f"🎯 Confidence: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
        print(f"📊 Prediction Entropy: {entropy:.4f} (normalized: {normalized_entropy:.4f})")

        print(f"\n📈 Confidence Scores for All Classes:")
        print("-" * 70)
        print(f"{'Rank':<6} {'Class':<15} {'Confidence':<12} {'Percentage':<12} {'Bar':<20}")
        print("-" * 70)

        # Sort classes by confidence for better display
        sorted_indices = np.argsort(confidence_scores)[::-1]

        for rank, class_idx in enumerate(sorted_indices, 1):
            class_name = class_mapping[class_idx]
            confidence = confidence_scores[class_idx]
            percentage = confidence * 100

            # Create visual bar
            bar_length = int(confidence * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            # Add visual indicator for predicted class
            indicator = "🥇" if class_idx == predicted_class_idx else f"{rank:2d}"

            print(f"{indicator:<6} {class_name:<15} {confidence:<12.4f} {percentage:<12.2f} {bar}")

        print("=" * 70)

        # Provide detailed interpretation
        print("🔍 Interpretation:")
        if max_confidence > 0.9:
            interpretation = "Very High Confidence - Excellent prediction"
            reliability = "🟢 Highly Reliable"
        elif max_confidence > 0.8:
            interpretation = "High Confidence - Good prediction"
            reliability = "🟢 Reliable"
        elif max_confidence > 0.7:
            interpretation = "Moderate-High Confidence - Acceptable prediction"
            reliability = "🟡 Moderately Reliable"
        elif max_confidence > 0.6:
            interpretation = "Moderate Confidence - Use with caution"
            reliability = "🟡 Somewhat Reliable"
        elif max_confidence > 0.5:
            interpretation = "Low Confidence - Uncertain prediction"
            reliability = "🟠 Low Reliability"
        else:
            interpretation = "Very Low Confidence - Highly uncertain"
            reliability = "🔴 Unreliable"

        print(f"   {interpretation}")
        print(f"   {reliability}")

        # Entropy interpretation
        if normalized_entropy < 0.3:
            entropy_desc = "Low entropy - Model is confident"
        elif normalized_entropy < 0.7:
            entropy_desc = "Medium entropy - Model has some uncertainty"
        else:
            entropy_desc = "High entropy - Model is very uncertain"

        print(f"   📊 {entropy_desc}")

        # Check for close predictions
        sorted_confidences = np.sort(confidence_scores)[::-1]
        if len(sorted_confidences) > 1:
            confidence_gap = sorted_confidences[0] - sorted_confidences[1]
            if confidence_gap < 0.1:
                print(f"   ⚠️  Close prediction: Gap to 2nd choice is only {confidence_gap:.3f}")

        # Arduino integration section
        print(f"\n🤖 Arduino Integration:")
        print("=" * 40)
        print(f"📤 Send Command: {predicted_class_idx}")
        print(f"🏷️  Class: '{predicted_class}'")
        print(f"🎯 Confidence: {max_confidence:.3f}")

        print(f"\n📋 Complete Class Mapping for Arduino:")
        for i, class_name in enumerate(class_mapping):
            status = " ← PREDICTED" if i == predicted_class_idx else ""
            print(f"   {i} → {class_name}{status}")

        # Usage example
        print(f"\n💡 Arduino Code Example:")
        print(f"   int command = {predicted_class_idx};  // Received from classification")
        print(f"   switch(command) {{")
        for i, class_name in enumerate(class_mapping):
            print(f"     case {i}: // {class_name}")
            print(f"       // Implement {class_name} vibration pattern")
            print(f"       break;")
        print(f"   }}")

        print("=" * 70)

        # Return comprehensive results
        results = {
            'predicted_class': predicted_class,
            'predicted_class_idx': int(predicted_class_idx),
            'confidence': float(max_confidence),
            'confidence_scores': confidence_scores.tolist(),
            'class_mapping': class_mapping,
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'interpretation': interpretation,
            'reliability': reliability,
            'arduino_command': int(predicted_class_idx)
        }

        return results

    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        traceback.print_exc()
        return None

def validate_audio_file(file_path):
    """
    Comprehensive validation of the input audio file.

    Args:
        file_path (str): Path to the audio file

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        print(f"🔍 Validating audio file: {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: File '{file_path}' does not exist")
            return False

        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            print(f"❌ Error: '{file_path}' is not a file")
            return False

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"❌ Error: File '{file_path}' is empty")
            return False

        if file_size < 1000:  # Less than 1KB
            print(f"⚠️  Warning: File is very small ({file_size} bytes) - may be corrupted")

        # Check file extension
        supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff', '.au']
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in supported_extensions:
            print(f"⚠️  Warning: '{file_ext}' files may not be fully supported")
            print(f"   Supported formats: {', '.join(supported_extensions)}")
            print(f"   Recommended: .wav")
            print(f"   Attempting to process anyway...")
        else:
            print(f"✅ File format '{file_ext}' is supported")

        # Check file permissions
        if not os.access(file_path, os.R_OK):
            print(f"❌ Error: Cannot read file '{file_path}' - permission denied")
            return False

        print(f"✅ File validation passed")
        return True

    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False

def print_usage():
    """Print comprehensive usage instructions."""
    print("\n" + "=" * 60)
    print("🎵 AUDIO CLASSIFICATION MODEL TESTING")
    print("=" * 60)
    print("\n📖 Usage:")
    print("   python3 test_model.py <path_to_audio_file>")

    print("\n💡 Examples:")
    print("   python3 test_model.py sample_audio.wav")
    print("   python3 test_model.py /path/to/audio/file.wav")
    print("   python3 test_model.py disturbance/test_audio.wav")
    print("   python3 test_model.py \"audio file with spaces.wav\"")

    print("\n🎵 Supported Audio Formats:")
    print("   • .wav (recommended - best compatibility)")
    print("   • .mp3 (widely supported)")
    print("   • .flac (lossless compression)")
    print("   • .m4a (AAC format)")
    print("   • .ogg (open source)")
    print("   • .aiff (uncompressed)")
    print("   • .au (basic format)")

    print("\n⚙️  Audio Processing:")
    print("   • Automatically normalized to 3 seconds duration")
    print("   • Resampled to 22050 Hz")
    print("   • Converted to mono")
    print("   • MFCC features extracted (13 coefficients)")

    print("\n🤖 Arduino Integration:")
    print("   The system outputs integer commands (0-3) for Arduino:")
    print("   • 0 → disturbance pattern")
    print("   • 1 → slow vibration pattern")
    print("   • 2 → medium vibration pattern")
    print("   • 3 → fast vibration pattern")

    print("\n📋 Prerequisites:")
    print("   1. Run dataset_preprocess.py first (creates data.json)")
    print("   2. Run train_model.py second (creates model.h5)")
    print("   3. Then use this script for testing")

    print("\n🔧 Troubleshooting:")
    print("   • File not found: Check the file path")
    print("   • Model not found: Run train_model.py first")
    print("   • Data not found: Run dataset_preprocess.py first")
    print("   • Low confidence: Try different audio or retrain with more data")

    print("=" * 60)

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='Audio Classification Model Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_model.py audio.wav
  python3 test_model.py /path/to/audio.wav
  python3 test_model.py "file with spaces.wav"

The system will classify the audio into one of 4 categories:
  0 - disturbance
  1 - slow
  2 - medium
  3 - fast
        """
    )

    parser.add_argument(
        'audio_file',
        help='Path to the audio file to classify'
    )

    parser.add_argument(
        '--model',
        default='model.h5',
        help='Path to the trained model file (default: model.h5)'
    )

    parser.add_argument(
        '--data',
        default='data.json',
        help='Path to the dataset file (default: data.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()

def main():
    """
    Main function to run the complete testing pipeline.
    """
    print("=" * 80)
    print("🎵 AUDIO CLASSIFICATION MODEL TESTING")
    print("=" * 80)

    try:
        # Parse command line arguments
        if len(sys.argv) == 1:
            print("❌ Error: No audio file provided")
            print_usage()
            return 1

        # Simple argument parsing for backward compatibility
        if len(sys.argv) == 2:
            audio_file_path = sys.argv[1]
            model_path = "model.h5"
            data_path = "data.json"
            verbose = False
        else:
            # Use argparse for advanced options
            args = parse_arguments()
            audio_file_path = args.audio_file
            model_path = args.model
            data_path = args.data
            verbose = args.verbose

        print(f"🎵 Input audio file: {audio_file_path}")
        print(f"🤖 Model file: {model_path}")
        print(f"📄 Data file: {data_path}")

        # Step 1: Validate input file
        print("\n" + "-" * 60)
        print("STEP 1: VALIDATING INPUT FILE")
        print("-" * 60)

        if not validate_audio_file(audio_file_path):
            print("❌ File validation failed")
            return 1

        # Step 2: Load model and configuration
        print("\n" + "-" * 60)
        print("STEP 2: LOADING MODEL AND CONFIGURATION")
        print("-" * 60)

        model, class_mapping, normalization_params = load_model_and_mapping(
            model_path, data_path
        )

        if model is None or class_mapping is None:
            print("❌ Failed to load model or configuration")
            return 1

        # Step 3: Preprocess audio file
        print("\n" + "-" * 60)
        print("STEP 3: PREPROCESSING AUDIO")
        print("-" * 60)

        features = preprocess_audio_file(audio_file_path, normalization_params)
        if features is None:
            print("❌ Failed to preprocess audio file")
            return 1

        # Step 4: Make prediction
        print("\n" + "-" * 60)
        print("STEP 4: MAKING PREDICTION")
        print("-" * 60)

        results = make_prediction(model, features, class_mapping)
        if results is None:
            print("❌ Failed to make prediction")
            return 1

        # Step 5: Summary
        print("\n" + "=" * 80)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        print(f"📁 Input File: {audio_file_path}")
        print(f"🏆 Predicted Class: {results['predicted_class']}")
        print(f"🎯 Confidence: {results['confidence']:.3f} ({results['confidence']*100:.1f}%)")
        print(f"🤖 Arduino Command: {results['arduino_command']}")

        if results['confidence'] < 0.7:
            print(f"\n💡 Tips for Better Results:")
            print(f"   • Ensure clear, high-quality audio")
            print(f"   • Record in a quiet environment")
            print(f"   • Use consistent recording conditions")
            print(f"   • Consider retraining with more diverse data")

        print("=" * 80)
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)