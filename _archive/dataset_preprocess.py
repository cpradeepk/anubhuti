#!/usr/bin/env python3
"""
Audio Dataset Preprocessing Script for Sound-to-Vibration Classification

This script preprocesses audio files from dataset folders and creates a structured
data.json file for training. It handles:
- Recursive scanning of dataset folders (disturbance/, slow/, medium/, fast/)
- Audio loading with librosa at 22050 Hz sampling rate
- Normalization to exactly 3 seconds (pad shorter, truncate longer from center)
- MFCC feature extraction with 13 coefficients
- Label creation based on folder names
- Error handling for corrupted files and empty folders
- Progress tracking with progress bar

Author: Audio Classification System
Date: 2025-08-30
"""

import os
import sys
import json
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings
from pathlib import Path
import traceback

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# Configuration constants - CRITICAL: These must match across all scripts
SAMPLE_RATE = 22050  # Hz - consistent across all scripts
DURATION = 3.0       # seconds - normalize all audio to this duration
N_MFCC = 13         # number of MFCC coefficients to extract
RANDOM_SEED = 42    # for reproducible results
HOP_LENGTH = 512    # MFCC hop length - must be consistent
N_FFT = 2048        # FFT window size - must be consistent

# Class mapping - must match folder names exactly
CLASS_MAPPING = ["disturbance", "slow", "medium", "fast"]
CLASS_TO_LABEL = {class_name: idx for idx, class_name in enumerate(CLASS_MAPPING)}

def normalize_audio_duration(audio_data, target_duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Normalize audio to exactly target_duration seconds.
    CRITICAL: This function must be identical across all scripts for consistency.

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
    CRITICAL: Parameters must be identical across all scripts for consistency.

    Args:
        audio_data (np.array): Input audio signal
        sample_rate (int): Audio sampling rate
        n_mfcc (int): Number of MFCC coefficients to extract

    Returns:
        np.array: MFCC features with shape (n_mfcc, time_frames) or None if failed
    """
    try:
        if len(audio_data) == 0:
            print("Warning: Empty audio data for MFCC extraction")
            return None

        # Ensure audio is float32 and normalized
        audio_data = audio_data.astype(np.float32)

        # Extract MFCC features with consistent parameters
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

        # Ensure consistent output shape
        if mfcc_features.shape[0] != n_mfcc:
            print(f"Warning: Expected {n_mfcc} MFCC coefficients, got {mfcc_features.shape[0]}")
            return None

        return mfcc_features.astype(np.float32)

    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        traceback.print_exc()
        return None

def process_audio_file(file_path, class_label):
    """
    Process a single audio file and extract features.

    Args:
        file_path (str): Path to the audio file
        class_label (int): Numeric class label

    Returns:
        tuple: (mfcc_features, label, file_path) or None if processing fails
    """
    try:
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None

        if not os.path.isfile(file_path):
            print(f"Error: Not a file - {file_path}")
            return None

        # Check file size (avoid processing empty files)
        file_size = os.path.getsize(file_path)
        if file_size < 1000:  # Less than 1KB is likely corrupted
            print(f"Warning: File too small (possibly corrupted) - {file_path}")
            return None

        # Load audio file with librosa
        try:
            audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
        except Exception as load_error:
            print(f"Error loading audio file {file_path}: {load_error}")
            return None

        # Validate loaded audio
        if audio_data is None or len(audio_data) == 0:
            print(f"Warning: Empty or invalid audio data in {file_path}")
            return None

        # Check for NaN or infinite values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            print(f"Warning: Invalid audio values (NaN/Inf) in {file_path}")
            return None

        # Normalize duration to exactly 3 seconds
        normalized_audio = normalize_audio_duration(audio_data)

        # Extract MFCC features
        mfcc_features = extract_mfcc_features(normalized_audio)

        if mfcc_features is None:
            print(f"Failed to extract MFCC features from {file_path}")
            return None

        # Validate MFCC features
        if np.any(np.isnan(mfcc_features)) or np.any(np.isinf(mfcc_features)):
            print(f"Warning: Invalid MFCC values (NaN/Inf) in {file_path}")
            return None

        # Convert to list for JSON serialization
        mfcc_list = mfcc_features.tolist()

        return mfcc_list, class_label, file_path

    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        traceback.print_exc()
        return None

def scan_dataset_folders(base_path="."):
    """
    Recursively scan dataset folders for .wav files.

    Args:
        base_path (str): Base directory containing class folders

    Returns:
        list: List of tuples (file_path, class_label, class_name)
    """
    audio_files = []
    base_path = Path(base_path)

    print(f"Scanning base directory: {base_path.absolute()}")

    # Check if base directory exists
    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_path}")
        return audio_files

    for class_name in CLASS_MAPPING:
        class_folder = base_path / class_name

        print(f"Checking class folder: {class_folder}")

        if not class_folder.exists():
            print(f"Warning: Class folder '{class_name}' not found at {class_folder}")
            continue

        if not class_folder.is_dir():
            print(f"Warning: '{class_name}' exists but is not a directory")
            continue

        # Get class label
        class_label = CLASS_TO_LABEL[class_name]

        # Find all audio files (support multiple formats but prefer .wav)
        audio_extensions = ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.FLAC"]
        found_files = []

        for extension in audio_extensions:
            found_files.extend(list(class_folder.rglob(extension)))

        if not found_files:
            print(f"Warning: No audio files found in {class_folder}")
            print(f"  Supported formats: {', '.join(audio_extensions)}")
            continue

        # Filter and validate files
        valid_files = []
        for audio_file in found_files:
            if audio_file.is_file() and os.path.getsize(audio_file) > 0:
                valid_files.append(audio_file)
            else:
                print(f"Skipping invalid file: {audio_file}")

        if not valid_files:
            print(f"Warning: No valid audio files found in {class_folder}")
            continue

        print(f"Found {len(valid_files)} valid audio files in '{class_name}' folder")

        for audio_file in valid_files:
            audio_files.append((str(audio_file), class_label, class_name))

    return audio_files

def create_dataset_json(audio_files, output_file="data.json"):
    """
    Process all audio files and create the structured data.json file.

    Args:
        audio_files (list): List of tuples (file_path, class_label, class_name)
        output_file (str): Output JSON file path

    Returns:
        bool: True if successful, False otherwise
    """
    if not audio_files:
        print("Error: No audio files found to process")
        print("Please ensure you have .wav files in the following folders:")
        for class_name in CLASS_MAPPING:
            print(f"  - {class_name}/")
        return False

    print(f"\nProcessing {len(audio_files)} audio files...")
    print("This may take a few minutes depending on the number of files...")

    # Initialize data structure with exact required format
    dataset = {
        "mapping": CLASS_MAPPING.copy(),  # Ensure it's a copy
        "labels": [],
        "mfcc": [],
        "files": []
    }

    # Process files with progress bar
    successful_files = 0
    failed_files = 0
    failed_file_list = []

    try:
        for file_path, class_label, class_name in tqdm(audio_files, desc="Processing audio files", unit="file"):
            result = process_audio_file(file_path, class_label)

            if result is not None:
                mfcc_features, label, processed_file_path = result

                # Validate data before adding
                if (isinstance(mfcc_features, list) and
                    isinstance(label, int) and
                    0 <= label < len(CLASS_MAPPING)):

                    # Add to dataset
                    dataset["mfcc"].append(mfcc_features)
                    dataset["labels"].append(label)
                    dataset["files"].append(processed_file_path)

                    successful_files += 1
                else:
                    print(f"Invalid data format for {file_path}")
                    failed_files += 1
                    failed_file_list.append(file_path)
            else:
                failed_files += 1
                failed_file_list.append(file_path)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return False
    except Exception as e:
        print(f"\nUnexpected error during processing: {e}")
        traceback.print_exc()
        return False

    # Print detailed processing summary
    print(f"\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Successfully processed: {successful_files} files")
    print(f"Failed to process: {failed_files} files")
    print(f"Total files attempted: {len(audio_files)}")
    print(f"Success rate: {(successful_files/len(audio_files)*100):.1f}%")

    if failed_files > 0:
        print(f"\nFailed files:")
        for failed_file in failed_file_list[:10]:  # Show first 10 failed files
            print(f"  - {failed_file}")
        if len(failed_file_list) > 10:
            print(f"  ... and {len(failed_file_list) - 10} more")

    if successful_files == 0:
        print("\nError: No files were successfully processed")
        print("Please check your audio files and try again")
        return False

    # Print class distribution
    print(f"\nClass Distribution:")
    total_samples = len(dataset["labels"])
    for i, class_name in enumerate(CLASS_MAPPING):
        count = dataset["labels"].count(i)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    # Validate dataset balance
    min_samples = min(dataset["labels"].count(i) for i in range(len(CLASS_MAPPING)))
    if min_samples == 0:
        print("\nWarning: Some classes have no samples. This will affect training quality.")
    elif min_samples < 5:
        print(f"\nWarning: Minimum class has only {min_samples} samples. Consider adding more data.")

    # Save to JSON file with error handling
    try:
        print(f"\nSaving dataset to {output_file}...")

        # Create backup if file exists
        if os.path.exists(output_file):
            backup_file = f"{output_file}.backup"
            os.rename(output_file, backup_file)
            print(f"Created backup: {backup_file}")

        # Save with proper formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        # Verify file was saved correctly
        file_size = os.path.getsize(output_file)
        print(f"Dataset successfully saved to {output_file}")
        print(f"File size: {file_size / 1024 / 1024:.2f} MB")
        print(f"Dataset contains {len(dataset['mfcc'])} samples")

        if len(dataset['mfcc']) > 0:
            mfcc_shape = np.array(dataset['mfcc'][0]).shape
            print(f"MFCC feature shape per sample: {mfcc_shape}")

        return True

    except Exception as e:
        print(f"Error saving dataset to {output_file}: {e}")
        traceback.print_exc()
        return False

def validate_dataset_structure():
    """
    Validate that the required dataset folders exist and are accessible.

    Returns:
        bool: True if all folders exist and are accessible, False otherwise
    """
    print("Validating dataset folder structure...")

    missing_folders = []
    inaccessible_folders = []

    for class_name in CLASS_MAPPING:
        folder_path = Path(class_name)

        if not folder_path.exists():
            missing_folders.append(class_name)
        elif not folder_path.is_dir():
            print(f"Error: '{class_name}' exists but is not a directory")
            inaccessible_folders.append(class_name)
        elif not os.access(folder_path, os.R_OK):
            print(f"Error: Cannot read from directory '{class_name}'")
            inaccessible_folders.append(class_name)

    if missing_folders:
        print("Error: Missing required dataset folders:")
        for folder in missing_folders:
            print(f"  - {folder}/")
        print("\nTo create missing folders, run:")
        print(f"mkdir -p {' '.join(missing_folders)}")
        return False

    if inaccessible_folders:
        print("Error: Some folders are not accessible:")
        for folder in inaccessible_folders:
            print(f"  - {folder}/")
        return False

    print("✅ All required folders exist and are accessible")
    return True

def print_system_info():
    """Print system information for debugging."""
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Librosa version: {librosa.__version__}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Audio parameters: {SAMPLE_RATE}Hz, {DURATION}s, {N_MFCC} MFCC coefficients")

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    print("=" * 80)
    print("AUDIO DATASET PREPROCESSING FOR SOUND-TO-VIBRATION CLASSIFICATION")
    print("=" * 80)

    # Print system information
    print_system_info()

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    print(f"\n✅ Random seed set to {RANDOM_SEED} for reproducible results")

    try:
        # Validate dataset structure
        print("\n" + "-" * 60)
        print("STEP 1: VALIDATING DATASET STRUCTURE")
        print("-" * 60)

        if not validate_dataset_structure():
            print("\n❌ Dataset validation failed. Please fix the issues above and try again.")
            return 1

        # Scan for audio files
        print("\n" + "-" * 60)
        print("STEP 2: SCANNING FOR AUDIO FILES")
        print("-" * 60)

        audio_files = scan_dataset_folders()

        if not audio_files:
            print("\n❌ No audio files found in dataset folders.")
            print("\nTo proceed, please add audio files (.wav, .mp3, .flac) to these folders:")
            for class_name in CLASS_MAPPING:
                print(f"  - {class_name}/")
            print("\nExample file structure:")
            print("  disturbance/")
            print("    ├── audio1.wav")
            print("    └── audio2.wav")
            print("  slow/")
            print("    ├── audio3.wav")
            print("    └── audio4.wav")
            print("  ... (and so on)")
            return 1

        # Process audio files and create dataset
        print("\n" + "-" * 60)
        print("STEP 3: PROCESSING AUDIO FILES")
        print("-" * 60)

        success = create_dataset_json(audio_files)

        if success:
            print("\n" + "=" * 80)
            print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nGenerated files:")
            print("  📄 data.json - Preprocessed dataset ready for training")
            print("\nNext steps:")
            print("  1. Run: python3 train_model.py")
            print("  2. After training, test with: python3 test_model.py <audio_file.wav>")
            print("\nFor Arduino integration, the system will output integer commands:")
            for i, class_name in enumerate(CLASS_MAPPING):
                print(f"  {i} → {class_name}")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("❌ PREPROCESSING FAILED")
            print("=" * 80)
            print("Please check the error messages above and try again.")
            print("Common issues:")
            print("  - Corrupted audio files")
            print("  - Unsupported audio formats")
            print("  - Insufficient disk space")
            print("  - Permission issues")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)