#!/usr/bin/env python3
"""
Create Test Audio Files for Audio Classification System

This script generates synthetic test audio files for all 4 classes to verify
the audio classification system works correctly. Each class gets a distinct
audio pattern that can be easily distinguished by the model.

Classes:
- disturbance: Low frequency with irregular patterns
- slow: Low-medium frequency with slow modulation
- medium: Medium frequency with moderate modulation
- fast: High frequency with rapid modulation

Author: Audio Classification System
Date: 2025-08-30
"""

import numpy as np
import soundfile as sf
import os
import sys
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Audio parameters - CRITICAL: Must match preprocessing script
SAMPLE_RATE = 22050  # Hz - consistent with preprocessing
DURATION = 3.0       # seconds - exactly 3 seconds
SAMPLES = int(SAMPLE_RATE * DURATION)
RANDOM_SEED = 42     # for reproducible test files

# Set random seed for reproducible results
np.random.seed(RANDOM_SEED)

def create_disturbance_audio(filename):
    """
    Create disturbance class audio with irregular, chaotic patterns.
    Uses low frequencies with random amplitude and frequency modulation.
    """
    t = np.linspace(0, DURATION, SAMPLES, False)

    # Base low frequency with random variations
    base_freq = 120 + np.random.normal(0, 20, SAMPLES)

    # Create chaotic signal with multiple components
    audio = np.zeros(SAMPLES)

    # Primary chaotic component
    audio += 0.4 * np.sin(2 * np.pi * base_freq * t)

    # Add irregular bursts
    for _ in range(5):
        start = np.random.randint(0, SAMPLES - 1000)
        end = start + np.random.randint(500, 1500)
        burst_freq = np.random.uniform(80, 200)
        audio[start:end] += 0.3 * np.sin(2 * np.pi * burst_freq * t[start:end])

    # Add noise for realism
    noise = np.random.normal(0, 0.15, SAMPLES)
    audio += noise

    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)

def create_slow_audio(filename):
    """
    Create slow class audio with low frequency and slow modulation.
    Uses steady, slow-changing patterns.
    """
    t = np.linspace(0, DURATION, SAMPLES, False)

    # Base frequency for slow rhythm
    base_freq = 200

    # Slow amplitude modulation (0.5 Hz)
    amp_mod = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)

    # Slow frequency modulation
    freq_mod = base_freq + 30 * np.sin(2 * np.pi * 0.3 * t)

    # Create smooth, slow audio
    audio = amp_mod * np.sin(2 * np.pi * freq_mod * t)

    # Add subtle harmonics
    audio += 0.2 * amp_mod * np.sin(2 * np.pi * freq_mod * 2 * t)

    # Add minimal noise
    noise = np.random.normal(0, 0.05, SAMPLES)
    audio += noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)

def create_medium_audio(filename):
    """
    Create medium class audio with medium frequency and moderate modulation.
    Uses balanced, moderate-paced patterns.
    """
    t = np.linspace(0, DURATION, SAMPLES, False)

    # Base frequency for medium rhythm
    base_freq = 400

    # Medium amplitude modulation (1.5 Hz)
    amp_mod = 0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t)

    # Medium frequency modulation
    freq_mod = base_freq + 50 * np.sin(2 * np.pi * 1.0 * t)

    # Create medium-paced audio
    audio = amp_mod * np.sin(2 * np.pi * freq_mod * t)

    # Add harmonics for richness
    audio += 0.3 * amp_mod * np.sin(2 * np.pi * freq_mod * 1.5 * t)
    audio += 0.15 * amp_mod * np.sin(2 * np.pi * freq_mod * 3 * t)

    # Add moderate noise
    noise = np.random.normal(0, 0.08, SAMPLES)
    audio += noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)

def create_fast_audio(filename):
    """
    Create fast class audio with high frequency and rapid modulation.
    Uses quick, energetic patterns.
    """
    t = np.linspace(0, DURATION, SAMPLES, False)

    # Base frequency for fast rhythm
    base_freq = 800

    # Fast amplitude modulation (3 Hz)
    amp_mod = 0.7 + 0.3 * np.sin(2 * np.pi * 3.0 * t)

    # Fast frequency modulation with multiple components
    freq_mod = base_freq + 100 * np.sin(2 * np.pi * 2.5 * t) + 50 * np.sin(2 * np.pi * 5.0 * t)

    # Create fast, energetic audio
    audio = amp_mod * np.sin(2 * np.pi * freq_mod * t)

    # Add multiple harmonics for complexity
    audio += 0.4 * amp_mod * np.sin(2 * np.pi * freq_mod * 1.25 * t)
    audio += 0.2 * amp_mod * np.sin(2 * np.pi * freq_mod * 2.5 * t)
    audio += 0.1 * amp_mod * np.sin(2 * np.pi * freq_mod * 5 * t)

    # Add more noise for energy
    noise = np.random.normal(0, 0.1, SAMPLES)
    audio += noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)

def create_test_audio_file(class_name, audio_generator):
    """
    Create a test audio file for a specific class.

    Args:
        class_name (str): Name of the class (folder name)
        audio_generator (function): Function to generate audio for this class

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure class folder exists
        class_folder = Path(class_name)
        if not class_folder.exists():
            print(f"⚠️  Creating folder: {class_folder}")
            class_folder.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = class_folder / f"test_{class_name}.wav"

        print(f"🎵 Generating {class_name} audio...")

        # Generate audio using the specific generator
        audio_data = audio_generator(str(filename))

        # Validate audio data
        if len(audio_data) != SAMPLES:
            print(f"❌ Error: Generated audio has wrong length: {len(audio_data)} != {SAMPLES}")
            return False

        # Check for invalid values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            print(f"❌ Error: Generated audio contains invalid values")
            return False

        # Save audio file
        sf.write(str(filename), audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')

        # Verify file was created and has correct size
        if not filename.exists():
            print(f"❌ Error: Failed to create file {filename}")
            return False

        file_size = filename.stat().st_size
        expected_size = SAMPLES * 2 + 44  # 16-bit samples + WAV header

        print(f"✅ Created: {filename}")
        print(f"   Duration: {DURATION} seconds")
        print(f"   Sample rate: {SAMPLE_RATE} Hz")
        print(f"   File size: {file_size:,} bytes")

        return True

    except Exception as e:
        print(f"❌ Error creating {class_name} audio: {e}")
        return False

def validate_folders():
    """
    Validate that all required class folders exist or can be created.

    Returns:
        bool: True if all folders are ready, False otherwise
    """
    class_names = ["disturbance", "slow", "medium", "fast"]

    print("🔍 Validating class folders...")

    for class_name in class_names:
        class_folder = Path(class_name)

        if not class_folder.exists():
            try:
                class_folder.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created folder: {class_folder}")
            except Exception as e:
                print(f"❌ Error creating folder {class_folder}: {e}")
                return False
        elif not class_folder.is_dir():
            print(f"❌ Error: {class_folder} exists but is not a directory")
            return False
        else:
            print(f"✅ Folder exists: {class_folder}")

    return True

def main():
    """
    Main function to create test audio files for all classes.
    """
    print("=" * 80)
    print("🎵 CREATING TEST AUDIO FILES FOR AUDIO CLASSIFICATION")
    print("=" * 80)

    print(f"📊 Audio Parameters:")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Duration: {DURATION} seconds")
    print(f"   Samples per file: {SAMPLES:,}")
    print(f"   Random Seed: {RANDOM_SEED}")

    try:
        # Validate folders
        print(f"\n" + "-" * 60)
        print("STEP 1: VALIDATING FOLDERS")
        print("-" * 60)

        if not validate_folders():
            print("❌ Folder validation failed")
            return 1

        # Create test files
        print(f"\n" + "-" * 60)
        print("STEP 2: GENERATING TEST AUDIO FILES")
        print("-" * 60)

        # Define class generators
        class_generators = [
            ("disturbance", create_disturbance_audio),
            ("slow", create_slow_audio),
            ("medium", create_medium_audio),
            ("fast", create_fast_audio)
        ]

        success_count = 0
        total_count = len(class_generators)

        for class_name, generator in class_generators:
            if create_test_audio_file(class_name, generator):
                success_count += 1
            print()  # Add spacing between files

        # Summary
        print("-" * 60)
        print(f"📊 Generation Summary:")
        print(f"   Successfully created: {success_count}/{total_count} files")
        print(f"   Failed: {total_count - success_count}/{total_count} files")

        if success_count == total_count:
            print("\n" + "=" * 80)
            print("✅ ALL TEST AUDIO FILES CREATED SUCCESSFULLY!")
            print("=" * 80)

            print(f"\n📁 Generated Files:")
            for class_name, _ in class_generators:
                filename = Path(class_name) / f"test_{class_name}.wav"
                if filename.exists():
                    print(f"   🎵 {filename}")

            print(f"\n🚀 Next Steps:")
            print(f"   1. Run preprocessing: python3 dataset_preprocess.py")
            print(f"   2. Train the model: python3 train_model.py")
            print(f"   3. Test predictions: python3 test_model.py <audio_file.wav>")

            print(f"\n💡 Note:")
            print(f"   These are synthetic test files for system verification.")
            print(f"   For real applications, replace with actual audio recordings.")

            print("=" * 80)
            return 0
        else:
            print(f"\n❌ Some files failed to generate. Check the errors above.")
            return 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Generation interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)