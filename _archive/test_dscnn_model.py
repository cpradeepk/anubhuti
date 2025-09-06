#!/usr/bin/env python3
"""
DS-CNN Model Testing Script

This script tests the trained DS-CNN model with proper input formatting.
It handles both 2D CNN input (DS-CNN) and 1D Dense input formats automatically.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import librosa
import argparse
from pathlib import Path

# Audio configuration (must match training)
SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 13
N_FRAMES = 130

def load_model_and_metadata(model_path="model.h5", metadata_path="model_metadata.json"):
    """
    Load the trained model and its metadata.
    """
    print("🤖 Loading DS-CNN model and metadata...")
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Load metadata
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded: {metadata.get('model_type', 'Unknown')}")
        else:
            print(f"⚠️  Metadata file not found: {metadata_path}")
        
        # Get class mapping
        class_mapping = ["disturbance", "soo", "hum", "hmm"]  # Default mapping
        if metadata and "class_mapping" in metadata:
            class_mapping = metadata["class_mapping"]
        
        print(f"📋 Classes: {class_mapping}")
        
        return model, metadata, class_mapping
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def preprocess_audio_for_dscnn(audio_file):
    """
    Preprocess audio file for DS-CNN model input.
    """
    print(f"🔄 Preprocessing audio for DS-CNN: {audio_file}")
    
    try:
        # Load audio
        audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        print(f"🎵 Audio loaded: {len(audio_data)} samples, {len(audio_data)/sr:.2f}s")
        
        # Normalize duration to exactly 3 seconds
        target_length = int(DURATION * SAMPLE_RATE)
        if len(audio_data) < target_length:
            # Pad with zeros
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        elif len(audio_data) > target_length:
            # Truncate
            audio_data = audio_data[:target_length]
        
        print(f"⏱️  Audio normalized: {len(audio_data)} samples, {len(audio_data)/sr:.2f}s")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=2048,
            hop_length=512,
            n_mels=40
        )
        
        print(f"🔊 MFCC extracted: {mfcc.shape}")
        
        # Ensure consistent frame count
        if mfcc.shape[1] < N_FRAMES:
            mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode='constant')
        elif mfcc.shape[1] > N_FRAMES:
            mfcc = mfcc[:, :N_FRAMES]
        
        print(f"📐 MFCC normalized: {mfcc.shape}")
        
        return mfcc
        
    except Exception as e:
        print(f"❌ Error preprocessing audio: {e}")
        return None

def format_input_for_model(mfcc_features, model):
    """
    Format MFCC features based on model input requirements.
    """
    model_input_shape = model.input_shape
    print(f"🔍 Model expects input shape: {model_input_shape}")
    
    if len(model_input_shape) == 4:  # DS-CNN format: (batch, height, width, channels)
        print("🏗️  Formatting for DS-CNN (2D CNN) model")
        
        # Reshape for DS-CNN: (1, n_mfcc, n_frames, 1)
        formatted_input = mfcc_features.reshape(1, N_MFCC, N_FRAMES, 1)
        
        # Normalize per sample (0-1 scaling)
        sample = formatted_input[0, :, :, 0]
        sample_min = sample.min()
        sample_max = sample.max()
        
        if sample_max > sample_min:
            formatted_input[0, :, :, 0] = (sample - sample_min) / (sample_max - sample_min)
        
        print("✅ Applied DS-CNN normalization (per-sample min-max)")
        
    else:  # Dense format: (batch, features)
        print("🏗️  Formatting for Dense model")
        
        # Flatten for dense model
        flattened = mfcc_features.flatten()
        
        # Basic normalization
        feature_min = flattened.min()
        feature_max = flattened.max()
        
        if feature_max > feature_min:
            normalized = (flattened - feature_min) / (feature_max - feature_min)
        else:
            normalized = flattened
        
        formatted_input = normalized.reshape(1, -1)
        print("✅ Applied Dense model normalization")
    
    print(f"✅ Input formatted: {formatted_input.shape}")
    print(f"   Range: [{formatted_input.min():.3f}, {formatted_input.max():.3f}]")
    
    return formatted_input.astype(np.float32)

def make_prediction(model, formatted_input, class_mapping):
    """
    Make prediction using the model.
    """
    print("🔮 Making prediction...")
    
    try:
        # Make prediction
        predictions = model.predict(formatted_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class_name = class_mapping[predicted_class_idx]
        
        print("✅ Prediction completed")
        
        return predicted_class_idx, predicted_class_name, confidence, predictions[0]
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None, None, None

def get_arduino_command(class_idx, class_name):
    """
    Get Arduino command based on class prediction.
    
    Motor Control Mapping:
    - "disturbance" (0) → No vibration (ignored)
    - "soo" (1) → Top motor vibrates
    - "hum" (2) → Bottom motor vibrates  
    - "hmm" (3) → Both motors vibrate simultaneously
    """
    arduino_commands = {
        0: 0,  # disturbance → no vibration
        1: 1,  # soo → top motor
        2: 2,  # hum → bottom motor
        3: 3   # hmm → both motors
    }
    
    return arduino_commands.get(class_idx, 0)

def test_dscnn_model(audio_file, model_path="model.h5", metadata_path="model_metadata.json"):
    """
    Test the DS-CNN model with an audio file.
    """
    print("=" * 80)
    print("🎯 DS-CNN MODEL TESTING")
    print("=" * 80)
    print(f"🎵 Audio file: {audio_file}")
    print(f"🤖 Model: {model_path}")
    print(f"📄 Metadata: {metadata_path}")
    print()
    
    # Validate input file
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    # Load model and metadata
    model, metadata, class_mapping = load_model_and_metadata(model_path, metadata_path)
    if model is None:
        return False
    
    # Preprocess audio
    mfcc_features = preprocess_audio_for_dscnn(audio_file)
    if mfcc_features is None:
        return False
    
    # Format input for model
    formatted_input = format_input_for_model(mfcc_features, model)
    if formatted_input is None:
        return False
    
    # Make prediction
    class_idx, class_name, confidence, all_predictions = make_prediction(
        model, formatted_input, class_mapping
    )
    
    if class_idx is None:
        return False
    
    # Get Arduino command
    arduino_command = get_arduino_command(class_idx, class_name)
    
    # Display results
    print()
    print("=" * 80)
    print("🎯 PREDICTION RESULTS")
    print("=" * 80)
    print(f"🎵 Audio File: {Path(audio_file).name}")
    print(f"🤖 Model Type: {metadata.get('model_type', 'Unknown') if metadata else 'Unknown'}")
    print()
    print(f"🎯 Predicted Class: {class_name}")
    print(f"📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    print(f"🔢 Class Index: {class_idx}")
    print(f"🤖 Arduino Command: {arduino_command}")
    print()
    
    # Show all class probabilities
    print("📊 All Class Probabilities:")
    print("-" * 40)
    for i, (class_name_i, prob) in enumerate(zip(class_mapping, all_predictions)):
        marker = "👉" if i == class_idx else "  "
        print(f"{marker} {class_name_i:<12}: {prob:.3f} ({prob*100:.1f}%)")
    
    print()
    
    # Motor control interpretation
    print("🤖 Arduino Motor Control:")
    print("-" * 40)
    motor_actions = {
        0: "No vibration (disturbance ignored)",
        1: "Top motor vibrates ('soo' sound)",
        2: "Bottom motor vibrates ('hum' sound)",
        3: "Both motors vibrate ('hmm' sound)"
    }
    print(f"   Command {arduino_command}: {motor_actions.get(arduino_command, 'Unknown')}")
    
    print()
    print("✅ Testing completed successfully!")
    
    return True

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Test DS-CNN Audio Classification Model")
    parser.add_argument("audio_file", help="Path to audio file to test")
    parser.add_argument("--model", default="model.h5", help="Path to model file")
    parser.add_argument("--metadata", default="model_metadata.json", help="Path to metadata file")
    
    args = parser.parse_args()
    
    success = test_dscnn_model(args.audio_file, args.model, args.metadata)
    
    if not success:
        print("❌ Testing failed!")
        sys.exit(1)
    
    print("🎉 Testing completed successfully!")

if __name__ == "__main__":
    main()
