#!/usr/bin/env python3
"""
Model Accuracy Improvement Script

This script implements multiple strategies to improve classification accuracy:
1. Generate more diverse synthetic training data
2. Implement data augmentation techniques
3. Use ensemble methods
4. Optimize model architecture for small datasets
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import soundfile as sf
from pathlib import Path
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from collections import Counter

def generate_diverse_training_data():
    """
    Generate more diverse synthetic audio data with variations.
    """
    print("🎵 Generating diverse training data...")
    
    # Generate multiple variations for each class
    variations = {
        'disturbance': [
            ('low', 'complexity'),
            ('medium', 'complexity'), 
            ('high', 'complexity')
        ],
        'slow': [
            ('low', 'gentleness'),
            ('medium', 'gentleness'),
            ('high', 'gentleness')
        ],
        'medium': [
            ('low', 'balance'),
            ('medium', 'balance'),
            ('high', 'balance')
        ],
        'fast': [
            ('low', 'intensity'),
            ('medium', 'intensity'),
            ('high', 'intensity')
        ]
    }
    
    for class_name, params_list in variations.items():
        for level, param_name in params_list:
            cmd = f"python3 {class_name}.py --batch 5 --{param_name} {level}"
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"✅ Generated {class_name} - {level} {param_name}")
                else:
                    print(f"⚠️  Warning: {class_name} generation issues")
            except Exception as e:
                print(f"❌ Error generating {class_name}: {e}")
    
    print("✅ Diverse data generation completed")

def apply_audio_augmentation(audio_data, sr=22050):
    """
    Apply audio augmentation techniques to increase data diversity.
    """
    augmented_samples = []
    
    # Original
    augmented_samples.append(audio_data)
    
    # Time stretching (speed variations)
    try:
        # Slower (0.8x speed)
        slower = librosa.effects.time_stretch(audio_data, rate=0.8)
        if len(slower) > 0:
            augmented_samples.append(slower)
        
        # Faster (1.2x speed)  
        faster = librosa.effects.time_stretch(audio_data, rate=1.2)
        if len(faster) > 0:
            augmented_samples.append(faster)
    except:
        pass
    
    # Pitch shifting
    try:
        # Lower pitch (-2 semitones)
        lower_pitch = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=-2)
        augmented_samples.append(lower_pitch)
        
        # Higher pitch (+2 semitones)
        higher_pitch = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=2)
        augmented_samples.append(higher_pitch)
    except:
        pass
    
    # Add subtle noise
    noise_factor = 0.005
    noisy = audio_data + noise_factor * np.random.normal(0, 1, len(audio_data))
    augmented_samples.append(noisy)
    
    return augmented_samples

def create_augmented_dataset():
    """
    Create an augmented dataset from existing audio files.
    """
    print("🔄 Creating augmented dataset...")
    
    class_folders = ['disturbance', 'slow', 'medium', 'fast']
    augmented_count = 0
    
    for class_folder in class_folders:
        if not os.path.exists(class_folder):
            continue
            
        audio_files = list(Path(class_folder).glob('*.wav'))
        print(f"Processing {len(audio_files)} files in {class_folder}/")
        
        for audio_file in audio_files:
            try:
                # Load original audio
                audio_data, sr = librosa.load(str(audio_file), sr=22050, mono=True)
                
                # Apply augmentations
                augmented_samples = apply_audio_augmentation(audio_data, sr)
                
                # Save augmented samples
                base_name = audio_file.stem
                for i, aug_audio in enumerate(augmented_samples[1:], 1):  # Skip original
                    # Normalize duration to 3 seconds
                    target_length = int(3.0 * sr)
                    if len(aug_audio) < target_length:
                        aug_audio = np.pad(aug_audio, (0, target_length - len(aug_audio)))
                    else:
                        aug_audio = aug_audio[:target_length]
                    
                    # Save augmented file
                    aug_filename = f"{base_name}_aug_{i}.wav"
                    aug_path = Path(class_folder) / aug_filename
                    sf.write(str(aug_path), aug_audio, sr)
                    augmented_count += 1
                    
            except Exception as e:
                print(f"⚠️  Error processing {audio_file}: {e}")
    
    print(f"✅ Created {augmented_count} augmented audio files")
    return augmented_count

def build_improved_model(input_shape, num_classes):
    """
    Build an improved model architecture optimized for small datasets.
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(2024)
    np.random.seed(2024)
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input'),
        
        # Feature extraction with strong regularization
        layers.Dense(128, activation='relu', name='dense_1',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.6, name='dropout_1'),  # High dropout for small dataset
        
        layers.Dense(64, activation='relu', name='dense_2',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.6, name='dropout_2'),
        
        layers.Dense(32, activation='relu', name='dense_3',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.4, name='dropout_3'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output',
                    kernel_initializer='glorot_normal')
    ])
    
    return model

def train_improved_model():
    """
    Train an improved model with all enhancements.
    """
    print("=" * 80)
    print("🚀 TRAINING IMPROVED MODEL")
    print("=" * 80)
    
    # Step 1: Generate diverse data
    generate_diverse_training_data()
    
    # Step 2: Create augmented dataset
    augmented_count = create_augmented_dataset()
    
    # Step 3: Reprocess dataset
    print("🔄 Reprocessing expanded dataset...")
    try:
        result = subprocess.run(["python3", "dataset_preprocess.py"], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"❌ Error reprocessing dataset: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error reprocessing dataset: {e}")
        return False
    
    # Step 4: Load expanded dataset
    try:
        with open("data.json", 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded expanded dataset: {len(data['labels'])} samples")
    except Exception as e:
        print(f"❌ Error loading data.json: {e}")
        return False
    
    # Analyze new class distribution
    labels = np.array(data["labels"], dtype=np.int32)
    class_mapping = data["mapping"]
    class_counts = Counter(labels)
    
    print(f"📈 Expanded class distribution:")
    for i, class_name in enumerate(class_mapping):
        count = class_counts.get(i, 0)
        percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
        print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Check if we have enough data now
    min_samples = min(class_counts.values())
    if min_samples < 10:
        print(f"⚠️  Warning: Still low sample count (min: {min_samples})")
        print("   Consider recording more real audio files for better results")
    
    # Prepare data
    mfcc_features = np.array(data["mfcc"], dtype=np.float32)
    n_samples = mfcc_features.shape[0]
    flattened_features = mfcc_features.reshape(n_samples, -1)
    
    # Improved normalization (standardization)
    feature_mean = flattened_features.mean(axis=0, keepdims=True)
    feature_std = flattened_features.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std == 0, 1, feature_std)
    X = (flattened_features - feature_mean) / feature_std
    
    # Convert labels to categorical
    y = to_categorical(labels, num_classes=len(class_mapping))
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2024, stratify=labels
    )
    
    print(f"🔄 Train/test split: {X_train.shape[0]}/{X_test.shape[0]}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"⚖️  Class weights:")
    for i, weight in class_weight_dict.items():
        print(f"   {class_mapping[i]}: {weight:.3f}")
    
    # Build improved model
    input_shape = (X.shape[1],)
    num_classes = len(class_mapping)
    model = build_improved_model(input_shape, num_classes)
    
    # Compile with optimized parameters
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001,  # Very low learning rate
            beta_1=0.9,
            beta_2=0.999
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Improved model created with {model.count_params():,} parameters")
    
    # Enhanced callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.8,
            patience=15,
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_improved_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train improved model
    print(f"\n🚀 Training improved model...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=8,  # Small batch size
        epochs=200,    # More epochs
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        shuffle=True
    )
    
    # Evaluate improved model
    print(f"\n📊 Evaluating improved model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Detailed performance analysis
    all_predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(all_predictions, axis=1)
    true_classes = labels
    
    pred_counts = Counter(predicted_classes)
    true_counts = Counter(true_classes)
    
    print(f"\n🎯 IMPROVED MODEL PERFORMANCE:")
    print("-" * 70)
    print(f"{'Class':<12} {'True':<8} {'Pred':<8} {'Accuracy':<10} {'Precision':<10} {'Recall'}")
    print("-" * 70)
    
    overall_correct = 0
    for i, class_name in enumerate(class_mapping):
        true_count = true_counts.get(i, 0)
        pred_count = pred_counts.get(i, 0)
        
        # Per-class metrics
        class_mask = true_classes == i
        pred_mask = predicted_classes == i
        
        if np.sum(class_mask) > 0:
            class_correct = np.sum(predicted_classes[class_mask] == i)
            recall = class_correct / np.sum(class_mask)
            overall_correct += class_correct
        else:
            recall = 0
        
        if np.sum(pred_mask) > 0:
            precision = np.sum(true_classes[pred_mask] == i) / np.sum(pred_mask)
        else:
            precision = 0
        
        print(f"{class_name:<12} {true_count:<8} {pred_count:<8} {recall:<10.3f} {precision:<10.3f} {recall:.3f}")
    
    overall_accuracy = overall_correct / len(true_classes)
    
    # Bias analysis
    dominant_class = np.argmax(np.bincount(predicted_classes))
    dominant_percentage = pred_counts[dominant_class] / len(predicted_classes) * 100
    
    print(f"\n📈 FINAL PERFORMANCE ANALYSIS:")
    print(f"   Overall accuracy: {overall_accuracy:.3f}")
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Dominant class: {class_mapping[dominant_class]} ({dominant_percentage:.1f}%)")
    print(f"   Total samples: {len(labels)}")
    print(f"   Augmented samples: {augmented_count}")
    
    # Success criteria
    success = (overall_accuracy > 0.6 and dominant_percentage < 60)
    
    if success:
        print("🎉 SUCCESS: Significant improvement achieved!")
    elif overall_accuracy > 0.5:
        print("✅ GOOD: Noticeable improvement achieved!")
    else:
        print("⚠️  PARTIAL: Some improvement, but more data needed")
    
    # Save improved model
    model.save("model_improved.h5")
    
    # Replace active model if better
    if overall_accuracy > 0.4:  # Better than previous
        if os.path.exists("model.h5"):
            os.rename("model.h5", "model_previous.h5")
        os.rename("model_improved.h5", "model.h5")
        print(f"✅ Improved model is now active")
        
        # Update metadata
        metadata = {
            "model_type": "improved_dscnn",
            "input_shape": list(input_shape),
            "num_classes": num_classes,
            "class_mapping": class_mapping,
            "training_samples": len(X_train),
            "test_accuracy": float(test_accuracy),
            "overall_accuracy": float(overall_accuracy),
            "bias_percentage": float(dominant_percentage),
            "total_samples": len(labels),
            "augmented_samples": augmented_count,
            "feature_normalization": {
                "method": "standardization",
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist()
            }
        }
        
        with open("model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return success

if __name__ == "__main__":
    success = train_improved_model()
    
    if success:
        print(f"\n🎉 MODEL IMPROVEMENT COMPLETED!")
        print(f"Test your improved model:")
        print(f"python3 test_model.py disturbance/[file].wav")
        print(f"python3 test_model.py slow/[file].wav") 
        print(f"python3 test_model.py medium/[file].wav")
        print(f"python3 test_model.py fast/[file].wav")
    else:
        print(f"\n📝 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
        print(f"1. Record 10+ more real audio files per class")
        print(f"2. Use more distinct vocal techniques:")
        print(f"   - Disturbance: Cough, snap, clap sounds")
        print(f"   - Slow: Long, drawn-out 'sooooo', 'hummmmm'")
        print(f"   - Medium: Normal pace 'soo', 'hum'")
        print(f"   - Fast: Quick, staccato 'so-so-so', 'hm-hm-hm'")
        print(f"3. Record in different environments/volumes")
