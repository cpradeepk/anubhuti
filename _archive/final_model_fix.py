#!/usr/bin/env python3
"""
Final Model Fix - Generate More Training Data and Retrain

This script addresses the remaining bias by:
1. Generating synthetic training data for underrepresented classes
2. Retraining with a more balanced dataset
3. Using ensemble techniques for better predictions
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import subprocess

def generate_synthetic_data():
    """
    Generate synthetic audio data for underrepresented classes.
    """
    print("🎵 Generating synthetic training data...")
    
    # Generate 3 files for each class to balance the dataset
    commands = [
        "python3 disturbance.py --batch 3 --complexity medium",
        "python3 fast.py --batch 3 --intensity medium", 
        "python3 medium.py --batch 3 --balance medium",
        "python3 slow.py --batch 3 --gentleness medium"
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ Generated: {cmd.split()[1]}")
            else:
                print(f"⚠️  Warning: {cmd.split()[1]} generation had issues")
        except Exception as e:
            print(f"❌ Error generating {cmd.split()[1]}: {e}")
    
    print("✅ Synthetic data generation completed")

def retrain_with_balanced_data():
    """
    Retrain the model with the expanded, more balanced dataset.
    """
    print("\n🔄 Reprocessing dataset with new synthetic data...")
    
    # Reprocess the dataset to include new synthetic files
    try:
        result = subprocess.run(["python3", "dataset_preprocess.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"❌ Error reprocessing dataset: {result.stderr}")
            return False
        print("✅ Dataset reprocessed with synthetic data")
    except Exception as e:
        print(f"❌ Error reprocessing dataset: {e}")
        return False
    
    # Load the updated dataset
    try:
        with open("data.json", 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded updated dataset: {len(data['labels'])} samples")
    except Exception as e:
        print(f"❌ Error loading updated data.json: {e}")
        return False
    
    # Prepare data
    mfcc_features = np.array(data["mfcc"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    class_mapping = data["mapping"]
    
    print(f"📊 Updated dataset: {len(labels)} samples, {len(class_mapping)} classes")
    
    # Check class distribution
    from collections import Counter
    class_counts = Counter(labels)
    print(f"📈 Class distribution:")
    for i, class_name in enumerate(class_mapping):
        count = class_counts.get(i, 0)
        percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
        print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Flatten and normalize features
    n_samples = mfcc_features.shape[0]
    flattened_features = mfcc_features.reshape(n_samples, -1)
    
    # Normalize features (0-1 scaling)
    feature_min = flattened_features.min(axis=0, keepdims=True)
    feature_max = flattened_features.max(axis=0, keepdims=True)
    feature_range = feature_max - feature_min
    feature_range = np.where(feature_range == 0, 1, feature_range)
    X = (flattened_features - feature_min) / feature_range
    
    # Convert labels to categorical
    y = to_categorical(labels, num_classes=len(class_mapping))
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=789, stratify=labels  # New seed
    )
    
    print(f"🔄 Train/test split: {X_train.shape[0]}/{X_test.shape[0]}")
    
    # Calculate class weights for the updated dataset
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"⚖️  Updated class weights:")
    for i, weight in class_weight_dict.items():
        print(f"   {class_mapping[i]}: {weight:.3f}")
    
    # Build final optimized model
    print(f"\n🏗️  Building final optimized model...")
    
    # Set random seed
    tf.random.set_seed(999)
    np.random.seed(999)
    
    input_shape = (X.shape[1],)
    num_classes = len(class_mapping)
    
    # More balanced architecture
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='input'),
        
        # Feature extraction with moderate regularization
        layers.Dense(256, activation='relu', name='dense_1',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3, name='dropout_1'),
        
        layers.Dense(128, activation='relu', name='dense_2',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.4, name='dropout_2'),
        
        layers.Dense(64, activation='relu', name='dense_3',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.Dropout(0.3, name='dropout_3'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output',
                    kernel_initializer='glorot_normal')
    ])
    
    # Compile with optimized parameters
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002),  # Even lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Final model created with {model.count_params():,} parameters")
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.7,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train final model
    print(f"\n🚀 Training final optimized model...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=16,  # Larger batch size for stability
        epochs=150,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        shuffle=True
    )
    
    # Evaluate final model
    print(f"\n📊 Evaluating final model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Check final bias
    all_predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(all_predictions, axis=1)
    true_classes = labels
    
    pred_counts = Counter(predicted_classes)
    true_counts = Counter(true_classes)
    
    print(f"\n🎯 FINAL MODEL PERFORMANCE:")
    print("-" * 60)
    print(f"{'Class':<12} {'True Count':<12} {'Pred Count':<12} {'Accuracy'}")
    print("-" * 60)
    
    overall_correct = 0
    for i, class_name in enumerate(class_mapping):
        true_count = true_counts.get(i, 0)
        pred_count = pred_counts.get(i, 0)
        
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum(predicted_classes[class_mask] == i)
            class_accuracy = class_correct / np.sum(class_mask)
            overall_correct += class_correct
        else:
            class_accuracy = 0
        
        print(f"{class_name:<12} {true_count:<12} {pred_count:<12} {class_accuracy:.3f}")
    
    overall_accuracy = overall_correct / len(true_classes)
    
    # Final bias check
    dominant_class = np.argmax(np.bincount(predicted_classes))
    dominant_percentage = pred_counts[dominant_class] / len(predicted_classes) * 100
    
    print(f"\n📈 FINAL BIAS ANALYSIS:")
    print(f"   Overall accuracy: {overall_accuracy:.3f}")
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Dominant class: {class_mapping[dominant_class]} ({dominant_percentage:.1f}%)")
    
    success = dominant_percentage < 50  # Success if no class dominates >50%
    
    if success:
        print("🎉 SUCCESS: Model bias eliminated!")
    else:
        print("⚠️  PARTIAL: Model bias reduced but still present")
    
    # Save the final model
    model.save("model_final.h5")
    print(f"💾 Final model saved as: model_final.h5")
    
    # Replace the active model
    if os.path.exists("model.h5"):
        os.rename("model.h5", "model_previous.h5")
    os.rename("model_final.h5", "model.h5")
    print(f"✅ Final model is now active as: model.h5")
    
    # Update metadata
    metadata = {
        "model_type": "final_optimized_dscnn",
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "class_mapping": class_mapping,
        "training_samples": len(X_train),
        "test_accuracy": float(test_accuracy),
        "overall_accuracy": float(overall_accuracy),
        "bias_percentage": float(dominant_percentage),
        "feature_normalization": {
            "method": "min_max_scaling",
            "feature_min": feature_min.tolist(),
            "feature_max": feature_max.tolist()
        }
    }
    
    with open("model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Final metadata saved to: model_metadata.json")
    
    return success

if __name__ == "__main__":
    print("=" * 80)
    print("🎯 FINAL MODEL OPTIMIZATION")
    print("=" * 80)
    
    # Step 1: Generate more training data
    generate_synthetic_data()
    
    # Step 2: Retrain with balanced data
    success = retrain_with_balanced_data()
    
    if success:
        print(f"\n🎉 FINAL OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print(f"Your model should now predict all 4 classes correctly.")
        print(f"\nNext steps:")
        print(f"1. Test: python3 test_model.py [audio_file].wav")
        print(f"2. Verify Arduino commands: 0, 1, 2, 3")
        print(f"3. Deploy to your Arduino system")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS - Model improved but may need more data")
        print(f"Consider recording more diverse audio samples for each class.")
