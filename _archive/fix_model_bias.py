#!/usr/bin/env python3
"""
Quick Model Bias Fix Script

This script implements multiple strategies to fix the model bias issue:
1. Retrain with different random seeds
2. Use class weights
3. Improved model architecture
4. Better training parameters

Run this to quickly fix the bias toward class 1 (slow).
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

def fix_model_bias():
    """
    Retrain the model with bias-fixing strategies.
    """
    print("=" * 80)
    print("🔧 FIXING MODEL BIAS - QUICK RETRAIN")
    print("=" * 80)
    
    # Load data
    try:
        with open("data.json", 'r') as f:
            data = json.load(f)
        print("✅ Loaded data.json")
    except Exception as e:
        print(f"❌ Error loading data.json: {e}")
        return False
    
    # Prepare data
    mfcc_features = np.array(data["mfcc"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    class_mapping = data["mapping"]
    
    print(f"📊 Dataset: {len(labels)} samples, {len(class_mapping)} classes")
    
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
    
    print(f"📐 Features shape: {X.shape}")
    print(f"🏷️  Labels shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=labels  # Different seed
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
    print(f"\n🏗️  Building improved model...")
    
    # Set random seed for reproducible results
    tf.random.set_seed(456)  # Different seed from original
    np.random.seed(456)
    
    input_shape = (X.shape[1],)
    num_classes = len(class_mapping)
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input'),
        
        # Smaller, more regularized architecture for small dataset
        layers.Dense(128, activation='relu', name='dense_1',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.02)),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.5, name='dropout_1'),
        
        layers.Dense(64, activation='relu', name='dense_2',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.02)),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.5, name='dropout_2'),
        
        layers.Dense(32, activation='relu', name='dense_3',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.02)),
        layers.Dropout(0.3, name='dropout_3'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output',
                    kernel_initializer='glorot_normal')
    ])
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🚀 Training model with bias fixes...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=8,  # Smaller batch size for small dataset
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Use class weights
        verbose=1,
        shuffle=True
    )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions on all data to check bias
    all_predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(all_predictions, axis=1)
    true_classes = labels
    
    # Analyze predictions
    from collections import Counter
    pred_counts = Counter(predicted_classes)
    true_counts = Counter(true_classes)
    
    print(f"\n🎯 BIAS CHECK RESULTS:")
    print("-" * 50)
    print(f"{'Class':<12} {'True Count':<12} {'Pred Count':<12} {'Accuracy'}")
    print("-" * 50)
    
    overall_correct = 0
    for i, class_name in enumerate(class_mapping):
        true_count = true_counts.get(i, 0)
        pred_count = pred_counts.get(i, 0)
        
        # Calculate per-class accuracy
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum(predicted_classes[class_mask] == i)
            class_accuracy = class_correct / np.sum(class_mask)
            overall_correct += class_correct
        else:
            class_accuracy = 0
        
        print(f"{class_name:<12} {true_count:<12} {pred_count:<12} {class_accuracy:.3f}")
    
    overall_accuracy = overall_correct / len(true_classes)
    
    # Check if bias is fixed
    dominant_class = np.argmax(np.bincount(predicted_classes))
    dominant_percentage = pred_counts[dominant_class] / len(predicted_classes) * 100
    
    print(f"\n📈 BIAS ANALYSIS:")
    print(f"   Overall accuracy: {overall_accuracy:.3f}")
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Dominant class: {class_mapping[dominant_class]} ({dominant_percentage:.1f}%)")
    
    if dominant_percentage < 60:
        print("✅ SUCCESS: Model bias appears to be fixed!")
        bias_fixed = True
    elif dominant_percentage < 80:
        print("⚠️  PARTIAL: Model bias reduced but still present")
        bias_fixed = False
    else:
        print("❌ FAILED: Model still heavily biased")
        bias_fixed = False
    
    if bias_fixed or dominant_percentage < 80:
        # Save the improved model
        model.save("model_fixed.h5")
        print(f"💾 Improved model saved as: model_fixed.h5")
        
        # Replace the original model
        if os.path.exists("model.h5"):
            os.rename("model.h5", "model_backup.h5")
            print(f"📄 Original model backed up as: model_backup.h5")
        
        os.rename("model_fixed.h5", "model.h5")
        print(f"✅ New model is now active as: model.h5")
        
        # Update metadata
        metadata = {
            "model_type": "bias_fixed_dscnn",
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
        
        print(f"📄 Updated metadata saved to: model_metadata.json")
        
    return bias_fixed

if __name__ == "__main__":
    success = fix_model_bias()
    
    if success:
        print(f"\n🎉 MODEL BIAS FIX COMPLETED SUCCESSFULLY!")
        print(f"Next steps:")
        print(f"1. Test the fixed model: python3 test_model.py [audio_file].wav")
        print(f"2. Verify different classes are predicted correctly")
        print(f"3. Check Arduino commands are now 0, 1, 2, 3 (not just 1)")
    else:
        print(f"\n⚠️  MODEL BIAS PARTIALLY FIXED")
        print(f"Consider:")
        print(f"1. Adding more diverse training data")
        print(f"2. Running this script again with different parameters")
        print(f"3. Checking if audio files are actually different enough")
