#!/usr/bin/env python3
"""
Model Training Diagnosis Script

This script analyzes your trained model to identify why it's biased toward one class.
It checks:
- Model predictions on training data
- Class-wise prediction patterns
- Model confidence distributions
- Training vs test performance
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from collections import Counter

def diagnose_model_bias():
    """
    Analyze the trained model for bias issues.
    """
    print("=" * 80)
    print("🤖 MODEL BIAS DIAGNOSIS")
    print("=" * 80)
    
    # Load data
    try:
        with open("data.json", 'r') as f:
            data = json.load(f)
        print("✅ Loaded data.json")
    except Exception as e:
        print(f"❌ Error loading data.json: {e}")
        return
    
    # Load model
    try:
        model = keras.models.load_model("model.h5")
        print("✅ Loaded model.h5")
    except Exception as e:
        print(f"❌ Error loading model.h5: {e}")
        return
    
    # Prepare data (same as training)
    mfcc_features = np.array(data["mfcc"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    class_mapping = data["mapping"]
    
    # Flatten and normalize features (same as training)
    n_samples = mfcc_features.shape[0]
    flattened_features = mfcc_features.reshape(n_samples, -1)
    
    # Normalize features
    feature_min = flattened_features.min(axis=0, keepdims=True)
    feature_max = flattened_features.max(axis=0, keepdims=True)
    feature_range = feature_max - feature_min
    feature_range = np.where(feature_range == 0, 1, feature_range)
    normalized_features = (flattened_features - feature_min) / feature_range
    
    # Convert labels to categorical
    y_categorical = to_categorical(labels, num_classes=len(class_mapping))
    
    print(f"\n📊 DATA PREPARATION:")
    print(f"   Original MFCC shape: {mfcc_features.shape}")
    print(f"   Flattened shape: {flattened_features.shape}")
    print(f"   Normalized range: [{normalized_features.min():.3f}, {normalized_features.max():.3f}]")
    
    # 1. ANALYZE PREDICTIONS ON ALL DATA
    print(f"\n🎯 PREDICTION ANALYSIS ON ALL DATA:")
    print("-" * 50)
    
    predictions = model.predict(normalized_features, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = labels
    
    # Count predictions per class
    pred_counts = Counter(predicted_classes)
    true_counts = Counter(true_classes)
    
    print(f"{'Class':<12} {'True Count':<12} {'Pred Count':<12} {'Accuracy':<10}")
    print("-" * 60)
    
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
        
        print(f"{class_name:<12} {true_count:<12} {pred_count:<12} {class_accuracy:<10.3f}")
    
    overall_accuracy = overall_correct / len(true_classes)
    print(f"\nOverall Accuracy: {overall_accuracy:.3f}")
    
    # 2. ANALYZE PREDICTION CONFIDENCE
    print(f"\n📈 CONFIDENCE ANALYSIS:")
    print("-" * 50)
    
    max_confidences = np.max(predictions, axis=1)
    
    print(f"Confidence Statistics:")
    print(f"   Mean: {np.mean(max_confidences):.3f}")
    print(f"   Std:  {np.std(max_confidences):.3f}")
    print(f"   Min:  {np.min(max_confidences):.3f}")
    print(f"   Max:  {np.max(max_confidences):.3f}")
    
    # Check if model is always confident about one class
    dominant_class = np.argmax(np.bincount(predicted_classes))
    dominant_class_name = class_mapping[dominant_class]
    dominant_percentage = pred_counts[dominant_class] / len(predicted_classes) * 100
    
    print(f"\n🎯 BIAS ANALYSIS:")
    print(f"   Dominant predicted class: {dominant_class_name} ({dominant_class})")
    print(f"   Dominance percentage: {dominant_percentage:.1f}%")
    
    if dominant_percentage > 70:
        print("🚨 CRITICAL: Model is heavily biased toward one class!")
    elif dominant_percentage > 50:
        print("⚠️  WARNING: Model shows bias toward one class")
    else:
        print("✅ Model predictions are reasonably distributed")
    
    # 3. ANALYZE INDIVIDUAL PREDICTIONS
    print(f"\n📋 DETAILED PREDICTION ANALYSIS:")
    print("-" * 70)
    print(f"{'File':<30} {'True':<12} {'Pred':<12} {'Confidence':<12} {'Correct'}")
    print("-" * 70)
    
    files = data.get("files", [])
    for i in range(min(len(files), 15)):  # Show first 15 files
        file_name = files[i].split('/')[-1][:28]  # Get filename, truncate
        true_class = class_mapping[true_classes[i]]
        pred_class = class_mapping[predicted_classes[i]]
        confidence = max_confidences[i]
        correct = "✅" if true_classes[i] == predicted_classes[i] else "❌"
        
        print(f"{file_name:<30} {true_class:<12} {pred_class:<12} {confidence:<12.3f} {correct}")
    
    if len(files) > 15:
        print(f"... and {len(files) - 15} more files")
    
    # 4. ANALYZE MODEL WEIGHTS
    print(f"\n⚖️  MODEL WEIGHT ANALYSIS:")
    print("-" * 50)
    
    # Get output layer weights
    output_layer = model.layers[-1]
    if hasattr(output_layer, 'get_weights'):
        weights, biases = output_layer.get_weights()
        
        print(f"Output layer biases:")
        for i, (class_name, bias) in enumerate(zip(class_mapping, biases)):
            print(f"   {class_name}: {bias:.6f}")
        
        # Check for bias in the biases (pun intended)
        bias_range = np.max(biases) - np.min(biases)
        if bias_range > 1.0:
            print(f"\n⚠️  WARNING: Large bias range ({bias_range:.3f})")
            print("   This could cause prediction bias")
            
            # Find most biased class
            max_bias_idx = np.argmax(biases)
            print(f"   Most biased toward: {class_mapping[max_bias_idx]} ({biases[max_bias_idx]:.3f})")
    
    # 5. RECOMMENDATIONS
    print(f"\n💡 DIAGNOSIS SUMMARY:")
    print("=" * 60)
    
    if dominant_percentage > 80:
        print("🚨 CRITICAL ISSUE: Model always predicts the same class")
        print("   LIKELY CAUSES:")
        print("   1. Model didn't learn properly (stuck in local minimum)")
        print("   2. Features are too similar between classes")
        print("   3. Learning rate too high or too low")
        print("   SOLUTIONS:")
        print("   1. Retrain with different random seed")
        print("   2. Adjust learning rate")
        print("   3. Add more diverse training data")
        
    elif dominant_percentage > 60:
        print("⚠️  MODERATE ISSUE: Model is biased toward one class")
        print("   SOLUTIONS:")
        print("   1. Use class weights during training")
        print("   2. Balance the dataset")
        print("   3. Adjust decision threshold")
        
    else:
        print("✅ Model predictions appear reasonably distributed")
        print("   Issue might be in specific test cases or data quality")
    
    return {
        'dominant_class': int(dominant_class),
        'dominant_percentage': float(dominant_percentage),
        'overall_accuracy': float(overall_accuracy),
        'prediction_counts': dict(pred_counts),
        'confidence_stats': {
            'mean': float(np.mean(max_confidences)),
            'std': float(np.std(max_confidences)),
            'min': float(np.min(max_confidences)),
            'max': float(np.max(max_confidences))
        }
    }

if __name__ == "__main__":
    results = diagnose_model_bias()
    
    if results:
        with open('model_diagnosis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Model diagnosis saved to: model_diagnosis.json")
