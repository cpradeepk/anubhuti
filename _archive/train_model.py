#!/usr/bin/env python3
"""
Audio Classification Model Training Script

This script trains a DS-CNN model for sound-to-vibration classification using
preprocessed MFCC features. It handles:
- Loading preprocessed data from data.json
- Train/test split (80/20) with stratification
- Building DS-CNN architecture with Dense layers
- Model compilation with categorical crossentropy loss
- Training with validation split and early stopping
- Model saving as model.h5
- Training metrics display and evaluation

Author: Audio Classification System
Date: 2025-08-30
"""

import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import warnings
import traceback
from pathlib import Path

# Suppress TensorFlow and other warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# Configuration constants - CRITICAL: Must match preprocessing script
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10  # for early stopping
MIN_DELTA = 0.001  # minimum change to qualify as improvement

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Mixed precision disabled to avoid compatibility issues with top_k_categorical_accuracy
# Using float32 for stable training
print("ℹ️  Using float32 precision for stable training")

def load_dataset(data_file="data.json"):
    """
    Load preprocessed dataset from JSON file with comprehensive validation.

    Args:
        data_file (str): Path to the data.json file

    Returns:
        tuple: (X, y, class_mapping, file_paths) or (None, None, None, None) if failed
    """
    try:
        print(f"Loading dataset from {data_file}...")

        # Check if file exists
        if not os.path.exists(data_file):
            print(f"❌ Error: {data_file} not found.")
            print("Please run dataset_preprocess.py first to create the dataset.")
            return None, None, None, None

        # Check file size
        file_size = os.path.getsize(data_file)
        if file_size == 0:
            print(f"❌ Error: {data_file} is empty.")
            return None, None, None, None

        print(f"📄 Dataset file size: {file_size / 1024 / 1024:.2f} MB")

        # Load JSON data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required keys
        required_keys = ["mfcc", "labels", "mapping", "files"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"❌ Error: Missing required keys in dataset: {missing_keys}")
            return None, None, None, None

        # Extract and validate data components
        try:
            mfcc_features = np.array(data["mfcc"], dtype=np.float32)
            labels = np.array(data["labels"], dtype=np.int32)
            class_mapping = data["mapping"]
            file_paths = data["files"]
        except Exception as e:
            print(f"❌ Error converting data to numpy arrays: {e}")
            return None, None, None, None

        # Validate data consistency
        if len(mfcc_features) != len(labels) or len(labels) != len(file_paths):
            print(f"❌ Error: Data length mismatch:")
            print(f"  MFCC features: {len(mfcc_features)}")
            print(f"  Labels: {len(labels)}")
            print(f"  File paths: {len(file_paths)}")
            return None, None, None, None

        # Validate MFCC features
        if len(mfcc_features) == 0:
            print("❌ Error: No MFCC features found in dataset")
            return None, None, None, None

        # Check for invalid values
        if np.any(np.isnan(mfcc_features)) or np.any(np.isinf(mfcc_features)):
            print("❌ Error: Dataset contains NaN or infinite values")
            return None, None, None, None

        # Validate labels
        unique_labels = np.unique(labels)
        expected_labels = np.arange(len(class_mapping))
        if not np.array_equal(np.sort(unique_labels), expected_labels):
            print(f"❌ Error: Label mismatch. Expected: {expected_labels}, Found: {unique_labels}")
            return None, None, None, None

        print(f"✅ Dataset loaded successfully:")
        print(f"  📊 Samples: {len(mfcc_features)}")
        print(f"  🏷️  Classes: {len(class_mapping)}")
        print(f"  📐 MFCC shape: {mfcc_features.shape}")
        print(f"  🎯 Class mapping: {class_mapping}")

        # Print detailed class distribution
        print(f"\n📈 Class Distribution:")
        total_samples = len(labels)
        for i, class_name in enumerate(class_mapping):
            count = np.sum(labels == i)
            percentage = (count / total_samples) * 100
            print(f"  {class_name:12}: {count:4d} samples ({percentage:5.1f}%)")

        # Check for class imbalance
        class_counts = [np.sum(labels == i) for i in range(len(class_mapping))]
        min_count = min(class_counts)
        max_count = max(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 3:
            print(f"⚠️  Warning: Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print("   Consider balancing your dataset for better performance")
        elif min_count < 10:
            print(f"⚠️  Warning: Some classes have very few samples (min: {min_count})")
            print("   Consider adding more data for better performance")
        else:
            print("✅ Dataset appears well-balanced")

        return mfcc_features, labels, class_mapping, file_paths

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in {data_file}: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Unexpected error loading dataset: {e}")
        traceback.print_exc()
        return None, None, None, None

def prepare_features(mfcc_features):
    """
    Prepare MFCC features for training by flattening and normalizing.
    CRITICAL: This preprocessing must be identical in test_model.py

    Args:
        mfcc_features (np.array): MFCC features with shape (samples, n_mfcc, time_frames)

    Returns:
        tuple: (normalized_features, normalization_params) for consistent preprocessing
    """
    print("🔄 Preparing features for training...")

    # Validate input shape
    if len(mfcc_features.shape) != 3:
        print(f"❌ Error: Expected 3D MFCC features, got shape {mfcc_features.shape}")
        return None, None

    n_samples, n_mfcc, time_frames = mfcc_features.shape
    print(f"📐 Input shape: {n_samples} samples × {n_mfcc} MFCC × {time_frames} time frames")

    # Flatten MFCC features for Dense layers
    # Shape: (samples, n_mfcc, time_frames) -> (samples, n_mfcc * time_frames)
    flattened_features = mfcc_features.reshape(n_samples, -1)
    print(f"📐 Flattened shape: {flattened_features.shape}")

    # Check for invalid values before normalization
    if np.any(np.isnan(flattened_features)) or np.any(np.isinf(flattened_features)):
        print("❌ Error: Features contain NaN or infinite values")
        return None, None

    # Calculate normalization parameters
    feature_min = flattened_features.min(axis=0, keepdims=True)
    feature_max = flattened_features.max(axis=0, keepdims=True)

    # Avoid division by zero
    feature_range = feature_max - feature_min
    feature_range = np.where(feature_range == 0, 1, feature_range)

    # Normalize features to [0, 1] range
    normalized_features = (flattened_features - feature_min) / feature_range

    # Store normalization parameters for later use in testing
    normalization_params = {
        'min': feature_min.flatten().tolist(),
        'max': feature_max.flatten().tolist(),
        'input_shape': mfcc_features.shape[1:],  # (n_mfcc, time_frames)
        'flattened_shape': flattened_features.shape[1]
    }

    print(f"✅ Features normalized:")
    print(f"   Min: {normalized_features.min():.6f}")
    print(f"   Max: {normalized_features.max():.6f}")
    print(f"   Mean: {normalized_features.mean():.6f}")
    print(f"   Std: {normalized_features.std():.6f}")

    # Validate normalization
    if np.any(np.isnan(normalized_features)) or np.any(np.isinf(normalized_features)):
        print("❌ Error: Normalization produced NaN or infinite values")
        return None, None

    return normalized_features.astype(np.float32), normalization_params

def create_model(input_shape, num_classes):
    """
    Create DS-CNN model architecture with Dense layers.
    Architecture: Dense(512) → BN → Dropout(0.3) → Dense(256) → BN → Dropout(0.4) →
                 Dense(128) → BN → Dropout(0.5) → Dense(num_classes, softmax)

    Args:
        input_shape (tuple): Shape of input features (flattened MFCC)
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled model ready for training
    """
    print(f"🏗️  Creating DS-CNN model:")
    print(f"   Input shape: {input_shape}")
    print(f"   Output classes: {num_classes}")

    try:
        # Build improved model architecture for small dataset
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape, name='input'),

            # First Dense block - Feature extraction (smaller for small dataset)
            layers.Dense(256, activation='relu', name='dense_1',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.01)),  # Higher regularization
            layers.BatchNormalization(name='bn_1'),
            layers.Dropout(0.4, name='dropout_1'),  # Higher dropout

            # Second Dense block - Feature refinement
            layers.Dense(128, activation='relu', name='dense_2',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(name='bn_2'),
            layers.Dropout(0.5, name='dropout_2'),

            # Third Dense block - Classification preparation
            layers.Dense(64, activation='relu', name='dense_3',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(name='bn_3'),
            layers.Dropout(0.5, name='dropout_3'),

            # Output layer - Classification
            layers.Dense(num_classes, activation='softmax', name='output',
                        kernel_initializer='glorot_normal')
        ])

        # Compile model with appropriate metrics and class weights
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0005,  # Reduced learning rate for better convergence
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        # Print model summary
        print("\n🏗️  Model Architecture:")
        print("=" * 80)
        model.summary()
        print("=" * 80)

        # Calculate and display model parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        print(f"📊 Model Parameters:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {non_trainable_params:,}")

        return model

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        traceback.print_exc()
        return None

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot and save comprehensive training history with multiple metrics.

    Args:
        history: Keras training history object
        save_path (str): Path to save the plot
    """
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History - Audio Classification Model', fontsize=16, fontweight='bold')

        # Plot 1: Accuracy
        ax1 = axes[0, 0]
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Plot 2: Loss
        ax2 = axes[0, 1]
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Precision and Recall (if available)
        ax3 = axes[1, 0]
        if 'precision' in history.history and 'val_precision' in history.history:
            ax3.plot(history.history['precision'], label='Training Precision', linewidth=2, color='green')
            ax3.plot(history.history['val_precision'], label='Validation Precision', linewidth=2, color='orange')
            ax3.plot(history.history['recall'], label='Training Recall', linewidth=2, color='purple', linestyle='--')
            ax3.plot(history.history['val_recall'], label='Validation Recall', linewidth=2, color='brown', linestyle='--')
            ax3.set_title('Precision & Recall', fontweight='bold')
            ax3.set_ylabel('Score')
        else:
            # Fallback to top-k accuracy if precision/recall not available
            if 'top_k_categorical_accuracy' in history.history:
                ax3.plot(history.history['top_k_categorical_accuracy'], label='Training Top-K Acc', linewidth=2, color='green')
                ax3.plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-K Acc', linewidth=2, color='orange')
                ax3.set_title('Top-K Categorical Accuracy', fontweight='bold')
                ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # Plot 4: Learning Rate (if available)
        ax4 = axes[1, 1]
        if 'lr' in history.history:
            ax4.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='red')
            ax4.set_title('Learning Rate Schedule', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Show training summary instead
            ax4.text(0.1, 0.8, 'Training Summary:', fontsize=12, fontweight='bold', transform=ax4.transAxes)

            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            total_epochs = len(history.history['accuracy'])

            summary_text = f"""
Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}
Total Epochs: {total_epochs}

Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}
Epoch of Best Val Acc: {np.argmax(history.history['val_accuracy']) + 1}
            """

            ax4.text(0.1, 0.6, summary_text, fontsize=10, transform=ax4.transAxes,
                    verticalalignment='top', fontfamily='monospace')
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
            ax4.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to free memory

        print(f"📊 Training history plot saved to {save_path}")

        # Also save a simple version for quick viewing
        simple_path = save_path.replace('.png', '_simple.png')
        fig_simple, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['loss'], label='Training', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(simple_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"📊 Simple training plot saved to {simple_path}")

    except Exception as e:
        print(f"⚠️  Warning: Could not save training plot: {e}")
        traceback.print_exc()

def evaluate_model(model, X_test, y_test, class_mapping):
    """
    Comprehensive model evaluation on test data with detailed metrics.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels (categorical)
        class_mapping: List of class names

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print("\n🔍 Evaluating model on test data...")

    try:
        # Get model predictions
        print("Making predictions on test set...")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate basic metrics
        test_results = model.evaluate(X_test, y_test, verbose=0)

        # Extract metrics (handle different numbers of metrics)
        metrics_names = model.metrics_names
        results_dict = dict(zip(metrics_names, test_results))

        print(f"\n📊 Test Results:")
        print("=" * 50)
        for metric_name, value in results_dict.items():
            if 'accuracy' in metric_name.lower():
                print(f"  {metric_name:25}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"  {metric_name:25}: {value:.4f}")

        # Detailed per-class analysis
        print(f"\n📈 Per-Class Performance:")
        print("=" * 70)
        print(f"{'Class':<12} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 70)

        overall_correct = 0
        overall_total = 0

        for i, class_name in enumerate(class_mapping):
            # Get samples for this class
            class_mask = y_true_classes == i
            class_count = np.sum(class_mask)

            if class_count > 0:
                # Calculate class-specific metrics
                class_predictions = y_pred_classes[class_mask]
                class_accuracy = np.mean(class_predictions == i)

                # Calculate precision and recall for this class
                true_positives = np.sum((y_pred_classes == i) & (y_true_classes == i))
                false_positives = np.sum((y_pred_classes == i) & (y_true_classes != i))
                false_negatives = np.sum((y_pred_classes != i) & (y_true_classes == i))

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

                print(f"{class_name:<12} {class_count:<8} {class_accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}")

                overall_correct += np.sum(class_predictions == i)
                overall_total += class_count
            else:
                print(f"{class_name:<12} {class_count:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

        print("-" * 70)
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
        print(f"{'Overall':<12} {overall_total:<8} {overall_accuracy:<10.3f}")

        # Confusion Matrix
        print(f"\n🔢 Confusion Matrix:")
        print("=" * 50)
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Print confusion matrix with class names
        print(f"True\\\\Pred", end="")
        for class_name in class_mapping:
            print(f"{class_name[:8]:<10}", end="")
        print()

        for i, class_name in enumerate(class_mapping):
            print(f"{class_name[:12]:<12}", end="")
            for j in range(len(class_mapping)):
                print(f"{cm[i, j]:<10}", end="")
            print()

        # Classification Report
        print(f"\n📋 Detailed Classification Report:")
        print("=" * 60)
        try:
            report = classification_report(y_true_classes, y_pred_classes,
                                         target_names=class_mapping,
                                         digits=4, zero_division=0)
            print(report)
        except Exception as e:
            print(f"Could not generate classification report: {e}")

        # Confidence analysis
        print(f"\n🎯 Prediction Confidence Analysis:")
        print("=" * 50)
        max_confidences = np.max(y_pred_proba, axis=1)

        confidence_ranges = [
            (0.9, 1.0, "Very High"),
            (0.8, 0.9, "High"),
            (0.6, 0.8, "Medium"),
            (0.4, 0.6, "Low"),
            (0.0, 0.4, "Very Low")
        ]

        for min_conf, max_conf, label in confidence_ranges:
            mask = (max_confidences >= min_conf) & (max_confidences < max_conf)
            count = np.sum(mask)
            percentage = (count / len(max_confidences)) * 100
            if count > 0:
                accuracy_in_range = np.mean(y_pred_classes[mask] == y_true_classes[mask])
                print(f"  {label:10} ({min_conf:.1f}-{max_conf:.1f}): {count:3d} samples ({percentage:5.1f}%) - Accuracy: {accuracy_in_range:.3f}")

        print(f"\nAverage confidence: {np.mean(max_confidences):.4f}")
        print(f"Confidence std dev: {np.std(max_confidences):.4f}")

        # Return comprehensive results
        evaluation_results = {
            'test_accuracy': results_dict.get('accuracy', 0),
            'test_loss': results_dict.get('loss', 0),
            'per_class_accuracy': {},
            'confusion_matrix': cm.tolist(),
            'average_confidence': float(np.mean(max_confidences)),
            'predictions': y_pred_classes.tolist(),
            'true_labels': y_true_classes.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }

        # Add per-class accuracies to results
        for i, class_name in enumerate(class_mapping):
            class_mask = y_true_classes == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred_classes[class_mask] == i)
                evaluation_results['per_class_accuracy'][class_name] = float(class_accuracy)

        return evaluation_results

    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        traceback.print_exc()
        return None

def save_training_metadata(normalization_params, class_mapping, model_config, training_results):
    """
    Save training metadata for use during inference.

    Args:
        normalization_params: Parameters used for feature normalization
        class_mapping: List of class names
        model_config: Model configuration parameters
        training_results: Training results and metrics
    """
    try:
        metadata = {
            'normalization_params': normalization_params,
            'class_mapping': class_mapping,
            'model_config': model_config,
            'training_results': training_results,
            'version': '1.0',
            'created_at': str(np.datetime64('now'))
        }

        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print("📄 Training metadata saved to model_metadata.json")

    except Exception as e:
        print(f"⚠️  Warning: Could not save training metadata: {e}")

def main():
    """
    Main function to run the complete training pipeline.
    """
    print("=" * 80)
    print("🎵 AUDIO CLASSIFICATION MODEL TRAINING")
    print("=" * 80)

    start_time = np.datetime64('now')

    try:
        # Step 1: Load dataset
        print("\n" + "-" * 60)
        print("STEP 1: LOADING DATASET")
        print("-" * 60)

        X, y, class_mapping, file_paths = load_dataset()
        if X is None:
            print("❌ Failed to load dataset. Exiting.")
            return 1

        # Step 2: Prepare features
        print("\n" + "-" * 60)
        print("STEP 2: PREPARING FEATURES")
        print("-" * 60)

        X_processed, normalization_params = prepare_features(X)
        if X_processed is None:
            print("❌ Failed to prepare features. Exiting.")
            return 1

        # Step 3: Prepare labels
        print("\n" + "-" * 60)
        print("STEP 3: PREPARING LABELS")
        print("-" * 60)

        print("🏷️  Converting labels to categorical format...")
        y_categorical = to_categorical(y, num_classes=len(class_mapping))
        print(f"✅ Labels converted: {y.shape} → {y_categorical.shape}")

        # Step 4: Split data
        print("\n" + "-" * 60)
        print("STEP 4: SPLITTING DATA")
        print("-" * 60)

        print(f"📊 Splitting data: {100-TEST_SIZE*100:.0f}% train, {TEST_SIZE*100:.0f}% test")

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_categorical,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y  # Ensure balanced split across classes
        )

        print(f"✅ Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Feature dimensions: {X_train.shape[1]}")

        # Validate split
        if X_train.shape[0] < len(class_mapping):
            print("⚠️  Warning: Very small training set. Results may be unreliable.")

        # Step 5: Create model
        print("\n" + "-" * 60)
        print("STEP 5: CREATING MODEL")
        print("-" * 60)

        input_shape = (X_train.shape[1],)
        num_classes = len(class_mapping)
        model = create_model(input_shape, num_classes)

        if model is None:
            print("❌ Failed to create model. Exiting.")
            return 1

        # Step 6: Setup training callbacks
        print("\n" + "-" * 60)
        print("STEP 6: SETTING UP TRAINING")
        print("-" * 60)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(2, PATIENCE//2),
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            )
        ]

        print(f"🔧 Training configuration:")
        print(f"   Max epochs: {EPOCHS}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Validation split: {VALIDATION_SPLIT}")
        print(f"   Early stopping patience: {PATIENCE}")
        print(f"   Learning rate: {LEARNING_RATE}")

        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(enumerate(class_weights))

        print(f"🔧 Class weights calculated:")
        for i, weight in class_weight_dict.items():
            print(f"   {class_mapping[i]}: {weight:.3f}")

        # Step 7: Train model
        print("\n" + "-" * 60)
        print("STEP 7: TRAINING MODEL")
        print("-" * 60)
        print("🚀 Starting training... This may take several minutes.")

        # Set different random seed for better convergence
        tf.random.set_seed(123)  # Different from preprocessing seed

        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            class_weight=class_weight_dict,  # Use class weights
            verbose=1,
            shuffle=True
        )

        print("✅ Training completed!")

        # Step 8: Plot training history
        print("\n" + "-" * 60)
        print("STEP 8: GENERATING TRAINING PLOTS")
        print("-" * 60)

        plot_training_history(history)

        # Step 9: Evaluate model
        print("\n" + "-" * 60)
        print("STEP 9: EVALUATING MODEL")
        print("-" * 60)

        evaluation_results = evaluate_model(model, X_test, y_test, class_mapping)

        if evaluation_results is None:
            print("⚠️  Warning: Model evaluation failed")
            test_accuracy = 0.0
        else:
            test_accuracy = evaluation_results.get('test_accuracy', 0.0)

        # Step 10: Save model and metadata
        print("\n" + "-" * 60)
        print("STEP 10: SAVING MODEL")
        print("-" * 60)

        model_path = "model.h5"
        print(f"💾 Saving trained model to {model_path}...")
        model.save(model_path, save_format='h5')

        # Save training metadata
        model_config = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history.history['loss']),
            'learning_rate': LEARNING_RATE
        }

        save_training_metadata(normalization_params, class_mapping, model_config, evaluation_results)

        # Final summary
        end_time = np.datetime64('now')
        training_duration = end_time - start_time

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        print(f"📊 Final Results:")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Training Duration: {training_duration}")
        print(f"   Epochs Trained: {len(history.history['loss'])}")

        print(f"\n📁 Generated Files:")
        print(f"   🤖 model.h5 - Trained model for inference")
        print(f"   🏆 best_model.h5 - Best model checkpoint")
        print(f"   📊 training_history.png - Training plots")
        print(f"   📊 training_history_simple.png - Simple training plots")
        print(f"   📄 model_metadata.json - Training metadata")

        print(f"\n🚀 Next Steps:")
        print(f"   1. Review training plots: open training_history.png")
        print(f"   2. Test the model: python3 test_model.py <audio_file.wav>")
        print(f"   3. For Arduino integration, use integer commands:")
        for i, class_name in enumerate(class_mapping):
            print(f"      {i} → {class_name}")

        if test_accuracy < 0.7:
            print(f"\n⚠️  Performance Tips (Current accuracy: {test_accuracy*100:.1f}%):")
            print(f"   - Add more diverse training data")
            print(f"   - Ensure balanced classes")
            print(f"   - Check audio quality and consistency")
            print(f"   - Consider data augmentation")

        print("=" * 80)
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)