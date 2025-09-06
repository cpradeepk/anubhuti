#!/usr/bin/env python3
"""
Optimized DS-CNN Model for Audio Classification

This implements a true Depthwise Separable CNN architecture optimized for:
- Small dataset performance (22 samples)
- Real-time inference on Raspberry Pi
- Scalability to larger datasets
- 4-class classification: "soo", "hum", "hmm", "disturbance"

Architecture: MFCC → Reshape → DS-CNN Layers → Global Pooling → Dense → Output
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from collections import Counter
import matplotlib.pyplot as plt

# Configuration for embedded deployment
SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 13
N_FRAMES = 130
INPUT_SHAPE = (N_MFCC, N_FRAMES, 1)  # Height, Width, Channels for CNN
NUM_CLASSES = 4

class DSCNNModel:
    """
    Optimized DS-CNN model for small dataset audio classification.
    """
    
    def __init__(self, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_mapping = ["disturbance", "soo", "hum", "hmm"]  # Updated mapping
        
    def depthwise_separable_conv_block(self, x, filters, kernel_size=(3, 3), 
                                     strides=(1, 1), dropout_rate=0.3, name_prefix=""):
        """
        Depthwise Separable Convolution Block optimized for small datasets.
        """
        # Depthwise Convolution
        x = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            depthwise_regularizer=keras.regularizers.l2(0.01),
            name=f'{name_prefix}_depthwise'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
        
        # Pointwise Convolution
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(0.01),
            name=f'{name_prefix}_pointwise'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu2')(x)
        
        # Dropout for regularization
        x = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x)
        
        return x
    
    def build_model(self):
        """
        Build optimized DS-CNN architecture for small dataset.
        """
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Initial convolution to increase channels
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='initial_conv'
        )(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.ReLU(name='initial_relu')(x)
        
        # DS-CNN Block 1 - Feature extraction
        x = self.depthwise_separable_conv_block(
            x, filters=64, kernel_size=(3, 3), strides=(1, 1),
            dropout_rate=0.4, name_prefix='dscnn_block1'
        )
        
        # DS-CNN Block 2 - Pattern recognition
        x = self.depthwise_separable_conv_block(
            x, filters=128, kernel_size=(3, 3), strides=(2, 2),
            dropout_rate=0.5, name_prefix='dscnn_block2'
        )
        
        # DS-CNN Block 3 - High-level features (smaller for small dataset)
        x = self.depthwise_separable_conv_block(
            x, filters=128, kernel_size=(3, 3), strides=(1, 1),
            dropout_rate=0.5, name_prefix='dscnn_block3'
        )
        
        # Global Average Pooling (better than Flatten for small datasets)
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers for classification
        x = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='dense1'
        )(x)
        x = layers.Dropout(0.6, name='dropout_dense1')(x)
        
        x = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='dense2'
        )(x)
        x = layers.Dropout(0.4, name='dropout_dense2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='DS_CNN_Audio_Classifier')
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile model with optimized parameters for small dataset.
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Use Adam optimizer with low learning rate
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Get detailed model summary.
        """
        if self.model is None:
            return "Model not built yet"
        
        return self.model.summary()

def prepare_data_for_dscnn():
    """
    Prepare data specifically for DS-CNN input format.
    """
    print("🔄 Preparing data for DS-CNN...")
    
    # Load preprocessed data
    try:
        with open("data.json", 'r') as f:
            data = json.load(f)
        print("✅ Loaded data.json")
    except Exception as e:
        print(f"❌ Error loading data.json: {e}")
        return None, None, None
    
    # Extract data
    mfcc_features = np.array(data["mfcc"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int32)
    class_mapping = data["mapping"]
    
    print(f"📊 Dataset: {len(labels)} samples, {len(class_mapping)} classes")
    print(f"📐 MFCC shape: {mfcc_features.shape}")
    
    # Reshape for CNN: (samples, height, width, channels)
    # MFCC shape: (samples, n_mfcc, n_frames) → (samples, n_mfcc, n_frames, 1)
    X = mfcc_features.reshape(-1, N_MFCC, N_FRAMES, 1)
    
    # Normalize features for CNN (0-1 scaling per sample)
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        sample = X[i, :, :, 0]
        sample_min = sample.min()
        sample_max = sample.max()
        if sample_max > sample_min:
            X_normalized[i, :, :, 0] = (sample - sample_min) / (sample_max - sample_min)
        else:
            X_normalized[i, :, :, 0] = sample
    
    # Convert labels to categorical
    y = to_categorical(labels, num_classes=len(class_mapping))
    
    print(f"✅ Data prepared for DS-CNN:")
    print(f"   Input shape: {X_normalized.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Value range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    
    return X_normalized, y, class_mapping

def train_dscnn_model():
    """
    Train DS-CNN model with advanced techniques for small dataset.
    """
    print("=" * 80)
    print("🚀 TRAINING OPTIMIZED DS-CNN MODEL")
    print("=" * 80)
    
    # Prepare data
    X, y, class_mapping = prepare_data_for_dscnn()
    if X is None:
        return False
    
    labels = np.argmax(y, axis=1)
    
    # Analyze class distribution
    class_counts = Counter(labels)
    print(f"📈 Class distribution:")
    for i, class_name in enumerate(class_mapping):
        count = class_counts.get(i, 0)
        percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
        print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
    
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
    
    # Create and build model
    dscnn = DSCNNModel(input_shape=INPUT_SHAPE, num_classes=len(class_mapping))
    model = dscnn.build_model()
    model = dscnn.compile_model(learning_rate=0.0001)
    
    print(f"\n🏗️  DS-CNN Model Architecture:")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Advanced training strategy for small dataset
    # Use K-Fold Cross Validation for better generalization
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accuracies = []
    
    best_model = None
    best_accuracy = 0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, labels)):
        print(f"\n📊 Training Fold {fold + 1}/3...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Rebuild model for each fold
        fold_model = dscnn.build_model()
        fold_model = dscnn.compile_model(learning_rate=0.0001)
        
        # Callbacks for this fold
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.7,
                patience=12,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Train fold
        history = fold_model.fit(
            X_train, y_train,
            batch_size=4,  # Very small batch for small dataset
            epochs=150,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0,
            shuffle=True
        )
        
        # Evaluate fold
        val_accuracy = max(history.history['val_accuracy'])
        fold_accuracies.append(val_accuracy)
        
        print(f"   Fold {fold + 1} validation accuracy: {val_accuracy:.3f}")
        
        # Keep best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = fold_model
    
    # Final evaluation
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\n📊 Cross-Validation Results:")
    print(f"   Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"   Best fold accuracy: {best_accuracy:.3f}")
    
    # Final training on all data with best hyperparameters
    print(f"\n🚀 Final training on complete dataset...")
    
    final_model = dscnn.build_model()
    final_model = dscnn.compile_model(learning_rate=0.0001)
    
    # Final callbacks
    final_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',  # Monitor training loss since we use all data
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.8,
            patience=15,
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_dscnn_model.h5',
            monitor='accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Final training
    final_history = final_model.fit(
        X, y,
        batch_size=4,
        epochs=200,
        callbacks=final_callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        shuffle=True
    )
    
    # Final evaluation on all data
    final_predictions = final_model.predict(X, verbose=0)
    predicted_classes = np.argmax(final_predictions, axis=1)
    true_classes = labels
    
    # Detailed performance analysis
    pred_counts = Counter(predicted_classes)
    true_counts = Counter(true_classes)
    
    print(f"\n🎯 FINAL DS-CNN PERFORMANCE:")
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
    
    print(f"\n📈 FINAL PERFORMANCE METRICS:")
    print(f"   Overall accuracy: {overall_accuracy:.3f}")
    print(f"   Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"   Dominant class: {class_mapping[dominant_class]} ({dominant_percentage:.1f}%)")
    print(f"   Model parameters: {final_model.count_params():,}")
    
    # Success criteria
    success = (overall_accuracy > 0.5 and dominant_percentage < 70)
    
    if success:
        print("🎉 SUCCESS: DS-CNN model trained successfully!")
    else:
        print("⚠️  PARTIAL: Model trained but needs more diverse data")
    
    # Save final model
    final_model.save("dscnn_model.h5")
    print(f"💾 DS-CNN model saved as: dscnn_model.h5")
    
    # Replace active model
    if os.path.exists("model.h5"):
        os.rename("model.h5", "model_backup.h5")
    os.rename("dscnn_model.h5", "model.h5")
    print(f"✅ DS-CNN model is now active")
    
    # Save metadata for deployment
    metadata = {
        "model_type": "DS_CNN_Audio_Classifier",
        "architecture": "Depthwise_Separable_CNN",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": len(class_mapping),
        "class_mapping": class_mapping,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mfcc": N_MFCC,
        "n_frames": N_FRAMES,
        "overall_accuracy": float(overall_accuracy),
        "cv_accuracy_mean": float(mean_accuracy),
        "cv_accuracy_std": float(std_accuracy),
        "dominant_class_percentage": float(dominant_percentage),
        "total_parameters": int(final_model.count_params()),
        "optimized_for": "raspberry_pi_deployment"
    }
    
    with open("dscnn_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update main metadata
    with open("model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 DS-CNN metadata saved for deployment")
    
    return success

if __name__ == "__main__":
    success = train_dscnn_model()
    
    if success:
        print(f"\n🎉 DS-CNN MODEL READY FOR DEPLOYMENT!")
        print(f"Next steps:")
        print(f"1. Test: python3 test_model.py [audio_file].wav")
        print(f"2. Deploy to Raspberry Pi")
        print(f"3. Implement real-time audio processing")
        print(f"4. Set up wireless communication with Arduino")
    else:
        print(f"\n📝 RECOMMENDATIONS:")
        print(f"1. Record more diverse audio samples")
        print(f"2. Use different vocal techniques for each class")
        print(f"3. Consider data augmentation techniques")
