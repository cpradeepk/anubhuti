#!/usr/bin/env python3
"""
YAMNet-based Audio Classification Training Script

This script trains a classifier on top of YAMNet embeddings for vocal sound classification.
Uses YAMNet as a frozen feature extractor and trains a dense neural network classifier.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from yamnet_utils import YAMNetProcessor, load_dataset, create_balanced_splits, save_model_metadata, aggregate_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YAMNetClassifierTrainer:
    """
    Trainer class for YAMNet-based audio classification.
    """
    
    def __init__(self, dataset_path: str, output_dir: str = "yamnet_models"):
        """
        Initialize trainer.
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Directory to save trained models and outputs
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Class mapping (compatible with existing Arduino integration)
        self.class_mapping = {
            'slow': 0,      # Maps to Arduino command 1 (top motor)
            'medium': 1,    # Maps to Arduino command 2 (bottom motor)
            'fast': 2,      # Maps to Arduino command 3 (both motors)
            'disturbance': 3 # Maps to Arduino command 0 (no vibration)
        }
        
        self.yamnet_processor = YAMNetProcessor()
        self.model = None
        self.history = None
        
        logger.info(f"🎯 YAMNet Classifier Trainer initialized")
        logger.info(f"   Dataset: {self.dataset_path}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Classes: {list(self.class_mapping.keys())}")
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load dataset and extract YAMNet embeddings.
        
        Returns:
            Tuple of (embeddings, labels, metadata)
        """
        logger.info("📊 Loading and preprocessing dataset...")
        
        # Load dataset
        file_paths, labels, class_names = load_dataset(self.dataset_path, self.class_mapping)
        
        if len(file_paths) == 0:
            raise ValueError("No audio files found in dataset!")
        
        # Extract embeddings for all files
        all_embeddings = []
        all_labels = []
        failed_files = []
        
        logger.info(f"🔄 Extracting YAMNet embeddings from {len(file_paths)} files...")
        
        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), 
                                                   total=len(file_paths),
                                                   desc="Processing audio files")):
            try:
                # Extract embeddings
                embeddings, metadata = self.yamnet_processor.process_audio_file(file_path)
                
                # Aggregate embeddings to single vector per file
                aggregated_embedding = aggregate_embeddings(embeddings, method='mean')
                
                all_embeddings.append(aggregated_embedding)
                all_labels.append(label)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to process {file_path}: {e}")
                failed_files.append(file_path)
                continue
        
        if len(all_embeddings) == 0:
            raise ValueError("No embeddings could be extracted!")
        
        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings)
        labels_array = np.array(all_labels)
        
        # Create metadata
        dataset_metadata = {
            'total_files': len(file_paths),
            'successful_files': len(all_embeddings),
            'failed_files': len(failed_files),
            'embedding_shape': embeddings_array.shape,
            'class_mapping': self.class_mapping,
            'class_names': class_names,
            'failed_file_list': failed_files
        }
        
        logger.info(f"✅ Dataset preprocessing completed:")
        logger.info(f"   Successful: {len(all_embeddings)}/{len(file_paths)} files")
        logger.info(f"   Embeddings shape: {embeddings_array.shape}")
        logger.info(f"   Labels shape: {labels_array.shape}")
        
        # Print class distribution
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        logger.info("   Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = class_names[label]
            percentage = (count / len(labels_array)) * 100
            logger.info(f"     {class_name}: {count} samples ({percentage:.1f}%)")
        
        return embeddings_array, labels_array, dataset_metadata
    
    def create_model(self, input_dim: int, num_classes: int) -> tf.keras.Model:
        """
        Create classifier model on top of YAMNet embeddings.
        
        Args:
            input_dim: Dimension of input embeddings (1024 for YAMNet)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"🏗️  Building classifier model...")
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(input_dim,), name='yamnet_embeddings'),
            
            # Dense layers with dropout for regularization
            tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.3, name='dropout_1'),
            
            tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
            tf.keras.layers.Dropout(0.4, name='dropout_2'),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"✅ Model created:")
        logger.info(f"   Input shape: ({input_dim},)")
        logger.info(f"   Output classes: {num_classes}")
        logger.info(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, embeddings: np.ndarray, labels: np.ndarray, 
                   validation_split: float = 0.2) -> Dict:
        """
        Train the classifier model.
        
        Args:
            embeddings: YAMNet embeddings array
            labels: Corresponding labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        logger.info("🚀 Starting model training...")
        
        # Convert labels to categorical
        num_classes = len(self.class_mapping)
        labels_categorical = tf.keras.utils.to_categorical(labels, num_classes)
        
        # Create model
        self.model = self.create_model(embeddings.shape[1], num_classes)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        logger.info("📊 Class weights for balanced training:")
        for class_id, weight in class_weight_dict.items():
            class_name = list(self.class_mapping.keys())[class_id]
            logger.info(f"   {class_name}: {weight:.3f}")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("🔄 Training in progress...")
        self.history = self.model.fit(
            embeddings, labels_categorical,
            batch_size=32,
            epochs=50,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Training completed!")
        
        return self.history.history
    
    def evaluate_model(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate trained model performance.
        
        Args:
            embeddings: Test embeddings
            labels: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("📊 Evaluating model performance...")
        
        # Convert labels to categorical
        labels_categorical = tf.keras.utils.to_categorical(labels, len(self.class_mapping))
        
        # Evaluate model
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            embeddings, labels_categorical, verbose=0
        )
        
        # Get predictions
        predictions = self.model.predict(embeddings, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Generate classification report
        class_names = list(self.class_mapping.keys())
        report = classification_report(
            labels, predicted_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(labels, predicted_classes)
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        logger.info(f"✅ Model evaluation completed:")
        logger.info(f"   Test Accuracy: {test_accuracy:.3f}")
        logger.info(f"   Test Precision: {test_precision:.3f}")
        logger.info(f"   Test Recall: {test_recall:.3f}")
        
        return evaluation_results

    def plot_training_history(self, save_path: str = None):
        """
        Plot and save training history.

        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YAMNet Classifier Training History', fontsize=16)

        # Plot training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "training_history.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Training history plot saved to {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: str = None):
        """
        Plot and save confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - YAMNet Classifier')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')

        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Confusion matrix plot saved to {save_path}")
        plt.close()

    def save_model_and_metadata(self, evaluation_results: Dict, dataset_metadata: Dict):
        """
        Save trained model and comprehensive metadata.

        Args:
            evaluation_results: Model evaluation results
            dataset_metadata: Dataset preprocessing metadata
        """
        # Save model
        model_path = self.output_dir / "yamnet_classifier.h5"
        self.model.save(model_path)
        logger.info(f"✅ Model saved to {model_path}")

        # Prepare comprehensive metadata
        metadata = {
            'model_info': {
                'model_type': 'YAMNet_Classifier',
                'architecture': 'YAMNet + Dense Classifier',
                'yamnet_url': self.yamnet_processor.model_url,
                'input_shape': list(self.model.input_shape[1:]),
                'output_shape': list(self.model.output_shape[1:]),
                'total_parameters': int(self.model.count_params()),
                'training_date': datetime.now().isoformat()
            },
            'dataset_info': dataset_metadata,
            'training_config': {
                'batch_size': 32,
                'max_epochs': 50,
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'loss_function': 'categorical_crossentropy',
                'validation_split': 0.2
            },
            'performance_metrics': evaluation_results,
            'class_mapping': self.class_mapping,
            'arduino_mapping': {
                'slow': 1,      # Top motor
                'medium': 2,    # Bottom motor
                'fast': 3,      # Both motors
                'disturbance': 0 # No vibration
            }
        }

        # Save metadata
        metadata_path = self.output_dir / "yamnet_model_metadata.json"
        save_model_metadata(metadata, metadata_path)

        logger.info(f"✅ Complete model and metadata saved to {self.output_dir}")

def main():
    """
    Main training function.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train YAMNet-based Audio Classifier")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", default="yamnet_models", help="Output directory for models")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation set ratio")

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = YAMNetClassifierTrainer(args.dataset, args.output)

        # Load and preprocess data
        embeddings, labels, dataset_metadata = trainer.load_and_preprocess_data()

        # Create balanced splits using sklearn directly
        from sklearn.model_selection import train_test_split

        # First split: separate test set
        train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
            embeddings, labels,
            test_size=args.test_split,
            stratify=labels,
            random_state=42
        )

        logger.info(f"📊 Data splits:")
        logger.info(f"   Train+Val: {len(train_val_embeddings)} samples")
        logger.info(f"   Test: {len(test_embeddings)} samples")
        logger.info(f"   Test classes: {np.unique(test_labels)}")
        logger.info(f"   Train+Val classes: {np.unique(train_val_labels)}")

        # Train model
        history = trainer.train_model(train_val_embeddings, train_val_labels)

        # Evaluate model
        evaluation_results = trainer.evaluate_model(test_embeddings, test_labels)

        # Generate plots
        trainer.plot_training_history()
        trainer.plot_confusion_matrix(
            np.array(evaluation_results['confusion_matrix']),
            evaluation_results['class_names']
        )

        # Save model and metadata
        trainer.save_model_and_metadata(evaluation_results, dataset_metadata)

        # Print final results
        print("\n" + "="*80)
        print("🎉 YAMNET CLASSIFIER TRAINING COMPLETED")
        print("="*80)
        print(f"📊 Final Test Accuracy: {evaluation_results['test_accuracy']:.3f}")
        print(f"📊 Final Test Precision: {evaluation_results['test_precision']:.3f}")
        print(f"📊 Final Test Recall: {evaluation_results['test_recall']:.3f}")
        print(f"💾 Model saved to: {trainer.output_dir}/yamnet_classifier.h5")
        print(f"📄 Metadata saved to: {trainer.output_dir}/yamnet_model_metadata.json")
        print(f"📈 Plots saved to: {trainer.output_dir}/")

        print("\n📋 Per-Class Performance:")
        for class_name in evaluation_results['class_names']:
            metrics = evaluation_results['classification_report'][class_name]
            print(f"   {class_name:12}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

        print("\n🚀 Ready for deployment! Use test_yamnet_model.py to test individual files.")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
