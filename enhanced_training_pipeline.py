#!/usr/bin/env python3
"""
Enhanced YAMNet Training Pipeline with Real-World Audio Integration
Optimized for classroom environment audio data
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow for optimal performance
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

class EnhancedYAMNetTrainer:
    def __init__(self, dataset_dir, real_world_audio=None, output_dir="yamnet_models"):
        self.dataset_dir = Path(dataset_dir)
        self.real_world_audio = real_world_audio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # YAMNet configuration
        self.sample_rate = 16000
        self.segment_duration = 5.0  # seconds
        self.hop_duration = 2.5      # seconds (50% overlap)
        
        # Class mapping for Arduino commands
        self.class_mapping = {
            'slow': 1,      # Top motor
            'medium': 2,    # Bottom motor  
            'fast': 3,      # Both motors
            'disturbance': 4 # Continue pattern
        }
        
        print("🚀 Enhanced YAMNet Training Pipeline Initialized")
        print(f"📁 Dataset directory: {self.dataset_dir}")
        print(f"🎵 Real-world audio: {self.real_world_audio}")
        print(f"💾 Output directory: {self.output_dir}")
    
    def load_yamnet_model(self):
        """Load pre-trained YAMNet model from TensorFlow Hub"""
        print("📥 Loading YAMNet model from TensorFlow Hub...")
        
        # Load YAMNet model
        yamnet_model_url = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(yamnet_model_url)
        
        print("✅ YAMNet model loaded successfully")
        return self.yamnet_model
    
    def segment_long_audio(self, audio_file, output_dir):
        """Segment long audio file into training chunks"""
        print(f"🔪 Segmenting audio file: {audio_file}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        duration = len(audio) / sr
        
        print(f"📊 Audio duration: {duration:.1f} seconds")
        print(f"📊 Sample rate: {sr} Hz")
        
        # Create output directory
        segments_dir = Path(output_dir) / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate segment parameters
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(self.hop_duration * sr)
        
        segments = []
        segment_count = 0
        
        # Extract segments with overlap
        for start_sample in range(0, len(audio) - segment_samples + 1, hop_samples):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            
            # Save segment
            segment_file = segments_dir / f"segment_{segment_count:04d}.wav"
            sf.write(segment_file, segment, sr)
            segments.append(str(segment_file))
            segment_count += 1
        
        print(f"✅ Created {len(segments)} audio segments")
        return segments
    
    def classify_segments_with_existing_model(self, segments, model_path=None):
        """Use existing model to pre-classify segments for labeling assistance"""
        if not model_path or not os.path.exists(model_path):
            print("⚠️  No existing model found for pre-classification")
            return {}
        
        print("🤖 Pre-classifying segments with existing model...")
        
        try:
            # Load existing model
            model = tf.keras.models.load_model(model_path)
            
            predictions = {}
            class_names = ['slow', 'medium', 'fast', 'disturbance']
            
            for segment_file in tqdm(segments, desc="Classifying segments"):
                # Load and preprocess audio
                audio, _ = librosa.load(segment_file, sr=self.sample_rate)
                
                # Get YAMNet embeddings
                embeddings = self.yamnet_model(audio)
                embeddings = tf.reduce_mean(embeddings, axis=0)
                embeddings = tf.expand_dims(embeddings, 0)
                
                # Predict with existing model
                pred = model.predict(embeddings, verbose=0)
                predicted_class = class_names[np.argmax(pred)]
                confidence = np.max(pred)
                
                predictions[segment_file] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': pred[0].tolist()
                }
            
            print(f"✅ Pre-classified {len(predictions)} segments")
            return predictions
            
        except Exception as e:
            print(f"⚠️  Pre-classification failed: {e}")
            return {}
    
    def create_labeling_interface(self, segments, predictions=None):
        """Create interactive labeling interface for manual review"""
        print("🏷️  Creating labeling interface...")
        
        # Create labeling directory
        labeling_dir = self.output_dir / "labeling"
        labeling_dir.mkdir(exist_ok=True)
        
        # Create labeling script
        labeling_script = labeling_dir / "label_segments.py"
        
        with open(labeling_script, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
Interactive Segment Labeling Tool
Usage: python3 label_segments.py
"""

import os
import json
import pygame
from pathlib import Path

class SegmentLabeler:
    def __init__(self):
        pygame.mixer.init(frequency=16000)
        self.segments = {json.dumps(segments)}
        self.predictions = {json.dumps(predictions or {})}
        self.labels = {{}}
        self.current_index = 0
        
        # Load existing labels if available
        self.labels_file = Path("segment_labels.json")
        if self.labels_file.exists():
            with open(self.labels_file) as f:
                self.labels = json.load(f)
    
    def play_segment(self, segment_file):
        """Play audio segment"""
        try:
            pygame.mixer.music.load(segment_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing {{segment_file}}: {{e}}")
    
    def label_segments(self):
        """Interactive labeling interface"""
        print("🎵 YAMNet Segment Labeling Tool")
        print("Commands: [p]lay, [1]slow, [2]medium, [3]fast, [4]disturbance, [s]kip, [q]uit")
        print("-" * 60)
        
        for i, segment in enumerate(self.segments):
            self.current_index = i
            segment_name = os.path.basename(segment)
            
            # Show prediction if available
            if segment in self.predictions:
                pred = self.predictions[segment]
                print(f"\\n[{{i+1}}/{{len(self.segments)}}] {{segment_name}}")
                print(f"Predicted: {{pred['predicted_class']}} (confidence: {{pred['confidence']:.3f}})")
            else:
                print(f"\\n[{{i+1}}/{{len(self.segments)}}] {{segment_name}}")
            
            # Check if already labeled
            if segment in self.labels:
                print(f"Current label: {{self.labels[segment]}}")
            
            while True:
                command = input("Command: ").lower().strip()
                
                if command == 'p':
                    self.play_segment(segment)
                elif command == '1':
                    self.labels[segment] = 'slow'
                    print("✅ Labeled as: slow")
                    break
                elif command == '2':
                    self.labels[segment] = 'medium'
                    print("✅ Labeled as: medium")
                    break
                elif command == '3':
                    self.labels[segment] = 'fast'
                    print("✅ Labeled as: fast")
                    break
                elif command == '4':
                    self.labels[segment] = 'disturbance'
                    print("✅ Labeled as: disturbance")
                    break
                elif command == 's':
                    print("⏭️  Skipped")
                    break
                elif command == 'q':
                    self.save_labels()
                    print("💾 Labels saved. Exiting...")
                    return
                else:
                    print("Invalid command. Use: p, 1, 2, 3, 4, s, q")
            
            # Auto-save every 10 segments
            if (i + 1) % 10 == 0:
                self.save_labels()
                print(f"💾 Auto-saved labels ({{len(self.labels)}} segments labeled)")
        
        self.save_labels()
        print(f"🎉 Labeling completed! {{len(self.labels)}} segments labeled")
    
    def save_labels(self):
        """Save labels to JSON file"""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)

if __name__ == "__main__":
    labeler = SegmentLabeler()
    labeler.label_segments()
''')
        
        print(f"✅ Labeling interface created: {labeling_script}")
        print(f"📋 To label segments, run: cd {labeling_dir} && python3 label_segments.py")
        
        return labeling_script
    
    def load_enhanced_dataset(self, labeled_segments_file=None):
        """Load original dataset plus labeled real-world segments"""
        print("📊 Loading enhanced dataset...")
        
        # Load original dataset
        X_original, y_original = self.load_original_dataset()
        print(f"📁 Original dataset: {len(X_original)} samples")
        
        # Load real-world segments if available
        X_realworld, y_realworld = [], []
        if labeled_segments_file and os.path.exists(labeled_segments_file):
            X_realworld, y_realworld = self.load_labeled_segments(labeled_segments_file)
            print(f"🌍 Real-world segments: {len(X_realworld)} samples")
        
        # Combine datasets
        X_combined = X_original + X_realworld
        y_combined = y_original + y_realworld
        
        print(f"📊 Combined dataset: {len(X_combined)} samples")
        
        # Show class distribution
        unique, counts = np.unique(y_combined, return_counts=True)
        for class_name, count in zip(unique, counts):
            print(f"  {class_name}: {count} samples")
        
        return X_combined, y_combined
    
    def load_original_dataset(self):
        """Load original audio dataset"""
        X, y = [], []
        
        for class_dir in self.dataset_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_mapping:
                class_name = class_dir.name
                
                for audio_file in class_dir.glob("*.wav"):
                    try:
                        # Load audio
                        audio, _ = librosa.load(audio_file, sr=self.sample_rate)
                        X.append(audio)
                        y.append(class_name)
                    except Exception as e:
                        print(f"⚠️  Error loading {audio_file}: {e}")
        
        return X, y
    
    def load_labeled_segments(self, labels_file):
        """Load labeled real-world segments"""
        with open(labels_file) as f:
            labels = json.load(f)
        
        X, y = [], []
        for segment_file, label in labels.items():
            if label in self.class_mapping:
                try:
                    audio, _ = librosa.load(segment_file, sr=self.sample_rate)
                    X.append(audio)
                    y.append(label)
                except Exception as e:
                    print(f"⚠️  Error loading {segment_file}: {e}")
        
        return X, y
    
    def extract_yamnet_features(self, audio_data):
        """Extract YAMNet embeddings from audio data"""
        print("🔍 Extracting YAMNet features...")
        
        features = []
        for i, audio in enumerate(tqdm(audio_data, desc="Extracting features")):
            try:
                # Get YAMNet embeddings
                embeddings = self.yamnet_model(audio)
                # Average embeddings over time
                feature_vector = tf.reduce_mean(embeddings, axis=0)
                features.append(feature_vector.numpy())
            except Exception as e:
                print(f"⚠️  Error extracting features for sample {i}: {e}")
                # Use zero vector as fallback
                features.append(np.zeros(1024))
        
        return np.array(features)
    
    def build_classifier(self, input_shape):
        """Build enhanced classifier on top of YAMNet"""
        print("🏗️  Building enhanced classifier...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.class_mapping), activation='softmax')
        ])
        
        # Use latest TensorFlow optimizer syntax
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=True  # Enable XLA compilation for faster training
        )
        
        return model
    
    def train_enhanced_model(self, X, y, validation_split=0.2, epochs=50):
        """Train the enhanced model"""
        print("🏋️  Training enhanced YAMNet model...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Extract features
        X_features = self.extract_yamnet_features(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y_encoded, test_size=validation_split, 
            random_state=42, stratify=y_encoded
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Validation set: {len(X_val)} samples")
        
        # Build model
        model = self.build_classifier(X_features.shape[1:])
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                self.output_dir / 'best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(self.output_dir / 'yamnet_classifier_enhanced.h5')
        
        # Save metadata
        metadata = {
            'classes': label_encoder.classes_.tolist(),
            'class_mapping': self.class_mapping,
            'model_architecture': 'YAMNet + Enhanced Dense Classifier',
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'total_parameters': model.count_params(),
            'sample_rate': self.sample_rate,
            'segment_duration': self.segment_duration
        }
        
        with open(self.output_dir / 'yamnet_model_metadata_enhanced.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Enhanced model training completed!")
        return model, history, label_encoder

def main():
    parser = argparse.ArgumentParser(description='Enhanced YAMNet Training Pipeline')
    parser.add_argument('--dataset', required=True, help='Path to original dataset directory')
    parser.add_argument('--real-world-audio', help='Path to 45-minute real-world audio file')
    parser.add_argument('--labeled-segments', help='Path to labeled segments JSON file')
    parser.add_argument('--output-dir', default='yamnet_models_enhanced', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EnhancedYAMNetTrainer(
        dataset_dir=args.dataset,
        real_world_audio=args.real_world_audio,
        output_dir=args.output_dir
    )
    
    # Load YAMNet
    trainer.load_yamnet_model()
    
    # Process real-world audio if provided
    if args.real_world_audio:
        print("🎵 Processing real-world audio...")
        segments = trainer.segment_long_audio(args.real_world_audio, args.output_dir)
        
        # Pre-classify segments if existing model available
        existing_model = Path("yamnet_models/yamnet_classifier.h5")
        predictions = trainer.classify_segments_with_existing_model(
            segments, existing_model if existing_model.exists() else None
        )
        
        # Create labeling interface
        trainer.create_labeling_interface(segments, predictions)
        
        print("📋 Next steps:")
        print("1. Run the labeling interface to classify segments")
        print("2. Re-run this script with --labeled-segments to train enhanced model")
        return
    
    # Load enhanced dataset
    X, y = trainer.load_enhanced_dataset(args.labeled_segments)
    
    # Train enhanced model
    model, history, label_encoder = trainer.train_enhanced_model(X, y, epochs=args.epochs)
    
    print("🎉 Enhanced YAMNet training pipeline completed!")

if __name__ == "__main__":
    main()
