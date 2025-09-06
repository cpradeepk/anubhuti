#!/usr/bin/env python3
"""
YAMNet Model Testing Script

This script tests the trained YAMNet-based classifier with individual audio files.
Provides detailed predictions and confidence scores for each class.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
from typing import Dict, Tuple
import logging

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from yamnet_utils import YAMNetProcessor, load_model_metadata, aggregate_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YAMNetModelTester:
    """
    Tester class for YAMNet-based audio classification model.
    """
    
    def __init__(self, model_path: str, metadata_path: str = None):
        """
        Initialize model tester.
        
        Args:
            model_path: Path to trained model (.h5 file)
            metadata_path: Path to model metadata JSON file
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.model_path.parent / "yamnet_model_metadata.json"
        
        # Load model and metadata
        self.model = None
        self.metadata = None
        self.class_names = None
        self.arduino_mapping = None
        self.yamnet_processor = YAMNetProcessor()
        
        self._load_model_and_metadata()
        
        logger.info(f"🎯 YAMNet Model Tester initialized")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Metadata: {self.metadata_path}")
        logger.info(f"   Classes: {self.class_names}")
    
    def _load_model_and_metadata(self):
        """Load trained model and metadata."""
        try:
            # Load model
            logger.info(f"🤖 Loading trained model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Input shape: {self.model.input_shape}")
            logger.info(f"   Output shape: {self.model.output_shape}")
            logger.info(f"   Parameters: {self.model.count_params():,}")
            
            # Load metadata
            if self.metadata_path.exists():
                self.metadata = load_model_metadata(self.metadata_path)
                
                # Extract class information
                class_mapping = self.metadata.get('class_mapping', {})
                self.class_names = [None] * len(class_mapping)
                for class_name, class_id in class_mapping.items():
                    self.class_names[class_id] = class_name
                
                # Extract Arduino mapping
                self.arduino_mapping = self.metadata.get('arduino_mapping', {})
                
                logger.info(f"✅ Metadata loaded successfully")
                logger.info(f"   Model type: {self.metadata.get('model_info', {}).get('model_type', 'Unknown')}")
                
            else:
                logger.warning(f"⚠️  Metadata file not found: {self.metadata_path}")
                # Default class names
                self.class_names = ['slow', 'medium', 'fast', 'disturbance']
                self.arduino_mapping = {'slow': 1, 'medium': 2, 'fast': 3, 'disturbance': 0}
            
        except Exception as e:
            logger.error(f"❌ Error loading model or metadata: {e}")
            raise
    
    def predict_single_file(self, audio_path: str) -> Dict:
        """
        Predict class for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            logger.info(f"🔄 Processing audio file: {audio_path}")
            
            # Validate file exists
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Extract YAMNet embeddings
            embeddings, metadata = self.yamnet_processor.process_audio_file(audio_path)
            
            # Aggregate embeddings to single vector
            aggregated_embedding = aggregate_embeddings(embeddings, method='mean')
            
            # Reshape for model input
            model_input = aggregated_embedding.reshape(1, -1)
            
            # Make prediction
            predictions = self.model.predict(model_input, verbose=0)
            predicted_probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(predicted_probabilities)
            predicted_class_name = self.class_names[predicted_class_idx]
            confidence = float(predicted_probabilities[predicted_class_idx])
            
            # Get Arduino command
            arduino_command = self.arduino_mapping.get(predicted_class_name, 0)
            
            # Prepare results
            results = {
                'file_path': str(audio_path),
                'file_name': audio_path.name,
                'predicted_class_idx': int(predicted_class_idx),
                'predicted_class_name': predicted_class_name,
                'confidence': confidence,
                'arduino_command': arduino_command,
                'all_probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(predicted_probabilities)
                },
                'audio_metadata': metadata
            }
            
            logger.info(f"✅ Prediction completed: {predicted_class_name} ({confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error predicting file {audio_path}: {e}")
            raise
    
    def print_prediction_results(self, results: Dict):
        """
        Print formatted prediction results.
        
        Args:
            results: Prediction results dictionary
        """
        print("\n" + "="*80)
        print("🎯 YAMNET CLASSIFIER PREDICTION RESULTS")
        print("="*80)
        
        print(f"🎵 File: {results['file_name']}")
        print(f"📁 Path: {results['file_path']}")
        
        # Audio metadata
        audio_meta = results['audio_metadata']
        print(f"⏱️  Duration: {audio_meta['duration_seconds']:.2f} seconds")
        print(f"🔊 Sample Rate: {audio_meta['sample_rate']} Hz")
        print(f"📊 YAMNet Frames: {audio_meta['num_frames']}")
        
        print(f"\n🎯 Predicted Class: {results['predicted_class_name']}")
        print(f"📊 Confidence: {results['confidence']:.3f} ({results['confidence']*100:.1f}%)")
        print(f"🔢 Class Index: {results['predicted_class_idx']}")
        print(f"🤖 Arduino Command: {results['arduino_command']}")
        
        # Arduino motor control explanation
        motor_actions = {
            0: "No vibration (disturbance ignored)",
            1: "Top motor vibrates ('slow' sound)",
            2: "Bottom motor vibrates ('medium' sound)",
            3: "Both motors vibrate ('fast' sound)"
        }
        print(f"🎮 Motor Action: {motor_actions.get(results['arduino_command'], 'Unknown')}")
        
        print(f"\n📊 All Class Probabilities:")
        print("-" * 40)
        
        # Sort probabilities for better display
        sorted_probs = sorted(results['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for class_name, probability in sorted_probs:
            marker = "👉" if class_name == results['predicted_class_name'] else "  "
            print(f"{marker} {class_name:<12}: {probability:.3f} ({probability*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def batch_predict(self, audio_files: list) -> Dict:
        """
        Predict classes for multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary containing batch prediction results
        """
        logger.info(f"🔄 Processing {len(audio_files)} audio files...")
        
        batch_results = {
            'total_files': len(audio_files),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'results': [],
            'failed_files': []
        }
        
        for audio_file in audio_files:
            try:
                result = self.predict_single_file(audio_file)
                batch_results['results'].append(result)
                batch_results['successful_predictions'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to process {audio_file}: {e}")
                batch_results['failed_files'].append({
                    'file': str(audio_file),
                    'error': str(e)
                })
                batch_results['failed_predictions'] += 1
        
        logger.info(f"✅ Batch prediction completed: "
                   f"{batch_results['successful_predictions']}/{len(audio_files)} successful")
        
        return batch_results

def main():
    """
    Main testing function.
    """
    parser = argparse.ArgumentParser(description="Test YAMNet-based Audio Classifier")
    parser.add_argument("audio_file", help="Path to audio file to test")
    parser.add_argument("--model", default="yamnet_models/yamnet_classifier.h5", 
                       help="Path to trained model file")
    parser.add_argument("--metadata", help="Path to model metadata file (optional)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--json-output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = YAMNetModelTester(args.model, args.metadata)
        
        # Make prediction
        results = tester.predict_single_file(args.audio_file)
        
        # Print results (unless quiet mode)
        if not args.quiet:
            tester.print_prediction_results(results)
        else:
            print(f"File: {results['file_name']}")
            print(f"Predicted Class: {results['predicted_class_name']}")
            print(f"Confidence: {results['confidence']:.3f} ({results['confidence']*100:.1f}%)")
            print(f"Arduino Command: {results['arduino_command']}")
        
        # Save to JSON if requested
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✅ Results saved to {args.json_output}")
        
        logger.info("🎉 Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
