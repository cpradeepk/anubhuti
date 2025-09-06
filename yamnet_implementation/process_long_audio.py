#!/usr/bin/env python3
"""
Long Audio Processing Script for YAMNet Classifier

This script processes long audio files using sliding window approach with YAMNet embeddings.
Implements chunk-based processing with overlap and aggregation strategies.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from collections import Counter
import time

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from yamnet_utils import YAMNetProcessor, load_model_metadata, aggregate_embeddings, chunk_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LongAudioProcessor:
    """
    Processor for long audio files using YAMNet-based classification.
    """
    
    def __init__(self, model_path: str, metadata_path: str = None,
                 chunk_duration: float = 5.0, overlap: float = 0.5):
        """
        Initialize long audio processor.
        
        Args:
            model_path: Path to trained model (.h5 file)
            metadata_path: Path to model metadata JSON file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap ratio between chunks (0.0 to 1.0)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.model_path.parent / "yamnet_model_metadata.json"
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        
        # Load model and metadata
        self.model = None
        self.metadata = None
        self.class_names = None
        self.arduino_mapping = None
        self.yamnet_processor = YAMNetProcessor()
        
        self._load_model_and_metadata()
        
        logger.info(f"🎯 Long Audio Processor initialized")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Chunk duration: {self.chunk_duration}s")
        logger.info(f"   Overlap: {self.overlap*100:.0f}%")
        logger.info(f"   Classes: {self.class_names}")
    
    def _load_model_and_metadata(self):
        """Load trained model and metadata."""
        try:
            # Load model
            logger.info(f"🤖 Loading trained model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded successfully")
            
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
                
            else:
                logger.warning(f"⚠️  Metadata file not found: {self.metadata_path}")
                # Default class names
                self.class_names = ['slow', 'medium', 'fast', 'disturbance']
                self.arduino_mapping = {'slow': 1, 'medium': 2, 'fast': 3, 'disturbance': 0}
            
        except Exception as e:
            logger.error(f"❌ Error loading model or metadata: {e}")
            raise
    
    def process_audio_chunks(self, audio_path: str) -> Dict:
        """
        Process long audio file using sliding window approach.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"🔄 Processing long audio file: {audio_path}")
            start_time = time.time()
            
            # Load and preprocess audio
            audio = self.yamnet_processor.preprocess_audio(audio_path)
            total_duration = len(audio) / self.yamnet_processor.YAMNET_SAMPLE_RATE
            
            logger.info(f"🎵 Audio loaded: {total_duration:.2f} seconds")
            
            # Check if chunking is needed
            if total_duration <= self.chunk_duration * 1.5:
                logger.info("🔄 Audio is short, processing as single chunk")
                return self._process_single_chunk(audio_path, audio)
            
            # Split audio into chunks
            chunks = chunk_audio(audio, self.chunk_duration, self.overlap)
            logger.info(f"📊 Created {len(chunks)} chunks ({self.chunk_duration}s each, {self.overlap*100:.0f}% overlap)")
            
            # Process each chunk
            chunk_results = []
            chunk_predictions = []
            chunk_confidences = []
            
            logger.info("🔄 Processing chunks...")
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                try:
                    # Extract embeddings for chunk
                    embeddings = self.yamnet_processor.extract_embeddings(chunk)
                    aggregated_embedding = aggregate_embeddings(embeddings, method='mean')
                    
                    # Make prediction
                    model_input = aggregated_embedding.reshape(1, -1)
                    predictions = self.model.predict(model_input, verbose=0)
                    predicted_probabilities = predictions[0]
                    
                    # Get predicted class
                    predicted_class_idx = np.argmax(predicted_probabilities)
                    predicted_class_name = self.class_names[predicted_class_idx]
                    confidence = float(predicted_probabilities[predicted_class_idx])
                    
                    chunk_result = {
                        'chunk_id': i,
                        'start_time': i * self.chunk_duration * (1 - self.overlap),
                        'end_time': (i * self.chunk_duration * (1 - self.overlap)) + self.chunk_duration,
                        'predicted_class_idx': int(predicted_class_idx),
                        'predicted_class_name': predicted_class_name,
                        'confidence': confidence,
                        'probabilities': predicted_probabilities.tolist()
                    }
                    
                    chunk_results.append(chunk_result)
                    chunk_predictions.append(predicted_class_name)
                    chunk_confidences.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to process chunk {i}: {e}")
                    continue
            
            # Aggregate results
            processing_time = time.time() - start_time
            aggregated_results = self._aggregate_chunk_results(
                chunk_results, chunk_predictions, chunk_confidences, 
                audio_path, total_duration, processing_time
            )
            
            logger.info(f"✅ Long audio processing completed in {processing_time:.2f}s")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"❌ Error processing long audio {audio_path}: {e}")
            raise
    
    def _process_single_chunk(self, audio_path: str, audio: np.ndarray) -> Dict:
        """
        Process audio as single chunk (for shorter files).
        
        Args:
            audio_path: Path to audio file
            audio: Preprocessed audio array
            
        Returns:
            Processing results dictionary
        """
        # Extract embeddings
        embeddings = self.yamnet_processor.extract_embeddings(audio)
        aggregated_embedding = aggregate_embeddings(embeddings, method='mean')
        
        # Make prediction
        model_input = aggregated_embedding.reshape(1, -1)
        predictions = self.model.predict(model_input, verbose=0)
        predicted_probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predicted_probabilities)
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = float(predicted_probabilities[predicted_class_idx])
        
        total_duration = len(audio) / self.yamnet_processor.YAMNET_SAMPLE_RATE
        
        return {
            'file_path': str(audio_path),
            'file_name': Path(audio_path).name,
            'total_duration': total_duration,
            'processing_method': 'single_chunk',
            'num_chunks': 1,
            'dominant_class': predicted_class_name,
            'dominant_confidence': confidence,
            'arduino_command': self.arduino_mapping.get(predicted_class_name, 0),
            'class_distribution': {predicted_class_name: 1.0},
            'chunk_results': [{
                'chunk_id': 0,
                'start_time': 0.0,
                'end_time': total_duration,
                'predicted_class_name': predicted_class_name,
                'confidence': confidence
            }],
            'all_probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(predicted_probabilities)
            }
        }
    
    def _aggregate_chunk_results(self, chunk_results: List[Dict], 
                               chunk_predictions: List[str],
                               chunk_confidences: List[float],
                               audio_path: str, total_duration: float,
                               processing_time: float) -> Dict:
        """
        Aggregate results from multiple chunks.
        
        Args:
            chunk_results: List of individual chunk results
            chunk_predictions: List of predicted class names
            chunk_confidences: List of confidence scores
            audio_path: Path to original audio file
            total_duration: Total audio duration
            processing_time: Total processing time
            
        Returns:
            Aggregated results dictionary
        """
        # Count class predictions
        class_counts = Counter(chunk_predictions)
        total_chunks = len(chunk_predictions)
        
        # Calculate class distribution
        class_distribution = {
            class_name: count / total_chunks 
            for class_name, count in class_counts.items()
        }
        
        # Determine dominant class (majority voting)
        dominant_class = class_counts.most_common(1)[0][0]
        dominant_count = class_counts.most_common(1)[0][1]
        dominant_percentage = dominant_count / total_chunks
        
        # Calculate average confidence for dominant class
        dominant_confidences = [
            conf for pred, conf in zip(chunk_predictions, chunk_confidences)
            if pred == dominant_class
        ]
        dominant_confidence = np.mean(dominant_confidences) if dominant_confidences else 0.0
        
        # Calculate overall confidence (weighted by chunk confidence)
        overall_confidence = np.mean(chunk_confidences)
        
        # Get Arduino command
        arduino_command = self.arduino_mapping.get(dominant_class, 0)
        
        # Calculate average probabilities across all chunks
        all_probs = np.array([chunk['probabilities'] for chunk in chunk_results])
        avg_probabilities = np.mean(all_probs, axis=0)
        
        return {
            'file_path': str(audio_path),
            'file_name': Path(audio_path).name,
            'total_duration': total_duration,
            'processing_time': processing_time,
            'processing_method': 'sliding_window',
            'chunk_duration': self.chunk_duration,
            'overlap_ratio': self.overlap,
            'num_chunks': total_chunks,
            'successful_chunks': len(chunk_results),
            'dominant_class': dominant_class,
            'dominant_confidence': float(dominant_confidence),
            'dominant_percentage': float(dominant_percentage),
            'overall_confidence': float(overall_confidence),
            'arduino_command': arduino_command,
            'class_distribution': class_distribution,
            'chunk_results': chunk_results,
            'all_probabilities': {
                self.class_names[i]: float(prob)
                for i, prob in enumerate(avg_probabilities)
            }
        }

    def process_long_audio(self, audio_path: str) -> Dict:
        """
        Main method to process long audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Processing results dictionary
        """
        return self.process_audio_chunks(audio_path)

    def print_long_audio_results(self, results: Dict):
        """
        Print formatted results for long audio processing.

        Args:
            results: Processing results dictionary
        """
        print("\n" + "="*80)
        print("🎯 LONG AUDIO PROCESSING RESULTS")
        print("="*80)

        print(f"🎵 File: {results['file_name']}")
        print(f"📁 Path: {results['file_path']}")
        print(f"⏱️  Duration: {results['total_duration']:.2f} seconds ({results['total_duration']/60:.1f} minutes)")
        print(f"🔄 Processing Time: {results['processing_time']:.2f} seconds")
        print(f"⚡ Processing Speed: {results['total_duration']/results['processing_time']:.1f}x real-time")

        if results['processing_method'] == 'sliding_window':
            print(f"📊 Chunks: {results['num_chunks']} ({results['chunk_duration']}s each, {results['overlap_ratio']*100:.0f}% overlap)")
            print(f"✅ Successful: {results['successful_chunks']}/{results['num_chunks']} chunks")

        print(f"\n🎯 Dominant Class: {results['dominant_class']}")
        print(f"📊 Dominant Confidence: {results['dominant_confidence']:.3f} ({results['dominant_confidence']*100:.1f}%)")
        print(f"📈 Dominant Percentage: {results['dominant_percentage']*100:.1f}% of chunks")
        print(f"🤖 Arduino Command: {results['arduino_command']}")

        # Arduino motor control explanation
        motor_actions = {
            0: "No vibration (disturbance ignored)",
            1: "Top motor vibrates ('slow' sound)",
            2: "Bottom motor vibrates ('medium' sound)",
            3: "Both motors vibrate ('fast' sound)"
        }
        print(f"🎮 Motor Action: {motor_actions.get(results['arduino_command'], 'Unknown')}")

        print(f"\n📊 Class Distribution:")
        print("-" * 40)
        sorted_distribution = sorted(results['class_distribution'].items(),
                                   key=lambda x: x[1], reverse=True)

        for class_name, percentage in sorted_distribution:
            chunk_count = int(percentage * results['num_chunks'])
            marker = "👉" if class_name == results['dominant_class'] else "  "
            print(f"{marker} {class_name:<12}: {chunk_count:3d} chunks ({percentage*100:.1f}%)")

        print(f"\n📊 Average Class Probabilities:")
        print("-" * 40)
        sorted_probs = sorted(results['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True)

        for class_name, probability in sorted_probs:
            marker = "👉" if class_name == results['dominant_class'] else "  "
            print(f"{marker} {class_name:<12}: {probability:.3f} ({probability*100:.1f}%)")

        print("\n" + "="*80)

def main():
    """
    Main function for long audio processing.
    """
    parser = argparse.ArgumentParser(description="Process Long Audio with YAMNet Classifier")
    parser.add_argument("audio_file", help="Path to long audio file")
    parser.add_argument("--model", default="yamnet_models/yamnet_classifier.h5",
                       help="Path to trained model file")
    parser.add_argument("--metadata", help="Path to model metadata file (optional)")
    parser.add_argument("--chunk-duration", type=float, default=5.0,
                       help="Duration of each chunk in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio between chunks (0.0 to 1.0)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--json-output", help="Save results to JSON file")
    parser.add_argument("--chunk-details", action="store_true",
                       help="Show detailed results for each chunk")

    args = parser.parse_args()

    try:
        # Validate arguments
        if not (0.0 <= args.overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 and 1.0")

        if args.chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")

        # Initialize processor
        processor = LongAudioProcessor(
            args.model, args.metadata,
            args.chunk_duration, args.overlap
        )

        # Process audio file
        results = processor.process_long_audio(args.audio_file)

        # Print results (unless quiet mode)
        if not args.quiet:
            processor.print_long_audio_results(results)

            # Show chunk details if requested
            if args.chunk_details and results['processing_method'] == 'sliding_window':
                print(f"\n📋 Detailed Chunk Results:")
                print("-" * 80)
                for chunk in results['chunk_results'][:10]:  # Show first 10 chunks
                    print(f"Chunk {chunk['chunk_id']:3d}: "
                          f"{chunk['start_time']:6.1f}s-{chunk['end_time']:6.1f}s → "
                          f"{chunk['predicted_class_name']:<12} "
                          f"({chunk['confidence']:.3f})")

                if len(results['chunk_results']) > 10:
                    print(f"... and {len(results['chunk_results']) - 10} more chunks")
        else:
            print(f"File: {results['file_name']}")
            print(f"Duration: {results['total_duration']:.2f} seconds")
            print(f"Dominant Class: {results['dominant_class']}")
            print(f"Confidence: {results['dominant_confidence']:.3f} ({results['dominant_confidence']*100:.1f}%)")
            print(f"Arduino Command: {results['arduino_command']}")

        # Save to JSON if requested
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✅ Results saved to {args.json_output}")

        logger.info("🎉 Long audio processing completed successfully!")

    except Exception as e:
        logger.error(f"❌ Long audio processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
