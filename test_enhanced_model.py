#!/usr/bin/env python3
"""
Test Enhanced YAMNet Model
Compares performance between original and enhanced models
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import json
import time
from pathlib import Path
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class EnhancedModelTester:
    def __init__(self, enhanced_model_path, original_model_path=None):
        self.enhanced_model_path = enhanced_model_path
        self.original_model_path = original_model_path
        self.sample_rate = 16000
        
        # Load YAMNet for feature extraction
        print("📥 Loading YAMNet model...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load enhanced model
        print(f"📥 Loading enhanced model: {enhanced_model_path}")
        self.enhanced_model = tf.keras.models.load_model(enhanced_model_path)
        
        # Load original model if provided
        self.original_model = None
        if original_model_path and os.path.exists(original_model_path):
            print(f"📥 Loading original model: {original_model_path}")
            self.original_model = tf.keras.models.load_model(original_model_path)
        
        # Load metadata
        metadata_path = Path(enhanced_model_path).parent / 'yamnet_model_metadata_enhanced.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                self.class_names = self.metadata['classes']
        else:
            self.class_names = ['slow', 'medium', 'fast', 'disturbance']
        
        print(f"✅ Models loaded. Classes: {self.class_names}")
    
    def extract_features(self, audio_file):
        """Extract YAMNet features from audio file"""
        # Load audio
        audio, _ = librosa.load(audio_file, sr=self.sample_rate)
        
        # Get YAMNet embeddings
        embeddings = self.yamnet_model(audio)
        feature_vector = tf.reduce_mean(embeddings, axis=0)
        
        return tf.expand_dims(feature_vector, 0)
    
    def predict_enhanced(self, audio_file):
        """Predict using enhanced model"""
        features = self.extract_features(audio_file)
        
        start_time = time.time()
        prediction = self.enhanced_model.predict(features, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prediction[0].tolist(),
            'inference_time_ms': inference_time
        }
    
    def predict_original(self, audio_file):
        """Predict using original model"""
        if self.original_model is None:
            return None
        
        features = self.extract_features(audio_file)
        
        start_time = time.time()
        prediction = self.original_model.predict(features, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prediction[0].tolist(),
            'inference_time_ms': inference_time
        }
    
    def test_single_file(self, audio_file):
        """Test single audio file"""
        print(f"\n🎵 Testing: {os.path.basename(audio_file)}")
        print("-" * 50)
        
        # Enhanced model prediction
        enhanced_result = self.predict_enhanced(audio_file)
        print(f"🚀 Enhanced Model:")
        print(f"   Predicted: {enhanced_result['predicted_class']}")
        print(f"   Confidence: {enhanced_result['confidence']:.3f} ({enhanced_result['confidence']*100:.1f}%)")
        print(f"   Inference Time: {enhanced_result['inference_time_ms']:.1f}ms")
        
        # Original model prediction (if available)
        if self.original_model is not None:
            original_result = self.predict_original(audio_file)
            print(f"📊 Original Model:")
            print(f"   Predicted: {original_result['predicted_class']}")
            print(f"   Confidence: {original_result['confidence']:.3f} ({original_result['confidence']*100:.1f}%)")
            print(f"   Inference Time: {original_result['inference_time_ms']:.1f}ms")
            
            # Comparison
            if enhanced_result['predicted_class'] == original_result['predicted_class']:
                print("✅ Both models agree")
            else:
                print("⚠️  Models disagree")
            
            confidence_diff = enhanced_result['confidence'] - original_result['confidence']
            if confidence_diff > 0:
                print(f"📈 Enhanced model +{confidence_diff:.3f} confidence")
            else:
                print(f"📉 Enhanced model {confidence_diff:.3f} confidence")
        
        return enhanced_result
    
    def test_dataset(self, dataset_dir):
        """Test entire dataset"""
        print(f"\n📊 Testing dataset: {dataset_dir}")
        print("=" * 60)
        
        results = {
            'enhanced': {'correct': 0, 'total': 0, 'times': []},
            'original': {'correct': 0, 'total': 0, 'times': []} if self.original_model else None
        }
        
        class_results = {class_name: {'correct': 0, 'total': 0} for class_name in self.class_names}
        
        for class_dir in Path(dataset_dir).iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_names:
                true_class = class_dir.name
                print(f"\n📁 Testing class: {true_class}")
                
                for audio_file in class_dir.glob("*.wav"):
                    # Enhanced model
                    enhanced_result = self.predict_enhanced(str(audio_file))
                    results['enhanced']['total'] += 1
                    results['enhanced']['times'].append(enhanced_result['inference_time_ms'])
                    
                    class_results[true_class]['total'] += 1
                    
                    if enhanced_result['predicted_class'] == true_class:
                        results['enhanced']['correct'] += 1
                        class_results[true_class]['correct'] += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    print(f"   {status} {audio_file.name}: {enhanced_result['predicted_class']} ({enhanced_result['confidence']:.3f})")
                    
                    # Original model
                    if self.original_model is not None:
                        original_result = self.predict_original(str(audio_file))
                        results['original']['total'] += 1
                        results['original']['times'].append(original_result['inference_time_ms'])
                        
                        if original_result['predicted_class'] == true_class:
                            results['original']['correct'] += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TESTING SUMMARY")
        print("=" * 60)
        
        # Enhanced model results
        enhanced_accuracy = results['enhanced']['correct'] / results['enhanced']['total']
        enhanced_avg_time = np.mean(results['enhanced']['times'])
        
        print(f"🚀 Enhanced Model:")
        print(f"   Accuracy: {enhanced_accuracy:.3f} ({enhanced_accuracy*100:.1f}%)")
        print(f"   Correct: {results['enhanced']['correct']}/{results['enhanced']['total']}")
        print(f"   Avg Inference Time: {enhanced_avg_time:.1f}ms")
        
        # Original model results
        if results['original'] is not None:
            original_accuracy = results['original']['correct'] / results['original']['total']
            original_avg_time = np.mean(results['original']['times'])
            
            print(f"📊 Original Model:")
            print(f"   Accuracy: {original_accuracy:.3f} ({original_accuracy*100:.1f}%)")
            print(f"   Correct: {results['original']['correct']}/{results['original']['total']}")
            print(f"   Avg Inference Time: {original_avg_time:.1f}ms")
            
            # Comparison
            accuracy_improvement = enhanced_accuracy - original_accuracy
            time_difference = enhanced_avg_time - original_avg_time
            
            print(f"\n📈 Improvement:")
            print(f"   Accuracy: {accuracy_improvement:+.3f} ({accuracy_improvement*100:+.1f}%)")
            print(f"   Speed: {time_difference:+.1f}ms")
        
        # Per-class results
        print(f"\n📋 Per-Class Results (Enhanced Model):")
        for class_name in self.class_names:
            if class_results[class_name]['total'] > 0:
                class_accuracy = class_results[class_name]['correct'] / class_results[class_name]['total']
                print(f"   {class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) - {class_results[class_name]['correct']}/{class_results[class_name]['total']}")
        
        return results
    
    def benchmark_performance(self, num_iterations=100):
        """Benchmark inference performance"""
        print(f"\n⚡ Performance Benchmark ({num_iterations} iterations)")
        print("-" * 50)
        
        # Create dummy audio for benchmarking
        dummy_audio = np.random.randn(self.sample_rate * 5)  # 5 seconds
        dummy_file = "/tmp/benchmark_audio.wav"
        
        import soundfile as sf
        sf.write(dummy_file, dummy_audio, self.sample_rate)
        
        # Benchmark enhanced model
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.predict_enhanced(dummy_file)
            times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"🚀 Enhanced Model Performance:")
        print(f"   Average: {avg_time:.1f}ms ± {std_time:.1f}ms")
        print(f"   Range: {min_time:.1f}ms - {max_time:.1f}ms")
        print(f"   Throughput: {1000/avg_time:.1f} inferences/second")
        
        # Clean up
        os.remove(dummy_file)
        
        return avg_time

def main():
    parser = argparse.ArgumentParser(description='Test Enhanced YAMNet Model')
    parser.add_argument('enhanced_model', help='Path to enhanced model (.h5 file)')
    parser.add_argument('--original-model', help='Path to original model for comparison')
    parser.add_argument('--test-file', help='Single audio file to test')
    parser.add_argument('--test-dataset', help='Dataset directory to test')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.enhanced_model):
        print(f"❌ Enhanced model not found: {args.enhanced_model}")
        sys.exit(1)
    
    # Initialize tester
    tester = EnhancedModelTester(args.enhanced_model, args.original_model)
    
    # Run tests
    if args.test_file:
        if os.path.exists(args.test_file):
            tester.test_single_file(args.test_file)
        else:
            print(f"❌ Test file not found: {args.test_file}")
    
    if args.test_dataset:
        if os.path.exists(args.test_dataset):
            tester.test_dataset(args.test_dataset)
        else:
            print(f"❌ Test dataset not found: {args.test_dataset}")
    
    if args.benchmark:
        tester.benchmark_performance()
    
    if not any([args.test_file, args.test_dataset, args.benchmark]):
        print("ℹ️  No test specified. Use --test-file, --test-dataset, or --benchmark")
        print("Example: python3 test_enhanced_model.py model.h5 --test-dataset ../dataset --benchmark")

if __name__ == "__main__":
    main()
