#!/usr/bin/env python3
"""
Manual Testing Script for YAMNet Pipeline

This script provides interactive manual testing with detailed results.
Run this to manually validate your YAMNet model performance.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import logging

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from test_yamnet_model import YAMNetModelTester

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)

class ManualTester:
    """Interactive manual testing for YAMNet pipeline."""
    
    def __init__(self, model_path: str = "yamnet_models/yamnet_classifier.h5"):
        self.model_path = model_path
        self.tester = YAMNetModelTester(model_path)
        self.test_results = []
        
    def test_single_file(self, file_path: str, expected_class: str = None) -> Dict:
        """Test a single audio file with detailed output."""
        print(f"\n🔄 Testing: {Path(file_path).name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = self.tester.predict_single_file(file_path)
            inference_time = (time.time() - start_time) * 1000
            
            # Display results
            print(f"📁 File: {result['file_name']}")
            print(f"⏱️  Duration: {result['audio_metadata']['duration_seconds']:.2f}s")
            print(f"🎯 Predicted Class: {result['predicted_class_name']}")
            print(f"📊 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
            print(f"🤖 Arduino Command: {result['arduino_command']}")
            print(f"⚡ Inference Time: {inference_time:.1f}ms")
            
            # Motor action explanation
            motor_actions = {
                0: "No vibration (disturbance ignored)",
                1: "Top motor vibrates ('slow' sound)",
                2: "Bottom motor vibrates ('medium' sound)",
                3: "Both motors vibrate ('fast' sound)"
            }
            print(f"🎮 Motor Action: {motor_actions.get(result['arduino_command'], 'Unknown')}")
            
            # Show all class probabilities
            print(f"\n📊 All Class Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for class_name, probability in sorted_probs:
                marker = "👉" if class_name == result['predicted_class_name'] else "  "
                print(f"{marker} {class_name:<12}: {probability:.3f} ({probability*100:.1f}%)")
            
            # Validation if expected class provided
            if expected_class:
                is_correct = result['predicted_class_name'] == expected_class
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"\n🎯 Validation: {status}")
                print(f"   Expected: {expected_class}")
                print(f"   Predicted: {result['predicted_class_name']}")
            
            # Store result
            test_result = {
                'file_path': file_path,
                'file_name': result['file_name'],
                'predicted_class': result['predicted_class_name'],
                'confidence': result['confidence'],
                'arduino_command': result['arduino_command'],
                'inference_time_ms': inference_time,
                'expected_class': expected_class,
                'correct': result['predicted_class_name'] == expected_class if expected_class else None
            }
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            print(f"❌ Error testing {file_path}: {e}")
            return {}
    
    def test_class_samples(self, class_name: str, dataset_path: str = "../", max_files: int = 3):
        """Test multiple samples from a specific class."""
        print(f"\n🎯 TESTING CLASS: {class_name.upper()}")
        print("=" * 60)
        
        class_path = Path(dataset_path) / class_name
        if not class_path.exists():
            print(f"❌ Class directory not found: {class_path}")
            return
        
        # Get audio files
        audio_files = list(class_path.glob('*.wav'))[:max_files]
        if not audio_files:
            print(f"❌ No audio files found in {class_path}")
            return
        
        print(f"Found {len(audio_files)} files to test...")
        
        correct_predictions = 0
        total_files = len(audio_files)
        
        for audio_file in audio_files:
            result = self.test_single_file(str(audio_file), class_name)
            if result.get('correct'):
                correct_predictions += 1
        
        # Class summary
        accuracy = correct_predictions / total_files if total_files > 0 else 0
        print(f"\n📊 CLASS SUMMARY: {class_name}")
        print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{total_files})")
        print(f"   Status: {'✅ PASS' if accuracy >= 0.7 else '❌ FAIL'}")
    
    def test_all_classes(self, dataset_path: str = "../", files_per_class: int = 2):
        """Test samples from all classes."""
        print("\n🧪 COMPREHENSIVE CLASS TESTING")
        print("=" * 80)
        
        classes = ['slow', 'medium', 'fast', 'disturbance']
        
        for class_name in classes:
            self.test_class_samples(class_name, dataset_path, files_per_class)
        
        # Overall summary
        self.print_overall_summary()
    
    def test_arduino_commands(self, dataset_path: str = "../"):
        """Test Arduino command mapping specifically."""
        print("\n🤖 ARDUINO COMMAND TESTING")
        print("=" * 60)
        
        # Expected commands
        expected_commands = {
            'slow': 1,
            'medium': 2,
            'fast': 3,
            'disturbance': 0
        }
        
        command_results = {}
        
        for class_name, expected_cmd in expected_commands.items():
            class_path = Path(dataset_path) / class_name
            if class_path.exists():
                audio_files = list(class_path.glob('*.wav'))
                if audio_files:
                    # Test first file from class
                    test_file = audio_files[0]
                    result = self.tester.predict_single_file(str(test_file))
                    
                    actual_cmd = result['arduino_command']
                    predicted_class = result['predicted_class_name']
                    
                    is_correct = (actual_cmd == expected_cmd and 
                                predicted_class == class_name)
                    
                    status = "✅" if is_correct else "❌"
                    print(f"{status} {class_name:<12}: Command {actual_cmd} (Expected: {expected_cmd})")
                    
                    command_results[class_name] = {
                        'expected': expected_cmd,
                        'actual': actual_cmd,
                        'correct': is_correct
                    }
        
        # Command mapping summary
        all_correct = all(result['correct'] for result in command_results.values())
        print(f"\n🎯 Arduino Command Test: {'✅ PASS' if all_correct else '❌ FAIL'}")
        
        return command_results
    
    def test_performance_benchmark(self, dataset_path: str = "../", num_files: int = 10):
        """Test inference speed performance."""
        print("\n⚡ PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Collect test files
        test_files = []
        for class_dir in ['slow', 'medium', 'fast', 'disturbance']:
            class_path = Path(dataset_path) / class_dir
            if class_path.exists():
                files = list(class_path.glob('*.wav'))[:3]
                test_files.extend(files)
        
        test_files = test_files[:num_files]
        
        if not test_files:
            print("❌ No test files found for performance testing")
            return
        
        print(f"Testing inference speed with {len(test_files)} files...")
        
        # Warm-up run
        self.tester.predict_single_file(str(test_files[0]))
        
        # Performance testing
        inference_times = []
        
        for i, test_file in enumerate(test_files):
            start_time = time.time()
            result = self.tester.predict_single_file(str(test_file))
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            print(f"  File {i+1:2d}: {inference_time:5.1f}ms - {test_file.name}")
        
        # Calculate metrics
        import numpy as np
        avg_time = np.mean(inference_times)
        max_time = np.max(inference_times)
        min_time = np.min(inference_times)
        
        print(f"\n📊 Performance Results:")
        print(f"   Average Time: {avg_time:.1f}ms")
        print(f"   Min/Max Time: {min_time:.1f}ms / {max_time:.1f}ms")
        print(f"   Throughput: {1000/avg_time:.1f} files/second")
        
        # Performance assessment
        target_time = 50  # ms
        meets_target = avg_time <= target_time
        status = "✅ PASS" if meets_target else "⚠️  SLOW"
        print(f"   {status}: Average time {'≤' if meets_target else '>'}{target_time}ms target")
        
        return {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'min_time_ms': min_time,
            'meets_target': meets_target
        }
    
    def print_overall_summary(self):
        """Print overall testing summary."""
        if not self.test_results:
            return
        
        print("\n📊 OVERALL TESTING SUMMARY")
        print("=" * 80)
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        correct_tests = sum(1 for r in self.test_results if r.get('correct'))
        overall_accuracy = correct_tests / total_tests if total_tests > 0 else 0
        
        avg_confidence = np.mean([r['confidence'] for r in self.test_results])
        avg_inference_time = np.mean([r['inference_time_ms'] for r in self.test_results])
        
        print(f"Total Tests: {total_tests}")
        print(f"Correct Predictions: {correct_tests}")
        print(f"Overall Accuracy: {overall_accuracy:.1%}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Inference Time: {avg_inference_time:.1f}ms")
        
        # Assessment
        if overall_accuracy >= 0.9:
            assessment = "🎉 EXCELLENT - Ready for production!"
        elif overall_accuracy >= 0.7:
            assessment = "✅ GOOD - Minor improvements possible"
        elif overall_accuracy >= 0.5:
            assessment = "⚠️  ACCEPTABLE - Needs improvement"
        else:
            assessment = "❌ POOR - Requires significant work"
        
        print(f"\nAssessment: {assessment}")
    
    def save_results(self, output_file: str = "manual_test_results.json"):
        """Save test results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\n💾 Test results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Interactive manual testing interface."""
    print("🧪 YAMNET MANUAL TESTING INTERFACE")
    print("=" * 80)
    
    tester = ManualTester()
    
    while True:
        print("\n📋 MANUAL TESTING OPTIONS:")
        print("1. Test single audio file")
        print("2. Test all classes (2 files each)")
        print("3. Test specific class")
        print("4. Test Arduino commands")
        print("5. Performance benchmark")
        print("6. View test summary")
        print("7. Save results and exit")
        print("0. Exit")
        
        choice = input("\nSelect option (0-7): ").strip()
        
        if choice == '1':
            file_path = input("Enter audio file path: ").strip()
            expected = input("Expected class (optional): ").strip() or None
            tester.test_single_file(file_path, expected)
            
        elif choice == '2':
            tester.test_all_classes()
            
        elif choice == '3':
            class_name = input("Enter class name (slow/medium/fast/disturbance): ").strip()
            tester.test_class_samples(class_name)
            
        elif choice == '4':
            tester.test_arduino_commands()
            
        elif choice == '5':
            tester.test_performance_benchmark()
            
        elif choice == '6':
            tester.print_overall_summary()
            
        elif choice == '7':
            tester.save_results()
            break
            
        elif choice == '0':
            break
            
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()
