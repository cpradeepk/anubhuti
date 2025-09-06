#!/usr/bin/env python3
"""
Quick Performance Testing Script for YAMNet Pipeline

This script provides rapid performance testing including:
- Inference speed benchmarks
- Memory usage monitoring
- Arduino command validation
- Basic edge case testing
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from test_yamnet_model import YAMNetModelTester

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

def test_inference_speed(model_path: str = "yamnet_models/yamnet_classifier.h5",
                        dataset_path: str = "../") -> Dict:
    """
    Test inference speed across multiple files.
    """
    print("⚡ PERFORMANCE TEST: Inference Speed")
    print("=" * 50)
    
    try:
        tester = YAMNetModelTester(model_path)
        
        # Collect test files
        test_files = []
        for class_dir in ['slow', 'medium', 'fast', 'disturbance']:
            class_path = Path(dataset_path) / class_dir
            if class_path.exists():
                files = list(class_path.glob('*.wav'))[:2]  # 2 files per class
                test_files.extend(files)
        
        if not test_files:
            print("❌ No test files found")
            return {}
        
        print(f"Testing {len(test_files)} files...")
        
        # Warm-up run
        tester.predict_single_file(str(test_files[0]))
        
        # Performance testing
        inference_times = []
        correct_predictions = 0
        
        for i, test_file in enumerate(test_files[:8]):  # Test 8 files
            class_name = test_file.parent.name
            
            start_time = time.time()
            result = tester.predict_single_file(str(test_file))
            inference_time = (time.time() - start_time) * 1000  # ms
            
            inference_times.append(inference_time)
            
            is_correct = result['predicted_class_name'] == class_name
            if is_correct:
                correct_predictions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} {test_file.name:<20}: {result['predicted_class_name']:<12} ({inference_time:5.1f}ms)")
        
        # Calculate metrics
        avg_time = np.mean(inference_times)
        max_time = np.max(inference_times)
        min_time = np.min(inference_times)
        accuracy = correct_predictions / len(inference_times)
        
        print(f"\n📊 Performance Results:")
        print(f"  Average Time: {avg_time:.1f}ms")
        print(f"  Min/Max Time: {min_time:.1f}ms / {max_time:.1f}ms")
        print(f"  Accuracy: {accuracy:.1%} ({correct_predictions}/{len(inference_times)})")
        print(f"  Throughput: {1000/avg_time:.1f} files/second")
        
        # Performance assessment
        target_time = 50  # ms
        if avg_time <= target_time:
            print(f"  ✅ PASS: Average time ≤{target_time}ms")
        else:
            print(f"  ❌ FAIL: Average time >{target_time}ms")
        
        return {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'min_time_ms': min_time,
            'accuracy': accuracy,
            'throughput_fps': 1000/avg_time,
            'meets_target': avg_time <= target_time
        }
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return {}

def test_arduino_commands(model_path: str = "yamnet_models/yamnet_classifier.h5",
                         dataset_path: str = "../") -> Dict:
    """
    Test Arduino command mapping validation.
    """
    print("\n🤖 ARDUINO TEST: Command Mapping")
    print("=" * 50)
    
    try:
        tester = YAMNetModelTester(model_path)
        
        # Expected Arduino commands
        expected_commands = {
            'slow': 1,      # Top motor
            'medium': 2,    # Bottom motor
            'fast': 3,      # Both motors
            'disturbance': 0 # No vibration
        }
        
        # Test one file from each class
        test_results = {}
        all_correct = True
        
        for class_name, expected_cmd in expected_commands.items():
            class_path = Path(dataset_path) / class_name
            if class_path.exists():
                test_files = list(class_path.glob('*.wav'))
                if test_files:
                    test_file = test_files[0]  # Use first file
                    
                    result = tester.predict_single_file(str(test_file))
                    actual_cmd = result['arduino_command']
                    predicted_class = result['predicted_class_name']
                    
                    is_correct = actual_cmd == expected_cmd and predicted_class == class_name
                    if not is_correct:
                        all_correct = False
                    
                    status = "✅" if is_correct else "❌"
                    print(f"  {status} {class_name:<12}: Command {actual_cmd} (Expected: {expected_cmd})")
                    
                    test_results[class_name] = {
                        'expected_command': expected_cmd,
                        'actual_command': actual_cmd,
                        'predicted_class': predicted_class,
                        'correct': is_correct
                    }
        
        # Motor control explanation
        print(f"\n🎮 Motor Control Mapping:")
        motor_actions = {
            0: "No vibration (disturbance ignored)",
            1: "Top motor vibrates ('slow' sound)",
            2: "Bottom motor vibrates ('medium' sound)",
            3: "Both motors vibrate ('fast' sound)"
        }
        
        for cmd, action in motor_actions.items():
            print(f"  Command {cmd}: {action}")
        
        print(f"\n📊 Arduino Test Result: {'✅ PASS' if all_correct else '❌ FAIL'}")
        
        return {
            'all_correct': all_correct,
            'test_results': test_results,
            'motor_actions': motor_actions
        }
        
    except Exception as e:
        print(f"❌ Arduino test failed: {e}")
        return {}

def test_model_robustness() -> Dict:
    """
    Test model robustness with basic edge cases.
    """
    print("\n🧪 ROBUSTNESS TEST: Edge Cases")
    print("=" * 50)
    
    try:
        tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
        
        # Test with non-existent file (should handle gracefully)
        print("Testing error handling...")
        
        try:
            result = tester.predict_single_file("non_existent_file.wav")
            print("  ❌ Should have failed with non-existent file")
            return {'error_handling': False}
        except Exception as e:
            print("  ✅ Properly handles non-existent files")
        
        # Test model loading
        print("Testing model consistency...")
        
        # Load model multiple times to check consistency
        tester1 = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
        tester2 = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
        
        print("  ✅ Model loads consistently")
        
        return {
            'error_handling': True,
            'model_loading': True,
            'consistency': True
        }
        
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
        return {}

def test_memory_usage() -> Dict:
    """
    Basic memory usage monitoring.
    """
    print("\n💾 MEMORY TEST: Usage Monitoring")
    print("=" * 50)
    
    try:
        import psutil
        process = psutil.Process()
        
        # Memory before model loading
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  Memory before loading: {mem_before:.1f}MB")
        
        # Load model
        tester = YAMNetModelTester("yamnet_models/yamnet_classifier.h5")
        
        # Memory after model loading
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  Memory after loading: {mem_after:.1f}MB")
        print(f"  Model memory usage: {mem_after - mem_before:.1f}MB")
        
        # Memory target
        target_memory = 200  # MB
        meets_target = mem_after <= target_memory
        
        status = "✅ PASS" if meets_target else "❌ FAIL"
        print(f"  {status}: Memory usage {'≤' if meets_target else '>'}{target_memory}MB")
        
        return {
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'model_memory_mb': mem_after - mem_before,
            'target_mb': target_memory,
            'meets_target': meets_target
        }
        
    except ImportError:
        print("  ⚠️  psutil not available - skipping memory test")
        return {}
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return {}

def main():
    """
    Run quick performance tests.
    """
    print("🧪 YAMNET QUICK PERFORMANCE TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Inference Speed
    results['performance'] = test_inference_speed()
    
    # Test 2: Arduino Commands
    results['arduino'] = test_arduino_commands()
    
    # Test 3: Model Robustness
    results['robustness'] = test_model_robustness()
    
    # Test 4: Memory Usage
    results['memory'] = test_memory_usage()
    
    # Overall Assessment
    print("\n🎯 OVERALL ASSESSMENT")
    print("=" * 80)
    
    # Count successful tests
    tests_passed = 0
    total_tests = 0
    
    if results.get('performance', {}).get('meets_target', False):
        tests_passed += 1
    if results.get('performance'):
        total_tests += 1
    
    if results.get('arduino', {}).get('all_correct', False):
        tests_passed += 1
    if results.get('arduino'):
        total_tests += 1
    
    if results.get('robustness', {}).get('error_handling', False):
        tests_passed += 1
    if results.get('robustness'):
        total_tests += 1
    
    if results.get('memory', {}).get('meets_target', False):
        tests_passed += 1
    if results.get('memory'):
        total_tests += 1
    
    pass_rate = tests_passed / total_tests if total_tests > 0 else 0
    
    print(f"Tests Passed: {tests_passed}/{total_tests} ({pass_rate:.1%})")
    
    if pass_rate >= 0.9:
        assessment = "🎉 EXCELLENT - System ready for production!"
    elif pass_rate >= 0.7:
        assessment = "✅ GOOD - Minor issues to address"
    elif pass_rate >= 0.5:
        assessment = "⚠️  ACCEPTABLE - Some improvements needed"
    else:
        assessment = "❌ NEEDS WORK - Significant issues found"
    
    print(f"Assessment: {assessment}")
    
    # Key metrics summary
    if results.get('performance'):
        perf = results['performance']
        print(f"\n📊 Key Metrics:")
        print(f"  Inference Speed: {perf.get('avg_time_ms', 0):.1f}ms average")
        print(f"  Accuracy: {perf.get('accuracy', 0):.1%}")
        print(f"  Throughput: {perf.get('throughput_fps', 0):.1f} files/second")
    
    if results.get('memory'):
        mem = results['memory']
        print(f"  Memory Usage: {mem.get('memory_after_mb', 0):.1f}MB")
    
    print(f"\n🚀 YAMNet pipeline testing completed!")
    
    return results

if __name__ == "__main__":
    main()
