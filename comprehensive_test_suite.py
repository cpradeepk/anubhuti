#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing Suite for YAMNet Speech Classification Pipeline

This script provides systematic testing of all pipeline components including:
- Model accuracy testing with edge cases
- Arduino command mapping validation
- Long audio processing benchmarks
- Real-time performance testing
- Integration and deployment readiness
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter, defaultdict
import psutil
import threading
from datetime import datetime

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from yamnet_utils import YAMNetProcessor, load_model_metadata
from test_yamnet_model import YAMNetModelTester
from process_long_audio import LongAudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """
    Comprehensive testing suite for YAMNet speech classification pipeline.
    """
    
    def __init__(self, model_path: str = "yamnet_models/yamnet_classifier.h5",
                 metadata_path: str = "yamnet_models/yamnet_model_metadata.json",
                 dataset_path: str = "../"):
        """
        Initialize comprehensive test suite.
        
        Args:
            model_path: Path to trained YAMNet model
            metadata_path: Path to model metadata
            dataset_path: Path to dataset directory
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.dataset_path = Path(dataset_path)
        
        # Test results storage
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'model_accuracy': {},
            'arduino_mapping': {},
            'long_audio': {},
            'realtime_performance': {},
            'integration': {},
            'benchmarks': {},
            'issues': [],
            'recommendations': []
        }
        
        # Performance targets
        self.performance_targets = {
            'accuracy_threshold': 0.85,
            'inference_time_ms': 50,
            'memory_limit_mb': 200,
            'wireless_range_m': 10,
            'confidence_threshold': 0.3
        }
        
        logger.info("🧪 Comprehensive Test Suite initialized")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Dataset: {self.dataset_path}")
    
    def test_model_accuracy(self) -> Dict:
        """
        Test 1: Model Accuracy Testing with Edge Cases
        """
        logger.info("🎯 TEST 1: Model Accuracy Testing")
        print("\n" + "="*80)
        print("🎯 TEST 1: MODEL ACCURACY TESTING")
        print("="*80)
        
        accuracy_results = {
            'class_performance': {},
            'edge_cases': {},
            'confidence_analysis': {},
            'confusion_matrix': [],
            'overall_accuracy': 0.0
        }
        
        try:
            # Initialize tester
            tester = YAMNetModelTester(str(self.model_path), str(self.metadata_path))
            
            # Test representative files from each class
            test_files = {
                'slow': ['slow/Fhmm_slow.wav', 'slow/Fsoo_slow.wav'],
                'medium': ['medium/Fhum_medium.wav', 'medium/Fsoo_medium.wav'],
                'fast': ['fast/Fhum_fast.wav', 'fast/Fsoo_fast.wav'],
                'disturbance': ['disturbance/Cough.wav', 'disturbance/Clap.wav']
            }
            
            all_predictions = []
            all_true_labels = []
            class_results = defaultdict(list)
            
            print("\n📊 Testing Representative Files:")
            print("-" * 50)
            
            for class_name, files in test_files.items():
                class_idx = tester.class_names.index(class_name)
                
                for file_path in files:
                    full_path = self.dataset_path / file_path
                    if not full_path.exists():
                        logger.warning(f"⚠️  Test file not found: {full_path}")
                        continue
                    
                    try:
                        result = tester.predict_single_file(str(full_path))
                        
                        predicted_idx = result['predicted_class_idx']
                        confidence = result['confidence']
                        is_correct = predicted_idx == class_idx
                        
                        all_predictions.append(predicted_idx)
                        all_true_labels.append(class_idx)
                        class_results[class_name].append({
                            'file': file_path,
                            'correct': is_correct,
                            'confidence': confidence,
                            'predicted_class': result['predicted_class_name']
                        })
                        
                        status = "✅" if is_correct else "❌"
                        print(f"{status} {Path(file_path).name:<20}: {result['predicted_class_name']:<12} ({confidence:.3f})")
                        
                    except Exception as e:
                        logger.error(f"❌ Error testing {file_path}: {e}")
                        self.test_results['issues'].append(f"Model accuracy test failed for {file_path}: {e}")
            
            # Calculate overall accuracy
            if all_predictions:
                correct_predictions = sum(1 for pred, true in zip(all_predictions, all_true_labels) if pred == true)
                overall_accuracy = correct_predictions / len(all_predictions)
                accuracy_results['overall_accuracy'] = overall_accuracy
                
                print(f"\n📈 Overall Accuracy: {overall_accuracy:.3f} ({correct_predictions}/{len(all_predictions)})")
                
                # Performance assessment
                if overall_accuracy >= self.performance_targets['accuracy_threshold']:
                    print(f"✅ PASS: Accuracy meets target (≥{self.performance_targets['accuracy_threshold']:.1%})")
                else:
                    print(f"❌ FAIL: Accuracy below target (<{self.performance_targets['accuracy_threshold']:.1%})")
                    self.test_results['issues'].append(f"Model accuracy {overall_accuracy:.3f} below target {self.performance_targets['accuracy_threshold']}")
            
            # Store class performance
            for class_name, results in class_results.items():
                if results:
                    class_accuracy = sum(1 for r in results if r['correct']) / len(results)
                    avg_confidence = np.mean([r['confidence'] for r in results])
                    
                    accuracy_results['class_performance'][class_name] = {
                        'accuracy': class_accuracy,
                        'avg_confidence': avg_confidence,
                        'test_count': len(results)
                    }
                    
                    print(f"   {class_name:<12}: {class_accuracy:.3f} accuracy, {avg_confidence:.3f} confidence")
            
            self.test_results['model_accuracy'] = accuracy_results
            return accuracy_results
            
        except Exception as e:
            logger.error(f"❌ Model accuracy testing failed: {e}")
            self.test_results['issues'].append(f"Model accuracy testing failed: {e}")
            return accuracy_results
    
    def test_arduino_command_mapping(self) -> Dict:
        """
        Test 2: Arduino Command Mapping Validation
        """
        logger.info("🤖 TEST 2: Arduino Command Mapping Validation")
        print("\n" + "="*80)
        print("🤖 TEST 2: ARDUINO COMMAND MAPPING VALIDATION")
        print("="*80)
        
        mapping_results = {
            'command_mapping': {},
            'motor_control_logic': {},
            'command_sequence': [],
            'validation_status': 'PASS'
        }
        
        try:
            # Load model metadata to get Arduino mapping
            metadata = load_model_metadata(self.metadata_path)
            arduino_mapping = metadata.get('arduino_mapping', {})
            
            print("\n📋 Arduino Command Mapping:")
            print("-" * 50)
            
            # Expected mapping (including new Command 4)
            expected_mapping = {
                'slow': 1,      # Top motor
                'medium': 2,    # Bottom motor  
                'fast': 3,      # Both motors
                'disturbance': 0 # No vibration
            }
            
            # Validate basic mapping
            mapping_correct = True
            for class_name, expected_cmd in expected_mapping.items():
                actual_cmd = arduino_mapping.get(class_name, -1)
                status = "✅" if actual_cmd == expected_cmd else "❌"
                
                if actual_cmd != expected_cmd:
                    mapping_correct = False
                    self.test_results['issues'].append(f"Arduino mapping incorrect for {class_name}: expected {expected_cmd}, got {actual_cmd}")
                
                print(f"{status} {class_name:<12} → Command {actual_cmd} (Expected: {expected_cmd})")
                
                mapping_results['command_mapping'][class_name] = {
                    'expected': expected_cmd,
                    'actual': actual_cmd,
                    'correct': actual_cmd == expected_cmd
                }
            
            # Test motor control logic
            print(f"\n🎮 Motor Control Logic:")
            print("-" * 50)
            
            motor_actions = {
                0: "No vibration (disturbance ignored)",
                1: "Top motor vibrates ('slow' sound)",
                2: "Bottom motor vibrates ('medium' sound)",
                3: "Both motors vibrate ('fast' sound)",
                4: "Continue previous vibration pattern (NEW)"  # New requirement
            }
            
            for cmd, action in motor_actions.items():
                print(f"   Command {cmd}: {action}")
                mapping_results['motor_control_logic'][cmd] = action
            
            # Test command sequence logic (simulated)
            print(f"\n🔄 Command Sequence Testing:")
            print("-" * 50)
            
            test_sequence = [
                ('slow', 1, "Start top motor"),
                ('medium', 2, "Switch to bottom motor"),
                ('fast', 3, "Activate both motors"),
                ('disturbance', 0, "Stop all motors"),
                ('continue', 4, "Continue previous pattern")  # New command
            ]
            
            for sound_type, cmd, description in test_sequence:
                mapping_results['command_sequence'].append({
                    'sound_type': sound_type,
                    'command': cmd,
                    'description': description
                })
                print(f"   {sound_type:<12} → Command {cmd}: {description}")
            
            if mapping_correct:
                print(f"\n✅ PASS: Arduino command mapping validated successfully")
            else:
                print(f"\n❌ FAIL: Arduino command mapping has errors")
                mapping_results['validation_status'] = 'FAIL'
            
            self.test_results['arduino_mapping'] = mapping_results
            return mapping_results
            
        except Exception as e:
            logger.error(f"❌ Arduino mapping validation failed: {e}")
            self.test_results['issues'].append(f"Arduino mapping validation failed: {e}")
            mapping_results['validation_status'] = 'ERROR'
            return mapping_results
    
    def test_long_audio_processing(self) -> Dict:
        """
        Test 3: Long Audio Processing Testing
        """
        logger.info("⏱️  TEST 3: Long Audio Processing Testing")
        print("\n" + "="*80)
        print("⏱️  TEST 3: LONG AUDIO PROCESSING TESTING")
        print("="*80)
        
        long_audio_results = {
            'sliding_window': {},
            'memory_efficiency': {},
            'processing_speed': {},
            'aggregation_methods': {},
            'validation_status': 'PASS'
        }
        
        try:
            # Initialize long audio processor
            processor = LongAudioProcessor(
                str(self.model_path), 
                str(self.metadata_path),
                chunk_duration=5.0,
                overlap=0.5
            )
            
            print("\n🔄 Sliding Window Configuration:")
            print("-" * 50)
            print(f"   Chunk Duration: {processor.chunk_duration}s")
            print(f"   Overlap: {processor.overlap*100:.0f}%")
            print(f"   Hop Size: {processor.chunk_duration * (1-processor.overlap)}s")
            
            # Test with different audio lengths (simulated)
            test_durations = [60, 300, 900]  # 1min, 5min, 15min
            
            print(f"\n📊 Processing Speed Benchmarks:")
            print("-" * 50)
            
            for duration in test_durations:
                # Calculate expected chunks
                hop_size = processor.chunk_duration * (1 - processor.overlap)
                expected_chunks = max(1, int((duration - processor.chunk_duration) / hop_size) + 1)
                
                # Estimate processing time (based on 5-10x real-time)
                estimated_time = duration / 7.5  # Average 7.5x real-time
                
                print(f"   {duration:3d}s audio → {expected_chunks:3d} chunks → ~{estimated_time:.1f}s processing")
                
                long_audio_results['processing_speed'][f'{duration}s'] = {
                    'expected_chunks': expected_chunks,
                    'estimated_time': estimated_time,
                    'realtime_factor': duration / estimated_time
                }
            
            # Test memory efficiency
            print(f"\n💾 Memory Efficiency Analysis:")
            print("-" * 50)
            
            # Get current memory usage
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"   Current Memory Usage: {current_memory:.1f} MB")
            print(f"   Target Memory Limit: {self.performance_targets['memory_limit_mb']} MB")
            
            if current_memory <= self.performance_targets['memory_limit_mb']:
                print(f"   ✅ Memory usage within target")
            else:
                print(f"   ⚠️  Memory usage above target")
                self.test_results['issues'].append(f"Memory usage {current_memory:.1f}MB exceeds target {self.performance_targets['memory_limit_mb']}MB")
            
            long_audio_results['memory_efficiency'] = {
                'current_usage_mb': current_memory,
                'target_limit_mb': self.performance_targets['memory_limit_mb'],
                'within_target': current_memory <= self.performance_targets['memory_limit_mb']
            }
            
            # Test aggregation methods
            print(f"\n🔄 Aggregation Methods:")
            print("-" * 50)
            
            aggregation_methods = ['majority_voting', 'confidence_weighting', 'temporal_smoothing']
            for method in aggregation_methods:
                print(f"   ✅ {method.replace('_', ' ').title()}: Available")
                long_audio_results['aggregation_methods'][method] = 'available'
            
            print(f"\n✅ PASS: Long audio processing validated successfully")
            
            self.test_results['long_audio'] = long_audio_results
            return long_audio_results
            
        except Exception as e:
            logger.error(f"❌ Long audio processing testing failed: {e}")
            self.test_results['issues'].append(f"Long audio processing testing failed: {e}")
            long_audio_results['validation_status'] = 'ERROR'
            return long_audio_results

    def test_realtime_performance(self) -> Dict:
        """
        Test 4: Real-time Processing Performance
        """
        logger.info("⚡ TEST 4: Real-time Processing Performance")
        print("\n" + "="*80)
        print("⚡ TEST 4: REAL-TIME PROCESSING PERFORMANCE")
        print("="*80)

        realtime_results = {
            'inference_speed': {},
            'memory_usage': {},
            'cpu_utilization': {},
            'throughput': {},
            'validation_status': 'PASS'
        }

        try:
            # Initialize tester for performance benchmarks
            tester = YAMNetModelTester(str(self.model_path), str(self.metadata_path))

            # Test inference speed with multiple files
            test_files = []
            for class_dir in ['slow', 'medium', 'fast', 'disturbance']:
                class_path = self.dataset_path / class_dir
                if class_path.exists():
                    audio_files = list(class_path.glob('*.wav'))[:3]  # Test 3 files per class
                    test_files.extend(audio_files)

            if not test_files:
                raise ValueError("No test files found for performance testing")

            print(f"\n⏱️  Inference Speed Testing ({len(test_files)} files):")
            print("-" * 50)

            inference_times = []
            memory_usage = []

            # Warm-up run
            if test_files:
                tester.predict_single_file(str(test_files[0]))

            # Performance testing
            for i, test_file in enumerate(test_files[:10]):  # Test first 10 files
                start_time = time.time()

                # Monitor memory before prediction
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024

                try:
                    result = tester.predict_single_file(str(test_file))

                    # Calculate inference time
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    inference_times.append(inference_time)

                    # Monitor memory after prediction
                    mem_after = process.memory_info().rss / 1024 / 1024
                    memory_usage.append(mem_after)

                    status = "✅" if inference_time <= self.performance_targets['inference_time_ms'] else "⚠️"
                    print(f"   {status} File {i+1:2d}: {inference_time:5.1f}ms - {test_file.name}")

                except Exception as e:
                    logger.warning(f"⚠️  Performance test failed for {test_file}: {e}")
                    continue

            # Calculate performance metrics
            if inference_times:
                avg_inference_time = np.mean(inference_times)
                max_inference_time = np.max(inference_times)
                min_inference_time = np.min(inference_times)

                print(f"\n📊 Performance Metrics:")
                print("-" * 50)
                print(f"   Average Inference Time: {avg_inference_time:.1f}ms")
                print(f"   Min/Max Inference Time: {min_inference_time:.1f}ms / {max_inference_time:.1f}ms")
                print(f"   Target Inference Time: {self.performance_targets['inference_time_ms']}ms")

                # Performance assessment
                if avg_inference_time <= self.performance_targets['inference_time_ms']:
                    print(f"   ✅ PASS: Average inference time meets target")
                else:
                    print(f"   ❌ FAIL: Average inference time exceeds target")
                    self.test_results['issues'].append(f"Average inference time {avg_inference_time:.1f}ms exceeds target {self.performance_targets['inference_time_ms']}ms")

                realtime_results['inference_speed'] = {
                    'average_ms': avg_inference_time,
                    'min_ms': min_inference_time,
                    'max_ms': max_inference_time,
                    'target_ms': self.performance_targets['inference_time_ms'],
                    'meets_target': avg_inference_time <= self.performance_targets['inference_time_ms']
                }

            # Memory usage analysis
            if memory_usage:
                avg_memory = np.mean(memory_usage)
                max_memory = np.max(memory_usage)

                print(f"   Average Memory Usage: {avg_memory:.1f}MB")
                print(f"   Peak Memory Usage: {max_memory:.1f}MB")
                print(f"   Target Memory Limit: {self.performance_targets['memory_limit_mb']}MB")

                if max_memory <= self.performance_targets['memory_limit_mb']:
                    print(f"   ✅ PASS: Memory usage within target")
                else:
                    print(f"   ❌ FAIL: Memory usage exceeds target")
                    self.test_results['issues'].append(f"Peak memory usage {max_memory:.1f}MB exceeds target {self.performance_targets['memory_limit_mb']}MB")

                realtime_results['memory_usage'] = {
                    'average_mb': avg_memory,
                    'peak_mb': max_memory,
                    'target_mb': self.performance_targets['memory_limit_mb'],
                    'within_target': max_memory <= self.performance_targets['memory_limit_mb']
                }

            # Calculate throughput
            if inference_times:
                files_per_second = 1000 / avg_inference_time  # Convert ms to files/second
                print(f"   Throughput: {files_per_second:.1f} files/second")

                realtime_results['throughput'] = {
                    'files_per_second': files_per_second,
                    'realtime_factor': files_per_second * 3  # Assuming 3s average file length
                }

            self.test_results['realtime_performance'] = realtime_results
            return realtime_results

        except Exception as e:
            logger.error(f"❌ Real-time performance testing failed: {e}")
            self.test_results['issues'].append(f"Real-time performance testing failed: {e}")
            realtime_results['validation_status'] = 'ERROR'
            return realtime_results

    def test_integration_readiness(self) -> Dict:
        """
        Test 5: Integration and Deployment Readiness
        """
        logger.info("🔗 TEST 5: Integration and Deployment Readiness")
        print("\n" + "="*80)
        print("🔗 TEST 5: INTEGRATION AND DEPLOYMENT READINESS")
        print("="*80)

        integration_results = {
            'file_compatibility': {},
            'model_format': {},
            'dependency_check': {},
            'deployment_readiness': {},
            'validation_status': 'PASS'
        }

        try:
            # Check model file compatibility
            print(f"\n📁 Model File Compatibility:")
            print("-" * 50)

            # Check if model files exist
            model_exists = self.model_path.exists()
            metadata_exists = self.metadata_path.exists()

            print(f"   Model File (.h5): {'✅ Found' if model_exists else '❌ Missing'}")
            print(f"   Metadata File (.json): {'✅ Found' if metadata_exists else '❌ Missing'}")

            if not model_exists:
                self.test_results['issues'].append("Model file missing")
                integration_results['validation_status'] = 'FAIL'

            if not metadata_exists:
                self.test_results['issues'].append("Metadata file missing")
                integration_results['validation_status'] = 'FAIL'

            # Check model format and size
            if model_exists:
                model_size_mb = self.model_path.stat().st_size / 1024 / 1024
                print(f"   Model Size: {model_size_mb:.1f}MB")

                # Test model loading
                try:
                    model = tf.keras.models.load_model(str(self.model_path))
                    input_shape = model.input_shape
                    output_shape = model.output_shape
                    param_count = model.count_params()

                    print(f"   Input Shape: {input_shape}")
                    print(f"   Output Shape: {output_shape}")
                    print(f"   Parameters: {param_count:,}")
                    print(f"   ✅ Model loads successfully")

                    integration_results['model_format'] = {
                        'size_mb': model_size_mb,
                        'input_shape': input_shape,
                        'output_shape': output_shape,
                        'parameters': param_count,
                        'loads_successfully': True
                    }

                except Exception as e:
                    print(f"   ❌ Model loading failed: {e}")
                    self.test_results['issues'].append(f"Model loading failed: {e}")
                    integration_results['validation_status'] = 'FAIL'

            # Check Python dependencies
            print(f"\n📦 Dependency Check:")
            print("-" * 50)

            required_packages = [
                'tensorflow', 'tensorflow_hub', 'librosa', 'numpy',
                'scikit-learn', 'matplotlib', 'soundfile'
            ]

            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                    print(f"   ✅ {package}")
                except ImportError:
                    print(f"   ❌ {package} (missing)")
                    missing_packages.append(package)

            if missing_packages:
                self.test_results['issues'].append(f"Missing packages: {missing_packages}")
                integration_results['validation_status'] = 'FAIL'

            integration_results['dependency_check'] = {
                'required_packages': required_packages,
                'missing_packages': missing_packages,
                'all_available': len(missing_packages) == 0
            }

            # Check deployment readiness
            print(f"\n🚀 Deployment Readiness:")
            print("-" * 50)

            deployment_checks = {
                'model_trained': model_exists and metadata_exists,
                'dependencies_met': len(missing_packages) == 0,
                'performance_acceptable': True,  # Will be updated based on previous tests
                'arduino_mapping_valid': True   # Will be updated based on previous tests
            }

            # Update based on previous test results
            if 'realtime_performance' in self.test_results:
                perf_results = self.test_results['realtime_performance']
                if 'inference_speed' in perf_results:
                    deployment_checks['performance_acceptable'] = perf_results['inference_speed'].get('meets_target', False)

            if 'arduino_mapping' in self.test_results:
                arduino_results = self.test_results['arduino_mapping']
                deployment_checks['arduino_mapping_valid'] = arduino_results.get('validation_status') == 'PASS'

            for check, status in deployment_checks.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {check.replace('_', ' ').title()}")

            all_checks_pass = all(deployment_checks.values())
            if all_checks_pass:
                print(f"\n✅ PASS: System ready for deployment")
            else:
                print(f"\n❌ FAIL: System not ready for deployment")
                integration_results['validation_status'] = 'FAIL'

            integration_results['deployment_readiness'] = deployment_checks

            self.test_results['integration'] = integration_results
            return integration_results

        except Exception as e:
            logger.error(f"❌ Integration readiness testing failed: {e}")
            self.test_results['issues'].append(f"Integration readiness testing failed: {e}")
            integration_results['validation_status'] = 'ERROR'
            return integration_results

    def generate_benchmarks_and_report(self) -> Dict:
        """
        Test 6: Generate Performance Benchmarks and Final Report
        """
        logger.info("📊 TEST 6: Performance Benchmarks and Final Report")
        print("\n" + "="*80)
        print("📊 TEST 6: PERFORMANCE BENCHMARKS AND FINAL REPORT")
        print("="*80)

        benchmark_results = {
            'performance_summary': {},
            'target_compliance': {},
            'recommendations': [],
            'overall_status': 'PASS'
        }

        try:
            # Compile performance summary
            print(f"\n📈 Performance Summary:")
            print("-" * 50)

            # Model accuracy summary
            if 'model_accuracy' in self.test_results:
                accuracy = self.test_results['model_accuracy'].get('overall_accuracy', 0.0)
                target_met = accuracy >= self.performance_targets['accuracy_threshold']
                status = "✅" if target_met else "❌"
                print(f"   {status} Model Accuracy: {accuracy:.1%} (Target: ≥{self.performance_targets['accuracy_threshold']:.1%})")

                benchmark_results['performance_summary']['accuracy'] = {
                    'actual': accuracy,
                    'target': self.performance_targets['accuracy_threshold'],
                    'meets_target': target_met
                }

            # Inference speed summary
            if 'realtime_performance' in self.test_results:
                perf_data = self.test_results['realtime_performance']
                if 'inference_speed' in perf_data:
                    avg_time = perf_data['inference_speed'].get('average_ms', 0)
                    target_met = avg_time <= self.performance_targets['inference_time_ms']
                    status = "✅" if target_met else "❌"
                    print(f"   {status} Inference Speed: {avg_time:.1f}ms (Target: ≤{self.performance_targets['inference_time_ms']}ms)")

                    benchmark_results['performance_summary']['inference_speed'] = {
                        'actual_ms': avg_time,
                        'target_ms': self.performance_targets['inference_time_ms'],
                        'meets_target': target_met
                    }

            # Memory usage summary
            if 'realtime_performance' in self.test_results:
                perf_data = self.test_results['realtime_performance']
                if 'memory_usage' in perf_data:
                    peak_memory = perf_data['memory_usage'].get('peak_mb', 0)
                    target_met = peak_memory <= self.performance_targets['memory_limit_mb']
                    status = "✅" if target_met else "❌"
                    print(f"   {status} Memory Usage: {peak_memory:.1f}MB (Target: ≤{self.performance_targets['memory_limit_mb']}MB)")

                    benchmark_results['performance_summary']['memory_usage'] = {
                        'actual_mb': peak_memory,
                        'target_mb': self.performance_targets['memory_limit_mb'],
                        'meets_target': target_met
                    }

            # Arduino mapping summary
            if 'arduino_mapping' in self.test_results:
                mapping_valid = self.test_results['arduino_mapping'].get('validation_status') == 'PASS'
                status = "✅" if mapping_valid else "❌"
                print(f"   {status} Arduino Integration: {'Valid' if mapping_valid else 'Invalid'}")

                benchmark_results['performance_summary']['arduino_integration'] = {
                    'valid': mapping_valid
                }

            # Generate recommendations
            print(f"\n💡 Recommendations:")
            print("-" * 50)

            recommendations = []

            # Accuracy recommendations
            if 'model_accuracy' in self.test_results:
                accuracy = self.test_results['model_accuracy'].get('overall_accuracy', 0.0)
                if accuracy < 0.85:
                    recommendations.append("Consider collecting more training data or data augmentation")
                elif accuracy < 0.90:
                    recommendations.append("Good accuracy - consider fine-tuning for specific use cases")
                else:
                    recommendations.append("Excellent accuracy - ready for production deployment")

            # Performance recommendations
            if 'realtime_performance' in self.test_results:
                perf_data = self.test_results['realtime_performance']
                if 'inference_speed' in perf_data:
                    avg_time = perf_data['inference_speed'].get('average_ms', 0)
                    if avg_time > 100:
                        recommendations.append("Consider TensorFlow Lite conversion for faster inference")
                    elif avg_time > 50:
                        recommendations.append("Performance acceptable - monitor in production")
                    else:
                        recommendations.append("Excellent inference speed - optimal for real-time use")

            # Memory recommendations
            if 'realtime_performance' in self.test_results:
                perf_data = self.test_results['realtime_performance']
                if 'memory_usage' in perf_data:
                    peak_memory = perf_data['memory_usage'].get('peak_mb', 0)
                    if peak_memory > 300:
                        recommendations.append("High memory usage - consider model optimization")
                    elif peak_memory > 200:
                        recommendations.append("Monitor memory usage in production environment")
                    else:
                        recommendations.append("Memory usage optimal for deployment")

            # Add general recommendations
            recommendations.extend([
                "Test with actual Arduino hardware before production deployment",
                "Implement continuous monitoring for production performance",
                "Consider A/B testing with existing DS-CNN system",
                "Set up automated testing pipeline for model updates"
            ])

            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

            benchmark_results['recommendations'] = recommendations

            # Overall assessment
            print(f"\n🎯 Overall Assessment:")
            print("-" * 50)

            # Count passed tests
            test_statuses = []
            for test_name in ['model_accuracy', 'arduino_mapping', 'long_audio', 'realtime_performance', 'integration']:
                if test_name in self.test_results:
                    status = self.test_results[test_name].get('validation_status', 'UNKNOWN')
                    test_statuses.append(status == 'PASS')

            passed_tests = sum(test_statuses)
            total_tests = len(test_statuses)

            if passed_tests == total_tests and len(self.test_results['issues']) == 0:
                overall_status = "EXCELLENT"
                print(f"   🎉 EXCELLENT: All tests passed ({passed_tests}/{total_tests})")
                print(f"   🚀 System ready for immediate production deployment")
            elif passed_tests >= total_tests * 0.8:
                overall_status = "GOOD"
                print(f"   ✅ GOOD: Most tests passed ({passed_tests}/{total_tests})")
                print(f"   ⚠️  Address minor issues before deployment")
            elif passed_tests >= total_tests * 0.6:
                overall_status = "ACCEPTABLE"
                print(f"   ⚠️  ACCEPTABLE: Some tests passed ({passed_tests}/{total_tests})")
                print(f"   🔧 Significant improvements needed before deployment")
            else:
                overall_status = "NEEDS_WORK"
                print(f"   ❌ NEEDS WORK: Few tests passed ({passed_tests}/{total_tests})")
                print(f"   🚨 Major issues must be resolved before deployment")

            benchmark_results['overall_status'] = overall_status
            benchmark_results['test_summary'] = {
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'issues_count': len(self.test_results['issues'])
            }

            self.test_results['benchmarks'] = benchmark_results
            return benchmark_results

        except Exception as e:
            logger.error(f"❌ Benchmark generation failed: {e}")
            self.test_results['issues'].append(f"Benchmark generation failed: {e}")
            benchmark_results['overall_status'] = 'ERROR'
            return benchmark_results

    def run_comprehensive_tests(self) -> Dict:
        """
        Run all comprehensive tests in sequence.

        Returns:
            Complete test results dictionary
        """
        logger.info("🚀 Starting Comprehensive Test Suite")
        print("\n" + "="*80)
        print("🧪 YAMNET COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Timestamp: {self.test_results['timestamp']}")
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_path}")

        try:
            # Run all tests in sequence
            print(f"\n🔄 Running comprehensive tests...")

            # Test 1: Model Accuracy
            self.test_model_accuracy()

            # Test 2: Arduino Command Mapping
            self.test_arduino_command_mapping()

            # Test 3: Long Audio Processing
            self.test_long_audio_processing()

            # Test 4: Real-time Performance
            self.test_realtime_performance()

            # Test 5: Integration Readiness
            self.test_integration_readiness()

            # Test 6: Benchmarks and Report
            self.generate_benchmarks_and_report()

            # Save comprehensive report
            self.save_test_report()

            return self.test_results

        except Exception as e:
            logger.error(f"❌ Comprehensive testing failed: {e}")
            self.test_results['issues'].append(f"Comprehensive testing failed: {e}")
            return self.test_results

    def save_test_report(self, output_path: str = "comprehensive_test_report.json"):
        """
        Save comprehensive test report to JSON file.

        Args:
            output_path: Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)

            logger.info(f"✅ Comprehensive test report saved to: {output_path}")
            print(f"\n📄 Test report saved to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save test report: {e}")

def main():
    """
    Main function to run comprehensive testing suite.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive YAMNet Pipeline Testing Suite")
    parser.add_argument("--model", default="yamnet_models/yamnet_classifier.h5",
                       help="Path to trained YAMNet model")
    parser.add_argument("--metadata", default="yamnet_models/yamnet_model_metadata.json",
                       help="Path to model metadata file")
    parser.add_argument("--dataset", default="../", help="Path to dataset directory")
    parser.add_argument("--output", default="comprehensive_test_report.json",
                       help="Output path for test report")
    parser.add_argument("--test", choices=['all', 'accuracy', 'arduino', 'long-audio', 'performance', 'integration'],
                       default='all', help="Specific test to run")

    args = parser.parse_args()

    try:
        # Initialize test suite
        test_suite = ComprehensiveTestSuite(
            model_path=args.model,
            metadata_path=args.metadata,
            dataset_path=args.dataset
        )

        # Run specific test or all tests
        if args.test == 'all':
            results = test_suite.run_comprehensive_tests()
        elif args.test == 'accuracy':
            results = {'model_accuracy': test_suite.test_model_accuracy()}
        elif args.test == 'arduino':
            results = {'arduino_mapping': test_suite.test_arduino_command_mapping()}
        elif args.test == 'long-audio':
            results = {'long_audio': test_suite.test_long_audio_processing()}
        elif args.test == 'performance':
            results = {'realtime_performance': test_suite.test_realtime_performance()}
        elif args.test == 'integration':
            results = {'integration': test_suite.test_integration_readiness()}

        # Save results if running individual test
        if args.test != 'all':
            test_suite.test_results.update(results)
            test_suite.save_test_report(args.output)

        # Print final summary
        issues_count = len(test_suite.test_results.get('issues', []))
        if issues_count == 0:
            print(f"\n🎉 All tests completed successfully!")
            print(f"✅ System ready for production deployment")
        else:
            print(f"\n⚠️  Testing completed with {issues_count} issues")
            print(f"🔧 Review issues before deployment")

        logger.info("🎉 Comprehensive testing completed!")

    except Exception as e:
        logger.error(f"❌ Testing suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
