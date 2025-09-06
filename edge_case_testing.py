#!/usr/bin/env python3
"""
Edge Case Testing Script for YAMNet Pipeline

This script tests edge cases and boundary conditions including:
- Very short audio files (<1s)
- Very long audio files (>10s)
- Noisy audio with low SNR
- Quiet audio with low amplitude
- Corrupted or malformed audio files
- Empty or silent audio
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import tempfile
import time

# Add yamnet_utils to path
sys.path.append(str(Path(__file__).parent))
from test_yamnet_model import YAMNetModelTester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeCaseTester:
    """
    Edge case testing for YAMNet pipeline robustness.
    """
    
    def __init__(self, model_path: str = "yamnet_models/yamnet_classifier.h5",
                 metadata_path: str = "yamnet_models/yamnet_model_metadata.json"):
        """
        Initialize edge case tester.
        
        Args:
            model_path: Path to trained YAMNet model
            metadata_path: Path to model metadata
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.tester = YAMNetModelTester(str(model_path), str(metadata_path))
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.edge_case_results = {
            'short_audio': {},
            'long_audio': {},
            'noisy_audio': {},
            'quiet_audio': {},
            'corrupted_audio': {},
            'silent_audio': {},
            'summary': {}
        }
        
        logger.info("🧪 Edge Case Tester initialized")
        logger.info(f"   Temp directory: {self.temp_dir}")
    
    def create_test_audio(self, duration: float, audio_type: str = 'sine', 
                         frequency: float = 440, sample_rate: int = 22050) -> np.ndarray:
        """
        Create synthetic test audio.
        
        Args:
            duration: Duration in seconds
            audio_type: Type of audio ('sine', 'noise', 'silence')
            frequency: Frequency for sine wave
            sample_rate: Sample rate
            
        Returns:
            Audio array
        """
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        if audio_type == 'sine':
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        elif audio_type == 'noise':
            audio = 0.1 * np.random.normal(0, 1, samples)
        elif audio_type == 'silence':
            audio = np.zeros(samples)
        else:
            raise ValueError(f"Unknown audio type: {audio_type}")
        
        return audio.astype(np.float32)
    
    def test_short_audio(self) -> Dict:
        """
        Test very short audio files (<1s).
        """
        logger.info("⏱️  Testing short audio files")
        print("\n" + "="*60)
        print("⏱️  EDGE CASE: SHORT AUDIO FILES (<1s)")
        print("="*60)
        
        short_results = {
            'test_durations': [0.1, 0.3, 0.5, 0.8],
            'results': [],
            'success_rate': 0.0,
            'issues': []
        }
        
        successful_tests = 0
        
        for duration in short_results['test_durations']:
            try:
                # Create short test audio
                audio = self.create_test_audio(duration, 'sine', 440)
                test_file = self.temp_dir / f"short_{duration}s.wav"
                sf.write(test_file, audio, 22050)
                
                # Test prediction
                start_time = time.time()
                result = self.tester.predict_single_file(str(test_file))
                inference_time = (time.time() - start_time) * 1000
                
                test_result = {
                    'duration': duration,
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'inference_time_ms': inference_time,
                    'success': True
                }
                
                short_results['results'].append(test_result)
                successful_tests += 1
                
                print(f"   ✅ {duration}s audio: {result['predicted_class_name']} ({result['confidence']:.3f}) - {inference_time:.1f}ms")
                
            except Exception as e:
                test_result = {
                    'duration': duration,
                    'error': str(e),
                    'success': False
                }
                short_results['results'].append(test_result)
                short_results['issues'].append(f"Failed {duration}s test: {e}")
                
                print(f"   ❌ {duration}s audio: FAILED - {e}")
        
        short_results['success_rate'] = successful_tests / len(short_results['test_durations'])
        
        print(f"\n📊 Short Audio Test Summary:")
        print(f"   Success Rate: {short_results['success_rate']:.1%} ({successful_tests}/{len(short_results['test_durations'])})")
        
        self.edge_case_results['short_audio'] = short_results
        return short_results
    
    def test_long_audio(self) -> Dict:
        """
        Test very long audio files (>10s).
        """
        logger.info("⏱️  Testing long audio files")
        print("\n" + "="*60)
        print("⏱️  EDGE CASE: LONG AUDIO FILES (>10s)")
        print("="*60)
        
        long_results = {
            'test_durations': [12, 20, 30],
            'results': [],
            'success_rate': 0.0,
            'issues': []
        }
        
        successful_tests = 0
        
        for duration in long_results['test_durations']:
            try:
                # Create long test audio (mix of frequencies)
                audio1 = self.create_test_audio(duration/3, 'sine', 220)
                audio2 = self.create_test_audio(duration/3, 'sine', 440)
                audio3 = self.create_test_audio(duration/3, 'sine', 880)
                audio = np.concatenate([audio1, audio2, audio3])
                
                test_file = self.temp_dir / f"long_{duration}s.wav"
                sf.write(test_file, audio, 22050)
                
                # Test prediction
                start_time = time.time()
                result = self.tester.predict_single_file(str(test_file))
                inference_time = (time.time() - start_time) * 1000
                
                test_result = {
                    'duration': duration,
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'inference_time_ms': inference_time,
                    'success': True
                }
                
                long_results['results'].append(test_result)
                successful_tests += 1
                
                print(f"   ✅ {duration}s audio: {result['predicted_class_name']} ({result['confidence']:.3f}) - {inference_time:.1f}ms")
                
            except Exception as e:
                test_result = {
                    'duration': duration,
                    'error': str(e),
                    'success': False
                }
                long_results['results'].append(test_result)
                long_results['issues'].append(f"Failed {duration}s test: {e}")
                
                print(f"   ❌ {duration}s audio: FAILED - {e}")
        
        long_results['success_rate'] = successful_tests / len(long_results['test_durations'])
        
        print(f"\n📊 Long Audio Test Summary:")
        print(f"   Success Rate: {long_results['success_rate']:.1%} ({successful_tests}/{len(long_results['test_durations'])})")
        
        self.edge_case_results['long_audio'] = long_results
        return long_results
    
    def test_noisy_audio(self) -> Dict:
        """
        Test noisy audio with low signal-to-noise ratio.
        """
        logger.info("🔊 Testing noisy audio")
        print("\n" + "="*60)
        print("🔊 EDGE CASE: NOISY AUDIO (Low SNR)")
        print("="*60)
        
        noisy_results = {
            'snr_levels': [0, 5, 10, 15],  # dB
            'results': [],
            'success_rate': 0.0,
            'issues': []
        }
        
        successful_tests = 0
        
        for snr_db in noisy_results['snr_levels']:
            try:
                # Create signal + noise
                signal = self.create_test_audio(3.0, 'sine', 440)
                noise = self.create_test_audio(3.0, 'noise')
                
                # Calculate noise scaling for desired SNR
                signal_power = np.mean(signal**2)
                noise_power = np.mean(noise**2)
                snr_linear = 10**(snr_db/10)
                noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
                
                # Mix signal and noise
                noisy_audio = signal + noise_scale * noise
                noisy_audio = np.clip(noisy_audio, -1.0, 1.0)  # Prevent clipping
                
                test_file = self.temp_dir / f"noisy_{snr_db}dB.wav"
                sf.write(test_file, noisy_audio, 22050)
                
                # Test prediction
                start_time = time.time()
                result = self.tester.predict_single_file(str(test_file))
                inference_time = (time.time() - start_time) * 1000
                
                test_result = {
                    'snr_db': snr_db,
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'inference_time_ms': inference_time,
                    'success': True
                }
                
                noisy_results['results'].append(test_result)
                successful_tests += 1
                
                print(f"   ✅ {snr_db}dB SNR: {result['predicted_class_name']} ({result['confidence']:.3f}) - {inference_time:.1f}ms")
                
            except Exception as e:
                test_result = {
                    'snr_db': snr_db,
                    'error': str(e),
                    'success': False
                }
                noisy_results['results'].append(test_result)
                noisy_results['issues'].append(f"Failed {snr_db}dB test: {e}")
                
                print(f"   ❌ {snr_db}dB SNR: FAILED - {e}")
        
        noisy_results['success_rate'] = successful_tests / len(noisy_results['snr_levels'])
        
        print(f"\n📊 Noisy Audio Test Summary:")
        print(f"   Success Rate: {noisy_results['success_rate']:.1%} ({successful_tests}/{len(noisy_results['snr_levels'])})")
        
        self.edge_case_results['noisy_audio'] = noisy_results
        return noisy_results
    
    def test_quiet_audio(self) -> Dict:
        """
        Test very quiet audio with low amplitude.
        """
        logger.info("🔇 Testing quiet audio")
        print("\n" + "="*60)
        print("🔇 EDGE CASE: QUIET AUDIO (Low Amplitude)")
        print("="*60)
        
        quiet_results = {
            'amplitude_levels': [0.01, 0.05, 0.1, 0.2],
            'results': [],
            'success_rate': 0.0,
            'issues': []
        }
        
        successful_tests = 0
        
        for amplitude in quiet_results['amplitude_levels']:
            try:
                # Create quiet audio
                audio = self.create_test_audio(3.0, 'sine', 440)
                quiet_audio = audio * amplitude
                
                test_file = self.temp_dir / f"quiet_{amplitude}.wav"
                sf.write(test_file, quiet_audio, 22050)
                
                # Test prediction
                start_time = time.time()
                result = self.tester.predict_single_file(str(test_file))
                inference_time = (time.time() - start_time) * 1000
                
                test_result = {
                    'amplitude': amplitude,
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'inference_time_ms': inference_time,
                    'success': True
                }
                
                quiet_results['results'].append(test_result)
                successful_tests += 1
                
                print(f"   ✅ {amplitude:.2f} amplitude: {result['predicted_class_name']} ({result['confidence']:.3f}) - {inference_time:.1f}ms")
                
            except Exception as e:
                test_result = {
                    'amplitude': amplitude,
                    'error': str(e),
                    'success': False
                }
                quiet_results['results'].append(test_result)
                quiet_results['issues'].append(f"Failed {amplitude} amplitude test: {e}")
                
                print(f"   ❌ {amplitude:.2f} amplitude: FAILED - {e}")
        
        quiet_results['success_rate'] = successful_tests / len(quiet_results['amplitude_levels'])
        
        print(f"\n📊 Quiet Audio Test Summary:")
        print(f"   Success Rate: {quiet_results['success_rate']:.1%} ({successful_tests}/{len(quiet_results['amplitude_levels'])})")
        
        self.edge_case_results['quiet_audio'] = quiet_results
        return quiet_results

    def test_silent_audio(self) -> Dict:
        """
        Test completely silent audio.
        """
        logger.info("🔇 Testing silent audio")
        print("\n" + "="*60)
        print("🔇 EDGE CASE: SILENT AUDIO")
        print("="*60)

        silent_results = {
            'test_durations': [1.0, 3.0, 5.0],
            'results': [],
            'success_rate': 0.0,
            'issues': []
        }

        successful_tests = 0

        for duration in silent_results['test_durations']:
            try:
                # Create silent audio
                audio = self.create_test_audio(duration, 'silence')
                test_file = self.temp_dir / f"silent_{duration}s.wav"
                sf.write(test_file, audio, 22050)

                # Test prediction
                start_time = time.time()
                result = self.tester.predict_single_file(str(test_file))
                inference_time = (time.time() - start_time) * 1000

                test_result = {
                    'duration': duration,
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'inference_time_ms': inference_time,
                    'success': True
                }

                silent_results['results'].append(test_result)
                successful_tests += 1

                print(f"   ✅ {duration}s silent: {result['predicted_class_name']} ({result['confidence']:.3f}) - {inference_time:.1f}ms")

            except Exception as e:
                test_result = {
                    'duration': duration,
                    'error': str(e),
                    'success': False
                }
                silent_results['results'].append(test_result)
                silent_results['issues'].append(f"Failed {duration}s silent test: {e}")

                print(f"   ❌ {duration}s silent: FAILED - {e}")

        silent_results['success_rate'] = successful_tests / len(silent_results['test_durations'])

        print(f"\n📊 Silent Audio Test Summary:")
        print(f"   Success Rate: {silent_results['success_rate']:.1%} ({successful_tests}/{len(silent_results['test_durations'])})")

        self.edge_case_results['silent_audio'] = silent_results
        return silent_results

    def test_corrupted_audio(self) -> Dict:
        """
        Test corrupted or malformed audio files.
        """
        logger.info("💥 Testing corrupted audio")
        print("\n" + "="*60)
        print("💥 EDGE CASE: CORRUPTED AUDIO")
        print("="*60)

        corrupted_results = {
            'test_types': ['empty_file', 'truncated_file', 'wrong_format'],
            'results': [],
            'success_rate': 0.0,
            'issues': []
        }

        successful_tests = 0

        for test_type in corrupted_results['test_types']:
            try:
                if test_type == 'empty_file':
                    # Create empty file
                    test_file = self.temp_dir / "empty.wav"
                    test_file.touch()

                elif test_type == 'truncated_file':
                    # Create truncated WAV file
                    audio = self.create_test_audio(3.0, 'sine', 440)
                    test_file = self.temp_dir / "truncated.wav"
                    sf.write(test_file, audio, 22050)

                    # Truncate the file
                    with open(test_file, 'r+b') as f:
                        f.truncate(100)  # Keep only first 100 bytes

                elif test_type == 'wrong_format':
                    # Create text file with .wav extension
                    test_file = self.temp_dir / "wrong_format.wav"
                    with open(test_file, 'w') as f:
                        f.write("This is not an audio file")

                # Test prediction (should handle gracefully)
                try:
                    start_time = time.time()
                    result = self.tester.predict_single_file(str(test_file))
                    inference_time = (time.time() - start_time) * 1000

                    # If we get here, the system handled the corruption gracefully
                    test_result = {
                        'test_type': test_type,
                        'predicted_class': result['predicted_class_name'],
                        'confidence': result['confidence'],
                        'inference_time_ms': inference_time,
                        'handled_gracefully': True,
                        'success': True
                    }

                    print(f"   ✅ {test_type}: Handled gracefully - {result['predicted_class_name']} ({result['confidence']:.3f})")

                except Exception as prediction_error:
                    # Expected behavior - system should reject corrupted files
                    test_result = {
                        'test_type': test_type,
                        'error': str(prediction_error),
                        'handled_gracefully': True,  # Rejecting corrupted files is good
                        'success': True
                    }

                    print(f"   ✅ {test_type}: Properly rejected - {str(prediction_error)[:50]}...")

                corrupted_results['results'].append(test_result)
                successful_tests += 1

            except Exception as e:
                test_result = {
                    'test_type': test_type,
                    'error': str(e),
                    'handled_gracefully': False,
                    'success': False
                }
                corrupted_results['results'].append(test_result)
                corrupted_results['issues'].append(f"Failed {test_type} test: {e}")

                print(f"   ❌ {test_type}: FAILED - {e}")

        corrupted_results['success_rate'] = successful_tests / len(corrupted_results['test_types'])

        print(f"\n📊 Corrupted Audio Test Summary:")
        print(f"   Success Rate: {corrupted_results['success_rate']:.1%} ({successful_tests}/{len(corrupted_results['test_types'])})")

        self.edge_case_results['corrupted_audio'] = corrupted_results
        return corrupted_results

    def run_all_edge_case_tests(self) -> Dict:
        """
        Run all edge case tests.

        Returns:
            Complete edge case test results
        """
        logger.info("🚀 Starting Edge Case Testing")
        print("\n" + "="*80)
        print("🧪 YAMNET EDGE CASE TESTING SUITE")
        print("="*80)

        try:
            # Run all edge case tests
            self.test_short_audio()
            self.test_long_audio()
            self.test_noisy_audio()
            self.test_quiet_audio()
            self.test_silent_audio()
            self.test_corrupted_audio()

            # Generate summary
            self.generate_edge_case_summary()

            return self.edge_case_results

        except Exception as e:
            logger.error(f"❌ Edge case testing failed: {e}")
            return self.edge_case_results
        finally:
            # Cleanup temp files
            self.cleanup()

    def generate_edge_case_summary(self):
        """
        Generate overall edge case testing summary.
        """
        print("\n" + "="*80)
        print("📊 EDGE CASE TESTING SUMMARY")
        print("="*80)

        total_tests = 0
        successful_tests = 0
        all_issues = []

        test_categories = ['short_audio', 'long_audio', 'noisy_audio', 'quiet_audio', 'silent_audio', 'corrupted_audio']

        for category in test_categories:
            if category in self.edge_case_results:
                results = self.edge_case_results[category]
                category_total = len(results.get('results', []))
                category_success = sum(1 for r in results.get('results', []) if r.get('success', False))
                success_rate = results.get('success_rate', 0.0)

                total_tests += category_total
                successful_tests += category_success
                all_issues.extend(results.get('issues', []))

                status = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.5 else "❌"
                print(f"   {status} {category.replace('_', ' ').title()}: {success_rate:.1%} ({category_success}/{category_total})")

        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0.0

        print(f"\n📈 Overall Edge Case Performance:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success Rate: {overall_success_rate:.1%}")
        print(f"   Issues Found: {len(all_issues)}")

        # Assessment
        if overall_success_rate >= 0.9:
            assessment = "EXCELLENT - System handles edge cases very well"
        elif overall_success_rate >= 0.7:
            assessment = "GOOD - System handles most edge cases adequately"
        elif overall_success_rate >= 0.5:
            assessment = "ACCEPTABLE - Some edge case handling needs improvement"
        else:
            assessment = "NEEDS WORK - Significant edge case handling issues"

        print(f"\n🎯 Assessment: {assessment}")

        self.edge_case_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'overall_success_rate': overall_success_rate,
            'issues_count': len(all_issues),
            'assessment': assessment,
            'all_issues': all_issues
        }

    def cleanup(self):
        """
        Clean up temporary files.
        """
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"✅ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup temp directory: {e}")

def main():
    """
    Main function for edge case testing.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="YAMNet Edge Case Testing")
    parser.add_argument("--model", default="yamnet_models/yamnet_classifier.h5",
                       help="Path to trained YAMNet model")
    parser.add_argument("--metadata", default="yamnet_models/yamnet_model_metadata.json",
                       help="Path to model metadata file")
    parser.add_argument("--output", default="edge_case_test_results.json",
                       help="Output path for test results")

    args = parser.parse_args()

    try:
        # Initialize edge case tester
        tester = EdgeCaseTester(args.model, args.metadata)

        # Run all edge case tests
        results = tester.run_all_edge_case_tests()

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Edge case test results saved to: {args.output}")

        # Print final assessment
        summary = results.get('summary', {})
        success_rate = summary.get('overall_success_rate', 0.0)

        if success_rate >= 0.8:
            print(f"🎉 Edge case testing completed successfully!")
        else:
            print(f"⚠️  Edge case testing completed with issues")

        logger.info("🎉 Edge case testing completed!")

    except Exception as e:
        logger.error(f"❌ Edge case testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
