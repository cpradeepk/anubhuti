#!/usr/bin/env python3
"""
Disturbance Class Audio Handler

This script handles audio files specifically for the 'disturbance' class in the
sound-to-vibration classification system. It provides functionality for:
- Generating disturbance-pattern audio files
- Validating existing disturbance audio files
- Processing and analyzing disturbance-specific audio characteristics
- Integration with the main classification pipeline

Disturbance audio characteristics:
- Irregular, chaotic patterns
- Low frequencies (80-200 Hz)
- Random amplitude variations
- Unpredictable timing and bursts
- High entropy and noise content

Author: Audio Classification System
Date: 2025-08-30
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import warnings
import traceback
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# Configuration constants - must match main system
SAMPLE_RATE = 22050  # Hz
DURATION = 3.0       # seconds
N_MFCC = 13         # MFCC coefficients
CLASS_NAME = "disturbance"
CLASS_LABEL = 0
RANDOM_SEED = 42

# Disturbance-specific parameters
DISTURBANCE_FREQ_RANGE = (80, 200)    # Hz - chaotic frequency range
DISTURBANCE_BURST_COUNT = (3, 8)      # Number of irregular bursts
DISTURBANCE_NOISE_LEVEL = 0.15        # High noise for chaos
DISTURBANCE_ENTROPY_THRESHOLD = 0.7   # Minimum entropy for validation

class DisturbanceAudioHandler:
    """
    Comprehensive handler for disturbance class audio files.
    """

    def __init__(self, folder_path="disturbance"):
        """
        Initialize the disturbance audio handler.

        Args:
            folder_path (str): Path to the disturbance audio folder
        """
        self.folder_path = Path(folder_path)
        self.class_name = CLASS_NAME
        self.class_label = CLASS_LABEL
        self.sample_rate = SAMPLE_RATE
        self.duration = DURATION

        # Ensure folder exists
        self.folder_path.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducible generation
        np.random.seed(RANDOM_SEED)

    def generate_disturbance_audio(self, filename=None, complexity_level="medium"):
        """
        Generate a disturbance-pattern audio file with chaotic characteristics.

        Args:
            filename (str): Output filename (auto-generated if None)
            complexity_level (str): "low", "medium", or "high" complexity

        Returns:
            tuple: (audio_data, filename) or (None, None) if failed
        """
        try:
            print(f"🌪️  Generating disturbance audio (complexity: {complexity_level})...")

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"disturbance_{complexity_level}_{timestamp}.wav"

            # Ensure .wav extension
            if not filename.lower().endswith('.wav'):
                filename += '.wav'

            # Set complexity parameters
            complexity_params = {
                "low": {"bursts": 3, "chaos_factor": 0.3, "noise_level": 0.1},
                "medium": {"bursts": 5, "chaos_factor": 0.5, "noise_level": 0.15},
                "high": {"bursts": 8, "chaos_factor": 0.8, "noise_level": 0.2}
            }

            params = complexity_params.get(complexity_level, complexity_params["medium"])

            # Generate time array
            samples = int(self.sample_rate * self.duration)
            t = np.linspace(0, self.duration, samples, False)

            # Initialize audio with base chaotic signal
            audio = np.zeros(samples, dtype=np.float32)

            # Base chaotic frequency modulation
            base_freq = np.random.uniform(*DISTURBANCE_FREQ_RANGE)
            chaos_mod = params["chaos_factor"] * np.random.normal(0, 20, samples)
            freq_signal = base_freq + chaos_mod

            # Primary chaotic component
            audio += 0.4 * np.sin(2 * np.pi * freq_signal * t)

            # Add irregular bursts
            for _ in range(params["bursts"]):
                # Random burst parameters
                start_sample = np.random.randint(0, samples - 1000)
                burst_duration = np.random.randint(500, 2000)
                end_sample = min(start_sample + burst_duration, samples)

                burst_freq = np.random.uniform(60, 250)
                burst_amplitude = np.random.uniform(0.2, 0.5)

                # Create burst with random envelope
                burst_samples = end_sample - start_sample
                burst_t = t[start_sample:end_sample]

                # Random envelope shape
                envelope_type = np.random.randint(0, 3)
                if envelope_type == 0:
                    envelope = np.exp(-3 * np.linspace(0, 1, burst_samples))  # Decay
                elif envelope_type == 1:
                    envelope = np.sin(np.pi * np.linspace(0, 1, burst_samples))  # Bell
                else:
                    envelope = np.ones(burst_samples) * np.random.uniform(0.5, 1.0)  # Flat

                burst_signal = burst_amplitude * envelope * np.sin(2 * np.pi * burst_freq * burst_t)
                audio[start_sample:end_sample] += burst_signal

            # Add chaotic harmonics
            for harmonic in [1.5, 2.3, 3.7]:
                harmonic_amp = 0.1 / harmonic
                harmonic_freq = freq_signal * harmonic
                audio += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)

            # Add high-level noise for chaos
            noise = np.random.normal(0, params["noise_level"], samples)
            audio += noise

            # Add random clicks and pops
            click_count = np.random.randint(5, 15)
            for _ in range(click_count):
                click_pos = np.random.randint(0, samples - 10)
                click_amplitude = np.random.uniform(0.1, 0.3)
                audio[click_pos:click_pos+5] += click_amplitude * np.random.normal(0, 1, 5)

            # Normalize to prevent clipping while maintaining chaos
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.85

            # Save audio file
            output_path = self.folder_path / filename
            sf.write(str(output_path), audio, self.sample_rate, format='WAV', subtype='PCM_16')

            print(f"✅ Generated disturbance audio: {output_path}")
            print(f"   Duration: {self.duration}s, Sample rate: {self.sample_rate}Hz")
            print(f"   Complexity: {complexity_level}, Bursts: {params['bursts']}")
            print(f"   File size: {output_path.stat().st_size:,} bytes")

            return audio, str(output_path)

        except Exception as e:
            print(f"❌ Error generating disturbance audio: {e}")
            traceback.print_exc()
            return None, None

    def validate_disturbance_audio(self, audio_file):
        """
        Validate if an audio file exhibits disturbance characteristics.

        Args:
            audio_file (str): Path to the audio file

        Returns:
            dict: Validation results with scores and recommendations
        """
        try:
            print(f"🔍 Validating disturbance audio: {audio_file}")

            # Load audio file
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)

            if len(audio_data) == 0:
                return {"valid": False, "error": "Empty audio file"}

            # Normalize duration
            target_samples = int(self.sample_rate * self.duration)
            if len(audio_data) != target_samples:
                # Pad or truncate as needed
                if len(audio_data) < target_samples:
                    audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)))
                else:
                    start_idx = (len(audio_data) - target_samples) // 2
                    audio_data = audio_data[start_idx:start_idx + target_samples]

            # Calculate disturbance-specific metrics
            results = {
                "valid": False,
                "scores": {},
                "recommendations": [],
                "file": audio_file
            }

            # 1. Frequency analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)

            # Check dominant frequency range
            low_freq_power = np.sum(magnitude[(freqs >= 50) & (freqs <= 300)])
            total_power = np.sum(magnitude[freqs >= 0])
            low_freq_ratio = low_freq_power / total_power if total_power > 0 else 0

            results["scores"]["low_frequency_dominance"] = float(low_freq_ratio)

            # 2. Entropy analysis (chaos measure)
            # Calculate spectral entropy
            psd = magnitude ** 2
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            max_entropy = np.log(len(psd_norm))
            normalized_entropy = spectral_entropy / max_entropy

            results["scores"]["spectral_entropy"] = float(normalized_entropy)

            # 3. Amplitude variation analysis
            # Calculate RMS in overlapping windows
            window_size = int(0.1 * self.sample_rate)  # 100ms windows
            hop_size = window_size // 2
            rms_values = []

            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)

            rms_values = np.array(rms_values)
            amplitude_variation = np.std(rms_values) / (np.mean(rms_values) + 1e-10)
            results["scores"]["amplitude_variation"] = float(amplitude_variation)

            # 4. Zero crossing rate (irregularity measure)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            mean_zcr = np.mean(zcr)
            zcr_variation = np.std(zcr)

            results["scores"]["zero_crossing_rate"] = float(mean_zcr)
            results["scores"]["zcr_variation"] = float(zcr_variation)

            # 5. Burst detection
            # Find sudden amplitude increases
            diff = np.diff(np.abs(audio_data))
            burst_threshold = np.std(diff) * 3
            bursts = np.sum(diff > burst_threshold)
            burst_density = bursts / len(diff)

            results["scores"]["burst_density"] = float(burst_density)

            # Overall validation scoring
            score_weights = {
                "low_frequency_dominance": 0.2,
                "spectral_entropy": 0.3,
                "amplitude_variation": 0.2,
                "zcr_variation": 0.15,
                "burst_density": 0.15
            }

            # Calculate weighted score
            weighted_score = 0
            for metric, weight in score_weights.items():
                if metric in results["scores"]:
                    # Normalize scores to 0-1 range based on expected disturbance values
                    if metric == "low_frequency_dominance":
                        normalized = min(results["scores"][metric] / 0.6, 1.0)
                    elif metric == "spectral_entropy":
                        normalized = min(results["scores"][metric] / 0.8, 1.0)
                    elif metric == "amplitude_variation":
                        normalized = min(results["scores"][metric] / 2.0, 1.0)
                    elif metric == "zcr_variation":
                        normalized = min(results["scores"][metric] / 0.1, 1.0)
                    elif metric == "burst_density":
                        normalized = min(results["scores"][metric] / 0.01, 1.0)
                    else:
                        normalized = results["scores"][metric]

                    weighted_score += weight * normalized

            results["overall_score"] = float(weighted_score)
            results["valid"] = weighted_score >= 0.6  # 60% threshold

            # Generate recommendations
            if results["scores"]["low_frequency_dominance"] < 0.3:
                results["recommendations"].append("Increase low-frequency content (80-200 Hz)")

            if results["scores"]["spectral_entropy"] < 0.5:
                results["recommendations"].append("Add more spectral complexity and chaos")

            if results["scores"]["amplitude_variation"] < 1.0:
                results["recommendations"].append("Increase amplitude variations and irregularity")

            if results["scores"]["burst_density"] < 0.005:
                results["recommendations"].append("Add more sudden bursts and transients")

            if not results["recommendations"]:
                results["recommendations"].append("Audio exhibits good disturbance characteristics")

            # Print validation results
            print(f"📊 Validation Results:")
            print(f"   Overall Score: {results['overall_score']:.3f}")
            print(f"   Valid Disturbance: {'✅ Yes' if results['valid'] else '❌ No'}")
            print(f"   Low Freq Dominance: {results['scores']['low_frequency_dominance']:.3f}")
            print(f"   Spectral Entropy: {results['scores']['spectral_entropy']:.3f}")
            print(f"   Amplitude Variation: {results['scores']['amplitude_variation']:.3f}")
            print(f"   Burst Density: {results['scores']['burst_density']:.6f}")

            if results["recommendations"]:
                print(f"💡 Recommendations:")
                for rec in results["recommendations"]:
                    print(f"   • {rec}")

            return results

        except Exception as e:
            print(f"❌ Error validating disturbance audio: {e}")
            traceback.print_exc()
            return {"valid": False, "error": str(e)}

    def process_folder(self):
        """
        Process all audio files in the disturbance folder.

        Returns:
            dict: Processing results and statistics
        """
        try:
            print(f"📁 Processing disturbance folder: {self.folder_path}")

            # Find all audio files
            audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
            audio_files = []

            for ext in audio_extensions:
                audio_files.extend(list(self.folder_path.glob(ext)))
                audio_files.extend(list(self.folder_path.glob(ext.upper())))

            if not audio_files:
                print("⚠️  No audio files found in disturbance folder")
                return {"processed": 0, "valid": 0, "invalid": 0, "files": []}

            print(f"Found {len(audio_files)} audio files")

            results = {
                "processed": 0,
                "valid": 0,
                "invalid": 0,
                "files": [],
                "statistics": {
                    "avg_score": 0,
                    "score_distribution": [],
                    "common_issues": []
                }
            }

            scores = []
            all_recommendations = []

            for audio_file in audio_files:
                print(f"\n🔍 Processing: {audio_file.name}")
                validation = self.validate_disturbance_audio(str(audio_file))

                file_result = {
                    "filename": audio_file.name,
                    "path": str(audio_file),
                    "valid": validation.get("valid", False),
                    "score": validation.get("overall_score", 0),
                    "recommendations": validation.get("recommendations", [])
                }

                results["files"].append(file_result)
                results["processed"] += 1

                if validation.get("valid", False):
                    results["valid"] += 1
                else:
                    results["invalid"] += 1

                if "overall_score" in validation:
                    scores.append(validation["overall_score"])

                all_recommendations.extend(validation.get("recommendations", []))

            # Calculate statistics
            if scores:
                results["statistics"]["avg_score"] = float(np.mean(scores))
                results["statistics"]["score_distribution"] = [float(s) for s in scores]

            # Find common issues
            from collections import Counter
            rec_counter = Counter(all_recommendations)
            results["statistics"]["common_issues"] = [
                {"issue": issue, "count": count}
                for issue, count in rec_counter.most_common(5)
            ]

            # Print summary
            print(f"\n📊 Processing Summary:")
            print(f"   Total files: {results['processed']}")
            print(f"   Valid disturbance files: {results['valid']}")
            print(f"   Invalid files: {results['invalid']}")
            print(f"   Average score: {results['statistics']['avg_score']:.3f}")

            if results["statistics"]["common_issues"]:
                print(f"   Common issues:")
                for issue in results["statistics"]["common_issues"][:3]:
                    print(f"     • {issue['issue']} ({issue['count']} files)")

            return results

        except Exception as e:
            print(f"❌ Error processing folder: {e}")
            traceback.print_exc()
            return {"processed": 0, "valid": 0, "invalid": 0, "files": []}

def main():
    """
    Main function for command-line usage.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Disturbance Class Audio Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 disturbance.py --generate                    # Generate test audio
  python3 disturbance.py --generate --complexity high  # Generate complex audio
  python3 disturbance.py --validate audio.wav          # Validate specific file
  python3 disturbance.py --process                     # Process entire folder
  python3 disturbance.py --batch 5 --complexity medium # Generate 5 medium files
        """
    )

    parser.add_argument('--generate', action='store_true',
                       help='Generate disturbance audio file')
    parser.add_argument('--validate', type=str,
                       help='Validate specific audio file')
    parser.add_argument('--process', action='store_true',
                       help='Process entire disturbance folder')
    parser.add_argument('--complexity', choices=['low', 'medium', 'high'],
                       default='medium', help='Complexity level for generation')
    parser.add_argument('--batch', type=int, default=1,
                       help='Number of files to generate')
    parser.add_argument('--folder', type=str, default='disturbance',
                       help='Disturbance folder path')
    parser.add_argument('--output', type=str,
                       help='Output filename for generation')

    args = parser.parse_args()

    # Initialize handler
    handler = DisturbanceAudioHandler(args.folder)

    print("=" * 80)
    print("🌪️  DISTURBANCE CLASS AUDIO HANDLER")
    print("=" * 80)

    try:
        if args.generate:
            print(f"Generating {args.batch} disturbance audio file(s)...")
            for i in range(args.batch):
                filename = args.output
                if args.batch > 1 and filename:
                    # Add number suffix for batch generation
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{i+1:02d}{ext}"

                audio, path = handler.generate_disturbance_audio(
                    filename=filename,
                    complexity_level=args.complexity
                )

                if audio is not None:
                    print(f"✅ Generated: {path}")
                else:
                    print(f"❌ Failed to generate file {i+1}")

        elif args.validate:
            print(f"Validating audio file: {args.validate}")
            results = handler.validate_disturbance_audio(args.validate)

            if results.get("valid"):
                print("✅ File is valid disturbance audio")
            else:
                print("❌ File does not meet disturbance criteria")

        elif args.process:
            print("Processing disturbance folder...")
            results = handler.process_folder()

            # Save results to JSON
            output_file = f"disturbance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Analysis saved to: {output_file}")

        else:
            print("No action specified. Use --help for usage information.")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()