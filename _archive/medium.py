#!/usr/bin/env python3
"""
Medium Class Audio Handler

This script handles audio files specifically for the 'medium' class in the
sound-to-vibration classification system. It provides functionality for:
- Generating medium-rhythm audio files
- Validating existing medium audio files
- Processing and analyzing medium-specific audio characteristics
- Integration with the main classification pipeline

Medium audio characteristics:
- Medium frequencies (300-600 Hz)
- Moderate modulation (1-2 Hz)
- Balanced, steady patterns
- Moderate harmonics
- Consistent energy levels

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
CLASS_NAME = "medium"
CLASS_LABEL = 2
RANDOM_SEED = 42

# Medium-specific parameters
MEDIUM_FREQ_RANGE = (300, 600)        # Hz - medium frequency range
MEDIUM_MODULATION_RANGE = (1.0, 2.0)  # Hz - moderate modulation
MEDIUM_HARMONIC_COUNT = 3              # Number of harmonics
MEDIUM_ENERGY_THRESHOLD = 0.6          # Moderate energy level
MEDIUM_STABILITY_THRESHOLD = 0.7       # Consistency measure

class MediumAudioHandler:
    """
    Comprehensive handler for medium class audio files.
    """

    def __init__(self, folder_path="medium"):
        """
        Initialize the medium audio handler.

        Args:
            folder_path (str): Path to the medium audio folder
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

    def generate_medium_audio(self, filename=None, balance_level="medium"):
        """
        Generate a medium-rhythm audio file with balanced characteristics.

        Args:
            filename (str): Output filename (auto-generated if None)
            balance_level (str): "low", "medium", or "high" balance

        Returns:
            tuple: (audio_data, filename) or (None, None) if failed
        """
        try:
            print(f"⚖️  Generating medium audio (balance: {balance_level})...")

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"medium_{balance_level}_{timestamp}.wav"

            # Ensure .wav extension
            if not filename.lower().endswith('.wav'):
                filename += '.wav'

            # Set balance parameters
            balance_params = {
                "low": {"base_freq": 350, "mod_freq": 1.0, "harmonics": 2, "stability": 0.5},
                "medium": {"base_freq": 450, "mod_freq": 1.5, "harmonics": 3, "stability": 0.7},
                "high": {"base_freq": 550, "mod_freq": 2.0, "harmonics": 3, "stability": 0.9}
            }

            params = balance_params.get(balance_level, balance_params["medium"])

            # Generate time array
            samples = int(self.sample_rate * self.duration)
            t = np.linspace(0, self.duration, samples, False)

            # Initialize audio
            audio = np.zeros(samples, dtype=np.float32)

            # Primary medium frequency with moderate modulation
            base_freq = params["base_freq"]
            mod_freq = params["mod_freq"]

            # Moderate amplitude modulation
            amp_mod = 0.6 + 0.4 * np.sin(2 * np.pi * mod_freq * t)

            # Moderate frequency modulation
            freq_mod = base_freq + 50 * np.sin(2 * np.pi * mod_freq * 0.7 * t)
            freq_mod += 25 * np.sin(2 * np.pi * mod_freq * 1.3 * t)

            # Primary medium signal
            audio += 0.7 * amp_mod * np.sin(2 * np.pi * freq_mod * t)

            # Add harmonics for richness
            for i in range(1, params["harmonics"] + 1):
                harmonic_freq = freq_mod * (1 + i * 0.5)
                harmonic_amp = 0.3 / i
                harmonic_mod = amp_mod * (0.7 + 0.3 * np.sin(2 * np.pi * mod_freq * i * 0.5 * t))

                audio += harmonic_amp * harmonic_mod * np.sin(2 * np.pi * harmonic_freq * t)

            # Add steady rhythmic component
            rhythm_freq = mod_freq * 1.5
            rhythm_envelope = 0.5 + 0.3 * np.sin(2 * np.pi * rhythm_freq * t)
            rhythm_signal = 0.25 * rhythm_envelope * np.sin(2 * np.pi * base_freq * 0.8 * t)
            audio += rhythm_signal

            # Add subtle texture
            texture_freq = base_freq * 1.2
            texture_mod = 0.1 + 0.05 * np.sin(2 * np.pi * mod_freq * 2 * t)
            texture_signal = 0.1 * texture_mod * np.sin(2 * np.pi * texture_freq * t)
            audio += texture_signal

            # Add controlled noise for naturalness
            noise_level = 0.03 * (1 - params["stability"] * 0.5)  # Less noise for higher stability
            noise = np.random.normal(0, noise_level, samples)
            audio += noise

            # Add moderate transients
            transient_count = int(mod_freq * self.duration)
            for _ in range(transient_count):
                pos = np.random.randint(0, samples - 200)
                transient_amp = 0.15
                transient_decay = np.exp(-np.linspace(0, 3, 200))
                transient_freq = np.random.uniform(400, 700)
                transient_signal = transient_amp * transient_decay * np.sin(2 * np.pi * transient_freq * t[pos:pos+200])
                audio[pos:pos+200] += transient_signal

            # Normalize with moderate level
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.8  # Moderate level

            # Save audio file
            output_path = self.folder_path / filename
            sf.write(str(output_path), audio, self.sample_rate, format='WAV', subtype='PCM_16')

            print(f"✅ Generated medium audio: {output_path}")
            print(f"   Duration: {self.duration}s, Sample rate: {self.sample_rate}Hz")
            print(f"   Balance: {balance_level}, Base freq: {params['base_freq']}Hz")
            print(f"   Modulation: {params['mod_freq']}Hz, Harmonics: {params['harmonics']}")
            print(f"   File size: {output_path.stat().st_size:,} bytes")

            return audio, str(output_path)

        except Exception as e:
            print(f"❌ Error generating medium audio: {e}")
            traceback.print_exc()
            return None, None

    def validate_medium_audio(self, audio_file):
        """
        Validate if an audio file exhibits medium rhythm characteristics.

        Args:
            audio_file (str): Path to the audio file

        Returns:
            dict: Validation results with scores and recommendations
        """
        try:
            print(f"🔍 Validating medium audio: {audio_file}")

            # Load audio file
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)

            if len(audio_data) == 0:
                return {"valid": False, "error": "Empty audio file"}

            # Normalize duration
            target_samples = int(self.sample_rate * self.duration)
            if len(audio_data) != target_samples:
                if len(audio_data) < target_samples:
                    audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)))
                else:
                    start_idx = (len(audio_data) - target_samples) // 2
                    audio_data = audio_data[start_idx:start_idx + target_samples]

            results = {
                "valid": False,
                "scores": {},
                "recommendations": [],
                "file": audio_file
            }

            # 1. Medium frequency content analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)

            # Check medium frequency dominance
            medium_freq_power = np.sum(magnitude[(freqs >= 250) & (freqs <= 750)])
            total_power = np.sum(magnitude[freqs >= 0])
            medium_freq_ratio = medium_freq_power / total_power if total_power > 0 else 0

            results["scores"]["medium_frequency_dominance"] = float(medium_freq_ratio)

            # 2. Stability analysis (consistency measure)
            # Calculate RMS in overlapping windows
            window_size = int(0.2 * self.sample_rate)  # 200ms windows
            hop_size = window_size // 2
            rms_values = []

            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)

            rms_values = np.array(rms_values)
            stability = 1.0 - (np.std(rms_values) / (np.mean(rms_values) + 1e-10))
            results["scores"]["stability"] = float(max(0, stability))

            # 3. Spectral centroid (balance measure)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            mean_centroid = np.mean(spectral_centroids)
            centroid_stability = 1.0 - (np.std(spectral_centroids) / (mean_centroid + 1e-10))

            results["scores"]["spectral_centroid"] = float(mean_centroid)
            results["scores"]["centroid_stability"] = float(max(0, centroid_stability))

            # 4. Tempo analysis
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            results["scores"]["estimated_tempo"] = float(tempo)

            # 5. Energy consistency
            rms = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=512)[0]
            mean_energy = np.mean(rms)
            energy_consistency = 1.0 - (np.std(rms) / (mean_energy + 1e-10))

            results["scores"]["mean_energy"] = float(mean_energy)
            results["scores"]["energy_consistency"] = float(max(0, energy_consistency))

            # Overall validation scoring
            score_weights = {
                "medium_frequency_dominance": 0.25,
                "stability": 0.2,
                "centroid_stability": 0.2,
                "energy_consistency": 0.2,
                "estimated_tempo": 0.15
            }

            # Calculate weighted score
            weighted_score = 0
            for metric, weight in score_weights.items():
                if metric in results["scores"]:
                    # Normalize scores based on expected medium values
                    if metric == "medium_frequency_dominance":
                        normalized = min(results["scores"][metric] / 0.5, 1.0)
                    elif metric in ["stability", "centroid_stability", "energy_consistency"]:
                        normalized = results["scores"][metric]  # Already 0-1
                    elif metric == "estimated_tempo":
                        # Medium tempo should be around 80-120 BPM
                        tempo_val = results["scores"][metric]
                        if 80 <= tempo_val <= 120:
                            normalized = 1.0
                        else:
                            normalized = max(0, 1.0 - abs(tempo_val - 100) / 50)
                    else:
                        normalized = results["scores"][metric]

                    weighted_score += weight * normalized

            results["overall_score"] = float(weighted_score)
            results["valid"] = weighted_score >= 0.6  # 60% threshold

            # Generate recommendations
            if results["scores"]["medium_frequency_dominance"] < 0.3:
                results["recommendations"].append("Increase medium-frequency content (300-600 Hz)")

            if results["scores"]["stability"] < 0.6:
                results["recommendations"].append("Improve amplitude stability and consistency")

            if results["scores"]["centroid_stability"] < 0.6:
                results["recommendations"].append("Stabilize spectral balance")

            if results["scores"]["energy_consistency"] < 0.6:
                results["recommendations"].append("Maintain more consistent energy levels")

            tempo_val = results["scores"]["estimated_tempo"]
            if tempo_val < 70 or tempo_val > 130:
                results["recommendations"].append("Adjust tempo to medium range (80-120 BPM)")

            if not results["recommendations"]:
                results["recommendations"].append("Audio exhibits good medium rhythm characteristics")

            # Print validation results
            print(f"📊 Validation Results:")
            print(f"   Overall Score: {results['overall_score']:.3f}")
            print(f"   Valid Medium Audio: {'✅ Yes' if results['valid'] else '❌ No'}")
            print(f"   Medium Freq Dominance: {results['scores']['medium_frequency_dominance']:.3f}")
            print(f"   Stability: {results['scores']['stability']:.3f}")
            print(f"   Energy Consistency: {results['scores']['energy_consistency']:.3f}")
            print(f"   Estimated Tempo: {results['scores']['estimated_tempo']:.0f} BPM")

            if results["recommendations"]:
                print(f"💡 Recommendations:")
                for rec in results["recommendations"]:
                    print(f"   • {rec}")

            return results

        except Exception as e:
            print(f"❌ Error validating medium audio: {e}")
            traceback.print_exc()
            return {"valid": False, "error": str(e)}

def main():
    """
    Main function for command-line usage.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Medium Class Audio Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 medium.py --generate                     # Generate test audio
  python3 medium.py --generate --balance high      # Generate balanced audio
  python3 medium.py --validate audio.wav           # Validate specific file
  python3 medium.py --batch 3 --balance medium     # Generate 3 medium files
        """
    )

    parser.add_argument('--generate', action='store_true',
                       help='Generate medium audio file')
    parser.add_argument('--validate', type=str,
                       help='Validate specific audio file')
    parser.add_argument('--balance', choices=['low', 'medium', 'high'],
                       default='medium', help='Balance level for generation')
    parser.add_argument('--batch', type=int, default=1,
                       help='Number of files to generate')
    parser.add_argument('--folder', type=str, default='medium',
                       help='Medium folder path')
    parser.add_argument('--output', type=str,
                       help='Output filename for generation')

    args = parser.parse_args()

    # Initialize handler
    handler = MediumAudioHandler(args.folder)

    print("=" * 80)
    print("⚖️  MEDIUM CLASS AUDIO HANDLER")
    print("=" * 80)

    try:
        if args.generate:
            print(f"Generating {args.batch} medium audio file(s)...")
            for i in range(args.batch):
                filename = args.output
                if args.batch > 1 and filename:
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{i+1:02d}{ext}"

                audio, path = handler.generate_medium_audio(
                    filename=filename,
                    balance_level=args.balance
                )

                if audio is not None:
                    print(f"✅ Generated: {path}")
                else:
                    print(f"❌ Failed to generate file {i+1}")

        elif args.validate:
            print(f"Validating audio file: {args.validate}")
            results = handler.validate_medium_audio(args.validate)

            if results.get("valid"):
                print("✅ File is valid medium audio")
            else:
                print("❌ File does not meet medium criteria")

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