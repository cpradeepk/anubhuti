#!/usr/bin/env python3
"""
Slow Class Audio Handler

This script handles audio files specifically for the 'slow' class in the
sound-to-vibration classification system. It provides functionality for:
- Generating slow-rhythm audio files
- Validating existing slow audio files
- Processing and analyzing slow-specific audio characteristics
- Integration with the main classification pipeline

Slow audio characteristics:
- Low-medium frequencies (200-400 Hz)
- Slow modulation (0.3-1.0 Hz)
- Gentle, gradual patterns
- Minimal harmonics
- Smooth energy transitions

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
CLASS_NAME = "slow"
CLASS_LABEL = 1
RANDOM_SEED = 42

# Slow-specific parameters
SLOW_FREQ_RANGE = (200, 400)         # Hz - low-medium frequency range
SLOW_MODULATION_RANGE = (0.3, 1.0)   # Hz - slow modulation
SLOW_HARMONIC_COUNT = 2               # Minimal harmonics
SLOW_SMOOTHNESS_THRESHOLD = 0.8       # Smoothness measure
SLOW_TEMPO_THRESHOLD = 60             # Maximum BPM equivalent

class SlowAudioHandler:
    """
    Comprehensive handler for slow class audio files.
    """

    def __init__(self, folder_path="slow"):
        """
        Initialize the slow audio handler.

        Args:
            folder_path (str): Path to the slow audio folder
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

    def generate_slow_audio(self, filename=None, gentleness_level="medium"):
        """
        Generate a slow-rhythm audio file with gentle characteristics.

        Args:
            filename (str): Output filename (auto-generated if None)
            gentleness_level (str): "low", "medium", or "high" gentleness

        Returns:
            tuple: (audio_data, filename) or (None, None) if failed
        """
        try:
            print(f"🐌 Generating slow audio (gentleness: {gentleness_level})...")

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"slow_{gentleness_level}_{timestamp}.wav"

            # Ensure .wav extension
            if not filename.lower().endswith('.wav'):
                filename += '.wav'

            # Set gentleness parameters
            gentleness_params = {
                "low": {"base_freq": 250, "mod_freq": 0.8, "harmonics": 2, "smoothness": 0.6},
                "medium": {"base_freq": 300, "mod_freq": 0.5, "harmonics": 2, "smoothness": 0.8},
                "high": {"base_freq": 350, "mod_freq": 0.3, "harmonics": 1, "smoothness": 0.9}
            }

            params = gentleness_params.get(gentleness_level, gentleness_params["medium"])

            # Generate time array
            samples = int(self.sample_rate * self.duration)
            t = np.linspace(0, self.duration, samples, False)

            # Initialize audio
            audio = np.zeros(samples, dtype=np.float32)

            # Primary slow frequency with gentle modulation
            base_freq = params["base_freq"]
            mod_freq = params["mod_freq"]

            # Slow, gentle amplitude modulation
            amp_mod = 0.5 + 0.3 * np.sin(2 * np.pi * mod_freq * t)

            # Slow frequency modulation
            freq_mod = base_freq + 30 * np.sin(2 * np.pi * mod_freq * 0.3 * t)

            # Primary slow signal
            audio += 0.6 * amp_mod * np.sin(2 * np.pi * freq_mod * t)

            # Add minimal harmonics for warmth
            for i in range(1, params["harmonics"] + 1):
                harmonic_freq = freq_mod * (1 + i * 0.5)
                harmonic_amp = 0.2 / (i + 1)
                harmonic_mod = amp_mod * (0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * i * 0.2 * t))

                audio += harmonic_amp * harmonic_mod * np.sin(2 * np.pi * harmonic_freq * t)

            # Add gentle sub-harmonic for depth
            sub_freq = base_freq * 0.5
            sub_mod = amp_mod * 0.3
            sub_signal = 0.15 * sub_mod * np.sin(2 * np.pi * sub_freq * t)
            audio += sub_signal

            # Add minimal noise for naturalness
            noise_level = 0.02 * (1 - params["smoothness"])
            noise = np.random.normal(0, noise_level, samples)
            audio += noise

            # Apply gentle envelope for smooth start/end
            envelope_samples = int(0.1 * self.sample_rate)  # 100ms fade
            envelope = np.ones(samples)

            # Fade in
            envelope[:envelope_samples] = np.linspace(0, 1, envelope_samples)
            # Fade out
            envelope[-envelope_samples:] = np.linspace(1, 0, envelope_samples)

            audio *= envelope

            # Normalize with gentle level
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.7  # Gentle level

            # Save audio file
            output_path = self.folder_path / filename
            sf.write(str(output_path), audio, self.sample_rate, format='WAV', subtype='PCM_16')

            print(f"✅ Generated slow audio: {output_path}")
            print(f"   Duration: {self.duration}s, Sample rate: {self.sample_rate}Hz")
            print(f"   Gentleness: {gentleness_level}, Base freq: {params['base_freq']}Hz")
            print(f"   Modulation: {params['mod_freq']}Hz, Harmonics: {params['harmonics']}")
            print(f"   File size: {output_path.stat().st_size:,} bytes")

            return audio, str(output_path)

        except Exception as e:
            print(f"❌ Error generating slow audio: {e}")
            traceback.print_exc()
            return None, None

    def validate_slow_audio(self, audio_file):
        """
        Validate if an audio file exhibits slow rhythm characteristics.

        Args:
            audio_file (str): Path to the audio file

        Returns:
            dict: Validation results with scores and recommendations
        """
        try:
            print(f"🔍 Validating slow audio: {audio_file}")

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

            # 1. Low-medium frequency content analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)

            # Check low-medium frequency dominance
            low_med_freq_power = np.sum(magnitude[(freqs >= 150) & (freqs <= 500)])
            total_power = np.sum(magnitude[freqs >= 0])
            low_med_freq_ratio = low_med_freq_power / total_power if total_power > 0 else 0

            results["scores"]["low_medium_frequency_dominance"] = float(low_med_freq_ratio)

            # 2. Smoothness analysis
            # Calculate derivative to measure abrupt changes
            diff = np.diff(audio_data)
            smoothness = 1.0 - (np.std(diff) / (np.std(audio_data) + 1e-10))
            results["scores"]["smoothness"] = float(max(0, smoothness))

            # 3. Tempo analysis (should be slow)
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            results["scores"]["estimated_tempo"] = float(tempo)

            # 4. Energy gentleness
            rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=1024)[0]
            mean_energy = np.mean(rms)
            energy_gentleness = 1.0 - (np.std(rms) / (mean_energy + 1e-10))

            results["scores"]["mean_energy"] = float(mean_energy)
            results["scores"]["energy_gentleness"] = float(max(0, energy_gentleness))

            # 5. Spectral rolloff (should be low for slow audio)
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            mean_rolloff = np.mean(rolloff)
            results["scores"]["spectral_rolloff"] = float(mean_rolloff)

            # Overall validation scoring
            score_weights = {
                "low_medium_frequency_dominance": 0.25,
                "smoothness": 0.25,
                "energy_gentleness": 0.2,
                "estimated_tempo": 0.2,
                "spectral_rolloff": 0.1
            }

            # Calculate weighted score
            weighted_score = 0
            for metric, weight in score_weights.items():
                if metric in results["scores"]:
                    # Normalize scores based on expected slow values
                    if metric == "low_medium_frequency_dominance":
                        normalized = min(results["scores"][metric] / 0.6, 1.0)
                    elif metric in ["smoothness", "energy_gentleness"]:
                        normalized = results["scores"][metric]  # Already 0-1
                    elif metric == "estimated_tempo":
                        # Slow tempo should be under 80 BPM
                        tempo_val = results["scores"][metric]
                        normalized = max(0, 1.0 - max(0, tempo_val - 80) / 40)
                    elif metric == "spectral_rolloff":
                        # Lower rolloff is better for slow
                        normalized = max(0, 1.0 - max(0, mean_rolloff - 1000) / 2000)
                    else:
                        normalized = results["scores"][metric]

                    weighted_score += weight * normalized

            results["overall_score"] = float(weighted_score)
            results["valid"] = weighted_score >= 0.6  # 60% threshold

            # Generate recommendations
            if results["scores"]["low_medium_frequency_dominance"] < 0.4:
                results["recommendations"].append("Increase low-medium frequency content (200-400 Hz)")

            if results["scores"]["smoothness"] < 0.7:
                results["recommendations"].append("Reduce abrupt changes for smoother transitions")

            if results["scores"]["energy_gentleness"] < 0.6:
                results["recommendations"].append("Make energy transitions more gentle")

            if results["scores"]["estimated_tempo"] > 90:
                results["recommendations"].append("Reduce tempo for slower rhythm (target <80 BPM)")

            if not results["recommendations"]:
                results["recommendations"].append("Audio exhibits good slow rhythm characteristics")

            # Print validation results
            print(f"📊 Validation Results:")
            print(f"   Overall Score: {results['overall_score']:.3f}")
            print(f"   Valid Slow Audio: {'✅ Yes' if results['valid'] else '❌ No'}")
            print(f"   Low-Med Freq Dominance: {results['scores']['low_medium_frequency_dominance']:.3f}")
            print(f"   Smoothness: {results['scores']['smoothness']:.3f}")
            print(f"   Energy Gentleness: {results['scores']['energy_gentleness']:.3f}")
            print(f"   Estimated Tempo: {results['scores']['estimated_tempo']:.0f} BPM")

            if results["recommendations"]:
                print(f"💡 Recommendations:")
                for rec in results["recommendations"]:
                    print(f"   • {rec}")

            return results

        except Exception as e:
            print(f"❌ Error validating slow audio: {e}")
            traceback.print_exc()
            return {"valid": False, "error": str(e)}

def main():
    """
    Main function for command-line usage.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Slow Class Audio Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 slow.py --generate                      # Generate test audio
  python3 slow.py --generate --gentleness high    # Generate gentle audio
  python3 slow.py --validate audio.wav            # Validate specific file
  python3 slow.py --batch 2 --gentleness medium   # Generate 2 medium files
        """
    )

    parser.add_argument('--generate', action='store_true',
                       help='Generate slow audio file')
    parser.add_argument('--validate', type=str,
                       help='Validate specific audio file')
    parser.add_argument('--gentleness', choices=['low', 'medium', 'high'],
                       default='medium', help='Gentleness level for generation')
    parser.add_argument('--batch', type=int, default=1,
                       help='Number of files to generate')
    parser.add_argument('--folder', type=str, default='slow',
                       help='Slow folder path')
    parser.add_argument('--output', type=str,
                       help='Output filename for generation')

    args = parser.parse_args()

    # Initialize handler
    handler = SlowAudioHandler(args.folder)

    print("=" * 80)
    print("🐌 SLOW CLASS AUDIO HANDLER")
    print("=" * 80)

    try:
        if args.generate:
            print(f"Generating {args.batch} slow audio file(s)...")
            for i in range(args.batch):
                filename = args.output
                if args.batch > 1 and filename:
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{i+1:02d}{ext}"

                audio, path = handler.generate_slow_audio(
                    filename=filename,
                    gentleness_level=args.gentleness
                )

                if audio is not None:
                    print(f"✅ Generated: {path}")
                else:
                    print(f"❌ Failed to generate file {i+1}")

        elif args.validate:
            print(f"Validating audio file: {args.validate}")
            results = handler.validate_slow_audio(args.validate)

            if results.get("valid"):
                print("✅ File is valid slow audio")
            else:
                print("❌ File does not meet slow criteria")

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