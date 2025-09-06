#!/usr/bin/env python3
"""
Fast Class Audio Handler

This script handles audio files specifically for the 'fast' class in the
sound-to-vibration classification system. It provides functionality for:
- Generating fast-rhythm audio files
- Validating existing fast audio files
- Processing and analyzing fast-specific audio characteristics
- Integration with the main classification pipeline

Fast audio characteristics:
- High frequencies (600-1200 Hz)
- Rapid modulation (2-5 Hz)
- Quick, energetic patterns
- Multiple harmonics
- High temporal activity

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
CLASS_NAME = "fast"
CLASS_LABEL = 3
RANDOM_SEED = 42

# Fast-specific parameters
FAST_FREQ_RANGE = (600, 1200)        # Hz - high frequency range
FAST_MODULATION_RANGE = (2.0, 5.0)   # Hz - rapid modulation
FAST_HARMONIC_COUNT = 4               # Number of harmonics
FAST_ENERGY_THRESHOLD = 0.8           # Minimum energy level
FAST_TEMPO_THRESHOLD = 120            # Minimum BPM equivalent

class FastAudioHandler:
    """
    Comprehensive handler for fast class audio files.
    """

    def __init__(self, folder_path="fast"):
        """
        Initialize the fast audio handler.

        Args:
            folder_path (str): Path to the fast audio folder
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

    def generate_fast_audio(self, filename=None, intensity_level="medium"):
        """
        Generate a fast-rhythm audio file with energetic characteristics.

        Args:
            filename (str): Output filename (auto-generated if None)
            intensity_level (str): "low", "medium", or "high" intensity

        Returns:
            tuple: (audio_data, filename) or (None, None) if failed
        """
        try:
            print(f"⚡ Generating fast audio (intensity: {intensity_level})...")

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fast_{intensity_level}_{timestamp}.wav"

            # Ensure .wav extension
            if not filename.lower().endswith('.wav'):
                filename += '.wav'

            # Set intensity parameters
            intensity_params = {
                "low": {"base_freq": 600, "mod_freq": 2.0, "harmonics": 2, "energy": 0.6},
                "medium": {"base_freq": 800, "mod_freq": 3.0, "harmonics": 3, "energy": 0.8},
                "high": {"base_freq": 1000, "mod_freq": 4.5, "harmonics": 4, "energy": 1.0}
            }

            params = intensity_params.get(intensity_level, intensity_params["medium"])

            # Generate time array
            samples = int(self.sample_rate * self.duration)
            t = np.linspace(0, self.duration, samples, False)

            # Initialize audio
            audio = np.zeros(samples, dtype=np.float32)

            # Primary fast frequency with rapid modulation
            base_freq = params["base_freq"]
            mod_freq = params["mod_freq"]

            # Fast amplitude modulation
            amp_mod = 0.7 + 0.3 * np.sin(2 * np.pi * mod_freq * t)

            # Fast frequency modulation with multiple components
            freq_mod = base_freq + 100 * np.sin(2 * np.pi * mod_freq * 0.8 * t)
            freq_mod += 50 * np.sin(2 * np.pi * mod_freq * 2.0 * t)

            # Primary fast signal
            audio += params["energy"] * amp_mod * np.sin(2 * np.pi * freq_mod * t)

            # Add harmonics for richness and energy
            for i in range(1, params["harmonics"] + 1):
                harmonic_freq = freq_mod * (1 + i * 0.25)  # Non-integer harmonics
                harmonic_amp = params["energy"] * 0.4 / i
                harmonic_mod = amp_mod * (0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * i * t))

                audio += harmonic_amp * harmonic_mod * np.sin(2 * np.pi * harmonic_freq * t)

            # Add rapid rhythmic pulses
            pulse_freq = mod_freq * 2  # Double the modulation frequency
            pulse_envelope = np.maximum(0, np.sin(2 * np.pi * pulse_freq * t))
            pulse_signal = 0.3 * params["energy"] * pulse_envelope * np.sin(2 * np.pi * base_freq * 1.5 * t)
            audio += pulse_signal

            # Add high-frequency sparkle
            sparkle_freq = base_freq * 2
            sparkle_mod = 0.2 + 0.1 * np.sin(2 * np.pi * mod_freq * 3 * t)
            sparkle_signal = 0.15 * params["energy"] * sparkle_mod * np.sin(2 * np.pi * sparkle_freq * t)
            audio += sparkle_signal

            # Add controlled noise for energy
            noise_level = 0.05 * params["energy"]
            noise = np.random.normal(0, noise_level, samples)
            audio += noise

            # Add rapid transients for percussive feel
            transient_count = int(mod_freq * self.duration * 2)  # More transients for fast
            for _ in range(transient_count):
                pos = np.random.randint(0, samples - 100)
                transient_amp = 0.2 * params["energy"]
                transient_decay = np.exp(-np.linspace(0, 5, 100))
                transient_freq = np.random.uniform(800, 1500)
                transient_signal = transient_amp * transient_decay * np.sin(2 * np.pi * transient_freq * t[pos:pos+100])
                audio[pos:pos+100] += transient_signal

            # Normalize to prevent clipping while maintaining energy
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.9  # Higher level for fast audio

            # Save audio file
            output_path = self.folder_path / filename
            sf.write(str(output_path), audio, self.sample_rate, format='WAV', subtype='PCM_16')

            print(f"✅ Generated fast audio: {output_path}")
            print(f"   Duration: {self.duration}s, Sample rate: {self.sample_rate}Hz")
            print(f"   Intensity: {intensity_level}, Base freq: {params['base_freq']}Hz")
            print(f"   Modulation: {params['mod_freq']}Hz, Harmonics: {params['harmonics']}")
            print(f"   File size: {output_path.stat().st_size:,} bytes")

            return audio, str(output_path)

        except Exception as e:
            print(f"❌ Error generating fast audio: {e}")
            traceback.print_exc()
            return None, None

    def validate_fast_audio(self, audio_file):
        """
        Validate if an audio file exhibits fast rhythm characteristics.

        Args:
            audio_file (str): Path to the audio file

        Returns:
            dict: Validation results with scores and recommendations
        """
        try:
            print(f"🔍 Validating fast audio: {audio_file}")

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

            # 1. High frequency content analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)

            # Check high frequency dominance
            high_freq_power = np.sum(magnitude[(freqs >= 500) & (freqs <= 1500)])
            total_power = np.sum(magnitude[freqs >= 0])
            high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0

            results["scores"]["high_frequency_dominance"] = float(high_freq_ratio)

            # 2. Temporal activity analysis
            # Calculate onset strength
            onset_frames = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
            onset_rate = len(librosa.onset.onset_detect(onset_strength=onset_frames, sr=self.sample_rate)) / self.duration

            results["scores"]["onset_rate"] = float(onset_rate)

            # 3. Spectral centroid (brightness measure)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            mean_centroid = np.mean(spectral_centroids)
            centroid_variation = np.std(spectral_centroids)

            results["scores"]["spectral_centroid"] = float(mean_centroid)
            results["scores"]["centroid_variation"] = float(centroid_variation)

            # 4. Energy analysis
            # RMS energy in short windows
            rms = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=512)[0]
            mean_energy = np.mean(rms)
            energy_variation = np.std(rms)

            results["scores"]["mean_energy"] = float(mean_energy)
            results["scores"]["energy_variation"] = float(energy_variation)

            # 5. Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            results["scores"]["estimated_tempo"] = float(tempo)

            # 6. Harmonic content analysis
            harmonic, percussive = librosa.effects.hpss(audio_data)
            harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio_data)) + 1e-10)

            results["scores"]["harmonic_content"] = float(harmonic_ratio)

            # Overall validation scoring
            score_weights = {
                "high_frequency_dominance": 0.25,
                "onset_rate": 0.2,
                "spectral_centroid": 0.15,
                "mean_energy": 0.15,
                "estimated_tempo": 0.15,
                "harmonic_content": 0.1
            }

            # Calculate weighted score
            weighted_score = 0
            for metric, weight in score_weights.items():
                if metric in results["scores"]:
                    # Normalize scores based on expected fast values
                    if metric == "high_frequency_dominance":
                        normalized = min(results["scores"][metric] / 0.4, 1.0)
                    elif metric == "onset_rate":
                        normalized = min(results["scores"][metric] / 10.0, 1.0)
                    elif metric == "spectral_centroid":
                        normalized = min(results["scores"][metric] / 2000.0, 1.0)
                    elif metric == "mean_energy":
                        normalized = min(results["scores"][metric] / 0.3, 1.0)
                    elif metric == "estimated_tempo":
                        normalized = min(results["scores"][metric] / 150.0, 1.0)
                    elif metric == "harmonic_content":
                        normalized = min(results["scores"][metric] / 0.7, 1.0)
                    else:
                        normalized = results["scores"][metric]

                    weighted_score += weight * normalized

            results["overall_score"] = float(weighted_score)
            results["valid"] = weighted_score >= 0.65  # 65% threshold for fast

            # Generate recommendations
            if results["scores"]["high_frequency_dominance"] < 0.2:
                results["recommendations"].append("Increase high-frequency content (600-1200 Hz)")

            if results["scores"]["onset_rate"] < 5.0:
                results["recommendations"].append("Add more rapid onsets and rhythmic activity")

            if results["scores"]["spectral_centroid"] < 1000:
                results["recommendations"].append("Increase spectral brightness and energy")

            if results["scores"]["estimated_tempo"] < 100:
                results["recommendations"].append("Increase tempo and rhythmic speed")

            if results["scores"]["mean_energy"] < 0.15:
                results["recommendations"].append("Boost overall energy and amplitude")

            if not results["recommendations"]:
                results["recommendations"].append("Audio exhibits good fast rhythm characteristics")

            # Print validation results
            print(f"📊 Validation Results:")
            print(f"   Overall Score: {results['overall_score']:.3f}")
            print(f"   Valid Fast Audio: {'✅ Yes' if results['valid'] else '❌ No'}")
            print(f"   High Freq Dominance: {results['scores']['high_frequency_dominance']:.3f}")
            print(f"   Onset Rate: {results['scores']['onset_rate']:.1f} /sec")
            print(f"   Spectral Centroid: {results['scores']['spectral_centroid']:.0f} Hz")
            print(f"   Estimated Tempo: {results['scores']['estimated_tempo']:.0f} BPM")
            print(f"   Mean Energy: {results['scores']['mean_energy']:.3f}")

            if results["recommendations"]:
                print(f"💡 Recommendations:")
                for rec in results["recommendations"]:
                    print(f"   • {rec}")

            return results

        except Exception as e:
            print(f"❌ Error validating fast audio: {e}")
            traceback.print_exc()
            return {"valid": False, "error": str(e)}

    def process_folder(self):
        """
        Process all audio files in the fast folder.

        Returns:
            dict: Processing results and statistics
        """
        try:
            print(f"📁 Processing fast folder: {self.folder_path}")

            # Find all audio files
            audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
            audio_files = []

            for ext in audio_extensions:
                audio_files.extend(list(self.folder_path.glob(ext)))
                audio_files.extend(list(self.folder_path.glob(ext.upper())))

            if not audio_files:
                print("⚠️  No audio files found in fast folder")
                return {"processed": 0, "valid": 0, "invalid": 0, "files": []}

            print(f"Found {len(audio_files)} audio files")

            results = {
                "processed": 0,
                "valid": 0,
                "invalid": 0,
                "files": [],
                "statistics": {
                    "avg_score": 0,
                    "avg_tempo": 0,
                    "avg_energy": 0,
                    "score_distribution": [],
                    "common_issues": []
                }
            }

            scores = []
            tempos = []
            energies = []
            all_recommendations = []

            for audio_file in audio_files:
                print(f"\n🔍 Processing: {audio_file.name}")
                validation = self.validate_fast_audio(str(audio_file))

                file_result = {
                    "filename": audio_file.name,
                    "path": str(audio_file),
                    "valid": validation.get("valid", False),
                    "score": validation.get("overall_score", 0),
                    "tempo": validation.get("scores", {}).get("estimated_tempo", 0),
                    "energy": validation.get("scores", {}).get("mean_energy", 0),
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

                if "scores" in validation:
                    if "estimated_tempo" in validation["scores"]:
                        tempos.append(validation["scores"]["estimated_tempo"])
                    if "mean_energy" in validation["scores"]:
                        energies.append(validation["scores"]["mean_energy"])

                all_recommendations.extend(validation.get("recommendations", []))

            # Calculate statistics
            if scores:
                results["statistics"]["avg_score"] = float(np.mean(scores))
                results["statistics"]["score_distribution"] = [float(s) for s in scores]

            if tempos:
                results["statistics"]["avg_tempo"] = float(np.mean(tempos))

            if energies:
                results["statistics"]["avg_energy"] = float(np.mean(energies))

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
            print(f"   Valid fast files: {results['valid']}")
            print(f"   Invalid files: {results['invalid']}")
            print(f"   Average score: {results['statistics']['avg_score']:.3f}")
            print(f"   Average tempo: {results['statistics']['avg_tempo']:.0f} BPM")
            print(f"   Average energy: {results['statistics']['avg_energy']:.3f}")

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
        description="Fast Class Audio Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 fast.py --generate                      # Generate test audio
  python3 fast.py --generate --intensity high     # Generate high-intensity audio
  python3 fast.py --validate audio.wav            # Validate specific file
  python3 fast.py --process                       # Process entire folder
  python3 fast.py --batch 3 --intensity medium    # Generate 3 medium files
        """
    )

    parser.add_argument('--generate', action='store_true',
                       help='Generate fast audio file')
    parser.add_argument('--validate', type=str,
                       help='Validate specific audio file')
    parser.add_argument('--process', action='store_true',
                       help='Process entire fast folder')
    parser.add_argument('--intensity', choices=['low', 'medium', 'high'],
                       default='medium', help='Intensity level for generation')
    parser.add_argument('--batch', type=int, default=1,
                       help='Number of files to generate')
    parser.add_argument('--folder', type=str, default='fast',
                       help='Fast folder path')
    parser.add_argument('--output', type=str,
                       help='Output filename for generation')

    args = parser.parse_args()

    # Initialize handler
    handler = FastAudioHandler(args.folder)

    print("=" * 80)
    print("⚡ FAST CLASS AUDIO HANDLER")
    print("=" * 80)

    try:
        if args.generate:
            print(f"Generating {args.batch} fast audio file(s)...")
            for i in range(args.batch):
                filename = args.output
                if args.batch > 1 and filename:
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{i+1:02d}{ext}"

                audio, path = handler.generate_fast_audio(
                    filename=filename,
                    intensity_level=args.intensity
                )

                if audio is not None:
                    print(f"✅ Generated: {path}")
                else:
                    print(f"❌ Failed to generate file {i+1}")

        elif args.validate:
            print(f"Validating audio file: {args.validate}")
            results = handler.validate_fast_audio(args.validate)

            if results.get("valid"):
                print("✅ File is valid fast audio")
            else:
                print("❌ File does not meet fast criteria")

        elif args.process:
            print("Processing fast folder...")
            results = handler.process_folder()

            # Save results to JSON
            output_file = f"fast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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