#!/usr/bin/env python3
"""
Dataset Validation Script for YAMNet Pipeline

This script validates the dataset structure, checks audio files, and provides
comprehensive statistics before training. Helps identify potential issues early.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import Counter, defaultdict
import librosa
import soundfile as sf
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    """
    Comprehensive dataset validation for YAMNet pipeline.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset validator.
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.expected_classes = ['slow', 'medium', 'fast', 'disturbance']
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        
        # Statistics storage
        self.validation_results = {
            'dataset_path': str(self.dataset_path),
            'structure_valid': False,
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'class_distribution': {},
            'format_distribution': {},
            'duration_stats': {},
            'sample_rate_stats': {},
            'issues': [],
            'recommendations': []
        }
        
        logger.info(f"🔍 Dataset Validator initialized for: {self.dataset_path}")
    
    def validate_structure(self) -> bool:
        """
        Validate dataset directory structure.
        
        Returns:
            True if structure is valid, False otherwise
        """
        logger.info("📁 Validating dataset structure...")
        
        if not self.dataset_path.exists():
            self.validation_results['issues'].append(f"Dataset directory does not exist: {self.dataset_path}")
            return False
        
        if not self.dataset_path.is_dir():
            self.validation_results['issues'].append(f"Dataset path is not a directory: {self.dataset_path}")
            return False
        
        # Check for expected class directories
        missing_classes = []
        existing_classes = []
        
        for class_name in self.expected_classes:
            class_dir = self.dataset_path / class_name
            if class_dir.exists() and class_dir.is_dir():
                existing_classes.append(class_name)
            else:
                missing_classes.append(class_name)
        
        if missing_classes:
            self.validation_results['issues'].append(f"Missing class directories: {missing_classes}")
        
        if not existing_classes:
            self.validation_results['issues'].append("No valid class directories found")
            return False
        
        # Check for unexpected directories
        unexpected_dirs = []
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name not in self.expected_classes:
                unexpected_dirs.append(item.name)
        
        if unexpected_dirs:
            self.validation_results['issues'].append(f"Unexpected directories found: {unexpected_dirs}")
            self.validation_results['recommendations'].append(
                f"Consider removing or renaming: {unexpected_dirs}"
            )
        
        logger.info(f"✅ Found {len(existing_classes)} valid class directories: {existing_classes}")
        if missing_classes:
            logger.warning(f"⚠️  Missing class directories: {missing_classes}")
        
        self.validation_results['structure_valid'] = len(existing_classes) >= 2
        return self.validation_results['structure_valid']
    
    def validate_audio_files(self) -> Dict:
        """
        Validate all audio files in the dataset.
        
        Returns:
            Dictionary with validation statistics
        """
        logger.info("🎵 Validating audio files...")
        
        file_stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'class_counts': defaultdict(int),
            'format_counts': defaultdict(int),
            'durations': [],
            'sample_rates': [],
            'invalid_files_list': []
        }
        
        # Process each class directory
        for class_name in self.expected_classes:
            class_dir = self.dataset_path / class_name
            if not class_dir.exists():
                continue
            
            logger.info(f"📂 Processing {class_name}/ directory...")
            
            # Find all potential audio files
            audio_files = []
            for ext in self.supported_formats:
                audio_files.extend(class_dir.glob(f"*{ext}"))
                audio_files.extend(class_dir.glob(f"*{ext.upper()}"))
            
            if not audio_files:
                self.validation_results['issues'].append(f"No audio files found in {class_name}/ directory")
                continue
            
            # Validate each audio file
            for audio_file in tqdm(audio_files, desc=f"Validating {class_name}", leave=False):
                file_stats['total_files'] += 1
                
                try:
                    # Try to load audio file
                    audio, sr = librosa.load(audio_file, sr=None, duration=1.0)  # Load first second for validation
                    
                    if len(audio) == 0:
                        raise ValueError("Empty audio file")
                    
                    # Get full file info
                    info = sf.info(audio_file)
                    duration = info.duration
                    sample_rate = info.samplerate
                    
                    # Record statistics
                    file_stats['valid_files'] += 1
                    file_stats['class_counts'][class_name] += 1
                    file_stats['format_counts'][audio_file.suffix.lower()] += 1
                    file_stats['durations'].append(duration)
                    file_stats['sample_rates'].append(sample_rate)
                    
                except Exception as e:
                    file_stats['invalid_files'] += 1
                    file_stats['invalid_files_list'].append({
                        'file': str(audio_file),
                        'class': class_name,
                        'error': str(e)
                    })
                    logger.warning(f"⚠️  Invalid file {audio_file.name}: {e}")
        
        # Update validation results
        self.validation_results.update({
            'total_files': file_stats['total_files'],
            'valid_files': file_stats['valid_files'],
            'invalid_files': file_stats['invalid_files'],
            'class_distribution': dict(file_stats['class_counts']),
            'format_distribution': dict(file_stats['format_counts'])
        })
        
        # Calculate duration statistics
        if file_stats['durations']:
            durations = np.array(file_stats['durations'])
            self.validation_results['duration_stats'] = {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
                'median': float(np.median(durations))
            }
        
        # Calculate sample rate statistics
        if file_stats['sample_rates']:
            sr_counts = Counter(file_stats['sample_rates'])
            self.validation_results['sample_rate_stats'] = {
                'unique_rates': list(sr_counts.keys()),
                'most_common': sr_counts.most_common(1)[0][0],
                'distribution': dict(sr_counts)
            }
        
        # Add invalid files to issues
        if file_stats['invalid_files_list']:
            self.validation_results['issues'].extend([
                f"Invalid file: {item['file']} ({item['error']})"
                for item in file_stats['invalid_files_list'][:5]  # Show first 5
            ])
            if len(file_stats['invalid_files_list']) > 5:
                self.validation_results['issues'].append(
                    f"... and {len(file_stats['invalid_files_list']) - 5} more invalid files"
                )
        
        logger.info(f"✅ Audio validation completed: {file_stats['valid_files']}/{file_stats['total_files']} valid files")
        
        return file_stats
    
    def analyze_dataset_balance(self):
        """
        Analyze dataset balance and provide recommendations.
        """
        logger.info("⚖️  Analyzing dataset balance...")
        
        class_counts = self.validation_results['class_distribution']
        if not class_counts:
            return
        
        total_files = sum(class_counts.values())
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Analyze balance
        if imbalance_ratio > 3.0:
            self.validation_results['issues'].append(
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
            )
            self.validation_results['recommendations'].append(
                "Consider collecting more data for underrepresented classes or using data augmentation"
            )
        elif imbalance_ratio > 2.0:
            self.validation_results['issues'].append(
                f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
            )
            self.validation_results['recommendations'].append(
                "Consider using class weights during training to handle imbalance"
            )
        
        # Check minimum samples per class
        if min_count < 10:
            self.validation_results['issues'].append(
                f"Very few samples in some classes (minimum: {min_count})"
            )
            self.validation_results['recommendations'].append(
                "Collect at least 20-50 samples per class for better performance"
            )
        elif min_count < 20:
            self.validation_results['recommendations'].append(
                "Consider collecting more samples per class (current minimum: {min_count})"
            )
    
    def generate_recommendations(self):
        """
        Generate specific recommendations based on validation results.
        """
        logger.info("💡 Generating recommendations...")
        
        # Sample rate recommendations
        if 'sample_rate_stats' in self.validation_results:
            sr_stats = self.validation_results['sample_rate_stats']
            if len(sr_stats['unique_rates']) > 1:
                self.validation_results['recommendations'].append(
                    f"Multiple sample rates detected: {sr_stats['unique_rates']}. "
                    "YAMNet will automatically resample to 16kHz, but consistent rates are preferred."
                )
        
        # Duration recommendations
        if 'duration_stats' in self.validation_results:
            dur_stats = self.validation_results['duration_stats']
            if dur_stats['std'] > dur_stats['mean']:
                self.validation_results['recommendations'].append(
                    f"High duration variance detected (std: {dur_stats['std']:.1f}s). "
                    "Consider normalizing audio lengths for more consistent training."
                )
            
            if dur_stats['min'] < 1.0:
                self.validation_results['recommendations'].append(
                    f"Very short audio files detected (min: {dur_stats['min']:.1f}s). "
                    "Consider removing files shorter than 1 second."
                )
        
        # Format recommendations
        format_dist = self.validation_results['format_distribution']
        if len(format_dist) > 2:
            self.validation_results['recommendations'].append(
                f"Multiple audio formats detected: {list(format_dist.keys())}. "
                "Consider standardizing to WAV format for best compatibility."
            )
    
    def print_validation_report(self):
        """
        Print comprehensive validation report.
        """
        results = self.validation_results
        
        print("\n" + "="*80)
        print("🔍 DATASET VALIDATION REPORT")
        print("="*80)
        
        print(f"📁 Dataset Path: {results['dataset_path']}")
        print(f"📊 Structure Valid: {'✅ Yes' if results['structure_valid'] else '❌ No'}")
        
        print(f"\n📈 File Statistics:")
        print(f"   Total Files: {results['total_files']}")
        print(f"   Valid Files: {results['valid_files']} ({results['valid_files']/results['total_files']*100:.1f}%)" if results['total_files'] > 0 else "   Valid Files: 0")
        print(f"   Invalid Files: {results['invalid_files']}")
        
        if results['class_distribution']:
            print(f"\n📊 Class Distribution:")
            total = sum(results['class_distribution'].values())
            for class_name, count in results['class_distribution'].items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   {class_name:<12}: {count:3d} files ({percentage:.1f}%)")
        
        if results['format_distribution']:
            print(f"\n🎵 Format Distribution:")
            for format_name, count in results['format_distribution'].items():
                print(f"   {format_name:<8}: {count} files")
        
        if results['duration_stats']:
            dur = results['duration_stats']
            print(f"\n⏱️  Duration Statistics:")
            print(f"   Mean: {dur['mean']:.2f}s ± {dur['std']:.2f}s")
            print(f"   Range: {dur['min']:.2f}s - {dur['max']:.2f}s")
            print(f"   Median: {dur['median']:.2f}s")
        
        if results['sample_rate_stats']:
            sr = results['sample_rate_stats']
            print(f"\n🔊 Sample Rate Statistics:")
            print(f"   Most Common: {sr['most_common']} Hz")
            print(f"   Unique Rates: {sr['unique_rates']}")
        
        if results['issues']:
            print(f"\n⚠️  Issues Found ({len(results['issues'])}):")
            for i, issue in enumerate(results['issues'][:10], 1):
                print(f"   {i}. {issue}")
            if len(results['issues']) > 10:
                print(f"   ... and {len(results['issues']) - 10} more issues")
        
        if results['recommendations']:
            print(f"\n💡 Recommendations ({len(results['recommendations'])}):")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if results['structure_valid'] and results['valid_files'] > 20 and len(results['issues']) < 3:
            print("   ✅ Dataset looks good for training!")
        elif results['structure_valid'] and results['valid_files'] > 10:
            print("   ⚠️  Dataset has some issues but should work for training")
        else:
            print("   ❌ Dataset needs significant improvements before training")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path: str = None):
        """
        Save validation report to JSON file.
        
        Args:
            output_path: Path to save report (optional)
        """
        if output_path is None:
            output_path = "dataset_validation_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"✅ Validation report saved to: {output_path}")
    
    def run_full_validation(self) -> Dict:
        """
        Run complete dataset validation pipeline.
        
        Returns:
            Validation results dictionary
        """
        logger.info("🚀 Starting full dataset validation...")
        
        # Step 1: Validate structure
        if not self.validate_structure():
            logger.error("❌ Dataset structure validation failed")
            return self.validation_results
        
        # Step 2: Validate audio files
        self.validate_audio_files()
        
        # Step 3: Analyze balance
        self.analyze_dataset_balance()
        
        # Step 4: Generate recommendations
        self.generate_recommendations()
        
        logger.info("✅ Dataset validation completed")
        
        return self.validation_results

def main():
    """
    Main function for dataset validation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Dataset for YAMNet Training")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("--output", help="Output path for validation report JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = DatasetValidator(args.dataset_path)
        
        # Run validation
        results = validator.run_full_validation()
        
        # Print report (unless quiet)
        if not args.quiet:
            validator.print_validation_report()
        
        # Save report
        validator.save_report(args.output)
        
        # Exit with appropriate code
        if results['structure_valid'] and results['valid_files'] > 0:
            logger.info("🎉 Dataset validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Dataset validation found critical issues")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
