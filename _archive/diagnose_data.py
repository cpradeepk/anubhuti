#!/usr/bin/env python3
"""
Data Diagnosis Script for Audio Classification Model Bias

This script analyzes your data.json file to identify potential causes of model bias:
- Class distribution imbalance
- Label correctness
- Feature similarity between classes
- Data quality issues

Run this first to diagnose the root cause of prediction bias.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def analyze_dataset(data_file="data.json"):
    """
    Comprehensive analysis of the training dataset.
    """
    print("=" * 80)
    print("🔍 AUDIO CLASSIFICATION DATA DIAGNOSIS")
    print("=" * 80)
    
    # Load data
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        print("✅ Successfully loaded data.json")
    except Exception as e:
        print(f"❌ Error loading {data_file}: {e}")
        return
    
    # Extract components
    labels = data.get("labels", [])
    mfcc_features = data.get("mfcc", [])
    file_paths = data.get("files", [])
    class_mapping = data.get("mapping", [])
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total samples: {len(labels)}")
    print(f"   Classes: {len(class_mapping)}")
    print(f"   Class mapping: {class_mapping}")
    print(f"   MFCC features shape: {np.array(mfcc_features).shape if mfcc_features else 'None'}")
    
    # 1. CLASS DISTRIBUTION ANALYSIS
    print(f"\n📈 CLASS DISTRIBUTION ANALYSIS:")
    print("-" * 50)
    
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"{'Class':<12} {'Label':<6} {'Count':<6} {'Percentage':<12} {'Files'}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_mapping):
        count = label_counts.get(i, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{class_name:<12} {i:<6} {count:<6} {percentage:<12.1f}% {'█' * int(percentage/5)}")
    
    # Check for severe imbalance
    if label_counts:
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n⚖️  IMBALANCE ANALYSIS:")
        print(f"   Max class samples: {max_count}")
        print(f"   Min class samples: {min_count}")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("🚨 CRITICAL: Severe class imbalance detected!")
            print("   This is likely causing your model bias.")
            print("   Solution: Balance your dataset or use class weights.")
        elif imbalance_ratio > 2:
            print("⚠️  WARNING: Moderate class imbalance detected.")
        else:
            print("✅ Class distribution is reasonably balanced.")
    
    # 2. LABEL-FILE MAPPING ANALYSIS
    print(f"\n📁 LABEL-FILE MAPPING ANALYSIS:")
    print("-" * 50)
    
    # Group files by predicted class from filename
    filename_class_mapping = {}
    label_errors = []
    
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        # Extract class from file path
        file_class = None
        for j, class_name in enumerate(class_mapping):
            if class_name in file_path.lower():
                file_class = j
                break
        
        if file_class is not None:
            if file_class != label:
                label_errors.append({
                    'file': file_path,
                    'expected_class': file_class,
                    'actual_label': label,
                    'expected_name': class_mapping[file_class],
                    'actual_name': class_mapping[label]
                })
        
        filename_class_mapping[file_path] = {
            'filename_class': file_class,
            'assigned_label': label
        }
    
    if label_errors:
        print(f"🚨 CRITICAL: {len(label_errors)} LABEL ERRORS DETECTED!")
        print("   Files with incorrect labels:")
        for error in label_errors[:10]:  # Show first 10 errors
            print(f"   ❌ {error['file']}")
            print(f"      Expected: {error['expected_name']} ({error['expected_class']})")
            print(f"      Got: {error['actual_name']} ({error['actual_label']})")
        
        if len(label_errors) > 10:
            print(f"   ... and {len(label_errors) - 10} more errors")
        
        print("\n💡 SOLUTION: This is likely your main problem!")
        print("   The preprocessing script assigned wrong labels to files.")
    else:
        print("✅ All file labels appear correct based on folder structure.")
    
    # 3. FEATURE SIMILARITY ANALYSIS
    print(f"\n🔊 FEATURE SIMILARITY ANALYSIS:")
    print("-" * 50)
    
    if mfcc_features:
        mfcc_array = np.array(mfcc_features)
        
        # Calculate mean features per class
        class_means = {}
        class_stds = {}
        
        for class_idx in range(len(class_mapping)):
            class_mask = np.array(labels) == class_idx
            if np.any(class_mask):
                class_features = mfcc_array[class_mask]
                class_means[class_idx] = np.mean(class_features, axis=0)
                class_stds[class_idx] = np.std(class_features, axis=0)
        
        # Calculate inter-class distances
        print("Inter-class feature distances:")
        for i in range(len(class_mapping)):
            for j in range(i+1, len(class_mapping)):
                if i in class_means and j in class_means:
                    distance = np.linalg.norm(class_means[i] - class_means[j])
                    print(f"   {class_mapping[i]} ↔ {class_mapping[j]}: {distance:.3f}")
        
        # Check for feature similarity
        all_distances = []
        for i in range(len(class_mapping)):
            for j in range(i+1, len(class_mapping)):
                if i in class_means and j in class_means:
                    distance = np.linalg.norm(class_means[i] - class_means[j])
                    all_distances.append(distance)
        
        if all_distances:
            avg_distance = np.mean(all_distances)
            print(f"\n   Average inter-class distance: {avg_distance:.3f}")
            
            if avg_distance < 5.0:
                print("⚠️  WARNING: Classes have very similar features!")
                print("   This could contribute to classification difficulty.")
            else:
                print("✅ Classes have sufficiently different features.")
    
    # 4. RECOMMENDATIONS
    print(f"\n💡 DIAGNOSIS SUMMARY & RECOMMENDATIONS:")
    print("=" * 60)
    
    if label_errors:
        print("🚨 PRIMARY ISSUE: Incorrect labels in data.json")
        print("   IMMEDIATE FIX: Run the label correction script")
        print("   COMMAND: python3 fix_labels.py")
    elif imbalance_ratio > 3:
        print("🚨 PRIMARY ISSUE: Severe class imbalance")
        print("   IMMEDIATE FIX: Balance dataset or use class weights")
        print("   COMMAND: python3 balance_dataset.py")
    elif len(set(labels)) < len(class_mapping):
        print("🚨 PRIMARY ISSUE: Missing classes in training data")
        print("   Some classes have no training samples")
    else:
        print("✅ Data appears structurally correct")
        print("   Issue might be in model training or architecture")
    
    return {
        'label_errors': label_errors,
        'imbalance_ratio': imbalance_ratio if 'imbalance_ratio' in locals() else 1.0,
        'class_counts': dict(label_counts),
        'total_samples': total_samples
    }

if __name__ == "__main__":
    results = analyze_dataset()
    
    # Save diagnosis results
    with open('diagnosis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Diagnosis results saved to: diagnosis_results.json")
    print("Run the recommended fix commands above to resolve the issues.")
