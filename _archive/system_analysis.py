#!/usr/bin/env python3
"""
Complete System Analysis for DS-CNN Audio Classification

This script provides comprehensive analysis of the current system performance,
identifies issues, and provides recommendations for improvement.
"""

import os
import json
import numpy as np
import tensorflow as tf
from collections import Counter
import subprocess

def analyze_current_system():
    """
    Analyze the current DS-CNN system performance.
    """
    print("=" * 80)
    print("📊 COMPLETE SYSTEM ANALYSIS")
    print("=" * 80)
    
    # 1. Model Analysis
    print("\n🤖 MODEL ANALYSIS:")
    print("-" * 50)
    
    try:
        model = tf.keras.models.load_model("model.h5")
        print(f"✅ Model loaded successfully")
        print(f"   Architecture: DS-CNN (Depthwise Separable CNN)")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output classes: {model.output_shape[1]}")
        
        # Load metadata
        if os.path.exists("model_metadata.json"):
            with open("model_metadata.json", 'r') as f:
                metadata = json.load(f)
            print(f"   Training accuracy: {metadata.get('overall_accuracy', 'Unknown'):.3f}")
            print(f"   CV accuracy: {metadata.get('cv_accuracy_mean', 'Unknown'):.3f}")
        
    except Exception as e:
        print(f"❌ Model analysis failed: {e}")
    
    # 2. Dataset Analysis
    print("\n📊 DATASET ANALYSIS:")
    print("-" * 50)
    
    try:
        with open("data.json", 'r') as f:
            data = json.load(f)
        
        labels = data["labels"]
        class_mapping = data["mapping"]
        
        print(f"   Total samples: {len(labels)}")
        print(f"   Classes: {len(class_mapping)}")
        
        class_counts = Counter(labels)
        print(f"   Class distribution:")
        for i, class_name in enumerate(class_mapping):
            count = class_counts.get(i, 0)
            percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
            print(f"     {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Check for imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print("   ⚠️  Dataset is imbalanced")
        else:
            print("   ✅ Dataset is reasonably balanced")
            
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
    
    # 3. Test All Audio Files
    print("\n🧪 TESTING ALL AUDIO FILES:")
    print("-" * 50)
    
    test_results = []
    class_folders = ["disturbance", "slow", "medium", "fast"]
    
    for class_idx, class_folder in enumerate(class_folders):
        if not os.path.exists(class_folder):
            continue
            
        audio_files = [f for f in os.listdir(class_folder) if f.endswith('.wav')]
        print(f"\n   Testing {class_folder} files ({len(audio_files)} files):")
        
        correct_predictions = 0
        total_files = len(audio_files)
        
        for audio_file in audio_files[:5]:  # Test first 5 files per class
            file_path = os.path.join(class_folder, audio_file)
            
            try:
                # Run test script and capture output
                result = subprocess.run(
                    ["python3", "test_dscnn_model.py", file_path],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    # Extract prediction from output
                    lines = result.stdout.split('\n')
                    predicted_class = None
                    confidence = None
                    
                    for line in lines:
                        if "Predicted Class:" in line:
                            predicted_class = line.split(":")[1].strip()
                        elif "Confidence:" in line:
                            confidence = float(line.split(":")[1].split("(")[0].strip())
                    
                    if predicted_class and confidence:
                        is_correct = (predicted_class == class_folder)
                        if is_correct:
                            correct_predictions += 1
                        
                        test_results.append({
                            'file': audio_file,
                            'true_class': class_folder,
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'correct': is_correct
                        })
                        
                        status = "✅" if is_correct else "❌"
                        print(f"     {status} {audio_file[:20]:<20}: {predicted_class} ({confidence:.3f})")
                
            except Exception as e:
                print(f"     ❌ {audio_file}: Error - {e}")
        
        if total_files > 0:
            accuracy = correct_predictions / min(total_files, 5)
            print(f"   Class accuracy: {accuracy:.3f} ({correct_predictions}/{min(total_files, 5)})")
    
    # 4. Overall Performance Analysis
    print("\n📈 OVERALL PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    if test_results:
        total_correct = sum(1 for r in test_results if r['correct'])
        total_tested = len(test_results)
        overall_accuracy = total_correct / total_tested
        
        print(f"   Overall accuracy: {overall_accuracy:.3f} ({total_correct}/{total_tested})")
        
        # Analyze predictions by class
        prediction_counts = Counter([r['predicted_class'] for r in test_results])
        print(f"   Prediction distribution:")
        for class_name, count in prediction_counts.items():
            percentage = (count / total_tested * 100)
            print(f"     {class_name}: {count} predictions ({percentage:.1f}%)")
        
        # Check for bias
        dominant_class = prediction_counts.most_common(1)[0]
        dominant_percentage = (dominant_class[1] / total_tested * 100)
        
        if dominant_percentage > 70:
            print(f"   🚨 CRITICAL: Model heavily biased toward '{dominant_class[0]}' ({dominant_percentage:.1f}%)")
        elif dominant_percentage > 50:
            print(f"   ⚠️  WARNING: Model biased toward '{dominant_class[0]}' ({dominant_percentage:.1f}%)")
        else:
            print(f"   ✅ Model predictions reasonably distributed")
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in test_results])
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        if avg_confidence < 0.4:
            print("   ⚠️  Low confidence - model is uncertain")
        elif avg_confidence > 0.7:
            print("   ✅ High confidence - model is confident")
        else:
            print("   ✅ Moderate confidence - acceptable for small dataset")
    
    # 5. System Readiness Assessment
    print("\n🎯 SYSTEM READINESS ASSESSMENT:")
    print("-" * 50)
    
    readiness_score = 0
    max_score = 10
    
    # Check model exists
    if os.path.exists("model.h5"):
        print("   ✅ Model file exists (+1)")
        readiness_score += 1
    else:
        print("   ❌ Model file missing (0)")
    
    # Check metadata exists
    if os.path.exists("model_metadata.json"):
        print("   ✅ Metadata file exists (+1)")
        readiness_score += 1
    else:
        print("   ❌ Metadata file missing (0)")
    
    # Check real-time processor
    if os.path.exists("realtime_audio_processor.py"):
        print("   ✅ Real-time processor ready (+1)")
        readiness_score += 1
    else:
        print("   ❌ Real-time processor missing (0)")
    
    # Check Arduino code
    if os.path.exists("arduino_wristband.ino"):
        print("   ✅ Arduino code ready (+1)")
        readiness_score += 1
    else:
        print("   ❌ Arduino code missing (0)")
    
    # Check DS-CNN test script
    if os.path.exists("test_dscnn_model.py"):
        print("   ✅ DS-CNN test script ready (+1)")
        readiness_score += 1
    else:
        print("   ❌ DS-CNN test script missing (0)")
    
    # Check deployment scripts
    if os.path.exists("setup_raspberry_pi.sh"):
        print("   ✅ Deployment scripts ready (+1)")
        readiness_score += 1
    else:
        print("   ❌ Deployment scripts missing (0)")
    
    # Check model performance
    if test_results:
        if overall_accuracy > 0.6:
            print("   ✅ Model accuracy acceptable (+2)")
            readiness_score += 2
        elif overall_accuracy > 0.4:
            print("   ⚠️  Model accuracy moderate (+1)")
            readiness_score += 1
        else:
            print("   ❌ Model accuracy too low (0)")
        
        if dominant_percentage < 60:
            print("   ✅ Model bias acceptable (+1)")
            readiness_score += 1
        else:
            print("   ❌ Model too biased (0)")
        
        if avg_confidence > 0.3:
            print("   ✅ Model confidence acceptable (+1)")
            readiness_score += 1
        else:
            print("   ❌ Model confidence too low (0)")
    
    print(f"\n   📊 READINESS SCORE: {readiness_score}/{max_score} ({readiness_score/max_score*100:.0f}%)")
    
    # 6. Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    
    if readiness_score >= 8:
        print("   🎉 SYSTEM READY FOR DEPLOYMENT!")
        print("   Next steps:")
        print("   1. Deploy to Raspberry Pi using setup_raspberry_pi.sh")
        print("   2. Upload Arduino code to wristband")
        print("   3. Test end-to-end system")
        print("   4. Fine-tune based on real-world performance")
        
    elif readiness_score >= 6:
        print("   ✅ SYSTEM MOSTLY READY - Minor improvements needed")
        print("   Priority fixes:")
        if test_results and overall_accuracy < 0.6:
            print("   - Improve model accuracy by adding more training data")
        if test_results and dominant_percentage > 60:
            print("   - Reduce model bias with balanced dataset")
        print("   - Test all components before deployment")
        
    else:
        print("   ⚠️  SYSTEM NEEDS SIGNIFICANT WORK")
        print("   Critical issues to address:")
        if not os.path.exists("model.h5"):
            print("   - Train and save the DS-CNN model")
        if test_results and overall_accuracy < 0.4:
            print("   - Collect more diverse training data")
        if test_results and dominant_percentage > 70:
            print("   - Fix severe model bias")
        print("   - Complete missing components")
    
    # 7. Next Steps
    print("\n🚀 IMMEDIATE NEXT STEPS:")
    print("-" * 50)
    
    if test_results and dominant_percentage > 60:
        print("   1. 🎵 Generate more diverse training data:")
        print("      python3 improve_model_accuracy.py")
        print("   2. 🔄 Retrain model with expanded dataset")
        print("   3. 🧪 Test improved model performance")
    
    print("   4. 🍓 Prepare Raspberry Pi deployment:")
    print("      bash setup_raspberry_pi.sh")
    print("   5. 🤖 Upload Arduino code to wristband")
    print("   6. 🔗 Test wireless communication")
    print("   7. 🎯 Perform end-to-end system test")
    
    return {
        'readiness_score': readiness_score,
        'max_score': max_score,
        'test_results': test_results,
        'overall_accuracy': overall_accuracy if test_results else 0,
        'dominant_percentage': dominant_percentage if test_results else 0
    }

if __name__ == "__main__":
    results = analyze_current_system()
    
    print(f"\n📄 Analysis complete. System readiness: {results['readiness_score']}/{results['max_score']}")
    
    # Save results
    with open("system_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("📄 Detailed results saved to: system_analysis_results.json")
