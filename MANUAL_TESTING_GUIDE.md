# 🧪 **Complete Manual Testing Guide for YAMNet Pipeline**

## 🚀 **Quick Manual Testing Commands**

### **Method 1: Interactive Testing Script**
```bash
cd yamnet_implementation/
python manual_test_script.py
```

This launches an interactive menu where you can:
- Test individual files
- Test all classes automatically
- Validate Arduino commands
- Run performance benchmarks
- View comprehensive summaries

### **Method 2: Direct Command Line Testing**

#### **Basic Single File Testing**
```bash
# Test each class with detailed output
python test_yamnet_model.py ../slow/Fhmm_slow.wav
python test_yamnet_model.py ../medium/Fhum_medium.wav  
python test_yamnet_model.py ../fast/Fhum_fast.wav
python test_yamnet_model.py ../disturbance/Cough.wav

# Quiet mode (just results)
python test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet
```

#### **Batch Testing Multiple Files**
```bash
# Test multiple files from each class
for file in ../slow/*.wav; do
    echo "Testing: $file"
    python test_yamnet_model.py "$file" --quiet
done

for file in ../medium/*.wav; do
    echo "Testing: $file"  
    python test_yamnet_model.py "$file" --quiet
done

for file in ../fast/*.wav; do
    echo "Testing: $file"
    python test_yamnet_model.py "$file" --quiet
done

for file in ../disturbance/*.wav; do
    echo "Testing: $file"
    python test_yamnet_model.py "$file" --quiet
done
```

## 📊 **Manual Testing Checklist**

### **✅ Test 1: Model Accuracy Validation**

**What to Test:**
- At least 2-3 files from each class
- Check prediction accuracy
- Verify confidence scores (should be >30%)

**Commands:**
```bash
# Slow class testing
python test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet
python test_yamnet_model.py ../slow/Fsoo_slow.wav --quiet
python test_yamnet_model.py ../slow/Mhmm_slow.wav --quiet

# Expected: All should predict "slow" class
```

**Success Criteria:**
- ≥70% accuracy per class
- Confidence scores >0.3
- No system crashes

### **✅ Test 2: Arduino Command Mapping**

**What to Test:**
- Verify correct Arduino commands for each class
- Check motor control mapping

**Commands:**
```bash
# Test one file from each class and check Arduino commands
python test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet
# Expected: Arduino Command: 1 (Top motor)

python test_yamnet_model.py ../medium/Fhum_medium.wav --quiet  
# Expected: Arduino Command: 2 (Bottom motor)

python test_yamnet_model.py ../fast/Fhum_fast.wav --quiet
# Expected: Arduino Command: 3 (Both motors)

python test_yamnet_model.py ../disturbance/Cough.wav --quiet
# Expected: Arduino Command: 0 (No vibration)
```

**Success Criteria:**
- slow → Command 1 ✅
- medium → Command 2 ✅  
- fast → Command 3 ✅
- disturbance → Command 0 ✅

### **✅ Test 3: Performance Benchmarking**

**What to Test:**
- Inference speed per file
- Memory usage
- System responsiveness

**Commands:**
```bash
# Time individual predictions
time python test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet
time python test_yamnet_model.py ../medium/Fhum_medium.wav --quiet
time python test_yamnet_model.py ../fast/Fhum_fast.wav --quiet
time python test_yamnet_model.py ../disturbance/Cough.wav --quiet
```

**Success Criteria:**
- First prediction: <2 seconds (includes model loading)
- Subsequent predictions: <50ms each
- No memory leaks or crashes

### **✅ Test 4: Edge Case Testing**

**What to Test:**
- Very short audio files
- Very long audio files  
- Non-existent files (error handling)
- Corrupted files (error handling)

**Commands:**
```bash
# Test error handling
python test_yamnet_model.py non_existent_file.wav
# Expected: Should show error message gracefully

# Test with different file types if available
python test_yamnet_model.py ../slow/some_file.mp3 --quiet
```

**Success Criteria:**
- Graceful error handling
- No system crashes
- Clear error messages

### **✅ Test 5: Long Audio Processing**

**What to Test:**
- Sliding window processing
- Chunk-based analysis
- Memory efficiency

**Commands:**
```bash
# Test long audio processing (if you have long files)
python process_long_audio.py long_audio_file.wav --chunk-details

# Or create a longer file by concatenating existing ones
# Then test with the long audio processor
```

**Success Criteria:**
- Processes without memory issues
- Provides chunk-by-chunk analysis
- Reasonable processing speed

## 🎯 **Step-by-Step Manual Testing Procedure**

### **Step 1: Environment Setup**
```bash
cd yamnet_implementation/
source ../venv/bin/activate  # Ensure virtual environment is active
```

### **Step 2: Quick Smoke Test**
```bash
# Test one file from each class to ensure basic functionality
python test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet
python test_yamnet_model.py ../medium/Fhum_medium.wav --quiet
python test_yamnet_model.py ../fast/Fhum_fast.wav --quiet
python test_yamnet_model.py ../disturbance/Cough.wav --quiet
```

**Expected Output Example:**
```
File: Fhmm_slow.wav
Predicted Class: slow
Confidence: 0.606 (60.6%)
Arduino Command: 1
```

### **Step 3: Comprehensive Class Testing**
```bash
# Run interactive testing script
python manual_test_script.py

# Select option 2: "Test all classes (2 files each)"
```

### **Step 4: Arduino Integration Validation**
```bash
# Run interactive testing script
python manual_test_script.py

# Select option 4: "Test Arduino commands"
```

### **Step 5: Performance Validation**
```bash
# Run interactive testing script  
python manual_test_script.py

# Select option 5: "Performance benchmark"
```

## 📊 **Expected Results & Success Criteria**

### **Excellent Performance (Ready for Production)**
- **Accuracy**: ≥90% across all classes
- **Inference Speed**: <50ms after model loading
- **Arduino Commands**: 100% correct mapping
- **Confidence**: Average >0.7
- **Error Handling**: Graceful failure modes

### **Good Performance (Minor Improvements)**
- **Accuracy**: 70-89% across all classes
- **Inference Speed**: 50-100ms after model loading
- **Arduino Commands**: ≥75% correct mapping
- **Confidence**: Average >0.5
- **Error Handling**: Most errors handled gracefully

### **Needs Improvement**
- **Accuracy**: <70% on any class
- **Inference Speed**: >100ms consistently
- **Arduino Commands**: <75% correct mapping
- **Confidence**: Average <0.5
- **Error Handling**: System crashes or unclear errors

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: Low Accuracy**
```bash
# Check if model files exist
ls -la yamnet_models/

# Verify dataset structure
python validate_dataset.py ../

# Retrain if necessary
python train_yamnet_model.py --dataset ../
```

### **Issue 2: Slow Performance**
```bash
# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# Consider TensorFlow Lite conversion for production
```

### **Issue 3: Arduino Command Errors**
```bash
# Check metadata file
cat yamnet_models/yamnet_model_metadata.json | grep -A 10 "arduino_mapping"

# Verify class mapping is correct
```

### **Issue 4: Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

## 🎉 **Manual Testing Complete!**

After completing manual testing, you should have:

1. **✅ Verified Model Accuracy**: All classes predict correctly
2. **✅ Validated Arduino Integration**: Commands map properly  
3. **✅ Confirmed Performance**: Speed meets requirements
4. **✅ Tested Edge Cases**: System handles errors gracefully
5. **✅ Documented Results**: Test results saved for reference

Your YAMNet speech classification pipeline is now thoroughly validated and ready for production deployment! 🚀
