#!/bin/bash
# Manual Testing Script for YAMNet Pipeline
# Run this script to perform comprehensive manual testing

echo "🧪 YAMNET MANUAL TESTING SUITE"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "test_yamnet_model.py" ]; then
    echo "❌ Error: Please run this script from yamnet_implementation/ directory"
    echo "   cd yamnet_implementation && bash run_manual_tests.sh"
    exit 1
fi

# Check if model exists
if [ ! -f "yamnet_models/yamnet_classifier.h5" ]; then
    echo "❌ Error: YAMNet model not found. Please train the model first."
    echo "   python train_yamnet_model.py --dataset ../"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# Test 1: Model Accuracy Testing
echo "🎯 TEST 1: MODEL ACCURACY TESTING"
echo "================================================"

declare -A test_results
total_tests=0
correct_tests=0

# Function to test a file and track results
test_file() {
    local file_path="$1"
    local expected_class="$2"
    local file_name=$(basename "$file_path")
    
    echo "Testing: $file_name"
    
    # Run test and capture output
    output=$(python test_yamnet_model.py "$file_path" --quiet 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        # Extract predicted class and confidence
        predicted_class=$(echo "$output" | grep "Predicted Class:" | cut -d: -f2 | xargs)
        confidence=$(echo "$output" | grep "Confidence:" | cut -d: -f2 | cut -d'(' -f1 | xargs)
        arduino_cmd=$(echo "$output" | grep "Arduino Command:" | cut -d: -f2 | xargs)
        
        total_tests=$((total_tests + 1))
        
        if [ "$predicted_class" = "$expected_class" ]; then
            echo "  ✅ $file_name: $predicted_class (Confidence: $confidence, Cmd: $arduino_cmd)"
            correct_tests=$((correct_tests + 1))
        else
            echo "  ❌ $file_name: $predicted_class (Expected: $expected_class, Confidence: $confidence)"
        fi
    else
        echo "  ❌ $file_name: FAILED TO PROCESS"
    fi
}

# Test slow class
echo ""
echo "Testing SLOW class files:"
echo "------------------------"
for file in ../slow/*.wav; do
    if [ -f "$file" ]; then
        test_file "$file" "slow"
    fi
done

# Test medium class  
echo ""
echo "Testing MEDIUM class files:"
echo "---------------------------"
for file in ../medium/*.wav; do
    if [ -f "$file" ]; then
        test_file "$file" "medium"
    fi
done

# Test fast class
echo ""
echo "Testing FAST class files:"
echo "-------------------------"
for file in ../fast/*.wav; do
    if [ -f "$file" ]; then
        test_file "$file" "fast"
    fi
done

# Test disturbance class
echo ""
echo "Testing DISTURBANCE class files:"
echo "--------------------------------"
for file in ../disturbance/*.wav; do
    if [ -f "$file" ]; then
        test_file "$file" "disturbance"
    fi
done

# Calculate accuracy
if [ $total_tests -gt 0 ]; then
    accuracy=$(echo "scale=1; $correct_tests * 100 / $total_tests" | bc -l)
    echo ""
    echo "📊 ACCURACY RESULTS:"
    echo "===================="
    echo "Total Tests: $total_tests"
    echo "Correct Predictions: $correct_tests"
    echo "Overall Accuracy: ${accuracy}%"
    
    if (( $(echo "$accuracy >= 85" | bc -l) )); then
        echo "✅ EXCELLENT: Accuracy ≥85%"
    elif (( $(echo "$accuracy >= 70" | bc -l) )); then
        echo "✅ GOOD: Accuracy ≥70%"
    else
        echo "⚠️  NEEDS IMPROVEMENT: Accuracy <70%"
    fi
fi

# Test 2: Arduino Command Validation
echo ""
echo "🤖 TEST 2: ARDUINO COMMAND VALIDATION"
echo "======================================"

echo "Testing Arduino command mapping..."

# Test one file from each class for Arduino commands
declare -A expected_commands=( ["slow"]=1 ["medium"]=2 ["fast"]=3 ["disturbance"]=0 )
arduino_tests_passed=0
arduino_total_tests=0

for class_name in slow medium fast disturbance; do
    # Find first file in class
    first_file=$(find "../$class_name" -name "*.wav" | head -1)
    
    if [ -f "$first_file" ]; then
        arduino_total_tests=$((arduino_total_tests + 1))
        
        output=$(python test_yamnet_model.py "$first_file" --quiet 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            predicted_class=$(echo "$output" | grep "Predicted Class:" | cut -d: -f2 | xargs)
            arduino_cmd=$(echo "$output" | grep "Arduino Command:" | cut -d: -f2 | xargs)
            expected_cmd=${expected_commands[$class_name]}
            
            if [ "$predicted_class" = "$class_name" ] && [ "$arduino_cmd" = "$expected_cmd" ]; then
                echo "  ✅ $class_name → Command $arduino_cmd (Correct)"
                arduino_tests_passed=$((arduino_tests_passed + 1))
            else
                echo "  ❌ $class_name → Command $arduino_cmd (Expected: $expected_cmd, Predicted: $predicted_class)"
            fi
        else
            echo "  ❌ $class_name → FAILED TO PROCESS"
        fi
    else
        echo "  ⚠️  $class_name → No test files found"
    fi
done

echo ""
echo "🎮 Motor Control Mapping:"
echo "  Command 0: No vibration (disturbance)"
echo "  Command 1: Top motor vibrates (slow)"
echo "  Command 2: Bottom motor vibrates (medium)"
echo "  Command 3: Both motors vibrate (fast)"

if [ $arduino_total_tests -gt 0 ]; then
    if [ $arduino_tests_passed -eq $arduino_total_tests ]; then
        echo "✅ Arduino Command Test: PASS ($arduino_tests_passed/$arduino_total_tests)"
    else
        echo "❌ Arduino Command Test: FAIL ($arduino_tests_passed/$arduino_total_tests)"
    fi
fi

# Test 3: Performance Benchmark
echo ""
echo "⚡ TEST 3: PERFORMANCE BENCHMARK"
echo "================================"

echo "Testing inference speed..."

# Test performance with a few files
perf_files=(
    "../slow/Fhmm_slow.wav"
    "../medium/Fhum_medium.wav"
    "../fast/Fhum_fast.wav"
    "../disturbance/Cough.wav"
)

total_time=0
file_count=0

for file in "${perf_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Testing: $(basename "$file")"
        
        # Time the prediction (excluding first run for warm-up)
        if [ $file_count -eq 0 ]; then
            # Warm-up run
            python test_yamnet_model.py "$file" --quiet >/dev/null 2>&1
        fi
        
        start_time=$(date +%s%N)
        python test_yamnet_model.py "$file" --quiet >/dev/null 2>&1
        end_time=$(date +%s%N)
        
        if [ $? -eq 0 ]; then
            duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
            echo "  ⏱️  $(basename "$file"): ${duration}ms"
            
            if [ $file_count -gt 0 ]; then  # Skip warm-up run from average
                total_time=$((total_time + duration))
            fi
        else
            echo "  ❌ $(basename "$file"): FAILED"
        fi
        
        file_count=$((file_count + 1))
    fi
done

if [ $file_count -gt 1 ]; then
    avg_time=$(( total_time / (file_count - 1) ))  # Exclude warm-up run
    echo ""
    echo "📊 Performance Results:"
    echo "  Average Inference Time: ${avg_time}ms"
    echo "  Throughput: $(echo "scale=1; 1000 / $avg_time" | bc -l) files/second"
    
    if [ $avg_time -le 50 ]; then
        echo "  ✅ EXCELLENT: Average time ≤50ms"
    elif [ $avg_time -le 100 ]; then
        echo "  ✅ GOOD: Average time ≤100ms"
    else
        echo "  ⚠️  SLOW: Average time >100ms"
    fi
fi

# Final Summary
echo ""
echo "🎯 FINAL TESTING SUMMARY"
echo "========================"

# Overall assessment
overall_status="EXCELLENT"

if [ $total_tests -gt 0 ]; then
    if (( $(echo "$accuracy < 70" | bc -l) )); then
        overall_status="NEEDS_IMPROVEMENT"
    elif (( $(echo "$accuracy < 85" | bc -l) )); then
        overall_status="GOOD"
    fi
fi

if [ $arduino_tests_passed -ne $arduino_total_tests ]; then
    if [ "$overall_status" = "EXCELLENT" ]; then
        overall_status="GOOD"
    elif [ "$overall_status" = "GOOD" ]; then
        overall_status="NEEDS_IMPROVEMENT"
    fi
fi

echo "Overall Assessment: $overall_status"

case $overall_status in
    "EXCELLENT")
        echo "🎉 All tests passed! System ready for production deployment."
        ;;
    "GOOD")
        echo "✅ Most tests passed. Minor improvements recommended."
        ;;
    "NEEDS_IMPROVEMENT")
        echo "⚠️  Some tests failed. Address issues before deployment."
        ;;
esac

echo ""
echo "📋 Next Steps:"
echo "1. Review any failed tests above"
echo "2. Test with actual Arduino hardware"
echo "3. Deploy to Raspberry Pi for real-world testing"
echo "4. Monitor performance in production environment"

echo ""
echo "🎉 Manual testing completed!"
