# 🚀 **Raspberry Pi 4 Deployment Guide - YAMNet Speech Classification Pipeline (Git-based)**

This guide provides step-by-step instructions for deploying the YAMNet speech classification pipeline on Raspberry Pi 4 using the GitHub repository.

## 📋 **Prerequisites**

- Raspberry Pi 4 (4GB RAM recommended) with Raspberry Pi OS installed
- SSH access to the Pi (via PuTTY or terminal)
- Internet connection on the Pi
- USB microphone (optional, for real-time testing)
- Basic Linux command line knowledge

## 🔧 **Step 1: System Preparation and Repository Setup**

### **1.1 Update System and Install Dependencies**

SSH into your Raspberry Pi and run:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential system packages
sudo apt install -y python3-pip python3-venv python3-dev git
sudo apt install -y portaudio19-dev libasound2-dev
sudo apt install -y alsa-utils pulseaudio pulseaudio-utils
sudo apt install -y libsndfile1-dev libflac-dev libvorbis-dev

# Install audio codec libraries
sudo apt install -y wget curl
```

### **1.2 Clone the Anubhuti Repository**

```bash
# Clone the repository
git clone https://github.com/cpradeepk/anubhuti.git
cd anubhuti

# Verify repository structure
ls -la
```

**Expected Output:**
```
drwxr-xr-x  3 pi pi  4096 Dec  6 10:30 yamnet_implementation
drwxr-xr-x  2 pi pi  4096 Dec  6 10:30 slow
drwxr-xr-x  2 pi pi  4096 Dec  6 10:30 medium
drwxr-xr-x  2 pi pi  4096 Dec  6 10:30 fast
drwxr-xr-x  2 pi pi  4096 Dec  6 10:30 disturbance
-rw-r--r--  1 pi pi 15234 Dec  6 10:30 README.md
-rw-r--r--  1 pi pi  8456 Dec  6 10:30 DEPLOYMENT_GUIDE.md
```

### **1.3 Verify Model Files**

```bash
# Check YAMNet model files
ls -la yamnet_implementation/yamnet_models/

# Verify critical files exist
ls -la yamnet_implementation/yamnet_models/yamnet_classifier.h5
ls -la yamnet_implementation/yamnet_models/yamnet_model_metadata.json
```

**Expected Output:**
```
-rw-r--r-- 1 pi pi 26214400 Dec  6 10:30 yamnet_classifier.h5
-rw-r--r-- 1 pi pi     2048 Dec  6 10:30 yamnet_model_metadata.json
```

## 🐍 **Step 2: Python Environment Setup**

### **2.1 Create Virtual Environment**

```bash
# Navigate to yamnet_implementation directory
cd yamnet_implementation/

# Create virtual environment
python3 -m venv yamnet_env

# Activate virtual environment
source yamnet_env/bin/activate

# Verify activation (prompt should show (yamnet_env))
which python
```

### **2.2 Install Python Dependencies**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install TensorFlow for Raspberry Pi (this may take 10-15 minutes)
pip install tensorflow==2.13.0

# Install TensorFlow Hub
pip install tensorflow-hub==0.14.0

# Install audio processing libraries
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install pyaudio
pip install pydub

# Install other dependencies
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import tensorflow_hub as hub; print('TensorFlow Hub imported successfully')"
```

**Expected Output:**
```
TensorFlow version: 2.13.0
TensorFlow Hub imported successfully
```

## 🎵 **Step 3: Audio System Configuration**

### **3.1 Configure Audio Input**

```bash
# List audio devices
arecord -l
aplay -l

# Test microphone (if USB microphone is connected)
arecord -D plughw:1,0 -f cd -t wav -d 5 test_recording.wav
aplay test_recording.wav

# Configure default audio device
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type asym
    playback.pcm "plughw:0,0"
    capture.pcm "plughw:1,0"
}
ctl.!default {
    type hw
    card 1
}
EOF
```

### **3.2 Test YAMNet Required Audio Format**

```bash
# Record in YAMNet's required format (16kHz mono)
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -t wav -d 5 test_16khz_mono.wav

# Verify format
python3 -c "
import soundfile as sf
data, sr = sf.read('test_16khz_mono.wav')
print(f'Sample rate: {sr}Hz, Channels: {data.ndim}, Duration: {len(data)/sr:.1f}s')
"
```

**Expected Output:**
```
Sample rate: 16000Hz, Channels: 1, Duration: 5.0s
```

## 🤖 **Step 4: Model Deployment and Testing**

### **4.1 Test Model Loading**

```bash
# Ensure you're in the yamnet_implementation directory with virtual environment active
cd ~/anubhuti/yamnet_implementation/
source yamnet_env/bin/activate

# Test model loading
python3 -c "
import tensorflow as tf
print('Loading YAMNet classifier...')
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print(f'✅ Model loaded successfully!')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')
print(f'Total parameters: {model.count_params():,}')
"
```

**Expected Output:**
```
✅ Model loaded successfully!
Input shape: (None, 1024)
Output shape: (None, 4)
Total parameters: 657,156
```

### **4.2 Test Audio Classification**

```bash
# Test with recorded audio
python3 test_yamnet_model.py test_16khz_mono.wav

# Test with sample dataset files
python3 test_yamnet_model.py ../slow/Fhmm_slow.wav --quiet
python3 test_yamnet_model.py ../medium/Fhum_medium.wav --quiet
python3 test_yamnet_model.py ../fast/Fhum_fast.wav --quiet
python3 test_yamnet_model.py ../disturbance/Cough.wav --quiet
```

**Expected Output:**
```
🎯 Predicted Class: slow
📊 Confidence: 0.606 (60.6%)
🤖 Arduino Command: 1
🎮 Motor Action: Top motor vibrates ('slow' sound)
⏱️ Processing Time: 1.234s
```

## 🎙️ **Step 5: Real-time Audio Testing**

### **5.1 Create Real-time Test Script**

```bash
# Create real-time testing script
cat > realtime_pi_test.py << 'EOF'
#!/usr/bin/env python3
import pyaudio
import numpy as np
import tempfile
import soundfile as sf
import time
import os
from test_yamnet_model import YAMNetModelTester

# Audio configuration for Pi
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

def main():
    print("🎙️ YAMNet Real-time Audio Classification on Raspberry Pi")
    print("=" * 55)
    
    # Initialize model
    print("Loading YAMNet model...")
    tester = YAMNetModelTester('yamnet_models/yamnet_classifier.h5')
    print("✅ Model loaded successfully!")
    
    # Initialize audio
    p = pyaudio.PyAudio()
    
    # List available input devices
    print("\n🔊 Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
    
    try:
        # Open audio stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print(f"\n🎯 Recording {RECORD_SECONDS}-second chunks for classification...")
        print("Press Ctrl+C to stop\n")
        
        while True:
            print("🎤 Recording...", end=" ", flush=True)
            
            # Record audio
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
            
            # Save to temporary file and classify
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, RATE)
                
                # Classify
                start_time = time.time()
                result = tester.predict_single_file(tmp_file.name)
                inference_time = (time.time() - start_time) * 1000
                
                # Display results
                print(f"✅ Class: {result['predicted_class_name']:<12} "
                      f"Confidence: {result['confidence']:.3f} "
                      f"Arduino: {result['arduino_command']} "
                      f"Time: {inference_time:.0f}ms")
                
                # Clean up
                os.unlink(tmp_file.name)
            
            time.sleep(0.5)  # Brief pause
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping real-time classification...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("✅ Audio stream closed successfully")

if __name__ == "__main__":
    main()
EOF

chmod +x realtime_pi_test.py
```

### **5.2 Run Real-time Testing**

```bash
# Run real-time audio classification
python3 realtime_pi_test.py
```

**Expected Output:**
```
🎙️ YAMNet Real-time Audio Classification on Raspberry Pi
=======================================================
Loading YAMNet model...
✅ Model loaded successfully!

🔊 Available audio input devices:
  Device 1: USB Audio Device (Channels: 1)

🎯 Recording 3-second chunks for classification...
Press Ctrl+C to stop

🎤 Recording... ✅ Class: slow         Confidence: 0.856 Arduino: 1 Time: 45ms
🎤 Recording... ✅ Class: medium       Confidence: 0.923 Arduino: 2 Time: 42ms
```

## 🔄 **Step 6: Future Updates and Maintenance**

### **6.1 Update Repository**

```bash
# Navigate to repository directory
cd ~/anubhuti

# Pull latest changes
git pull origin main

# Check what changed
git log --oneline -5

# If Python dependencies changed, update them
cd yamnet_implementation/
source yamnet_env/bin/activate
pip install -r requirements.txt --upgrade
```

### **6.2 System Health Check**

```bash
# Create health check script
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess
import psutil

def check_deployment():
    print("🔍 YAMNet Pi Deployment Health Check")
    print("=" * 40)
    
    checks = []
    
    # Check 1: Repository and model files
    checks.append(("Repository exists", os.path.exists('/home/pi/anubhuti')))
    checks.append(("Model file exists", os.path.exists('/home/pi/anubhuti/yamnet_implementation/yamnet_models/yamnet_classifier.h5')))
    
    # Check 2: Python packages
    try:
        import tensorflow as tf
        checks.append(("TensorFlow imported", True))
        checks.append(("TensorFlow version OK", tf.__version__.startswith('2.')))
    except ImportError:
        checks.append(("TensorFlow imported", False))
        checks.append(("TensorFlow version OK", False))
    
    try:
        import tensorflow_hub as hub
        checks.append(("TensorFlow Hub imported", True))
    except ImportError:
        checks.append(("TensorFlow Hub imported", False))
    
    try:
        import librosa
        checks.append(("Librosa imported", True))
    except ImportError:
        checks.append(("Librosa imported", False))
    
    # Check 3: System resources
    memory = psutil.virtual_memory()
    checks.append(("Sufficient memory", memory.available > 500 * 1024 * 1024))  # 500MB
    
    # Check 4: Audio system
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        has_audio_input = 'card' in result.stdout
        checks.append(("Audio input device detected", has_audio_input))
    except:
        checks.append(("Audio input device detected", False))
    
    # Display results
    print()
    passed = 0
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("🎉 Deployment health check SUCCESSFUL!")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
    
    return passed == len(checks)

if __name__ == "__main__":
    success = check_deployment()
    sys.exit(0 if success else 1)
EOF

chmod +x health_check.py
python3 health_check.py
```

## 🚨 **Troubleshooting**

### **Common Issues and Solutions**

#### **Issue 1: Git Clone Fails**
```bash
# Check internet connection
ping -c 3 github.com

# If DNS issues, try with IP
git clone https://140.82.112.3/cpradeepk/anubhuti.git
```

#### **Issue 2: TensorFlow Installation Fails**
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Retry TensorFlow installation
pip install tensorflow==2.13.0 --no-cache-dir
```

#### **Issue 3: Audio Device Not Found**
```bash
# Check USB connections
lsusb | grep -i audio

# Restart audio services
sudo systemctl restart alsa-state
pulseaudio --kill
pulseaudio --start

# Test audio again
arecord -l
```

#### **Issue 4: Model Loading Errors**
```bash
# Check model file integrity
ls -la yamnet_implementation/yamnet_models/yamnet_classifier.h5
md5sum yamnet_implementation/yamnet_models/yamnet_classifier.h5

# If corrupted, re-clone repository
cd ~
rm -rf anubhuti
git clone https://github.com/cpradeepk/anubhuti.git
```

## 🎉 **Deployment Complete!**

Your YAMNet speech classification pipeline is now successfully deployed on Raspberry Pi 4! You can:

1. **Run real-time classification**: `python3 realtime_pi_test.py`
2. **Test individual files**: `python3 test_yamnet_model.py audio_file.wav`
3. **Monitor system health**: `python3 health_check.py`
4. **Update the system**: `git pull origin main`

**Next Steps:**
- Connect Arduino wristband for haptic feedback testing
- Deploy in classroom environment
- Set up as a system service for automatic startup

**Your YAMNet pipeline is ready to help students with precise vocal sound detection!** 🎵🤖✨
