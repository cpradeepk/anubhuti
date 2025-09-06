#!/bin/bash
# Raspberry Pi Setup Script for Audio Classification System
# This script installs all dependencies and configures the system for deployment

echo "==============================================="
echo "🍓 RASPBERRY PI AUDIO CLASSIFICATION SETUP"
echo "==============================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and pip
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install system dependencies for audio processing
echo "🔊 Installing audio system dependencies..."
sudo apt install -y \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    pkg-config

# Install additional system tools
echo "🛠️  Installing system tools..."
sudo apt install -y \
    git \
    curl \
    wget \
    htop \
    screen \
    vim \
    alsa-utils \
    pulseaudio

# Create project directory
echo "📁 Creating project directory..."
PROJECT_DIR="/home/pi/audio_classification"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow Lite for Raspberry Pi
echo "🤖 Installing TensorFlow Lite..."
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl

# Install Python packages
echo "📚 Installing Python packages..."
pip install \
    numpy==1.21.6 \
    librosa==0.9.2 \
    soundfile==0.10.3 \
    sounddevice==0.4.5 \
    scikit-learn==1.1.3 \
    matplotlib==3.5.3 \
    pyserial==3.5 \
    requests==2.28.1

# Try to install TensorFlow (fallback to TensorFlow Lite if fails)
echo "🧠 Installing TensorFlow..."
pip install tensorflow==2.9.0 || {
    echo "⚠️  TensorFlow installation failed, using TensorFlow Lite only"
    pip install tflite-runtime
}

# Configure audio system
echo "🔊 Configuring audio system..."

# Add user to audio group
sudo usermod -a -G audio pi

# Configure ALSA
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
EOF

# Configure PulseAudio for low latency
mkdir -p ~/.config/pulse
cat > ~/.config/pulse/daemon.conf << 'EOF'
default-sample-format = s16le
default-sample-rate = 22050
default-sample-channels = 1
default-fragments = 2
default-fragment-size-msec = 25
EOF

# Create systemd service for auto-start
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/audio-classification.service > /dev/null << 'EOF'
[Unit]
Description=Audio Classification Service
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/audio_classification
Environment=PATH=/home/pi/audio_classification/venv/bin
ExecStart=/home/pi/audio_classification/venv/bin/python realtime_audio_processor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service (but don't start it yet)
sudo systemctl enable audio-classification.service

# Create configuration file
echo "📝 Creating configuration file..."
cat > config.json << 'EOF'
{
    "audio": {
        "sample_rate": 22050,
        "duration": 3.0,
        "buffer_size": 66150,
        "hop_length": 11025,
        "n_mfcc": 13,
        "n_frames": 130
    },
    "model": {
        "path": "model.h5",
        "metadata_path": "model_metadata.json",
        "confidence_threshold": 0.3
    },
    "arduino": {
        "serial_port": "/dev/ttyUSB0",
        "baud_rate": 9600,
        "wireless_ip": "192.168.1.100",
        "wireless_port": 8080
    },
    "motor_mapping": {
        "0": "NO_VIBRATION",
        "1": "TOP_MOTOR",
        "2": "BOTTOM_MOTOR", 
        "3": "BOTH_MOTORS"
    }
}
EOF

# Create startup script
echo "🚀 Creating startup script..."
cat > start_system.sh << 'EOF'
#!/bin/bash
# Audio Classification System Startup Script

echo "🎵 Starting Audio Classification System..."

# Activate virtual environment
source /home/pi/audio_classification/venv/bin/activate

# Check if model exists
if [ ! -f "model.h5" ]; then
    echo "❌ Model file not found! Please copy your trained model to this directory."
    exit 1
fi

# Check audio devices
echo "🔊 Available audio devices:"
python -c "import sounddevice as sd; print(sd.query_devices())"

# Start the real-time processor
echo "🚀 Starting real-time audio processor..."
python realtime_audio_processor.py

EOF

chmod +x start_system.sh

# Create test script
echo "🧪 Creating test script..."
cat > test_system.sh << 'EOF'
#!/bin/bash
# System Test Script

echo "🧪 Testing Audio Classification System..."

# Activate virtual environment
source /home/pi/audio_classification/venv/bin/activate

# Test audio input
echo "🎤 Testing microphone input..."
python -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds of audio...')
audio = sd.rec(int(3 * 22050), samplerate=22050, channels=1)
sd.wait()
print(f'Audio recorded: {audio.shape}, RMS: {np.sqrt(np.mean(audio**2)):.4f}')
"

# Test model loading
echo "🤖 Testing model loading..."
python -c "
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('model.h5')
    print(f'✅ Model loaded successfully: {model.count_params():,} parameters')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"

# Test Arduino communication
echo "📡 Testing Arduino communication..."
python -c "
import serial
import time
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)
    ser.write(b'1,0.8,200,500\n')
    print('✅ Serial communication test sent')
    ser.close()
except Exception as e:
    print(f'⚠️  Serial communication failed: {e}')
"

echo "✅ System test completed!"
EOF

chmod +x test_system.sh

# Create model conversion script for TensorFlow Lite
echo "📱 Creating TensorFlow Lite conversion script..."
cat > convert_to_tflite.py << 'EOF'
#!/usr/bin/env python3
"""
Convert trained model to TensorFlow Lite for better Raspberry Pi performance.
"""

import tensorflow as tf
import numpy as np
import json

def convert_model_to_tflite(model_path="model.h5", output_path="model.tflite"):
    """
    Convert Keras model to TensorFlow Lite format.
    """
    print(f"🔄 Converting {model_path} to TensorFlow Lite...")
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite model saved: {output_path}")
        print(f"   Original size: {len(open(model_path, 'rb').read()):,} bytes")
        print(f"   TFLite size: {len(tflite_model):,} bytes")
        
        # Test the converted model
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TensorFlow Lite model verified")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    convert_model_to_tflite()
EOF

chmod +x convert_to_tflite.py

# Create deployment checklist
echo "📋 Creating deployment checklist..."
cat > DEPLOYMENT_CHECKLIST.md << 'EOF'
# 🚀 Raspberry Pi Deployment Checklist

## Pre-deployment Setup
- [ ] Raspberry Pi OS installed and updated
- [ ] SSH enabled for remote access
- [ ] WiFi configured and connected
- [ ] Audio input device (USB microphone) connected
- [ ] Arduino wristband paired/connected

## File Transfer
Copy these files from your development machine to `/home/pi/audio_classification/`:
- [ ] `model.h5` - Trained DS-CNN model
- [ ] `model_metadata.json` - Model metadata
- [ ] `realtime_audio_processor.py` - Real-time processing script
- [ ] `dscnn_metadata.json` - DS-CNN specific metadata

## System Setup
- [ ] Run setup script: `bash setup_raspberry_pi.sh`
- [ ] Test system: `bash test_system.sh`
- [ ] Convert model to TensorFlow Lite: `python convert_to_tflite.py`

## Arduino Setup
- [ ] Upload `arduino_wristband.ino` to Arduino
- [ ] Configure WiFi credentials in Arduino code
- [ ] Test vibration motors
- [ ] Verify serial/wireless communication

## System Testing
- [ ] Test microphone input: `arecord -d 3 -f cd test.wav`
- [ ] Test model inference: `python test_model.py test.wav`
- [ ] Test real-time processing: `python realtime_audio_processor.py --test-mode`
- [ ] Test Arduino communication: Send test commands

## Production Deployment
- [ ] Start service: `sudo systemctl start audio-classification.service`
- [ ] Check service status: `sudo systemctl status audio-classification.service`
- [ ] Monitor logs: `journalctl -u audio-classification.service -f`
- [ ] Test end-to-end: Speak commands and verify wristband response

## Troubleshooting
- Check audio devices: `aplay -l` and `arecord -l`
- Check serial devices: `ls /dev/tty*`
- Check service logs: `journalctl -u audio-classification.service`
- Test individual components using test scripts
EOF

# Set permissions
chmod +x *.py
chmod +x *.sh

echo ""
echo "✅ Raspberry Pi setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your trained model files to this directory:"
echo "   - model.h5"
echo "   - model_metadata.json"
echo "   - realtime_audio_processor.py"
echo ""
echo "2. Test the system:"
echo "   bash test_system.sh"
echo ""
echo "3. Start the audio classification service:"
echo "   bash start_system.sh"
echo ""
echo "4. For production deployment:"
echo "   sudo systemctl start audio-classification.service"
echo ""
echo "📁 Project directory: $PROJECT_DIR"
echo "📖 See DEPLOYMENT_CHECKLIST.md for detailed instructions"
