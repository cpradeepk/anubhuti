# YAMNet Pipeline Deployment Guide

## 🚀 Production Deployment Instructions

This guide provides detailed instructions for deploying the YAMNet speech classification pipeline in production environments.

## 📋 Pre-Deployment Checklist

### System Requirements
- **Hardware**: Raspberry Pi 4+ (4GB RAM recommended) or equivalent Linux system
- **Operating System**: Raspberry Pi OS, Ubuntu 20.04+, or compatible Linux distribution
- **Python**: Version 3.8 or higher
- **Storage**: Minimum 2GB free space for models and dependencies
- **Audio**: USB microphone or audio input device
- **Network**: WiFi or Ethernet for Arduino communication

### Software Dependencies
- TensorFlow 2.9+
- TensorFlow Hub 0.12+
- librosa 0.9+
- NumPy, scikit-learn, matplotlib
- Audio processing libraries (soundfile, pydub)

## 🔧 Step-by-Step Deployment

### Step 1: System Preparation

#### Raspberry Pi Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git portaudio19-dev

# Install audio system dependencies
sudo apt install -y alsa-utils pulseaudio pulseaudio-utils

# Test audio input
arecord -l  # List audio devices
arecord -d 5 test.wav  # Record 5-second test
```

#### Create Project Directory
```bash
# Create deployment directory
mkdir -p ~/yamnet_deployment
cd ~/yamnet_deployment

# Copy project files (adjust paths as needed)
scp -r user@source:/path/to/yamnet_implementation/* .
```

### Step 2: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv yamnet_env
source yamnet_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### Step 3: Model Validation

```bash
# Verify model files exist
ls -la yamnet_models/
# Expected files:
# - yamnet_classifier.h5
# - yamnet_model_metadata.json

# Test model loading
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print(f'Model loaded successfully: {model.input_shape} -> {model.output_shape}')
"
```

### Step 4: Audio System Configuration

#### Configure Audio Input
```bash
# List audio devices
cat /proc/asound/cards

# Test microphone
arecord -D plughw:1,0 -f cd -t wav -d 5 test_mic.wav
aplay test_mic.wav

# Set default audio device (edit ~/.asoundrc)
cat > ~/.asoundrc << EOF
pcm.!default {
    type asym
    playback.pcm "plughw:0,0"
    capture.pcm "plughw:1,0"
}
EOF
```

#### Test Audio Processing
```bash
# Test with sample audio file
python test_yamnet_model.py test_audio.wav

# Expected output should show:
# - Predicted class
# - Confidence score
# - Arduino command
# - Processing time
```

### Step 5: Arduino Integration Setup

#### Serial Communication Setup
```bash
# Install serial communication library
pip install pyserial

# Check available serial ports
python -c "
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f'Port: {port.device}, Description: {port.description}')
"

# Test Arduino connection (adjust port as needed)
python -c "
import serial
import time
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)  # Wait for connection
    arduino.write(b'1')  # Send test command
    arduino.close()
    print('Arduino connection successful')
except Exception as e:
    print(f'Arduino connection failed: {e}')
"
```

#### Wireless Communication Setup (Optional)
```bash
# For WiFi-based Arduino communication
# Configure network settings on Arduino
# Test network connectivity
ping arduino_device_ip

# Test wireless command sending
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_UDP)
sock.sendto(b'1', ('arduino_ip', 8080))
sock.close()
print('Wireless command sent')
"
```

### Step 6: Real-time Processing Setup

#### Create Systemd Service (Optional)
```bash
# Create service file
sudo tee /etc/systemd/system/yamnet-classifier.service << EOF
[Unit]
Description=YAMNet Speech Classifier
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yamnet_deployment
Environment=PATH=/home/pi/yamnet_deployment/yamnet_env/bin
ExecStart=/home/pi/yamnet_deployment/yamnet_env/bin/python realtime_audio_processor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable yamnet-classifier.service
sudo systemctl start yamnet-classifier.service

# Check service status
sudo systemctl status yamnet-classifier.service
```

## 🧪 Deployment Testing

### Test 1: Basic Functionality
```bash
# Test individual file processing
python test_yamnet_model.py sample_audio.wav

# Expected results:
# - Correct class prediction
# - Reasonable confidence (>0.3)
# - Appropriate Arduino command
# - Fast processing (<100ms)
```

### Test 2: Real-time Processing
```bash
# Test real-time audio classification
python realtime_audio_processor.py --duration 30

# Monitor for:
# - Continuous audio processing
# - Stable memory usage
# - Consistent performance
# - Proper Arduino commands
```

### Test 3: Arduino Integration
```bash
# Test Arduino command sending
python -c "
from test_yamnet_model import YAMNetModelTester
import serial

tester = YAMNetModelTester('yamnet_models/yamnet_classifier.h5')
arduino = serial.Serial('/dev/ttyUSB0', 9600)

# Test each class
test_files = [
    ('slow_sample.wav', 1),
    ('medium_sample.wav', 2),
    ('fast_sample.wav', 3),
    ('disturbance_sample.wav', 0)
]

for file_path, expected_cmd in test_files:
    result = tester.predict_single_file(file_path)
    actual_cmd = result['arduino_command']
    arduino.write(str(actual_cmd).encode())
    print(f'{file_path}: Expected {expected_cmd}, Got {actual_cmd}')

arduino.close()
"
```

### Test 4: Performance Monitoring
```bash
# Monitor system resources during operation
python -c "
import psutil
import time
from test_yamnet_model import YAMNetModelTester

tester = YAMNetModelTester('yamnet_models/yamnet_classifier.h5')

print('Monitoring system resources...')
for i in range(10):
    # Simulate processing
    result = tester.predict_single_file('test_audio.wav')
    
    # Check resources
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    print(f'Iteration {i+1}: CPU {cpu_percent}%, Memory {memory.percent}%')
    time.sleep(1)
"
```

## 🔧 Production Configuration

### Performance Optimization

#### TensorFlow Lite Conversion (Recommended for Pi)
```bash
# Convert model to TensorFlow Lite for better performance
python -c "
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('yamnet_models/yamnet_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print('TensorFlow Lite model saved successfully')
"
```

#### Memory Optimization
```bash
# Configure TensorFlow memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce logging

# Limit TensorFlow threads for Raspberry Pi
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
```

### Security Configuration

#### Firewall Setup
```bash
# Configure firewall for Arduino communication
sudo ufw allow 8080/udp  # For wireless Arduino communication
sudo ufw allow 22/tcp    # For SSH access
sudo ufw enable
```

#### User Permissions
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Add user to dialout group for serial communication
sudo usermod -a -G dialout $USER

# Logout and login for changes to take effect
```

## 📊 Monitoring and Maintenance

### Log Configuration
```bash
# Create log directory
mkdir -p ~/yamnet_deployment/logs

# Configure logging in Python scripts
# Add to realtime_audio_processor.py:
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/yamnet_classifier.log'),
        logging.StreamHandler()
    ]
)
```

### Health Monitoring
```bash
# Create health check script
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
import psutil
import os
import json
from datetime import datetime

def health_check():
    health = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'model_file_exists': os.path.exists('yamnet_models/yamnet_classifier.h5'),
        'service_running': 'yamnet-classifier' in [p.name() for p in psutil.process_iter()]
    }
    
    with open('logs/health_status.json', 'w') as f:
        json.dump(health, f, indent=2)
    
    print(f"Health check completed: {health}")
    return health

if __name__ == "__main__":
    health_check()
EOF

chmod +x health_check.py

# Run health check
python health_check.py
```

### Automated Backup
```bash
# Create backup script
cat > backup_models.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/pi/yamnet_backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/yamnet_models_$DATE.tar.gz yamnet_models/
echo "Backup created: $BACKUP_DIR/yamnet_models_$DATE.tar.gz"

# Keep only last 5 backups
ls -t $BACKUP_DIR/yamnet_models_*.tar.gz | tail -n +6 | xargs rm -f
EOF

chmod +x backup_models.sh

# Add to crontab for daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /home/pi/yamnet_deployment/backup_models.sh") | crontab -
```

## 🚨 Troubleshooting Production Issues

### Common Production Problems

#### Issue 1: High CPU Usage
```bash
# Check process usage
top -p $(pgrep -f yamnet)

# Solution: Enable TensorFlow Lite
# Solution: Reduce audio processing frequency
# Solution: Implement prediction caching
```

#### Issue 2: Memory Leaks
```bash
# Monitor memory usage over time
watch -n 5 'ps aux | grep yamnet'

# Solution: Restart service periodically
# Add to crontab: 0 */6 * * * sudo systemctl restart yamnet-classifier
```

#### Issue 3: Audio Input Problems
```bash
# Check audio devices
arecord -l
pulseaudio --check

# Restart audio system
sudo systemctl restart pulseaudio
sudo systemctl restart alsa-state
```

#### Issue 4: Arduino Communication Failures
```bash
# Check serial connection
ls -la /dev/ttyUSB*
dmesg | grep tty

# Reset Arduino connection
sudo systemctl restart yamnet-classifier
```

## 🎯 Production Best Practices

1. **Monitoring**: Implement comprehensive logging and health checks
2. **Backup**: Regular automated backups of models and configurations
3. **Updates**: Staged deployment process for model updates
4. **Security**: Regular security updates and access control
5. **Performance**: Continuous performance monitoring and optimization
6. **Reliability**: Automatic restart mechanisms and error recovery
7. **Documentation**: Maintain deployment logs and configuration documentation

## 📈 Scaling Considerations

### Multi-Device Deployment
- Use configuration management tools (Ansible, Puppet)
- Implement centralized model distribution
- Monitor fleet health and performance
- Coordinate Arduino wristband assignments

### Cloud Integration
- Consider edge computing for preprocessing
- Implement model versioning and A/B testing
- Use cloud monitoring and alerting
- Enable remote configuration updates

This deployment guide ensures reliable, production-ready deployment of the YAMNet speech classification pipeline with comprehensive monitoring, maintenance, and troubleshooting procedures.
