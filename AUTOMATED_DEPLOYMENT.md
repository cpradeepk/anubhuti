# 🚀 **Automated Deployment Guide - YAMNet Speech Classification Pipeline**

This guide provides instructions for automated deployment and management of the YAMNet speech classification pipeline on Raspberry Pi using the provided deployment scripts.

## 📋 **Overview**

The automated deployment system includes:

- **`deploy.sh`**: Complete automated deployment script
- **`update.sh`**: Update and maintenance script with rollback capabilities
- **Git-based deployment**: Easy updates and version control
- **Comprehensive error handling**: Robust deployment with logging
- **Backup and rollback**: Safe update process with automatic backups

## 🎯 **Quick Start - One-Command Deployment**

### **Method 1: Direct Script Execution (Recommended)**

SSH into your Raspberry Pi and run:

```bash
# Download and run the deployment script directly
curl -fsSL https://raw.githubusercontent.com/cpradeepk/anubhuti/main/deploy.sh | bash
```

### **Method 2: Clone and Deploy**

```bash
# Clone repository first
git clone https://github.com/cpradeepk/anubhuti.git
cd anubhuti

# Run deployment script
./deploy.sh
```

## 📊 **Deployment Process Overview**

The automated deployment script performs these steps:

1. **System Check**: Verifies Raspberry Pi compatibility and requirements
2. **Package Installation**: Updates system and installs dependencies
3. **Repository Setup**: Clones/updates the Anubhuti repository
4. **Python Environment**: Creates virtual environment and installs packages
5. **Audio Configuration**: Sets up audio system for microphone input
6. **Model Testing**: Verifies YAMNet model deployment
7. **Summary Generation**: Creates deployment report and next steps

## 🔧 **Deployment Script Features**

### **`deploy.sh` - Main Deployment Script**

#### **Key Features:**
- ✅ **Idempotent**: Safe to run multiple times
- ✅ **Progress Indicators**: Visual feedback during installation
- ✅ **Comprehensive Logging**: Detailed logs for troubleshooting
- ✅ **Error Handling**: Graceful failure with helpful error messages
- ✅ **Backup Creation**: Automatic backup of existing installations
- ✅ **System Verification**: Pre and post-deployment checks

#### **Usage:**
```bash
# Basic deployment
./deploy.sh

# The script will automatically:
# - Check system requirements
# - Install all dependencies
# - Set up the YAMNet pipeline
# - Test the deployment
# - Provide next steps
```

#### **Expected Output:**
```
╔══════════════════════════════════════════════════════════════╗
║          YAMNet Speech Classification Pipeline               ║
║              Automated Deployment Script                    ║
║                  for Raspberry Pi 4                         ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting YAMNet deployment process...
📝 Logging to: /home/pi/yamnet_deployment.log
🔍 Checking if running on Raspberry Pi...
✅ Detected: Raspberry Pi 4 Model B Rev 1.4
🔍 Checking system requirements...
✅ Python version: 3.9.2
✅ Available memory: 1024MB
✅ Available disk space: 8.5G
📦 Updating system packages...
📁 Setting up repository...
✅ Repository cloned successfully
🐍 Setting up Python virtual environment...
✅ Virtual environment created
📚 Installing Python dependencies...
Installing TensorFlow (this may take 10-15 minutes)...
✅ Python dependencies installed successfully
🎵 Configuring audio system...
✅ Audio configuration created
🧪 Testing model deployment...
✅ Model deployment test passed
📋 Creating deployment summary...
✅ Deployment summary created: /home/pi/yamnet_deployment_summary.txt

╔══════════════════════════════════════════════════════════════╗
║                    🎉 DEPLOYMENT SUCCESSFUL! 🎉              ║
╚══════════════════════════════════════════════════════════════╝
```

### **`update.sh` - Update and Maintenance Script**

#### **Key Features:**
- ✅ **Git-based Updates**: Pull latest changes from repository
- ✅ **Automatic Backups**: Creates backup before each update
- ✅ **Rollback Capability**: Easy rollback to previous version
- ✅ **Dependency Management**: Updates Python packages when needed
- ✅ **Update Verification**: Tests deployment after updates
- ✅ **Multiple Update Modes**: Check-only, force update, rollback options

#### **Usage:**
```bash
# Check for updates without applying
./update.sh --check-only

# Apply available updates
./update.sh

# Force update even if no changes detected
./update.sh --force

# Rollback to previous version
./update.sh --rollback

# Show help
./update.sh --help
```

#### **Update Process:**
1. **Backup Creation**: Automatic backup of current installation
2. **Change Detection**: Check for repository updates
3. **Update Application**: Pull latest changes via git
4. **Dependency Updates**: Install new/updated Python packages
5. **Testing**: Verify updated deployment works correctly
6. **Summary**: Generate update report

## 📁 **File Structure After Deployment**

```
/home/pi/anubhuti/                          # Main repository
├── deploy.sh                               # Deployment script
├── update.sh                               # Update script
├── README.md                               # Main documentation
├── RASPBERRY_PI_DEPLOYMENT.md              # Pi-specific guide
├── AUTOMATED_DEPLOYMENT.md                 # This file
├── yamnet_implementation/                  # Core implementation
│   ├── yamnet_env/                         # Python virtual environment
│   ├── yamnet_models/                      # Trained models
│   │   ├── yamnet_classifier.h5            # Main model file
│   │   └── yamnet_model_metadata.json     # Model metadata
│   ├── test_yamnet_model.py               # Testing script
│   ├── train_yamnet_model.py              # Training script
│   ├── yamnet_utils.py                    # Utility functions
│   └── requirements.txt                   # Python dependencies
├── slow/                                   # Audio dataset
├── medium/                                 # Audio dataset
├── fast/                                   # Audio dataset
└── disturbance/                           # Audio dataset

/home/pi/                                   # User home directory
├── yamnet_deployment.log                  # Deployment log
├── yamnet_update.log                      # Update log
├── yamnet_deployment_summary.txt          # Deployment summary
├── yamnet_update_summary.txt              # Update summary
└── anubhuti_backups/                      # Automatic backups
    ├── anubhuti_backup_20241206_143022/   # Timestamped backups
    └── anubhuti_backup_20241206_151545/
```

## 🧪 **Post-Deployment Testing**

After successful deployment, test your system:

### **1. Basic Model Test**
```bash
cd ~/anubhuti/yamnet_implementation/
source yamnet_env/bin/activate

# Test model loading
python3 -c "
import tensorflow as tf
model = tf.keras.models.load_model('yamnet_models/yamnet_classifier.h5')
print('✅ Model loaded successfully!')
"
```

### **2. Audio Classification Test**
```bash
# Test with sample audio files
python3 test_yamnet_model.py ../slow/Fhmm_slow.wav
python3 test_yamnet_model.py ../medium/Fhum_medium.wav
python3 test_yamnet_model.py ../fast/Fhum_fast.wav
python3 test_yamnet_model.py ../disturbance/Cough.wav
```

### **3. Real-time Audio Test** (if microphone connected)
```bash
# Run real-time classification
python3 realtime_pi_test.py
```

### **4. System Health Check**
```bash
# Run comprehensive health check
python3 health_check.py
```

## 🔄 **Update Workflow**

### **Regular Updates**
```bash
cd ~/anubhuti

# Check for updates
./update.sh --check-only

# Apply updates if available
./update.sh
```

### **Emergency Rollback**
```bash
cd ~/anubhuti

# Rollback to previous version
./update.sh --rollback
```

## 📊 **Monitoring and Maintenance**

### **Log Files**
- **Deployment Log**: `~/yamnet_deployment.log`
- **Update Log**: `~/yamnet_update.log`
- **System Status**: `~/yamnet_deployment_summary.txt`

### **Backup Management**
- **Backup Location**: `~/anubhuti_backups/`
- **Automatic Cleanup**: Keeps last 5 backups
- **Manual Backup**: Created before each update

### **Performance Monitoring**
```bash
# Check system resources
free -h
df -h
vcgencmd measure_temp

# Monitor YAMNet processes
ps aux | grep yamnet
```

## 🚨 **Troubleshooting**

### **Common Issues and Solutions**

#### **1. NumPy 2.x Compatibility Issue (Most Common)**
```bash
# Quick fix for NumPy 2.x compatibility with TensorFlow
cd ~/anubhuti
./fix_numpy_compatibility.sh

# Or manual fix:
cd ~/anubhuti/yamnet_implementation/
source yamnet_env/bin/activate
pip uninstall numpy -y
pip install "numpy<2.0" --no-cache-dir
pip uninstall tensorflow -y
pip install tensorflow==2.13.0 --no-cache-dir
```

#### **2. Deployment Fails During TensorFlow Installation**
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Retry deployment
./deploy.sh
```

#### **3. Git Clone Fails**
```bash
# Check internet connection
ping -c 3 github.com

# Try with different DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# Retry deployment
./deploy.sh
```

#### **4. Audio System Issues**
```bash
# Check audio devices
arecord -l

# Restart audio services
sudo systemctl restart alsa-state
pulseaudio --kill && pulseaudio --start

# Test audio
arecord -d 3 test.wav && aplay test.wav
```

#### **5. Model Loading Errors**
```bash
# Check model file integrity
ls -la ~/anubhuti/yamnet_implementation/yamnet_models/yamnet_classifier.h5

# Re-download if corrupted
cd ~
rm -rf anubhuti
./deploy.sh
```

#### **6. Update Failures**
```bash
# Rollback to previous version
cd ~/anubhuti
./update.sh --rollback

# Check logs for details
tail -50 ~/yamnet_update.log
```

## 🎯 **Advanced Configuration**

### **Custom Installation Directory**
```bash
# Edit deploy.sh before running
nano deploy.sh
# Change: INSTALL_DIR="$HOME/anubhuti"
# To: INSTALL_DIR="/opt/yamnet"
```

### **Service Installation** (Auto-start)
```bash
# Create systemd service
sudo tee /etc/systemd/system/yamnet.service << 'EOF'
[Unit]
Description=YAMNet Speech Classifier
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/anubhuti/yamnet_implementation
Environment=PATH=/home/pi/anubhuti/yamnet_implementation/yamnet_env/bin
ExecStart=/home/pi/anubhuti/yamnet_implementation/yamnet_env/bin/python realtime_pi_test.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable yamnet.service
sudo systemctl start yamnet.service
```

## 🎉 **Success Indicators**

Your deployment is successful when you see:

✅ **Deployment Script Completes**: No errors during `deploy.sh` execution  
✅ **Model Loads Successfully**: TensorFlow can load the YAMNet classifier  
✅ **Audio Classification Works**: Test files produce correct predictions  
✅ **Real-time Processing**: Live audio classification with <50ms latency  
✅ **Health Check Passes**: All system components verified  

## 📞 **Support and Next Steps**

### **Immediate Next Steps:**
1. **Connect USB Microphone**: For real-time audio input
2. **Test Real-time Classification**: Run `realtime_pi_test.py`
3. **Connect Arduino Wristband**: For haptic feedback integration
4. **Deploy in Classroom**: Real-world testing with students

### **Long-term Maintenance:**
1. **Regular Updates**: Run `./update.sh` weekly
2. **Monitor Performance**: Check logs and system resources
3. **Backup Management**: Verify automatic backups are working
4. **Model Improvements**: Update models as new versions are released

**Your YAMNet speech classification pipeline is now fully automated and ready for production deployment! 🎵🤖✨**
