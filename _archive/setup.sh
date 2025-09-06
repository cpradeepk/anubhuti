#!/bin/bash
# Setup script for Audio Classification System

echo "=========================================="
echo "Audio Classification System Setup"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Install required packages
echo "Installing required Python packages..."
pip3 install librosa numpy soundfile scikit-learn tensorflow keras matplotlib tqdm

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✓ All packages installed successfully"
else
    echo "✗ Package installation failed. Please check your pip installation."
    exit 1
fi

# Make Python scripts executable
chmod +x dataset_preprocess.py
chmod +x train_model.py
chmod +x test_model.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add .wav audio files to the dataset folders:"
echo "   - disturbance/"
echo "   - slow/"
echo "   - medium/"
echo "   - fast/"
echo ""
echo "2. Run the preprocessing script:"
echo "   python3 dataset_preprocess.py"
echo ""
echo "3. Train the model:"
echo "   python3 train_model.py"
echo ""
echo "4. Test with an audio file:"
echo "   python3 test_model.py <path_to_audio_file.wav>"
echo ""
echo "See README.md for detailed instructions."
echo "=========================================="