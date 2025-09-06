#!/usr/bin/env python3
"""
Real-time Audio Processing Pipeline for Raspberry Pi

This system:
1. Captures live audio from microphone
2. Processes audio in real-time using sliding window
3. Extracts MFCC features
4. Classifies using DS-CNN model
5. Sends wireless commands to Arduino wristband

Optimized for Raspberry Pi performance and low latency.
"""

import os
import json
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import threading
import queue
import time
from collections import deque
import socket
import serial
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Audio configuration
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
HOP_LENGTH = int(SAMPLE_RATE * 0.5)  # 0.5 second hop for real-time processing
N_MFCC = 13
N_FRAMES = 130

# Communication configuration
ARDUINO_SERIAL_PORT = '/dev/ttyUSB0'  # Adjust for your setup
ARDUINO_BAUD_RATE = 9600
WIRELESS_IP = '192.168.1.100'  # Arduino WiFi IP
WIRELESS_PORT = 8080

class RealTimeAudioProcessor:
    """
    Real-time audio processing system for Raspberry Pi.
    """
    
    def __init__(self, model_path="model.h5", metadata_path="model_metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.class_mapping = ["disturbance", "soo", "hum", "hmm"]
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=BUFFER_SIZE)
        self.audio_queue = queue.Queue()
        
        # Processing state
        self.is_running = False
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.processing_thread = None
        
        # Communication
        self.arduino_serial = None
        self.arduino_socket = None
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.prediction_history = deque(maxlen=10)
        
        self.load_model()
        self.setup_communication()
    
    def load_model(self):
        """
        Load the trained DS-CNN model and metadata.
        """
        try:
            print("🤖 Loading DS-CNN model...")
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.class_mapping = self.metadata.get("class_mapping", self.class_mapping)
                print(f"✅ Metadata loaded: {len(self.class_mapping)} classes")
            
            # Warm up model with dummy input
            dummy_input = np.zeros((1, N_MFCC, N_FRAMES, 1), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            print("✅ Model warmed up for inference")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_communication(self):
        """
        Setup communication channels with Arduino.
        """
        print("📡 Setting up Arduino communication...")
        
        # Try serial communication first
        try:
            self.arduino_serial = serial.Serial(ARDUINO_SERIAL_PORT, ARDUINO_BAUD_RATE, timeout=1)
            print(f"✅ Serial communication established: {ARDUINO_SERIAL_PORT}")
        except Exception as e:
            print(f"⚠️  Serial communication failed: {e}")
        
        # Try wireless communication as backup
        try:
            self.arduino_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"✅ Wireless communication ready: {WIRELESS_IP}:{WIRELESS_PORT}")
        except Exception as e:
            print(f"⚠️  Wireless communication setup failed: {e}")
    
    def extract_mfcc_features(self, audio_data):
        """
        Extract MFCC features from audio data for DS-CNN input.
        """
        try:
            # Ensure audio is the right length
            if len(audio_data) < BUFFER_SIZE:
                audio_data = np.pad(audio_data, (0, BUFFER_SIZE - len(audio_data)))
            elif len(audio_data) > BUFFER_SIZE:
                audio_data = audio_data[:BUFFER_SIZE]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC,
                n_fft=2048,
                hop_length=512,
                n_mels=40
            )
            
            # Ensure consistent frame count
            if mfcc.shape[1] < N_FRAMES:
                mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode='constant')
            elif mfcc.shape[1] > N_FRAMES:
                mfcc = mfcc[:, :N_FRAMES]
            
            # Reshape for DS-CNN: (1, n_mfcc, n_frames, 1)
            mfcc_reshaped = mfcc.reshape(1, N_MFCC, N_FRAMES, 1)
            
            # Normalize (0-1 scaling)
            mfcc_min = mfcc_reshaped.min()
            mfcc_max = mfcc_reshaped.max()
            if mfcc_max > mfcc_min:
                mfcc_normalized = (mfcc_reshaped - mfcc_min) / (mfcc_max - mfcc_min)
            else:
                mfcc_normalized = mfcc_reshaped
            
            return mfcc_normalized.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error extracting MFCC: {e}")
            return None
    
    def classify_audio(self, mfcc_features):
        """
        Classify audio using DS-CNN model.
        """
        try:
            start_time = time.time()
            
            # Make prediction
            predictions = self.model.predict(mfcc_features, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return predicted_class, confidence, predictions[0]
            
        except Exception as e:
            print(f"❌ Error in classification: {e}")
            return None, 0.0, None
    
    def send_arduino_command(self, class_id, confidence, audio_characteristics=None):
        """
        Send command to Arduino based on classification.
        
        Motor Control Mapping:
        - "soo" (class 1) → Top motor vibrates
        - "hum" (class 2) → Bottom motor vibrates  
        - "hmm" (class 3) → Both motors vibrate simultaneously
        - "disturbance" (class 0) → Ignored (no vibration)
        """
        if class_id == 0:  # Disturbance - ignore
            return
        
        # Create command based on class
        command_data = {
            'class': int(class_id),
            'confidence': float(confidence),
            'timestamp': time.time()
        }
        
        # Add vibration pattern based on audio characteristics
        if audio_characteristics is not None:
            # Calculate rhythm and intensity from audio
            intensity = min(int(confidence * 255), 255)  # 0-255 PWM value
            duration = max(int(confidence * 1000), 100)  # Duration in ms
            
            command_data.update({
                'intensity': intensity,
                'duration': duration,
                'pattern': self.get_vibration_pattern(class_id, confidence)
            })
        
        # Send via serial
        if self.arduino_serial and self.arduino_serial.is_open:
            try:
                command_str = f"{class_id},{confidence:.3f},{command_data.get('intensity', 128)},{command_data.get('duration', 500)}\n"
                self.arduino_serial.write(command_str.encode())
                print(f"📤 Serial: {command_str.strip()}")
            except Exception as e:
                print(f"❌ Serial send error: {e}")
        
        # Send via wireless as backup
        if self.arduino_socket:
            try:
                command_json = json.dumps(command_data)
                self.arduino_socket.sendto(command_json.encode(), (WIRELESS_IP, WIRELESS_PORT))
                print(f"📡 Wireless: {self.class_mapping[class_id]} ({confidence:.3f})")
            except Exception as e:
                print(f"❌ Wireless send error: {e}")
    
    def get_vibration_pattern(self, class_id, confidence):
        """
        Generate vibration pattern based on class and confidence.
        """
        patterns = {
            1: "TOP_MOTOR",      # "soo" → Top motor
            2: "BOTTOM_MOTOR",   # "hum" → Bottom motor  
            3: "BOTH_MOTORS"     # "hmm" → Both motors
        }
        return patterns.get(class_id, "NO_VIBRATION")
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for real-time audio capture.
        """
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Add audio data to buffer
        audio_chunk = indata[:, 0]  # Use first channel
        self.audio_buffer.extend(audio_chunk)
        
        # Add to processing queue if buffer is full
        if len(self.audio_buffer) >= BUFFER_SIZE:
            audio_array = np.array(list(self.audio_buffer))
            if not self.audio_queue.full():
                self.audio_queue.put(audio_array.copy())
    
    def processing_loop(self):
        """
        Main processing loop for audio classification.
        """
        print("🔄 Starting audio processing loop...")
        
        while self.is_running:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=1.0)
                    
                    # Extract MFCC features
                    mfcc_features = self.extract_mfcc_features(audio_data)
                    if mfcc_features is None:
                        continue
                    
                    # Classify audio
                    predicted_class, confidence, all_predictions = self.classify_audio(mfcc_features)
                    if predicted_class is None:
                        continue
                    
                    # Update prediction history
                    self.prediction_history.append((predicted_class, confidence))
                    
                    # Apply smoothing (majority vote over last 3 predictions)
                    if len(self.prediction_history) >= 3:
                        recent_predictions = [p[0] for p in list(self.prediction_history)[-3:]]
                        smoothed_prediction = max(set(recent_predictions), key=recent_predictions.count)
                        smoothed_confidence = np.mean([p[1] for p in list(self.prediction_history)[-3:] if p[0] == smoothed_prediction])
                    else:
                        smoothed_prediction = predicted_class
                        smoothed_confidence = confidence
                    
                    # Only send command if confidence is above threshold
                    if smoothed_confidence > 0.3:  # 30% confidence threshold
                        self.send_arduino_command(smoothed_prediction, smoothed_confidence, all_predictions)
                        
                        # Update state
                        self.last_prediction = smoothed_prediction
                        self.prediction_confidence = smoothed_confidence
                        
                        # Print status
                        class_name = self.class_mapping[smoothed_prediction]
                        print(f"🎯 {class_name}: {smoothed_confidence:.3f} | Avg processing: {np.mean(self.processing_times):.3f}s")
                
                else:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(0.1)
    
    def start_realtime_processing(self):
        """
        Start real-time audio processing.
        """
        print("=" * 80)
        print("🎵 STARTING REAL-TIME AUDIO PROCESSING")
        print("=" * 80)
        print(f"📊 Configuration:")
        print(f"   Sample rate: {SAMPLE_RATE} Hz")
        print(f"   Buffer duration: {DURATION} seconds")
        print(f"   Classes: {', '.join(self.class_mapping)}")
        print(f"   Model: {self.model_path}")
        print("=" * 80)
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=HOP_LENGTH,
                dtype=np.float32
            ):
                print("🎤 Audio stream started. Press Ctrl+C to stop...")
                print("🔊 Speak 'soo', 'hum', or 'hmm' sounds to test classification")
                
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping audio processing...")
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
        finally:
            self.stop_processing()
    
    def stop_processing(self):
        """
        Stop real-time processing.
        """
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
        
        if self.arduino_socket:
            self.arduino_socket.close()
        
        print("✅ Real-time processing stopped")

def main():
    """
    Main function for real-time audio processing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Audio Classification for Raspberry Pi")
    parser.add_argument('--model', default='model.h5', help='Path to trained model')
    parser.add_argument('--metadata', default='model_metadata.json', help='Path to model metadata')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0', help='Arduino serial port')
    parser.add_argument('--wireless-ip', default='192.168.1.100', help='Arduino WiFi IP')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (no Arduino)')
    
    args = parser.parse_args()
    
    # Update global configuration
    global ARDUINO_SERIAL_PORT, WIRELESS_IP
    ARDUINO_SERIAL_PORT = args.serial_port
    WIRELESS_IP = args.wireless_ip
    
    try:
        # Create and start processor
        processor = RealTimeAudioProcessor(args.model, args.metadata)
        
        if args.test_mode:
            print("🧪 Running in test mode (no Arduino communication)")
            processor.arduino_serial = None
            processor.arduino_socket = None
        
        processor.start_realtime_processing()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
