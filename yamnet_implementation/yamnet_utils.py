#!/usr/bin/env python3
"""
YAMNet Utility Functions

This module provides helper functions for YAMNet-based audio classification,
including audio preprocessing, embedding extraction, and data handling.
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YAMNet configuration constants
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16000
YAMNET_EMBEDDING_SIZE = 1024

class YAMNetProcessor:
    """
    YAMNet processor for audio classification tasks.
    Handles model loading, audio preprocessing, and embedding extraction.
    """
    
    def __init__(self, model_url: str = YAMNET_MODEL_URL):
        """
        Initialize YAMNet processor.
        
        Args:
            model_url: URL to YAMNet model on TensorFlow Hub
        """
        self.model_url = model_url
        self.yamnet_model = None
        self.class_names = None
        self._load_yamnet()
    
    def _load_yamnet(self):
        """Load YAMNet model from TensorFlow Hub."""
        try:
            logger.info(f"Loading YAMNet model from {self.model_url}")
            self.yamnet_model = hub.load(self.model_url)
            
            # Load class names
            class_map_path = self.yamnet_model.class_map_path().numpy()
            self.class_names = self._load_class_names(class_map_path)
            
            logger.info(f"✅ YAMNet model loaded successfully")
            logger.info(f"   Model supports {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"❌ Failed to load YAMNet model: {e}")
            raise
    
    def _load_class_names(self, class_map_path: bytes) -> List[str]:
        """Load YAMNet class names from CSV file."""
        try:
            class_map_csv = tf.io.read_file(class_map_path)
            class_map_csv = tf.strings.split(class_map_csv, sep='\n')
            class_names = []
            for line in class_map_csv:
                if tf.strings.length(line) > 0:
                    parts = tf.strings.split(line, sep=',')
                    class_names.append(parts[2].numpy().decode('utf-8'))
            return class_names
        except Exception as e:
            logger.warning(f"Could not load class names: {e}")
            return []
    
    def preprocess_audio(self, audio_path: Union[str, Path], 
                        target_sr: int = YAMNET_SAMPLE_RATE) -> np.ndarray:
        """
        Preprocess audio file for YAMNet input.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (16kHz for YAMNet)
            
        Returns:
            Preprocessed audio as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            logger.debug(f"Loaded audio: {len(audio)} samples at {sr}Hz")
            
            # Resample to target sample rate if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                logger.debug(f"Resampled to {target_sr}Hz")
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing audio {audio_path}: {e}")
            raise
    
    def extract_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract YAMNet embeddings from audio.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            YAMNet embeddings (N, 1024) where N is number of frames
        """
        try:
            # Convert to TensorFlow tensor
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            
            # Extract embeddings using YAMNet
            scores, embeddings, spectrogram = self.yamnet_model(audio_tensor)
            
            return embeddings.numpy()
            
        except Exception as e:
            logger.error(f"❌ Error extracting embeddings: {e}")
            raise
    
    def process_audio_file(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Complete processing pipeline for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Extract embeddings
            embeddings = self.extract_embeddings(audio)
            
            # Create metadata
            metadata = {
                'file_path': str(audio_path),
                'duration_seconds': len(audio) / YAMNET_SAMPLE_RATE,
                'num_frames': embeddings.shape[0],
                'embedding_shape': embeddings.shape,
                'sample_rate': YAMNET_SAMPLE_RATE
            }
            
            logger.debug(f"Processed {Path(audio_path).name}: "
                        f"{embeddings.shape[0]} frames, "
                        f"{metadata['duration_seconds']:.2f}s")
            
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"❌ Error processing {audio_path}: {e}")
            raise

def load_dataset(dataset_path: Union[str, Path], 
                class_mapping: Dict[str, int] = None) -> Tuple[List, List, List]:
    """
    Load dataset from directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        class_mapping: Optional custom class mapping
        
    Returns:
        Tuple of (file_paths, labels, class_names)
    """
    if class_mapping is None:
        class_mapping = {
            'slow': 0,
            'medium': 1, 
            'fast': 2,
            'disturbance': 3
        }
    
    dataset_path = Path(dataset_path)
    file_paths = []
    labels = []
    class_names = list(class_mapping.keys())
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
    
    for class_name, class_id in class_mapping.items():
        class_dir = dataset_path / class_name
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        # Find all audio files in class directory
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(class_dir.glob(f"*{ext}"))
            audio_files.extend(class_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(audio_files)} files in {class_name}/")
        
        for audio_file in audio_files:
            file_paths.append(str(audio_file))
            labels.append(class_id)
    
    logger.info(f"✅ Dataset loaded: {len(file_paths)} files, {len(class_names)} classes")
    
    return file_paths, labels, class_names

def save_model_metadata(metadata: Dict, save_path: Union[str, Path]):
    """
    Save model metadata to JSON file.
    
    Args:
        metadata: Dictionary containing model metadata
        save_path: Path to save metadata file
    """
    try:
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✅ Metadata saved to {save_path}")
    except Exception as e:
        logger.error(f"❌ Error saving metadata: {e}")
        raise

def load_model_metadata(metadata_path: Union[str, Path]) -> Dict:
    """
    Load model metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Dictionary containing model metadata
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✅ Metadata loaded from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"❌ Error loading metadata: {e}")
        raise

def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if file is a valid audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid audio file, False otherwise
    """
    try:
        # Try to load audio file
        audio, sr = librosa.load(file_path, sr=None, duration=1.0)
        return len(audio) > 0 and sr > 0
    except:
        return False

def create_balanced_splits(file_paths: List[str], labels: List[int],
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_state: int = 42) -> Dict:
    """
    Create balanced train/validation/test splits.

    Args:
        file_paths: List of file paths
        labels: List of corresponding labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with split indices
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )

    # Second split: val vs test
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )

    return {
        'train': {'files': train_files, 'labels': train_labels},
        'val': {'files': val_files, 'labels': val_labels},
        'test': {'files': test_files, 'labels': test_labels}
    }

def aggregate_embeddings(embeddings: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Aggregate multiple embedding frames into single representation.

    Args:
        embeddings: Array of shape (n_frames, embedding_dim)
        method: Aggregation method ('mean', 'max', 'median')

    Returns:
        Single embedding vector of shape (embedding_dim,)
    """
    if method == 'mean':
        return np.mean(embeddings, axis=0)
    elif method == 'max':
        return np.max(embeddings, axis=0)
    elif method == 'median':
        return np.median(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def chunk_audio(audio: np.ndarray, chunk_duration: float = 5.0,
                overlap: float = 0.5, sample_rate: int = YAMNET_SAMPLE_RATE) -> List[np.ndarray]:
    """
    Split audio into overlapping chunks.

    Args:
        audio: Audio array
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap ratio (0.0 to 1.0)
        sample_rate: Audio sample rate

    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(chunk_samples * (1 - overlap))

    chunks = []
    start = 0

    while start + chunk_samples <= len(audio):
        chunk = audio[start:start + chunk_samples]
        chunks.append(chunk)
        start += hop_samples

    # Add final chunk if remaining audio is significant
    if start < len(audio) and (len(audio) - start) > chunk_samples * 0.5:
        final_chunk = audio[start:]
        # Pad if necessary
        if len(final_chunk) < chunk_samples:
            final_chunk = np.pad(final_chunk, (0, chunk_samples - len(final_chunk)))
        chunks.append(final_chunk)

    return chunks
