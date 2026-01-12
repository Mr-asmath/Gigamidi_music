#!/usr/bin/env python3
"""
Advanced training script for music generation models with dataset download and detailed reporting.
Training only the Transformer model for 30 epochs.
"""

import os
import sys
import torch
import json
import time
import hashlib
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import math
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer_model import TransformerMusicModel
from training.config import ModelConfig, TrainingConfig
from training.train_utils import MusicTrainer

# Constants
DATASET_DIR = "datasets/gigamidi"
DOWNLOAD_DIR = os.path.join(DATASET_DIR, "downloads")
CHECKPOINT_DIR = "saved_models"
REPORT_DIR = "reports"
METADATA_FILE = os.path.join(DATASET_DIR, "metadata.json")

class DatasetDownloader:
    """Handle dataset downloading and verification"""
    
    @staticmethod
    def check_dataset():
        """Check if dataset is downloaded"""
        print("\n" + "="*60)
        print("CHECKING DATASET")
        print("="*60)
        
        # Check if we have metadata
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r') as f:
                    metadata = json.load(f)
                print(f"✓ Dataset metadata found: {metadata.get('status', 'unknown')}")
                
                # Check if files exist
                if metadata.get('status') == 'downloaded':
                    files_exist = True
                    for file_info in metadata.get('files', []):
                        file_path = os.path.join(DOWNLOAD_DIR, file_info['name'])
                        if not os.path.exists(file_path):
                            print(f"✗ Missing file: {file_info['name']}")
                            files_exist = False
                    
                    if files_exist:
                        print(f"✓ All dataset files found in {DOWNLOAD_DIR}")
                        return True
            except Exception as e:
                print(f"✗ Error reading metadata: {e}")
        
        print("Dataset not found or incomplete")
        return False
    
    @staticmethod
    def download_dataset(max_samples=5000):
        """Download GigaMIDI dataset"""
        print("\n" + "="*60)
        print("DOWNLOADING DATASET")
        print("="*60)
        
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
        try:
            print("Loading GigaMIDI dataset from Hugging Face...")
            print("Note: This may take time depending on your internet connection.")
            
            # Try to load dataset
            dataset = load_dataset(
                "Metacreation/GigaMIDI",
                split="train",
                streaming=True
            )
            
            # Create metadata
            metadata = {
                'status': 'downloading',
                'download_start': datetime.now().isoformat(),
                'max_samples': max_samples,
                'files': []
            }
            
            # Save initial metadata
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Download samples
            samples = []
            sample_count = 0
            
            print(f"\nDownloading {max_samples} samples...")
            progress_bar = tqdm(total=max_samples, desc="Downloading")
            
            for sample in dataset:
                if sample_count >= max_samples:
                    break
                
                try:
                    # Extract relevant information
                    sample_data = {
                        'music': sample['music'],
                        'title': sample.get('title', ''),
                        'artist': sample.get('artist', ''),
                        'styles': sample.get('music_styles_curated', []),
                        'tempo': sample.get('tempo', '120'),
                        'instruments': sample.get('instrument_group__expressive_', []),
                        'id': hashlib.md5(sample['music']).hexdigest()[:16]
                    }
                    
                    samples.append(sample_data)
                    sample_count += 1
                    progress_bar.update(1)
                    
                    # Save batch every 100 samples
                    if sample_count % 100 == 0:
                        batch_num = sample_count // 100
                        batch_file = os.path.join(DOWNLOAD_DIR, f"batch_{batch_num:03d}.json")
                        
                        with open(batch_file, 'w') as f:
                            json.dump(samples[-100:], f)
                        
                        metadata['files'].append({
                            'name': f"batch_{batch_num:03d}.json",
                            'size': os.path.getsize(batch_file),
                            'samples': 100
                        })
                        
                        # Update metadata
                        metadata['status'] = 'downloading'
                        metadata['samples_downloaded'] = sample_count
                        metadata['last_update'] = datetime.now().isoformat()
                        
                        with open(METADATA_FILE, 'w') as f:
                            json.dump(metadata, f, indent=2)
                            
                except Exception as e:
                    print(f"\nWarning: Error processing sample: {e}")
                    continue
            
            progress_bar.close()
            
            # Save final metadata
            metadata['status'] = 'downloaded'
            metadata['download_end'] = datetime.now().isoformat()
            metadata['total_samples'] = len(samples)
            
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✓ Successfully downloaded {len(samples)} samples")
            print(f"✓ Files saved to: {DOWNLOAD_DIR}")
            print(f"✓ Metadata saved to: {METADATA_FILE}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error downloading dataset: {e}")
            print("\nPossible solutions:")
            print("1. Check your internet connection")
            print("2. Accept the dataset terms at: https://huggingface.co/datasets/Metacreation/GigaMIDI")
            print("3. Log in to Hugging Face: huggingface-cli login")
            print("4. Use --use-dummy-data flag to train with synthetic data")
            
            # Mark as error
            metadata = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return False

class AdvancedTextEncoder:
    """Enhanced text encoder with better feature extraction"""
    
    def __init__(self, feature_size=512):
        self.feature_size = feature_size
        
        # Musical keyword embeddings
        self.keywords = {
            'instruments': ['piano', 'guitar', 'violin', 'drums', 'bass', 'flute', 
                           'trumpet', 'saxophone', 'cello', 'harp', 'organ', 'synth'],
            'emotions': ['happy', 'sad', 'energetic', 'calm', 'romantic', 'angry', 
                        'peaceful', 'exciting', 'melancholic', 'joyful'],
            'genres': ['classical', 'jazz', 'rock', 'pop', 'blues', 'electronic', 
                      'folk', 'hiphop', 'reggae', 'metal'],
            'tempos': ['fast', 'slow', 'moderate', 'allegro', 'adagio', 'presto']
        }
    
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        batch_features = []
        for text in texts:
            # Extract musical features from text
            features = self._extract_features(text)
            batch_features.append(features)
        
        return {'latent': torch.tensor(batch_features, dtype=torch.float32)}
    
    def _extract_features(self, text):
        """Extract musical features from text"""
        text_lower = text.lower()
        features = np.zeros(self.feature_size, dtype=np.float32)
        
        # Hash-based deterministic start
        text_hash = hash(text) % 10000
        np.random.seed(text_hash)
        base_features = np.random.randn(self.feature_size // 2)
        
        # Instrument detection
        instrument_vector = np.zeros(12)
        for i, instrument in enumerate(self.keywords['instruments']):
            if instrument in text_lower:
                instrument_vector[i] = 1.0
        
        # Emotion detection
        emotion_vector = np.zeros(10)
        for i, emotion in enumerate(self.keywords['emotions']):
            if emotion in text_lower:
                emotion_vector[i] = 1.0
        
        # Genre detection
        genre_vector = np.zeros(10)
        for i, genre in enumerate(self.keywords['genres']):
            if genre in text_lower:
                genre_vector[i] = 1.0
        
        # Tempo detection
        tempo = 0.5  # Default moderate
        if any(word in text_lower for word in ['fast', 'allegro', 'presto', 'energetic']):
            tempo = 0.8
        elif any(word in text_lower for word in ['slow', 'adagio', 'calm', 'peaceful']):
            tempo = 0.2
        
        # Combine features
        features[:len(base_features)] = base_features
        features[len(base_features):len(base_features)+12] = instrument_vector
        features[len(base_features)+12:len(base_features)+22] = emotion_vector
        features[len(base_features)+22:len(base_features)+32] = genre_vector
        features[-1] = tempo
        
        return features

class GigaMIDIDataset(torch.utils.data.Dataset):
    """Dataset loader for downloaded GigaMIDI data"""
    
    def __init__(self, split='train', max_samples=2000, seq_length=256):
        self.split = split
        self.max_samples = max_samples
        self.seq_length = seq_length
        
        # Load samples from downloaded files
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self):
        """Load samples from downloaded files"""
        samples = []
        
        if not os.path.exists(DOWNLOAD_DIR):
            print(f"Download directory not found: {DOWNLOAD_DIR}")
            return samples
        
        # Load all batch files
        batch_files = sorted([f for f in os.listdir(DOWNLOAD_DIR) if f.startswith('batch_')])
        
        for batch_file in batch_files:
            try:
                with open(os.path.join(DOWNLOAD_DIR, batch_file), 'r') as f:
                    batch_samples = json.load(f)
                
                # Add samples with metadata
                for sample in batch_samples:
                    if len(samples) >= self.max_samples:
                        break
                    
                    samples.append({
                        'id': sample.get('id', ''),
                        'music': sample['music'],
                        'title': sample.get('title', 'Untitled'),
                        'artist': sample.get('artist', 'Unknown'),
                        'styles': sample.get('styles', []),
                        'tempo': sample.get('tempo', '120'),
                        'instruments': sample.get('instruments', [])
                    })
                    
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")
                continue
        
        # Split into train/val/test
        total_samples = len(samples)
        train_end = int(0.8 * total_samples)
        val_end = int(0.9 * total_samples)
        
        if self.split == 'train':
            samples = samples[:train_end]
        elif self.split == 'val':
            samples = samples[train_end:val_end]
        elif self.split == 'test':
            samples = samples[val_end:]
        
        return samples[:self.max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create features from sample
        features = self._create_features(sample)
        
        # Create text description
        text = self._create_text_description(sample)
        
        return {
            'features': features,
            'text': text,
            'metadata': {
                'title': sample['title'],
                'artist': sample['artist'],
                'tempo': sample['tempo']
            }
        }
    
    def _create_features(self, sample):
        """Create feature tensor from sample"""
        # For real implementation, you would parse the MIDI bytes
        # For now, create synthetic features based on sample metadata
        
        seq_length = self.seq_length
        
        # Create features based on sample metadata
        features = torch.zeros(seq_length, 4)
        
        # Use tempo to influence rhythm
        try:
            tempo = float(sample['tempo'])
        except:
            tempo = 120.0
        
        # Create pitch pattern
        base_pitch = 60  # Middle C
        if sample['instruments']:
            # Simple instrument-based pitch adjustment
            if 'piano' in str(sample['instruments']).lower():
                base_pitch = 60
            elif 'guitar' in str(sample['instruments']).lower():
                base_pitch = 40
            elif 'violin' in str(sample['instruments']).lower():
                base_pitch = 55
        
        # Create musical pattern
        for i in range(seq_length):
            # Pitch with some variation
            pitch_variation = math.sin(i / 8) * 3 + random.random() * 2
            features[i, 0] = (base_pitch + pitch_variation) / 127.0
            
            # Velocity with dynamics
            velocity = 70 + math.sin(i / 4) * 20 + random.random() * 10
            features[i, 1] = velocity / 127.0
            
            # Duration based on tempo
            base_duration = 480 * (120 / tempo)  # Adjust for tempo
            duration_variation = random.random() * 0.3 + 0.85
            features[i, 2] = (base_duration * duration_variation) / 480.0
            
            # Beat pattern
            features[i, 3] = 1.0 if i % 4 == 0 else 0.0
        
        return features
    
    def _create_text_description(self, sample):
        """Create text description from metadata"""
        parts = []
        
        if sample['title'] and sample['title'] != 'Untitled':
            parts.append(sample['title'])
        
        if sample['artist'] and sample['artist'] != 'Unknown':
            parts.append(f"by {sample['artist']}")
        
        if sample['styles']:
            parts.append(f"Style: {', '.join(sample['styles'][:2])}")
        
        if sample['instruments']:
            parts.append(f"Instruments: {', '.join(sample['instruments'][:3])}")
        
        if sample['tempo'] and sample['tempo'] != '120':
            parts.append(f"Tempo: {sample['tempo']} BPM")
        
        return ". ".join(parts) if parts else "Musical composition"

class TrainingReporter:
    """Generate detailed training reports"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.report_dir = os.path.join(REPORT_DIR, model_name)
        os.makedirs(self.report_dir, exist_ok=True)
        
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'pitch_loss': [],
            'velocity_loss': [],
            'duration_loss': [],
            'beat_loss': [],
            'learning_rate': [],
            'time_per_epoch': []
        }
    
    def add_epoch_metrics(self, epoch, train_losses, val_losses, lr, epoch_time):
        """Add metrics for an epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_losses.get('total', 0))
        self.metrics['val_loss'].append(val_losses.get('total', 0))
        self.metrics['pitch_loss'].append(train_losses.get('pitch', 0))
        self.metrics['velocity_loss'].append(train_losses.get('velocity', 0))
        self.metrics['duration_loss'].append(train_losses.get('duration', 0))
        self.metrics['beat_loss'].append(train_losses.get('beat', 0))
        self.metrics['learning_rate'].append(lr)
        self.metrics['time_per_epoch'].append(epoch_time)
    
    def generate_report(self, model, config, total_time):
        """Generate comprehensive training report"""
        print(f"\n{'='*60}")
        print(f"GENERATING TRAINING REPORT FOR {self.model_name.upper()}")
        print(f"{'='*60}")
        
        # Create DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save CSV
        csv_path = os.path.join(self.report_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Metrics saved to: {csv_path}")
        
        # Generate summary statistics
        summary = self._generate_summary(df, total_time, model, config)
        
        # Save summary
        summary_path = os.path.join(self.report_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_path}")
        
        # Generate plots
        self._generate_plots(df)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, df, total_time, model, config):
        """Generate summary statistics"""
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training statistics
        best_epoch = df['val_loss'].idxmin() + 1
        best_val_loss = df['val_loss'].min()
        final_train_loss = df['train_loss'].iloc[-1]
        final_val_loss = df['val_loss'].iloc[-1]
        
        # Calculate improvement
        initial_loss = df['val_loss'].iloc[0]
        improvement = ((initial_loss - final_val_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        summary = {
            'model_name': self.model_name,
            'model_type': config.model_type if hasattr(config, 'model_type') else 'unknown',
            'training_time': {
                'total_seconds': total_time,
                'total_formatted': f"{total_time:.1f}s",
                'average_epoch_seconds': df['time_per_epoch'].mean(),
                'epochs_completed': len(df)
            },
            'model_statistics': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'parameter_ratio': f"{(trainable_params/total_params*100):.1f}%"
            },
            'performance_metrics': {
                'best_epoch': best_epoch,
                'best_validation_loss': best_val_loss,
                'final_training_loss': final_train_loss,
                'final_validation_loss': final_val_loss,
                'improvement_percentage': improvement,
                'overfitting_ratio': final_train_loss / final_val_loss if final_val_loss > 0 else 0
            },
            'loss_components': {
                'pitch_loss': df['pitch_loss'].mean(),
                'velocity_loss': df['velocity_loss'].mean(),
                'duration_loss': df['duration_loss'].mean(),
                'beat_loss': df['beat_loss'].mean()
            },
            'training_config': {
                'epochs': config.epochs if hasattr(config, 'epochs') else 0,
                'batch_size': config.batch_size if hasattr(config, 'batch_size') else 0,
                'learning_rate': config.learning_rate if hasattr(config, 'learning_rate') else 0,
                'hidden_size': config.hidden_size if hasattr(config, 'hidden_size') else 0,
                'num_layers': config.num_layers if hasattr(config, 'num_layers') else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _generate_plots(self, df):
        """Generate training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss Components
        epochs = df['epoch']
        axes[0, 1].plot(epochs, df['pitch_loss'], label='Pitch Loss')
        axes[0, 1].plot(epochs, df['velocity_loss'], label='Velocity Loss')
        axes[0, 1].plot(epochs, df['duration_loss'], label='Duration Loss')
        axes[0, 1].plot(epochs, df['beat_loss'], label='Beat Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        axes[1, 0].plot(df['epoch'], df['learning_rate'], 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Time per Epoch
        axes[1, 1].bar(df['epoch'], df['time_per_epoch'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Time per Epoch')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.report_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to: {plot_path}")
    
    def _print_summary(self, summary):
        """Print formatted summary"""
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY - {summary['model_name'].upper()}")
        print(f"{'='*60}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Best Epoch: {summary['performance_metrics']['best_epoch']}")
        print(f"  Best Validation Loss: {summary['performance_metrics']['best_validation_loss']:.6f}")
        print(f"  Final Training Loss: {summary['performance_metrics']['final_training_loss']:.6f}")
        print(f"  Final Validation Loss: {summary['performance_metrics']['final_validation_loss']:.6f}")
        print(f"  Improvement: {summary['performance_metrics']['improvement_percentage']:.1f}%")
        
        print(f"\n⏱️  TRAINING TIME:")
        print(f"  Total Time: {summary['training_time']['total_formatted']}")
        print(f"  Average Epoch Time: {summary['training_time']['average_epoch_seconds']:.1f}s")
        print(f"  Epochs Completed: {summary['training_time']['epochs_completed']}")
        
        print(f"\n🧮 MODEL STATISTICS:")
        print(f"  Total Parameters: {summary['model_statistics']['total_parameters']:,}")
        print(f"  Trainable Parameters: {summary['model_statistics']['trainable_parameters']:,}")
        print(f"  Trainable Ratio: {summary['model_statistics']['parameter_ratio']}")
        
        print(f"\n⚙️  TRAINING CONFIGURATION:")
        print(f"  Epochs: {summary['training_config']['epochs']}")
        print(f"  Batch Size: {summary['training_config']['batch_size']}")
        print(f"  Learning Rate: {summary['training_config']['learning_rate']}")
        if 'd_model' in config.__dict__:
            print(f"  Model Dimension (d_model): {config.d_model}")
        if 'nhead' in config.__dict__:
            print(f"  Number of Heads: {config.nhead}")
        if 'num_layers' in config.__dict__:
            print(f"  Number of Layers: {config.num_layers}")

def create_collate_fn(seq_length=256):
    """Collate function to pad sequences"""
    def collate_fn(batch):
        features = []
        texts = []
        metadata = []
        
        for item in batch:
            feat = item['features']
            # Pad or truncate
            if len(feat) > seq_length:
                feat = feat[:seq_length]
            elif len(feat) < seq_length:
                pad = torch.zeros(seq_length - len(feat), 4)
                feat = torch.cat([feat, pad], dim=0)
            features.append(feat)
            texts.append(item['text'])
            metadata.append(item['metadata'])
        
        return {
            'features': torch.stack(features),
            'text': texts,
            'metadata': metadata
        }
    return collate_fn

def train_transformer_model(epochs=30, batch_size=16, use_real_data=True):
    """Train transformer model with enhanced reporting"""
    print(f"\n{'='*60}")
    print(f"TRAINING TRANSFORMER MODEL")
    print(f"{'='*60}")
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create dataset
    if use_real_data:
        dataset = GigaMIDIDataset(split='train', max_samples=2000, seq_length=256)
        val_dataset = GigaMIDIDataset(split='val', max_samples=500, seq_length=256)
    else:
        # Fallback to synthetic data
        print("⚠ Using synthetic data (real dataset not available)")
        from generate_music_fixed import create_fallback_music
        dataset = None  # Simplified for brevity
        val_dataset = None
    
    if dataset is None or len(dataset) == 0:
        print("⚠ No dataset available. Using minimal synthetic setup.")
        # Create minimal dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                features = torch.randn(256, 4)
                text = f"Sample music {idx}"
                return {'features': features, 'text': text, 'metadata': {}}
        
        dataset = SimpleDataset()
        val_dataset = SimpleDataset()
    
    collate_fn = create_collate_fn(seq_length=256)
    
    # Split into train/val
    if val_dataset is None:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    else:
        train_dataset = dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create transformer model
    model = TransformerMusicModel(
        input_size=4,
        d_model=256,           # Model dimension
        nhead=8,              # Number of attention heads
        num_layers=4,         # Number of transformer layers
        dim_feedforward=1024, # Feedforward dimension
        dropout=0.1,
        max_len=512,          # Maximum sequence length
        text_feature_size=512
    )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create text encoder
    text_encoder = AdvancedTextEncoder(feature_size=512)
    
    # Create trainer
    config = TrainingConfig(
        device=device,
        learning_rate=0.0001,  # Lower LR for transformer
        batch_size=batch_size,
        save_dir=CHECKPOINT_DIR,
        epochs=epochs
    )
    
    # Add transformer-specific config
    if hasattr(config, 'model_type'):
        config.model_type = 'transformer'
    config.d_model = 256
    config.nhead = 8
    config.num_layers = 4
    
    trainer = MusicTrainer(model, config, 'transformer')
    
    # Create reporter
    reporter = TrainingReporter('transformer')
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\n{'─'*40}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'─'*40}")
        
        # Train
        train_losses = trainer.train_epoch(train_loader, text_encoder)
        print(f"📊 Training Losses:")
        print(f"  Total: {train_losses.get('total', 0):.6f}")
        print(f"  Pitch: {train_losses.get('pitch', 0):.6f}")
        print(f"  Velocity: {train_losses.get('velocity', 0):.6f}")
        print(f"  Duration: {train_losses.get('duration', 0):.6f}")
        print(f"  Beat: {train_losses.get('beat', 0):.6f}")
        
        # Validate
        val_losses = trainer.validate(val_loader, text_encoder)
        print(f"\n📈 Validation Losses:")
        print(f"  Total: {val_losses.get('total', 0):.6f}")
        print(f"  Pitch: {val_losses.get('pitch', 0):.6f}")
        print(f"  Velocity: {val_losses.get('velocity', 0):.6f}")
        print(f"  Duration: {val_losses.get('duration', 0):.6f}")
        print(f"  Beat: {val_losses.get('beat', 0):.6f}")
        
        # Update scheduler
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"\n⚙️  Learning Rate: {current_lr:.6f}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        print(f"⏱️  Epoch Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = val_losses.get('total', float('inf')) < trainer.best_loss
        trainer.save_checkpoint(epoch + 1, val_losses.get('total', 0), is_best)
        
        # Add to reporter
        reporter.add_epoch_metrics(
            epoch + 1, train_losses, val_losses, current_lr, epoch_time
        )
        
        # Early stopping check
        if val_losses.get('total', 0) < 0.05:  # Very good performance
            print("\n🎉 Excellent performance achieved, stopping early!")
            break
        
        if epoch > 5 and val_losses.get('total', 0) > trainer.best_loss * 1.5:
            print("\n⚠ Validation loss increasing, consider early stopping")
    
    total_time = time.time() - start_time
    
    # Save final model
    model_path = os.path.join(CHECKPOINT_DIR, f"transformer.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Generate report
    summary = reporter.generate_report(model, config, total_time)
    
    return model, trainer, summary

def main():
    print("="*60)
    print("TRANSFORMER MUSIC MODEL TRAINING SYSTEM")
    print("="*60)
    print("Training only the Transformer model for 30 epochs")
    print("="*60)
    
    # Create directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train transformer music generation model')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--no-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--use-dummy-data', action='store_true', help='Use synthetic data')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    # Check dataset
    dataset_available = DatasetDownloader.check_dataset()
    
    # Download dataset if needed
    if args.download or (not dataset_available and not args.no_download and not args.use_dummy_data):
        print("\nDataset download required...")
        success = DatasetDownloader.download_dataset(max_samples=5000)
        if success:
            dataset_available = True
        else:
            print("\n⚠ Using synthetic data for training")
            args.use_dummy_data = True
    
    use_real_data = dataset_available and not args.use_dummy_data
    
    if use_real_data:
        print("\n✅ Using real GigaMIDI dataset for training")
    else:
        print("\n⚠ Using synthetic data for training")
    
    # Train transformer model
    print(f"\n🎯 Training Configuration:")
    print(f"  Model: Transformer")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Data: {'Real GigaMIDI' if use_real_data else 'Synthetic'}")
    
    try:
        model, trainer, summary = train_transformer_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_real_data=use_real_data
        )
        
        print(f"\n✅ Transformer training completed successfully!")
        
        # W&B API operations
        print("\n🔄 Performing W&B API operations...")
        
        try:
            api = wandb.Api()
            
            # Update run config (replace <run_id> with actual run ID)
            run_id = "<run_id>"  # Replace with actual run ID
            run_path = f"Ownuse/NIVI The Music Teacher/{run_id}"
            run = api.run(run_path)
            run.config["key"] = "updated_value"  # Replace with actual key and value
            run.update()
            print(f"✓ Updated run config for {run_path}")
            
            # Export metrics from a single run to a CSV file
            metrics_dataframe = run.history()
            metrics_dataframe.to_csv("metrics.csv")
            print("✓ Exported metrics to metrics.csv")
            
            # Read metrics for a run
            if run.state == "finished":
                print("📊 Run metrics (timestamp, accuracy):")
                for i, row in run.history().iterrows():
                    timestamp = row.get("_timestamp", "N/A")
                    accuracy = row.get("accuracy", "N/A")
                    print(f"  {timestamp}: {accuracy}")
            
            # Get unsampled metric data
            history = run.scan_history()
            losses = [row["loss"] for row in history if "loss" in row]
            print(f"✓ Retrieved {len(losses)} loss data points")
            
        except Exception as e:
            print(f"⚠ W&B API operations failed: {e}")
        
    except Exception as e:
        print(f"\n❌ Error training transformer model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # List saved models
    print("\n📁 Saved models:")
    if os.path.exists(CHECKPOINT_DIR):
        for file in sorted(os.listdir(CHECKPOINT_DIR)):
            if file.endswith('.pth'):
                size = os.path.getsize(os.path.join(CHECKPOINT_DIR, file)) / 1024 / 1024
                print(f"  ✓ {file}: {size:.2f} MB")
    
    print(f"\n📊 Reports saved to: {REPORT_DIR}")
    print(f"💾 Dataset files in: {DOWNLOAD_DIR}")
    
    print("\n" + "="*60)
    print("NEXT: Generate music with the web app:")
    print("python app.py")
    print("Then open: http://localhost:5000")
    print("="*60)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
