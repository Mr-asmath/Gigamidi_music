#!/usr/bin/env python3
"""
Working training script for music generation models.
"""

import os
import sys
import torch
from typing import Dict, List
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lstm_model import LSTMMusicModel
from training.config import ModelConfig, TrainingConfig
from training.train_utils import MusicTrainer

# Create a simple text encoder for testing
class SimpleTextEncoder:
    """Simple text encoder for testing"""
    def __init__(self, feature_size=512):
        self.feature_size = feature_size
    
    def __call__(self, texts):
        # Convert list to batch
        if isinstance(texts, str):
            texts = [texts]
        
        # Create random features based on text length
        batch_features = []
        for text in texts:
            # Simple hash-based deterministic features
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            features = np.random.randn(self.feature_size).astype(np.float32)
            batch_features.append(features)
        
        return {'latent': torch.tensor(batch_features)}

# Create a simple dataset
class SimpleMusicDataset:
    def __init__(self, num_samples=1000, seq_length=128):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.texts = [
            "happy piano melody",
            "sad violin piece",
            "energetic guitar solo",
            "calm flute music",
            "epic orchestral soundtrack",
            "jazzy saxophone tune",
            "rock drum beat",
            "classical string quartet",
            "electronic synth wave",
            "blues harmonica"
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random sequence length
        actual_length = random.randint(64, self.seq_length)
        
        # Create features
        features = torch.zeros(actual_length, 4)
        
        # Pitch: mostly in middle range with some pattern
        base_pitch = random.randint(48, 72)  # C3 to C5
        for i in range(actual_length):
            # Create a simple melody pattern
            pitch_variation = np.sin(i / 8) * 5 + np.random.randn() * 2
            features[i, 0] = (base_pitch + pitch_variation) / 127.0
        
        # Velocity: dynamic variation
        features[:, 1] = (60 + torch.randn(actual_length) * 20).clamp(20, 100) / 127.0
        
        # Duration: mostly quarter notes with variation
        features[:, 2] = (240 + torch.randn(actual_length) * 60).clamp(120, 480) / 480.0
        
        # Beat: every 4th position is a beat
        for i in range(actual_length):
            features[i, 3] = 1.0 if i % 4 == 0 else 0.0
        
        # Random text from list
        text = random.choice(self.texts)
        
        return {
            'features': features,
            'text': text
        }

def create_collate_fn(seq_length=128):
    """Collate function to pad sequences"""
    def collate_fn(batch):
        features = []
        texts = []
        
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
        
        return {
            'features': torch.stack(features),
            'text': texts
        }
    return collate_fn

def train_single_model(model_type='lstm', epochs=10, batch_size=8):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create dataset
    dataset = SimpleMusicDataset(num_samples=500, seq_length=128)
    collate_fn = create_collate_fn(seq_length=128)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
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
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    if model_type == 'lstm':
        model = LSTMMusicModel(
            input_size=4,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            text_feature_size=512
        )
    elif model_type == 'transformer':
        from models.transformer_model import TransformerMusicModel
        model = TransformerMusicModel(
            input_size=4,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_len=256,
            text_feature_size=512
        )
    elif model_type == 'cnn':
        from models.cnn_model import CNNMusicModel
        model = CNNMusicModel(
            input_channels=4,
            hidden_channels=[32, 64, 128],
            text_feature_size=512,
            output_size=4
        )
    elif model_type == 'hybrid':
        from models.hybrid_model import HybridMusicModel
        model = HybridMusicModel(
            input_size=4,
            cnn_channels=[32, 64],
            lstm_hidden=64,
            lstm_layers=2,
            text_feature_size=512,
            dropout=0.2
        )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create text encoder
    text_encoder = SimpleTextEncoder(feature_size=512)
    
    # Create trainer
    config = TrainingConfig(
        device=device,
        learning_rate=0.001,
        batch_size=batch_size,
        save_dir="saved_models"
    )
    
    trainer = MusicTrainer(model, config, model_type)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_losses = trainer.train_epoch(train_loader, text_encoder)
        print(f"Train Loss: {train_losses.get('total', 0):.4f}")
        
        # Validate
        val_losses = trainer.validate(val_loader, text_encoder)
        print(f"Val Loss: {val_losses.get('total', 0):.4f}")
        
        # Save checkpoint
        is_best = val_losses.get('total', float('inf')) < trainer.best_loss
        trainer.save_checkpoint(epoch + 1, val_losses.get('total', 0), is_best)
        
        # Early stopping check
        if val_losses.get('total', 0) < 0.1:  # Good enough
            print("Good performance achieved, stopping early")
            break
    
    # Save final model
    model_path = os.path.join("saved_models", f"{model_type}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    return model, trainer

def main():
    print("="*60)
    print("MUSIC MODEL TRAINING SYSTEM")
    print("="*60)
    
    # Create output directory
    os.makedirs("saved_models", exist_ok=True)
    
    # Train models one by one
    models_to_train = ['lstm', 'transformer', 'cnn', 'hybrid']
    
    for model_type in models_to_train:
        try:
            model, trainer = train_single_model(
                model_type=model_type,
                epochs=5,  # Few epochs for testing
                batch_size=4
            )
        except Exception as e:
            print(f"\n✗ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # List saved models
    print("\nSaved models:")
    if os.path.exists("saved_models"):
        for file in os.listdir("saved_models"):
            if file.endswith('.pth'):
                size = os.path.getsize(os.path.join("saved_models", file)) / 1024
                print(f"  ✓ {file}: {size:.1f} KB")
    
    print("\n" + "="*60)
    print("NEXT: Generate music with:")
    print("python generate_music_fixed.py --text 'your description'")
    print("="*60)

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()