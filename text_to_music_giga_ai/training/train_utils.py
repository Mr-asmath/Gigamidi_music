# training/train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, List, Optional
import os

class MusicTrainer:
    """Training utilities for music models"""
    
    def __init__(self, model, config, model_name):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Get learning rate from config
        lr = getattr(config, 'learning_rate', 0.001)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss functions
        self.pitch_loss = nn.CrossEntropyLoss()
        self.velocity_loss = nn.CrossEntropyLoss()
        self.duration_loss = nn.MSELoss()
        self.beat_loss = nn.BCELoss()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
    def compute_loss(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss"""
        losses = {}
        
        # Pitch loss
        if 'pitch' in predictions and 'pitch' in targets:
            pitch_pred = predictions['pitch'].reshape(-1, predictions['pitch'].size(-1))
            pitch_target = targets['pitch'].reshape(-1).long()
            losses['pitch'] = self.pitch_loss(pitch_pred, pitch_target)
        
        # Velocity loss
        if 'velocity' in predictions and 'velocity' in targets:
            vel_pred = predictions['velocity'].reshape(-1, predictions['velocity'].size(-1))
            vel_target = targets['velocity'].reshape(-1).long()
            losses['velocity'] = self.velocity_loss(vel_pred, vel_target)
        
        # Duration loss
        if 'duration' in predictions and 'duration' in targets:
            dur_pred = predictions['duration'].reshape(-1)
            dur_target = targets['duration'].reshape(-1).float()
            losses['duration'] = self.duration_loss(dur_pred, dur_target)
        
        # Beat loss
        if 'beat' in predictions and 'beat' in targets:
            beat_pred = predictions['beat'].reshape(-1)
            beat_target = targets['beat'].reshape(-1).float()
            losses['beat'] = self.beat_loss(beat_pred, beat_target)
        
        # Weighted total loss
        total_loss = (
            losses.get('pitch', 0) * getattr(self.config, 'pitch_weight', 1.0) +
            losses.get('velocity', 0) * getattr(self.config, 'velocity_weight', 0.5) +
            losses.get('duration', 0) * getattr(self.config, 'duration_weight', 0.3) +
            losses.get('beat', 0) * getattr(self.config, 'beat_weight', 0.2)
        )
        
        return total_loss, losses
    
    def train_epoch(self, dataloader: DataLoader, text_encoder) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f"Training {self.model_name}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            features = batch['features'].float().to(self.device)
            text_descriptions = batch['text']
            
            # Encode text
            with torch.no_grad():
                text_features = text_encoder(text_descriptions)
                if isinstance(text_features, dict) and 'latent' in text_features:
                    text_features = text_features['latent'].to(self.device)
                else:
                    text_features = text_features.to(self.device)
            
            # Get targets from features
            targets = {
                'pitch': (features[:, :, 0] * 127).long().clamp(0, 127),
                'velocity': (features[:, :, 1] * 127).long().clamp(0, 127),
                'duration': features[:, :, 2],
                'beat': features[:, :, 3]
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features, text_features)
            
            # Compute loss
            loss, losses = self.compute_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = 0
                epoch_losses[loss_name] += loss_value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average losses
        num_batches = len(dataloader)
        avg_losses = {name: value / num_batches for name, value in epoch_losses.items()}
        avg_losses['total'] = total_loss / num_batches
        
        return avg_losses
    
    def validate(self, dataloader: DataLoader, text_encoder) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        val_losses = {}
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validating {self.model_name}")
            for batch in pbar:
                # Move data to device
                features = batch['features'].float().to(self.device)
                text_descriptions = batch['text']
                
                # Encode text
                text_features = text_encoder(text_descriptions)
                if isinstance(text_features, dict) and 'latent' in text_features:
                    text_features = text_features['latent'].to(self.device)
                else:
                    text_features = text_features.to(self.device)
                
                # Get targets
                targets = {
                    'pitch': (features[:, :, 0] * 127).long().clamp(0, 127),
                    'velocity': (features[:, :, 1] * 127).long().clamp(0, 127),
                    'duration': features[:, :, 2],
                    'beat': features[:, :, 3]
                }
                
                # Forward pass
                predictions = self.model(features, text_features)
                
                # Compute loss
                loss, losses = self.compute_loss(predictions, targets)
                
                # Update statistics
                total_loss += loss.item()
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = 0
                    val_losses[loss_name] += loss_value.item()
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Average losses
        num_batches = len(dataloader)
        avg_losses = {name: value / num_batches for name, value in val_losses.items()}
        avg_losses['total'] = total_loss / num_batches
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_name': self.model_name
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.save_dir,
            f"{self.model_name}_checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.save_dir, f"{self.model_name}_best.pth")
            torch.save(checkpoint, best_path)
            self.best_loss = loss
        
        # Save model weights separately (for easy loading)
        model_path = os.path.join(self.config.save_dir, f"{self.model_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        if is_best:
            print(f"New best model: {best_path}")