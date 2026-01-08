import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class LSTMMusicModel(nn.Module):
    """LSTM-based music generation model"""
    
    def __init__(self, 
                 input_size=4,
                 hidden_size=512,
                 num_layers=3,
                 output_size=128,
                 dropout=0.2,
                 text_feature_size=512):
        super().__init__()
        
        # Text conditioning
        self.text_projection = nn.Linear(text_feature_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size + hidden_size,  # Combine MIDI features + text
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layers for different MIDI parameters
        self.pitch_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 128)  # 0-127 MIDI pitches
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 128)  # 0-127 velocities
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Duration in ticks
        )
        
        self.beat_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Beat probability
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, 
                midi_features: torch.Tensor,
                text_features: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            midi_features: (batch_size, seq_len, input_size)
            text_features: (batch_size, text_feature_size)
            hidden_state: Optional LSTM hidden state
            
        Returns:
            Dict of predictions and hidden state
        """
        batch_size, seq_len, _ = midi_features.shape
        
        # Project text features and repeat for sequence
        text_projected = self.text_projection(text_features)  # (batch_size, hidden_size)
        text_repeated = text_projected.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
        
        # Concatenate MIDI features with text conditioning
        combined = torch.cat([midi_features, text_repeated], dim=-1)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(combined, hidden_state)
        
        # Generate predictions for each MIDI parameter
        predictions = {
            'pitch': self.pitch_head(lstm_out),  # (batch_size, seq_len, 128)
            'velocity': self.velocity_head(lstm_out),  # (batch_size, seq_len, 128)
            'duration': self.duration_head(lstm_out),  # (batch_size, seq_len, 1)
            'beat': torch.sigmoid(self.beat_head(lstm_out)),  # (batch_size, seq_len, 1)
            'hidden_state': hidden
        }
        
        return predictions
    
    def generate(self, 
                 text_features: torch.Tensor,
                 initial_sequence: Optional[torch.Tensor] = None,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 top_k: int = 50) -> Dict[str, torch.Tensor]:
        """Generate music sequence"""
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Initialize with starting sequence or zeros
        if initial_sequence is None:
            current_seq = torch.zeros(batch_size, 1, 4, device=device)
        else:
            current_seq = initial_sequence
            
        # Initialize hidden state
        hidden = None
        
        # Store generated sequences
        generated_pitches = []
        generated_velocities = []
        generated_durations = []
        generated_beats = []
        
        for step in range(max_length):
            # Forward pass
            with torch.no_grad():
                predictions = self(current_seq, text_features, hidden)
                hidden = predictions['hidden_state']
                
                # Sample from predictions with temperature
                pitch_logits = predictions['pitch'][:, -1, :] / temperature
                velocity_logits = predictions['velocity'][:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k_pitch = min(top_k, pitch_logits.size(-1))
                    top_k_velocity = min(top_k, velocity_logits.size(-1))
                    
                    pitch_values, pitch_indices = torch.topk(pitch_logits, top_k_pitch)
                    velocity_values, velocity_indices = torch.topk(velocity_logits, top_k_velocity)
                    
                    pitch_logits = torch.full_like(pitch_logits, float('-inf'))
                    velocity_logits = torch.full_like(velocity_logits, float('-inf'))
                    
                    pitch_logits.scatter_(-1, pitch_indices, pitch_values)
                    velocity_logits.scatter_(-1, velocity_indices, velocity_values)
                
                # Sample pitches and velocities
                pitch_probs = F.softmax(pitch_logits, dim=-1)
                velocity_probs = F.softmax(velocity_logits, dim=-1)
                
                pitch = torch.multinomial(pitch_probs, 1).float() / 127.0
                velocity = torch.multinomial(velocity_probs, 1).float() / 127.0
                
                # Get duration and beat
                duration = torch.sigmoid(predictions['duration'][:, -1, :])
                beat = (predictions['beat'][:, -1, :] > 0.5).float()
                
                # Create next feature vector
                next_features = torch.cat([pitch, velocity, duration, beat], dim=-1).unsqueeze(1)
                
                # Update sequence
                current_seq = torch.cat([current_seq, next_features], dim=1)
                
                # Store generated values
                generated_pitches.append(pitch * 127)
                generated_velocities.append(velocity * 127)
                generated_durations.append(duration * 480)  # Scale to ticks
                generated_beats.append(beat)
        
        return {
            'pitches': torch.cat(generated_pitches, dim=1),
            'velocities': torch.cat(generated_velocities, dim=1),
            'durations': torch.cat(generated_durations, dim=1),
            'beats': torch.cat(generated_beats, dim=1)
        }