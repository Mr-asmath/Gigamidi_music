import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerMusicModel(nn.Module):
    """Transformer-based music generation model"""
    
    def __init__(self,
                 input_size=4,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_len=2000,
                 text_feature_size=512):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        self.text_projection = nn.Linear(text_feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.pitch_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for concatenated features
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 128)
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 128)
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def create_mask(self, sz):
        """Create causal mask for generation"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, 
                midi_features: torch.Tensor,
                text_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            midi_features: (batch_size, seq_len, input_size)
            text_features: (batch_size, text_feature_size)
            mask: Optional attention mask
        """
        batch_size, seq_len = midi_features.shape[:2]
        
        # Project inputs
        midi_projected = self.input_projection(midi_features) * math.sqrt(self.d_model)
        text_projected = self.text_projection(text_features).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add positional encoding to MIDI features
        midi_projected = self.pos_encoder(midi_projected)
        
        # Concatenate text features at the beginning of sequence
        combined = torch.cat([text_projected, midi_projected], dim=1)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.create_mask(combined.size(1)).to(combined.device)
        
        # Transformer encoding
        encoded = self.transformer_encoder(combined, mask=mask)
        
        # Separate text encoding and MIDI encoding
        text_encoded = encoded[:, 0, :]  # (batch_size, d_model)
        midi_encoded = encoded[:, 1:, :]  # (batch_size, seq_len, d_model)
        
        # Repeat text encoding for each position
        text_repeated = text_encoded.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate text and MIDI encodings
        combined_features = torch.cat([midi_encoded, text_repeated], dim=-1)
        
        # Generate predictions
        predictions = {
            'pitch': self.pitch_head(combined_features),
            'velocity': self.velocity_head(combined_features),
            'duration': self.duration_head(combined_features)
        }
        
        return predictions
    
    def generate(self,
                 text_features: torch.Tensor,
                 initial_sequence: Optional[torch.Tensor] = None,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 top_p: float = 0.9) -> Dict[str, torch.Tensor]:
        """Generate music sequence autoregressively"""
        batch_size = text_features.shape[0]
        device = text_features.device
        
        if initial_sequence is None:
            generated = torch.zeros(batch_size, 1, 4, device=device)
        else:
            generated = initial_sequence
            
        for i in range(max_length):
            # Create mask for current sequence length
            current_len = generated.shape[1]
            mask = self.create_mask(current_len + 1).to(device)  # +1 for text token
            
            # Forward pass
            with torch.no_grad():
                predictions = self(generated, text_features, mask)
                
                # Get predictions for the last position
                pitch_logits = predictions['pitch'][:, -1, :] / temperature
                velocity_logits = predictions['velocity'][:, -1, :] / temperature
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_pitch_logits, sorted_pitch_indices = torch.sort(pitch_logits, descending=True)
                    sorted_velocity_logits, sorted_velocity_indices = torch.sort(velocity_logits, descending=True)
                    
                    pitch_cum_probs = torch.cumsum(F.softmax(sorted_pitch_logits, dim=-1), dim=-1)
                    velocity_cum_probs = torch.cumsum(F.softmax(sorted_velocity_logits, dim=-1), dim=-1)
                    
                    pitch_mask = pitch_cum_probs > top_p
                    velocity_mask = velocity_cum_probs > top_p
                    
                    pitch_logits[sorted_pitch_indices[pitch_mask]] = float('-inf')
                    velocity_logits[sorted_velocity_indices[velocity_mask]] = float('-inf')
                
                # Sample next values
                pitch_probs = F.softmax(pitch_logits, dim=-1)
                velocity_probs = F.softmax(velocity_logits, dim=-1)
                
                pitch = torch.multinomial(pitch_probs, 1).float() / 127.0
                velocity = torch.multinomial(velocity_probs, 1).float() / 127.0
                
                # Get duration
                duration = torch.sigmoid(predictions['duration'][:, -1, :])
                
                # Create next feature vector (simple beat pattern)
                beat = torch.tensor([[1.0] if i % 4 == 0 else [0.0]] * batch_size, device=device).unsqueeze(-1)
                
                next_features = torch.cat([pitch, velocity, duration, beat], dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_features], dim=1)
        
        # Extract features from generated sequence (skip the first zero vector if needed)
        final_seq = generated[:, 1:, :] if initial_sequence is None else generated
        
        return {
            'sequence': final_seq,
            'pitches': final_seq[:, :, 0] * 127,
            'velocities': final_seq[:, :, 1] * 127,
            'durations': final_seq[:, :, 2] * 480,
            'beats': final_seq[:, :, 3]
        }