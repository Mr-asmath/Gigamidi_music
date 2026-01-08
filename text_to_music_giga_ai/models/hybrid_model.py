import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class HybridMusicModel(nn.Module):
    """Hybrid CNN-LSTM music generation model"""
    
    def __init__(self,
                 input_size=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=256,
                 lstm_layers=2,
                 text_feature_size=512,
                 dropout=0.2):
        super().__init__()
        
        # CNN feature extractor
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text conditioning for CNN
        self.text_cnn_projection = nn.Linear(text_feature_size, cnn_channels[2])
        
        # LSTM sequence model
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2] * 2,  # CNN features + text
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Text conditioning for LSTM
        self.text_lstm_projection = nn.Linear(text_feature_size, lstm_hidden * 2)  # *2 for bidirectional
        
        # Output heads
        self.pitch_head = nn.Sequential(
            nn.Linear(lstm_hidden * 4, lstm_hidden),  # *4 for bidirectional and concatenated features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 128)
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_hidden * 4, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 128)
        )
        
        self.duration_head = nn.Linear(lstm_hidden * 4, 1)
        self.beat_head = nn.Linear(lstm_hidden * 4, 1)
        
        # Upsampling to restore sequence length
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
    def forward(self, midi_features: torch.Tensor, text_features: torch.Tensor):
        """
        Args:
            midi_features: (batch_size, seq_len, input_size)
            text_features: (batch_size, text_feature_size)
        """
        batch_size, seq_len, _ = midi_features.shape
        
        # CNN processing: (batch_size, channels, seq_len)
        x = midi_features.transpose(1, 2)
        cnn_features = self.cnn_feature_extractor(x)
        
        # Apply text conditioning to CNN features
        text_cnn = self.text_cnn_projection(text_features).unsqueeze(-1)  # (batch_size, cnn_channels[2], 1)
        text_cnn_expanded = text_cnn.repeat(1, 1, cnn_features.shape[-1])
        cnn_features = cnn_features + text_cnn_expanded
        
        # Upsample if needed (CNN might reduce sequence length)
        if cnn_features.shape[-1] < seq_len:
            cnn_features = self.upsample(cnn_features)
            cnn_features = cnn_features[:, :, :seq_len]
        
        # Prepare for LSTM: (batch_size, seq_len, features)
        lstm_input = cnn_features.transpose(1, 2)
        
        # Add text conditioning for LSTM
        text_lstm = self.text_lstm_projection(text_features).unsqueeze(1)  # (batch_size, 1, lstm_hidden*2)
        text_lstm_expanded = text_lstm.repeat(1, seq_len, 1)
        lstm_input = torch.cat([lstm_input, text_lstm_expanded], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)
        
        # Generate predictions
        predictions = {
            'pitch': self.pitch_head(lstm_out),
            'velocity': self.velocity_head(lstm_out),
            'duration': torch.sigmoid(self.duration_head(lstm_out)),
            'beat': torch.sigmoid(self.beat_head(lstm_out))
        }
        
        return predictions
    
    def generate(self,
                 text_features: torch.Tensor,
                 initial_sequence: Optional[torch.Tensor] = None,
                 max_length: int = 512,
                 temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate music using hybrid approach"""
        batch_size = text_features.shape[0]
        device = text_features.device
        
        if initial_sequence is None:
            current_seq = torch.zeros(batch_size, 1, 4, device=device)
        else:
            current_seq = initial_sequence
        
        generated_features = []
        
        for i in range(max_length):
            with torch.no_grad():
                # Forward pass
                predictions = self(current_seq, text_features)
                
                # Sample next values
                pitch_probs = F.softmax(predictions['pitch'][:, -1, :] / temperature, dim=-1)
                velocity_probs = F.softmax(predictions['velocity'][:, -1, :] / temperature, dim=-1)
                
                pitch = torch.multinomial(pitch_probs, 1).float() / 127.0
                velocity = torch.multinomial(velocity_probs, 1).float() / 127.0
                duration = predictions['duration'][:, -1, :]
                beat = (predictions['beat'][:, -1, :] > 0.5).float()
                
                # Combine into feature vector
                next_features = torch.cat([pitch, velocity, duration, beat], dim=-1).unsqueeze(1)
                current_seq = torch.cat([current_seq, next_features], dim=1)
                
                # Store for output
                generated_features.append({
                    'pitch': pitch * 127,
                    'velocity': velocity * 127,
                    'duration': duration * 480,
                    'beat': beat
                })
        
        # Combine all generated features
        combined = {
            'pitches': torch.cat([f['pitch'] for f in generated_features], dim=1),
            'velocities': torch.cat([f['velocity'] for f in generated_features], dim=1),
            'durations': torch.cat([f['duration'] for f in generated_features], dim=1),
            'beats': torch.cat([f['beat'] for f in generated_features], dim=1)
        }
        
        return combined