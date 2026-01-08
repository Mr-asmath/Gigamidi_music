import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class CNNMusicModel(nn.Module):
    """CNN-based music pattern generation model"""
    
    def __init__(self,
                 input_channels=4,
                 hidden_channels=[64, 128, 256, 512],
                 kernel_sizes=[3, 3, 3, 3],
                 text_feature_size=512,
                 output_size=4):
        super().__init__()
        
        # Text conditioning projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_size, hidden_channels[0] * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels[0] * 4, hidden_channels[0])
        )
        
        # CNN layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.text_projections = nn.ModuleList()
        
        in_channels = input_channels
        for i, (hidden, kernel) in enumerate(zip(hidden_channels, kernel_sizes)):
            # Conv layer
            conv = nn.Conv1d(
                in_channels + (hidden_channels[0] if i == 0 else 0),  # Add text channels for first layer
                hidden,
                kernel_size=kernel,
                padding=kernel // 2
            )
            self.conv_layers.append(conv)
            
            # Layer normalization
            self.norm_layers.append(nn.LayerNorm([hidden, 1]))  # Will reshape in forward
            
            # Text projection for this layer (if needed)
            if i < len(hidden_channels) - 1:
                text_proj = nn.Linear(hidden_channels[0], hidden)
                self.text_projections.append(text_proj)
            
            in_channels = hidden
        
        # Upsampling layers to generate output
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels[-1], hidden_channels[-1] // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels[-1] // 2, hidden_channels[-1] // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels[-1] // 4, output_size, kernel_size=4, stride=2, padding=1)
        )
        
        # Output heads for different parameters
        self.output_heads = nn.ModuleDict({
            'pitch': nn.Conv1d(output_size, 128, kernel_size=1),
            'velocity': nn.Conv1d(output_size, 128, kernel_size=1),
            'duration': nn.Conv1d(output_size, 1, kernel_size=1),
            'beat': nn.Conv1d(output_size, 1, kernel_size=1)
        })
        
    def forward(self, midi_features: torch.Tensor, text_features: torch.Tensor):
        """
        Args:
            midi_features: (batch_size, seq_len, input_channels)
            text_features: (batch_size, text_feature_size)
        """
        batch_size, seq_len, input_channels = midi_features.shape
        
        # Reshape for CNN: (batch_size, channels, seq_len)
        x = midi_features.transpose(1, 2)
        
        # Project text features
        text_projected = self.text_projection(text_features)  # (batch_size, hidden_channels[0])
        
        # Add text conditioning as additional channels for first layer
        text_expanded = text_projected.unsqueeze(-1).repeat(1, 1, seq_len)  # (batch_size, hidden_channels[0], seq_len)
        x = torch.cat([x, text_expanded], dim=1)
        
        # CNN processing with residual connections
        residuals = []
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            # Convolution
            x = conv(x)
            
            # Add text conditioning if not last layer
            if i < len(self.text_projections):
                text_adjusted = self.text_projections[i](text_projected)
                text_adjusted = text_adjusted.unsqueeze(-1).repeat(1, 1, x.shape[-1])
                x = x + text_adjusted
            
            # LayerNorm (need to handle spatial dimension)
            x_orig_shape = x.shape
            x = x.view(batch_size, x_orig_shape[1], -1)  # Flatten spatial dim
            x = norm(x)
            x = x.view(x_orig_shape)  # Restore shape
            
            # ReLU and store residual
            x = F.relu(x)
            residuals.append(x)
            
            # Add residual connection every other layer
            if i >= 2 and i % 2 == 0:
                x = x + residuals[i-2]
        
        # Upsample to original sequence length
        x = self.upsample(x)
        
        # Ensure correct output length
        if x.shape[-1] != seq_len:
            x = F.interpolate(x, size=seq_len, mode='linear', align_corners=False)
        
        # Generate outputs for each parameter
        outputs = {}
        for name, head in self.output_heads.items():
            output = head(x)
            
            # Transpose back to (batch_size, seq_len, features)
            if name in ['pitch', 'velocity']:
                outputs[name] = output.transpose(1, 2)  # (batch_size, seq_len, 128)
            else:
                outputs[name] = output.transpose(1, 2)  # (batch_size, seq_len, 1)
        
        # Apply activations
        outputs['beat'] = torch.sigmoid(outputs['beat'])
        outputs['duration'] = torch.sigmoid(outputs['duration'])
        
        return outputs
    
    def generate(self,
                 text_features: torch.Tensor,
                 seed_pattern: Optional[torch.Tensor] = None,
                 pattern_length: int = 64,
                 num_patterns: int = 8) -> Dict[str, torch.Tensor]:
        """Generate music by creating and repeating patterns"""
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Create seed pattern or use provided one
        if seed_pattern is None:
            # Generate random seed pattern
            seed = torch.randn(batch_size, pattern_length, 4, device=device)
        else:
            seed = seed_pattern
            
        # Process through CNN
        with torch.no_grad():
            outputs = self(seed, text_features)
            
            # Extract the pattern
            pattern = {
                'pitches': torch.argmax(outputs['pitch'], dim=-1).float(),
                'velocities': torch.argmax(outputs['velocity'], dim=-1).float(),
                'durations': outputs['duration'].squeeze(-1) * 480,
                'beats': (outputs['beat'].squeeze(-1) > 0.5).float()
            }
            
            # Repeat pattern
            repeated_pattern = {}
            for key in pattern:
                repeated = pattern[key].unsqueeze(1).repeat(1, num_patterns, 1)
                repeated_pattern[key] = repeated.view(batch_size, -1)
            
            return repeated_pattern