# training/config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for each model type"""
    model_type: str  # 'lstm', 'transformer', 'cnn', 'hybrid'
    hidden_size: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    
    # Transformer specific
    nhead: int = 8
    dim_feedforward: int = 2048
    
    # CNN specific
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Hybrid specific
    lstm_hidden: int = 256
    lstm_layers: int = 2
    
    # Add device parameter
    device: str = "cuda"

@dataclass
class TrainingConfig:
    """Global training configuration"""
    # Dataset parameters
    dataset_path: str = "Metacreation/GigaMIDI"
    max_sequence_length: int = 512
    max_samples: int = 10000
    batch_size: int = 32
    
    # Model parameters
    text_feature_size: int = 512
    
    # Training parameters
    learning_rate: float = 0.001
    epochs: int = 50
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    
    # Paths
    save_dir: str = "saved_models"
    log_dir: str = "logs"
    
    # Device
    device: str = "cuda"
    
    # Checkpointing
    save_every: int = 5
    
    # Loss weights
    pitch_weight: float = 1.0
    velocity_weight: float = 0.5
    duration_weight: float = 0.3
    beat_weight: float = 0.2
    
    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print(f"CUDA not available, using {self.device}")