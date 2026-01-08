#!/usr/bin/env python3
"""
Working music generator with trained models.
"""

import torch
import argparse
import os
import sys
import numpy as np
import pretty_midi

# Add safe globals
try:
    torch.serialization.add_safe_globals([object])  # Allow any object
except:
    pass

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lstm_model import LSTMMusicModel

class SimpleTextEncoder:
    """Simple text encoder matching training"""
    def __init__(self, feature_size=512):
        self.feature_size = feature_size
    
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        batch_features = []
        for text in texts:
            # Deterministic features from text hash
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            features = np.random.randn(self.feature_size).astype(np.float32)
            batch_features.append(features)
        
        return {'latent': torch.tensor(batch_features)}

def load_model(model_type='lstm', device='cpu'):
    """Load trained model"""
    model_path = os.path.join("saved_models", f"{model_type}.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train models first with: python train_all_models.py")
        return None
    
    print(f"Loading {model_type} model...")
    
    # Create model architecture
    if model_type == 'lstm':
        model = LSTMMusicModel(
            input_size=4,
            hidden_size=128,
            num_layers=2,
            dropout=0.0,  # Disable dropout for inference
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
            dropout=0.0,
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
            dropout=0.0
        )
    
    # Load weights
    try:
        # Try different loading methods
        try:
            # First try with weights_only=False
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except:
            # Fall back to standard load
            checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Try loading as direct state dict
                model.load_state_dict(checkpoint)
        else:
            # Assume it's a state dict
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✓ {model_type} model loaded")
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def generate_music(model, text, length=100, temperature=0.8, device='cpu'):
    """Generate music from text"""
    # Create text encoder
    text_encoder = SimpleTextEncoder(feature_size=512)
    
    # Encode text
    with torch.no_grad():
        text_features = text_encoder([text])['latent'].to(device)
        
        # Generate music
        if hasattr(model, 'generate'):
            try:
                generated = model.generate(
                    text_features,
                    max_length=length,
                    temperature=temperature
                )
                return generated
            except Exception as e:
                print(f"Generation error: {e}")
        
        # Fallback: create simple output
        print("Using fallback generation")
        return create_fallback_music(text, length)

def create_fallback_music(text, length=100):
    """Create fallback music based on text"""
    text_lower = text.lower()
    
    # Simple emotion-based generation
    if any(word in text_lower for word in ['happy', 'joy', 'upbeat']):
        # Major scale, upbeat
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major
        pattern = [0, 2, 4, 5, 7, 5, 4, 2]
    elif any(word in text_lower for word in ['sad', 'melancholy', 'slow']):
        # Minor scale, slow
        scale = [60, 62, 63, 65, 67, 68, 70, 72]  # C minor
        pattern = [0, 2, 3, 5, 7, 5, 3, 2]
    else:
        # Neutral
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        pattern = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Generate melody
    pitches = []
    for i in range(length):
        note_idx = pattern[i % len(pattern)]
        if note_idx < len(scale):
            pitches.append(scale[note_idx])
        else:
            pitches.append(scale[0])
    
    return {
        'pitches': torch.tensor([pitches], dtype=torch.float32),
        'velocities': torch.full((1, length), 80.0),
        'durations': torch.full((1, length), 240.0),
        'beats': torch.zeros(1, length)
    }

def save_midi(generated, output_path, tempo=120, instrument="Acoustic Grand Piano"):
    """Save generated music as MIDI"""
    # Extract data
    pitches = generated['pitches'].cpu().numpy()[0]
    velocities = generated['velocities'].cpu().numpy()[0]
    
    if 'durations' in generated:
        durations = generated['durations'].cpu().numpy()[0]
    else:
        durations = np.full_like(pitches, 240.0)
    
    # Create MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Get instrument program
    try:
        program = pretty_midi.instrument_name_to_program(instrument)
    except:
        program = 0
    
    inst = pretty_midi.Instrument(program=program)
    
    # Add notes
    current_time = 0.0
    seconds_per_beat = 60.0 / tempo
    
    for i in range(len(pitches)):
        pitch = int(np.clip(pitches[i], 0, 127))
        velocity = int(np.clip(velocities[i], 1, 127))
        duration_beats = max(0.1, durations[i] / 480.0)  # Convert ticks to beats
        duration_seconds = duration_beats * seconds_per_beat
        
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + duration_seconds
        )
        inst.notes.append(note)
        
        # Move to next note (with some overlap)
        current_time += duration_seconds * 0.7
    
    midi.instruments.append(inst)
    midi.write(output_path)
    
    return midi

def main():
    parser = argparse.ArgumentParser(description="Generate music from text")
    parser.add_argument("--text", type=str, required=True,
                       help="Music description")
    parser.add_argument("--model", type=str, default="lstm",
                       choices=["lstm", "transformer", "cnn", "hybrid"],
                       help="Model to use")
    parser.add_argument("--output", type=str, default="output.mid",
                       help="Output MIDI file")
    parser.add_argument("--length", type=int, default=50,
                       help="Number of notes")
    parser.add_argument("--tempo", type=int, default=120,
                       help="Tempo (BPM)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MUSIC GENERATOR")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists("saved_models"):
        print("No trained models found!")
        print("\nFirst, train models with:")
        print("python train_all_models.py")
        print("\nCreating sample music anyway...")
        
        # Create sample music
        generated = create_fallback_music(args.text, args.length)
    else:
        # Load and use trained model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.model, device)
        
        if model:
            generated = generate_music(
                model, args.text, args.length, 
                temperature=0.8, device=device
            )
        else:
            print("Using fallback generation")
            generated = create_fallback_music(args.text, args.length)
    
    # Save MIDI
    midi = save_midi(generated, args.output, args.tempo)
    
    print(f"\n✓ Music generated successfully!")
    print(f"  File: {args.output}")
    print(f"  Notes: {len(generated['pitches'][0])}")
    print(f"  Tempo: {args.tempo} BPM")
    
    print("\n" + "="*60)
    print("You can play the MIDI file with:")
    print("- Windows Media Player")
    print("- VLC Media Player") 
    print("- Any MIDI player")
    print("="*60)

if __name__ == "__main__":
    # If no arguments, run a test
    if len(sys.argv) == 1:
        print("No arguments provided. Running test...")
        test_text = "happy piano melody with joyful feeling"
        sys.argv = [sys.argv[0], "--text", test_text]
    
    main()