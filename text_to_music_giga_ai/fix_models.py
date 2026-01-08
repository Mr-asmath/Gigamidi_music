# fix_models.py
import torch
import os

def fix_model_files(model_dir="saved_models"):
    """Fix saved model files to be compatible with PyTorch 2.6+"""
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} not found")
        return
    
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(model_dir, filename)
            print(f"\nFixing {filename}...")
            
            try:
                # Load the file
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                
                # Extract just the state dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        # Try to find any tensor dict
                        state_dict = {}
                        for key, value in checkpoint.items():
                            if isinstance(value, torch.Tensor) or hasattr(value, 'keys'):
                                state_dict[key] = value
                        if not state_dict:
                            print(f"  Could not extract state dict from {filename}")
                            continue
                else:
                    print(f"  Unexpected format in {filename}")
                    continue
                
                # Save just the state dict
                torch.save(state_dict, filepath)
                print(f"  ✓ Fixed {filename} ({len(state_dict)} parameters)")
                
            except Exception as e:
                print(f"  ✗ Error fixing {filename}: {e}")

if __name__ == "__main__":
    fix_model_files()