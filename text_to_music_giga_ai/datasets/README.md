# GigaMIDI Dataset Setup

## Dataset Access

The GigaMIDI dataset is hosted on Hugging Face and requires acceptance of terms:

1. Visit: https://huggingface.co/datasets/Metacreation/GigaMIDI
2. Accept the terms of use (CC BY-NC 4.0 license)
3. The dataset will be automatically downloaded when running the training script

## Dataset Structure

The dataset contains:
- 2.1M+ MIDI files
- Metadata including:
  - Title and artist
  - Music styles
  - Instrument information
  - Tempo and musical features
  - Loop annotations

## Automatic Download

The training script automatically:
1. Checks if you have access to the dataset
2. Downloads it using the Hugging Face `datasets` library
3. Processes MIDI files into training features

## Manual Download (Optional)

If you want to download manually:
```bash
# Install datasets library
pip install datasets

# Download dataset
from datasets import load_dataset
dataset = load_dataset("Metacreation/GigaMIDI")

# Or download specific split
train_data = load_dataset("Metacreation/GigaMIDI", split="train")