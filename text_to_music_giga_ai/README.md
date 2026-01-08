# Text-to-Music Generation with GigaMIDI

A multi-model AI system for generating music from text descriptions using the GigaMIDI dataset.

## Features
- Multiple model architectures: LSTM, Transformer, CNN, Hybrid
- Text-to-MIDI generation
- GigaMIDI dataset integration
- Pre-trained model saving/loading
- Real-time music generation

## Installation
```bash
pip install -r requirements.txt


### 4. **`preprocessing/__init__.py`**
```python
from .midi_processor import MIDIProcessor
from .tokenizer import MIDITokenizer
from .text_encoder import TextEncoder

__all__ = ['MIDIProcessor', 'MIDITokenizer', 'TextEncoder']