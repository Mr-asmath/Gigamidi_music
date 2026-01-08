import torch
from typing import List, Dict, Tuple
from miditok import REMI, TokenizerConfig
from symusic import Score
import numpy as np

class MIDITokenizer:
    """Tokenize MIDI sequences using REMI tokenization"""
    
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.config = TokenizerConfig(
            num_velocities=32,
            use_chords=True,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=True,
            num_tempos=32,
            tempo_range=(30, 250),
            beat_res={(0, 4): 8, (4, 12): 4},
            num_durations=32,
            special_tokens=["PAD", "MASK", "BOS", "EOS"]
        )
        self.tokenizer = REMI(self.config)
        
    def tokenize_score(self, score: Score) -> List[int]:
        """Tokenize a MIDI score"""
        tokens = self.tokenizer(score)
        return tokens.ids if hasattr(tokens, 'ids') else tokens
    
    def detokenize(self, tokens: List[int]) -> Score:
        """Convert tokens back to MIDI score"""
        return self.tokenizer.tokens_to_score(tokens)
    
    def create_vocabulary(self, dataset_samples=1000):
        """Create vocabulary from dataset samples"""
        # This would normally train the tokenizer on the dataset
        # For now, we'll use the pre-trained REMI vocabulary
        pass
    
    def encode_batch(self, scores: List[Score]) -> Dict[str, torch.Tensor]:
        """Encode batch of scores"""
        batch_tokens = [self.tokenize_score(score) for score in scores]
        
        # Pad sequences
        max_len = max(len(tokens) for tokens in batch_tokens)
        padded_tokens = []
        attention_masks = []
        
        for tokens in batch_tokens:
            padded = tokens + [self.tokenizer["PAD"]] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)
            attention_masks.append(mask)
        
        return {
            "input_ids": torch.LongTensor(padded_tokens),
            "attention_mask": torch.LongTensor(attention_masks)
        }