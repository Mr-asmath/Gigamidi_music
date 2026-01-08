import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import numpy as np

class TextEncoder(nn.Module):
    """Encode text descriptions into musical feature vectors"""
    
    def __init__(self, model_name="bert-base-uncased", hidden_size=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Musical feature dimensions
        self.emotion_encoder = nn.Linear(hidden_size, 4)  # happy, sad, energetic, calm
        self.style_encoder = nn.Linear(hidden_size, 8)    # classical, jazz, pop, rock, etc.
        self.instrument_encoder = nn.Linear(hidden_size, 16)  # instrument types
        self.tempo_encoder = nn.Linear(hidden_size, 1)    # tempo prediction
        
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text into musical features"""
        # Tokenize text
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to musical feature space
        projected = self.projection(cls_embedding)
        
        # Extract specific musical features
        features = {
            "emotion": torch.sigmoid(self.emotion_encoder(projected)),
            "style": torch.softmax(self.style_encoder(projected), dim=-1),
            "instruments": torch.sigmoid(self.instrument_encoder(projected)),
            "tempo": torch.sigmoid(self.tempo_encoder(projected)) * 200 + 60,  # 60-260 BPM
            "latent": projected
        }
        
        return features
    
    def extract_musical_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract musical keywords from text"""
        musical_terms = {
            "instruments": ["piano", "guitar", "violin", "drums", "bass", "flute", 
                           "trumpet", "saxophone", "cello", "harp", "organ", "synth"],
            "emotions": ["happy", "sad", "energetic", "calm", "romantic", "angry", 
                        "peaceful", "exciting", "melancholic", "joyful"],
            "genres": ["classical", "jazz", "rock", "pop", "blues", "electronic", 
                      "folk", "hiphop", "reggae", "metal"],
            "tempo": ["fast", "slow", "moderate", "allegro", "adagio", "presto"],
            "dynamics": ["loud", "soft", "crescendo", "diminuendo", "forte", "piano"]
        }
        
        text_lower = text.lower()
        extracted = {k: [] for k in musical_terms.keys()}
        
        for category, terms in musical_terms.items():
            for term in terms:
                if term in text_lower:
                    extracted[category].append(term)
        
        return extracted