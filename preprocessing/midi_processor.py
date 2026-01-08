import numpy as np
from typing import List, Dict, Tuple, Optional
import pretty_midi
from symusic import Score
import librosa
import torch
from dataclasses import dataclass

@dataclass
class MIDIFeatures:
    """Extracted features from MIDI file"""
    pitch: np.ndarray
    velocity: np.ndarray
    duration: np.ndarray
    tempo: float
    instrument: str
    beats: np.ndarray
    chords: List[str]

class MIDIProcessor:
    """Process MIDI files and extract features"""
    
    def __init__(self, max_sequence_length=512):
        self.max_sequence_length = max_sequence_length
        
    def load_midi(self, midi_bytes: bytes) -> Score:
        """Load MIDI from bytes using symusic"""
        try:
            score = Score.from_midi(midi_bytes)
            return score
        except Exception as e:
            print(f"Error loading MIDI: {e}")
            return None
    
    def extract_features(self, score: Score) -> MIDIFeatures:
        """Extract musical features from MIDI score"""
        features = []
        
        for track in score.tracks:
            if len(track.notes) > 0:
                # Extract note information
                pitches = [note.pitch for note in track.notes]
                velocities = [note.velocity for note in track.notes]
                durations = [note.duration for note in track.notes]
                start_times = [note.start for note in track.notes]
                
                # Calculate tempo (simplified)
                if len(start_times) > 1:
                    intervals = np.diff(sorted(start_times))
                    avg_interval = np.mean(intervals[intervals > 0])
                    tempo = 60 / (avg_interval * score.ticks_per_quarter / 480) if avg_interval > 0 else 120
                else:
                    tempo = 120
                
                # Identify instrument
                program = track.program if hasattr(track, 'program') else 0
                instrument = self._get_instrument_name(program)
                
                # Extract beat information
                beats = self._extract_beats(start_times, score.ticks_per_quarter)
                
                # Extract chords (simplified)
                chords = self._extract_chords(pitches, start_times)
                
                midi_features = MIDIFeatures(
                    pitch=np.array(pitches),
                    velocity=np.array(velocities),
                    duration=np.array(durations),
                    tempo=tempo,
                    instrument=instrument,
                    beats=beats,
                    chords=chords
                )
                features.append(midi_features)
        
        return features[0] if features else None
    
    def _get_instrument_name(self, program: int) -> str:
        """Convert MIDI program number to instrument name"""
        instruments = {
            0: "piano", 1: "piano", 2: "piano",
            24: "guitar", 25: "guitar", 26: "guitar",
            32: "bass", 33: "bass",
            40: "violin", 41: "viola", 42: "cello",
            48: "strings", 49: "strings", 50: "strings",
            56: "trumpet", 57: "trombone", 58: "tuba",
            64: "saxophone", 65: "saxophone", 66: "saxophone",
            72: "flute", 73: "flute", 74: "flute",
            80: "synth", 81: "synth", 82: "synth",
            88: "pad", 89: "pad", 90: "pad",
            96: "drums", 97: "drums", 98: "drums"
        }
        return instruments.get(program, "unknown")
    
    def _extract_beats(self, start_times: List[float], tpq: int) -> np.ndarray:
        """Extract beat positions"""
        if len(start_times) < 2:
            return np.array([])
        
        start_times = np.array(start_times)
        beats = []
        quarter_note = tpq
        
        for i in range(0, int(np.max(start_times)), quarter_note):
            if np.any(np.abs(start_times - i) < quarter_note / 4):
                beats.append(i / quarter_note)
        
        return np.array(beats)
    
    def _extract_chords(self, pitches: List[int], start_times: List[float]) -> List[str]:
        """Extract simple chord information"""
        if len(pitches) < 3:
            return []
        
        chords = []
        for i in range(0, len(pitches) - 2, 3):
            chord_pitches = sorted(set(pitches[i:i+3]))
            if len(chord_pitches) >= 3:
                root = chord_pitches[0] % 12
                chord_name = self._get_chord_name(root, chord_pitches)
                chords.append(chord_name)
        
        return chords
    
    def _get_chord_name(self, root: int, pitches: List[int]) -> str:
        """Get chord name from pitches"""
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        intervals = set()
        
        for pitch in pitches:
            intervals.add((pitch - root) % 12)
        
        if {0, 4, 7} <= intervals:
            return f"{notes[root]} major"
        elif {0, 3, 7} <= intervals:
            return f"{notes[root]} minor"
        elif {0, 4, 7, 10} <= intervals:
            return f"{notes[root]} dominant 7th"
        else:
            return f"{notes[root]} chord"
    
    def create_sequence_tensor(self, features: MIDIFeatures) -> torch.Tensor:
        """Convert features to sequence tensor"""
        # Combine features into a sequence
        seq_length = min(len(features.pitch), self.max_sequence_length)
        
        # Create feature matrix: [pitch, velocity, duration, beat_position]
        sequence = np.zeros((seq_length, 4))
        
        for i in range(seq_length):
            sequence[i, 0] = features.pitch[i] / 127.0  # Normalized pitch
            sequence[i, 1] = features.velocity[i] / 127.0  # Normalized velocity
            sequence[i, 2] = min(features.duration[i] / 480.0, 1.0)  # Normalized duration
            sequence[i, 3] = 1.0 if i in features.beats else 0.0  # Beat indicator
        
        return torch.FloatTensor(sequence)