#!/usr/bin/env python3
"""
Flask Web Application for Text-to-Music AI
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import torch
import numpy as np
import pretty_midi
import uuid
from datetime import datetime
import threading

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import music generation functions
from generate_music_fixed import (
    load_model, generate_music, save_midi, 
    create_fallback_music, SimpleTextEncoder
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Global model cache
MODEL_CACHE = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_type):
    """Get model from cache or load it"""
    if model_type not in MODEL_CACHE:
        MODEL_CACHE[model_type] = load_model(model_type, DEVICE)
    return MODEL_CACHE.get(model_type)

def generate_music_file(text, model_type, length, tempo, instrument):
    """Generate music and save to file"""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"music_{timestamp}_{uuid.uuid4().hex[:8]}.mid"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Try to load model
        model = get_model(model_type)
        
        if model:
            # Generate with trained model
            generated = generate_music(
                model, text, length, 
                temperature=0.8, device=DEVICE
            )
        else:
            # Use fallback
            generated = create_fallback_music(text, length)
        
        # Save MIDI file
        midi = save_midi(generated, filepath, tempo, instrument)
        
        # Convert MIDI to WAV for web playback (optional)
        # wav_path = convert_midi_to_wav(filepath)
        
        return {
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'message': 'Music generated successfully!',
            'notes': len(generated['pitches'][0]),
            'tempo': tempo,
            'instrument': instrument
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Error generating music'
        }

@app.route('/')
def index():
    """Render main page"""
    # Check if models are trained
    models_trained = os.path.exists('saved_models') and len(os.listdir('saved_models')) > 0
    
    return render_template('index.html', 
                         models_trained=models_trained,
                         device=DEVICE)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate music from text"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        model_type = data.get('model', 'lstm')
        length = int(data.get('length', 50))
        tempo = int(data.get('tempo', 120))
        instrument = data.get('instrument', 'Acoustic Grand Piano')
        
        if not text:
            return jsonify({
                'success': False,
                'message': 'Please enter text description'
            })
        
        # Validate inputs
        length = max(10, min(length, 500))  # Limit 10-500 notes
        tempo = max(40, min(tempo, 240))    # Limit 40-240 BPM
        
        # Generate music in background thread
        result = generate_music_file(text, model_type, length, tempo, instrument)
        
        if result['success']:
            # Get file URL for web access
            file_url = f"/static/audio/{result['filename']}"
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'file_url': file_url,
                'filename': result['filename'],
                'notes': result['notes'],
                'tempo': result['tempo'],
                'instrument': result['instrument'],
                'preview_text': text[:50] + '...' if len(text) > 50 else text
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Generation failed'),
                'error': result.get('error', '')
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/train', methods=['POST'])
def train_models():
    """Start model training"""
    try:
        # Start training in background thread
        def train_background():
            os.system('python train_all_models.py')
        
        thread = threading.Thread(target=train_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started in background. Check terminal for progress.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting training: {str(e)}'
        })

@app.route('/check-models')
def check_models():
    """Check if models are trained"""
    models = []
    if os.path.exists('saved_models'):
        for file in os.listdir('saved_models'):
            if file.endswith('.pth'):
                size = os.path.getsize(os.path.join('saved_models', file)) / 1024
                models.append({
                    'name': file.replace('.pth', ''),
                    'size_kb': round(size, 1)
                })
    
    return jsonify({
        'trained': len(models) > 0,
        'models': models,
        'count': len(models)
    })

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated MIDI file"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'success': False, 'message': 'File not found'})

@app.route('/list-audio')
def list_audio():
    """List generated audio files"""
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for file in sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True):
            if file.endswith('.mid'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
                size = os.path.getsize(filepath) / 1024
                files.append({
                    'name': file,
                    'size_kb': round(size, 1),
                    'url': f"/static/audio/{file}"
                })
    
    return jsonify({'files': files[:10]})  # Return last 10 files

@app.route('/clear-audio', methods=['POST'])
def clear_audio():
    """Clear generated audio files"""
    try:
        count = 0
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                if file.endswith('.mid'):
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
                    count += 1
        
        return jsonify({
            'success': True,
            'message': f'Cleared {count} audio files'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

if __name__ == '__main__':
    print("="*60)
    print("AI MUSIC GENERATOR WEB APP")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Audio folder: {app.config['UPLOAD_FOLDER']}")
    print("Starting server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)