document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const modelSelect = document.getElementById('modelSelect');
    const instrumentSelect = document.getElementById('instrumentSelect');
    const lengthInput = document.getElementById('lengthInput');
    const lengthValue = document.getElementById('lengthValue');
    const tempoInput = document.getElementById('tempoInput');
    const tempoValue = document.getElementById('tempoValue');
    const generateBtn = document.getElementById('generateBtn');
    const trainBtn = document.getElementById('trainBtn');
    const loadingState = document.getElementById('loadingState');
    const resultState = document.getElementById('resultState');
    const emptyState = document.getElementById('emptyState');
    const recentList = document.getElementById('recentList');
    const modelsList = document.getElementById('modelsList');
    const modelsStatus = document.getElementById('modelsStatus');
    const refreshModels = document.getElementById('refreshModels');
    const clearAudio = document.getElementById('clearAudio');
    const downloadBtn = document.getElementById('downloadBtn');
    const regenerateBtn = document.getElementById('regenerateBtn');
    const shareBtn = document.getElementById('shareBtn');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const stopBtn = document.getElementById('stopBtn');
    const visualizer = document.getElementById('visualizer');
    const audioPlayer = document.getElementById('audioPlayer');
    
    // State
    let currentMusic = null;
    let isPlaying = false;
    let visualizationInterval = null;
    let recentGenerations = [];
    
    // Initialize
    init();
    
    function init() {
        updateCharCount();
        checkModels();
        loadRecentGenerations();
        setupEventListeners();
        setupVisualization();
    }
    
    function setupEventListeners() {
        // Text input
        textInput.addEventListener('input', updateCharCount);
        
        // Range inputs
        lengthInput.addEventListener('input', () => {
            lengthValue.textContent = lengthInput.value;
        });
        
        tempoInput.addEventListener('input', () => {
            tempoValue.textContent = tempoInput.value;
        });
        
        // Generate button
        generateBtn.addEventListener('click', generateMusic);
        
        // Train button
        trainBtn.addEventListener('click', trainModels);
        
        // Example prompts
        document.querySelectorAll('.prompt-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                textInput.value = this.dataset.text;
                updateCharCount();
            });
        });
        
        // Model refresh
        refreshModels.addEventListener('click', checkModels);
        
        // Clear audio
        clearAudio.addEventListener('click', clearAudioFiles);
        
        // Player controls
        playBtn.addEventListener('click', playMusic);
        pauseBtn.addEventListener('click', pauseMusic);
        stopBtn.addEventListener('click', stopMusic);
        
        // Action buttons
        downloadBtn.addEventListener('click', downloadMusic);
        regenerateBtn.addEventListener('click', regenerateMusic);
        shareBtn.addEventListener('click', shareMusic);
        
        // Enter key for textarea
        textInput.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generateMusic();
            }
        });
    }
    
    function updateCharCount() {
        const length = textInput.value.length;
        charCount.textContent = `${length}/500 characters`;
        charCount.style.color = length > 500 ? '#f56565' : '#718096';
    }
    
    function checkModels() {
        modelsStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking models...';
        
        fetch('/check-models')
            .then(response => response.json())
            .then(data => {
                updateModelsStatus(data);
                updateModelsList(data.models);
            })
            .catch(error => {
                showToast('Error checking models', 'error');
                console.error('Error:', error);
            });
    }
    
    function updateModelsStatus(data) {
        if (data.trained) {
            modelsStatus.innerHTML = `<i class="fas fa-check-circle"></i> ${data.count} models trained`;
            modelsStatus.style.color = '#48bb78';
            generateBtn.disabled = false;
        } else {
            modelsStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No models trained';
            modelsStatus.style.color = '#f56565';
            generateBtn.disabled = true;
        }
    }
    
    function updateModelsList(models) {
        modelsList.innerHTML = '';
        
        if (!models || models.length === 0) {
            modelsList.innerHTML = `
                <div class="model-item">
                    <div class="model-name">
                        <i class="fas fa-robot"></i>
                        <span>No models trained</span>
                    </div>
                    <div class="model-status status-not-loaded">
                        Not Loaded
                    </div>
                </div>
            `;
            return;
        }
        
        models.forEach(model => {
            const modelItem = document.createElement('div');
            modelItem.className = 'model-item';
            modelItem.innerHTML = `
                <div class="model-name">
                    <i class="fas fa-brain"></i>
                    <span>${model.name.toUpperCase()} Model</span>
                </div>
                <div class="model-size">${model.size_kb} KB</div>
                <div class="model-status status-loaded">
                    Loaded
                </div>
            `;
            modelsList.appendChild(modelItem);
        });
    }
    
    function generateMusic() {
        const text = textInput.value.trim();
        if (!text) {
            showToast('Please enter text description', 'error');
            return;
        }
        
        if (text.length > 500) {
            showToast('Text too long (max 500 characters)', 'error');
            return;
        }
        
        // Show loading state
        loadingState.style.display = 'block';
        resultState.style.display = 'none';
        emptyState.style.display = 'none';
        generateBtn.disabled = true;
        
        // Prepare data
        const data = {
            text: text,
            model: modelSelect.value,
            length: parseInt(lengthInput.value),
            tempo: parseInt(tempoInput.value),
            instrument: instrumentSelect.value
        };
        
        const startTime = Date.now();
        
        // Send request
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            const generationTime = (Date.now() - startTime) / 1000;
            
            if (result.success) {
                currentMusic = result;
                
                // Update UI
                document.getElementById('musicTitle').textContent = 
                    result.preview_text || 'Generated Music';
                document.getElementById('noteCount').textContent = result.notes;
                document.getElementById('resultTempo').textContent = result.tempo;
                document.getElementById('resultInstrument').textContent = 
                    result.instrument.split('(')[0].trim();
                document.getElementById('modelUsed').textContent = 
                    modelSelect.options[modelSelect.selectedIndex].text;
                document.getElementById('genTime').textContent = 
                    `${generationTime.toFixed(1)}s`;
                
                // Set audio player source
                audioPlayer.src = result.file_url;
                
                // Show result
                loadingState.style.display = 'none';
                resultState.style.display = 'block';
                
                // Add to recent generations
                addRecentGeneration(text, result);
                
                // Check models again
                checkModels();
                
                showToast('Music generated successfully!', 'success');
            } else {
                loadingState.style.display = 'none';
                emptyState.style.display = 'block';
                showToast(result.message || 'Generation failed', 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingState.style.display = 'none';
            emptyState.style.display = 'block';
            showToast('Network error. Please try again.', 'error');
        })
        .finally(() => {
            generateBtn.disabled = false;
        });
    }
    
    function trainModels() {
        if (!confirm('Training models may take several minutes. Continue?')) {
            return;
        }
        
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
        
        fetch('/train', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                showToast(result.message, 'success');
                // Start checking models after a delay
                setTimeout(checkModels, 5000);
            } else {
                showToast(result.message, 'error');
            }
        })
        .catch(error => {
            showToast('Error starting training', 'error');
            console.error('Error:', error);
        })
        .finally(() => {
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="fas fa-robot"></i> Train Models';
        });
    }
    
    function addRecentGeneration(text, result) {
        const generation = {
            id: Date.now(),
            text: text.length > 50 ? text.substring(0, 50) + '...' : text,
            timestamp: new Date().toLocaleTimeString(),
            file_url: result.file_url,
            filename: result.filename
        };
        
        recentGenerations.unshift(generation);
        if (recentGenerations.length > 5) {
            recentGenerations = recentGenerations.slice(0, 5);
        }
        
        updateRecentList();
        saveRecentGenerations();
    }
    
    function updateRecentList() {
        recentList.innerHTML = '';
        
        if (recentGenerations.length === 0) {
            recentList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-music"></i>
                    <p>No music generated yet</p>
                </div>
            `;
            return;
        }
        
        recentGenerations.forEach(gen => {
            const item = document.createElement('div');
            item.className = 'recent-item';
            item.innerHTML = `
                <div class="recent-text" title="${gen.text}">
                    <i class="fas fa-music"></i> ${gen.text}
                </div>
                <div class="recent-actions">
                    <button class="btn-small" onclick="playRecent('${gen.file_url}')">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="btn-small" onclick="downloadRecent('${gen.filename}')">
                        <i class="fas fa-download"></i>
                    </button>
                </div>
            `;
            recentList.appendChild(item);
        });
    }
    
    function loadRecentGenerations() {
        const saved = localStorage.getItem('recentGenerations');
        if (saved) {
            try {
                recentGenerations = JSON.parse(saved);
                updateRecentList();
            } catch (e) {
                console.error('Error loading recent generations:', e);
            }
        }
    }
    
    function saveRecentGenerations() {
        localStorage.setItem('recentGenerations', JSON.stringify(recentGenerations));
    }
    
    function setupVisualization() {
        // Create visualization bars
        for (let i = 0; i < 32; i++) {
            const bar = document.createElement('div');
            bar.className = 'visualizer-bar';
            bar.style.setProperty('--i', i);
            visualizer.appendChild(bar);
        }
    }
    
    function playMusic() {
        if (!currentMusic || !audioPlayer.src) {
            showToast('No music loaded', 'error');
            return;
        }
        
        audioPlayer.play();
        isPlaying = true;
        startVisualization();
        
        playBtn.disabled = true;
        pauseBtn.disabled = false;
        stopBtn.disabled = false;
        
        showToast('Playing music...', 'success');
    }
    
    function pauseMusic() {
        audioPlayer.pause();
        isPlaying = false;
        stopVisualization();
        
        playBtn.disabled = false;
        pauseBtn.disabled = true;
    }
    
    function stopMusic() {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
        isPlaying = false;
        stopVisualization();
        
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        stopBtn.disabled = true;
    }
    
    function startVisualization() {
        visualizationInterval = setInterval(() => {
            const bars = visualizer.querySelectorAll('.visualizer-bar');
            bars.forEach(bar => {
                const randomHeight = 20 + Math.random() * 80;
                bar.style.height = `${randomHeight}%`;
            });
        }, 100);
    }
    
    function stopVisualization() {
        if (visualizationInterval) {
            clearInterval(visualizationInterval);
            visualizationInterval = null;
            
            const bars = visualizer.querySelectorAll('.visualizer-bar');
            bars.forEach(bar => {
                bar.style.height = '20%';
            });
        }
    }
    
    function downloadMusic() {
        if (!currentMusic) {
            showToast('No music to download', 'error');
            return;
        }
        
        window.location.href = `/download/${currentMusic.filename}`;
        showToast('Download started', 'success');
    }
    
    function regenerateMusic() {
        if (!textInput.value.trim()) {
            showToast('Please enter text first', 'error');
            return;
        }
        
        generateMusic();
    }
    
    function shareMusic() {
        if (!currentMusic) {
            showToast('No music to share', 'error');
            return;
        }
        
        if (navigator.share) {
            navigator.share({
                title: 'AI Generated Music',
                text: `Check out this AI-generated music: "${textInput.value}"`,
                url: window.location.origin + currentMusic.file_url
            })
            .then(() => showToast('Shared successfully!', 'success'))
            .catch(error => console.error('Error sharing:', error));
        } else {
            // Fallback: copy to clipboard
            const shareText = `AI Generated Music: "${textInput.value}" - Download: ${window.location.origin}${currentMusic.file_url}`;
            navigator.clipboard.writeText(shareText)
                .then(() => showToast('Link copied to clipboard!', 'success'))
                .catch(() => showToast('Cannot share on this device', 'error'));
        }
    }
    
    function clearAudioFiles() {
        if (!confirm('Clear all generated audio files?')) {
            return;
        }
        
        fetch('/clear-audio', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(result => {
            showToast(result.message, result.success ? 'success' : 'error');
            if (result.success) {
                recentGenerations = [];
                updateRecentList();
                saveRecentGenerations();
            }
        })
        .catch(error => {
            showToast('Error clearing files', 'error');
            console.error('Error:', error);
        });
    }
    
    function showToast(message, type = 'info') {
        // Remove existing toasts
        document.querySelectorAll('.toast').forEach(toast => toast.remove());
        
        // Create new toast
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = type === 'success' ? 'check-circle' : 
                     type === 'error' ? 'exclamation-circle' : 'info-circle';
        
        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Hide after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    // Audio player events
    audioPlayer.addEventListener('play', () => {
        isPlaying = true;
        startVisualization();
    });
    
    audioPlayer.addEventListener('pause', () => {
        isPlaying = false;
        stopVisualization();
    });
    
    audioPlayer.addEventListener('ended', () => {
        isPlaying = false;
        stopVisualization();
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        stopBtn.disabled = true;
    });
    
    // Global functions for recent items
    window.playRecent = function(fileUrl) {
        audioPlayer.src = fileUrl;
        playMusic();
    };
    
    window.downloadRecent = function(filename) {
        window.location.href = `/download/${filename}`;
    };
    
    // Check models every 30 seconds
    setInterval(checkModels, 30000);
    
    // Auto-check models on page load
    checkModels();
});