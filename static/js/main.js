// API Base URL
const API_BASE = window.location.origin;

// ============================================================================
// PREDICTION FUNCTIONS
// ============================================================================

async function predictSingle() {
    const fileInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    const previewDiv = document.getElementById('imagePreview');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showError(resultDiv, 'Please select an image');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Show image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewDiv.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
    
    // Show loading
    resultDiv.innerHTML = '<div class="loading"></div><p>Analyzing image...</p>';
    resultDiv.className = 'result-box';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResult(resultDiv, data);
        } else {
            showError(resultDiv, data.error || 'Prediction failed');
        }
    } catch (error) {
        showError(resultDiv, 'Network error: ' + error.message);
    }
}

function displayPredictionResult(container, data) {
    const diseaseClass = data.prediction.replace('Potato___', '').replace('_', ' ');
    const confidence = (data.confidence * 100).toFixed(2);
    const interpretation = data.interpretation;
    
    let html = `
        <div class="result-box success">
            <h4>🔍 Prediction Results</h4>
            <p><strong>Disease:</strong> ${interpretation.disease}</p>
            <p><strong>Confidence:</strong> ${confidence}% (${interpretation.confidence_level})</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%"></div>
            </div>
            <p><strong>Inference Time:</strong> ${data.inference_time_ms}ms</p>
            
            <h5 style="margin-top: 1rem;">Description:</h5>
            <p>${interpretation.description}</p>
            
            <h5>Recommendation:</h5>
            <p>${interpretation.recommendation}</p>
            
            <h5>All Confidences:</h5>
            <ul>
    `;
    
    for (const [cls, conf] of Object.entries(data.all_confidences)) {
        const cleanClass = cls.replace('Potato___', '').replace('_', ' ');
        html += `<li>${cleanClass}: ${(conf * 100).toFixed(2)}%</li>`;
    }
    
    html += `
            </ul>
        </div>
    `;
    
    container.innerHTML = html;
}

async function predictBatch() {
    const fileInput = document.getElementById('batchInput');
    const resultsDiv = document.getElementById('batchResults');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showError(resultsDiv, 'Please select images');
        return;
    }
    
    resultsDiv.innerHTML = '<div class="loading"></div><p>Processing batch...</p>';
    
    try {
        const formData = new FormData();
        for (let file of fileInput.files) {
            formData.append('files', file);
        }
        
        const response = await fetch(`${API_BASE}/api/predict/batch`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(resultsDiv, data.predictions);
        } else {
            showError(resultsDiv, data.error || 'Batch prediction failed');
        }
    } catch (error) {
        showError(resultsDiv, 'Network error: ' + error.message);
    }
}

function displayBatchResults(container, predictions) {
    let html = `<h4>Batch Prediction Results (${predictions.length} images)</h4>`;
    
    predictions.forEach((pred, index) => {
        const diseaseClass = pred.predicted_class.replace('Potato___', '').replace('_', ' ');
        const confidence = (pred.confidence * 100).toFixed(2);
        
        html += `
            <div class="batch-item">
                <span><strong>Image ${index + 1}:</strong> ${diseaseClass}</span>
                <span><strong>Confidence:</strong> ${confidence}%</span>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// ============================================================================
// DASHBOARD FUNCTIONS
// ============================================================================

async function updateDashboard() {
    try {
        // Update uptime
        const uptimeResponse = await fetch(`${API_BASE}/api/monitor/uptime`);
        const uptimeData = await uptimeResponse.json();
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement) {
            uptimeElement.textContent = uptimeData.uptime_formatted;
        }
        
        // Update stats
        const statsResponse = await fetch(`${API_BASE}/api/monitor/stats`);
        const statsData = await statsResponse.json();
        
        // Update prediction stats
        const totalPredElement = document.getElementById('totalPredictions');
        if (totalPredElement) {
            totalPredElement.textContent = statsData.total_predictions || 0;
        }
        
        const cpuElement = document.getElementById('cpuUsage');
        if (cpuElement) {
            cpuElement.textContent = statsData.system_stats.cpu_percent.toFixed(1) + '%';
        }
        
        const memoryElement = document.getElementById('memoryUsage');
        if (memoryElement) {
            memoryElement.textContent = statsData.system_stats.memory_percent.toFixed(1) + '%';
        }
        
        // Update detailed prediction stats
        if (statsData.prediction_stats.total_predictions) {
            const avgConf = document.getElementById('avgConfidence');
            if (avgConf) {
                avgConf.textContent = (statsData.prediction_stats.average_confidence * 100).toFixed(2) + '%';
            }
            
            const avgTime = document.getElementById('avgInferenceTime');
            if (avgTime) {
                avgTime.textContent = statsData.prediction_stats.average_inference_time_ms.toFixed(2) + 'ms';
            }
            
            const minTime = document.getElementById('minInferenceTime');
            if (minTime) {
                minTime.textContent = statsData.prediction_stats.min_inference_time_ms.toFixed(2) + 'ms';
            }
            
            const maxTime = document.getElementById('maxInferenceTime');
            if (maxTime) {
                maxTime.textContent = statsData.prediction_stats.max_inference_time_ms.toFixed(2) + 'ms';
            }
        }
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

async function updateTrainingStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/train/status`);
        const data = await response.json();
        
        const statusElement = document.getElementById('trainingStatus');
        const progressElement = document.getElementById('trainingProgress');
        const messageElement = document.getElementById('trainingMessage');
        const progressFill = document.getElementById('progressFill');
        
        if (statusElement) {
            statusElement.textContent = data.is_training ? 'Training' : 'Idle';
            statusElement.style.color = data.is_training ? '#e74c3c' : '#27ae60';
        }
        
        if (progressElement) {
            progressElement.textContent = data.progress + '%';
        }
        
        if (messageElement) {
            messageElement.textContent = data.message;
        }
        
        if (progressFill) {
            progressFill.style.width = data.progress + '%';
        }
    } catch (error) {
        console.error('Error updating training status:', error);
    }
}

async function uploadTrainingData() {
    const fileInput = document.getElementById('trainingDataInput');
    const classLabel = document.getElementById('classLabel').value;
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select files to upload');
        return;
    }
    
    try {
        const formData = new FormData();
        for (let file of fileInput.files) {
            formData.append('files', file);
        }
        formData.append('class_label', classLabel);
        
        const response = await fetch(`${API_BASE}/api/data/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(`Successfully uploaded ${data.uploaded_files} files for class: ${data.class_label}`);
            fileInput.value = '';
        } else {
            alert('Upload failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    }
}

async function startRetraining() {
    const epochs = document.getElementById('retrainEpochs').value;
    
    if (!confirm(`Start retraining with ${epochs} epochs? This may take several minutes.`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/train/retrain`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data_dir: 'data/upload',
                epochs: parseInt(epochs)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Retraining started! Check the training status above.');
        } else {
            alert('Failed to start retraining: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    }
}

// ============================================================================
// VISUALIZATION FUNCTIONS
// ============================================================================

async function initializeCharts() {
    try {
        const response = await fetch(`${API_BASE}/api/visualizations/data`);
        const data = await response.json();
        
        if (data.success) {
            createClassDistributionChart(data.class_distribution);
            createDiseaseImpactChart();
            createPerformanceChart();
            
            // Update total images
            const totalImagesElement = document.getElementById('totalImages');
            if (totalImagesElement) {
                totalImagesElement.textContent = data.total_images.toLocaleString();
            }
        }
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

function createClassDistributionChart(distribution) {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) return;
    
    const labels = distribution.labels.map(l => l.replace('Potato___', '').replace('_', ' '));
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Number of Images',
                data: distribution.values,
                backgroundColor: [
                    'rgba(231, 76, 60, 0.7)',
                    'rgba(230, 126, 34, 0.7)',
                    'rgba(46, 204, 113, 0.7)'
                ],
                borderColor: [
                    'rgba(231, 76, 60, 1)',
                    'rgba(230, 126, 34, 1)',
                    'rgba(46, 204, 113, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Training Data Distribution by Class'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createDiseaseImpactChart() {
    const ctx = document.getElementById('diseaseImpactChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Severity', 'Spread Rate', 'Crop Loss', 'Detection Difficulty', 'Treatment Cost'],
            datasets: [
                {
                    label: 'Early Blight',
                    data: [7, 6, 7, 5, 6],
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    borderColor: 'rgba(231, 76, 60, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Late Blight',
                    data: [9, 9, 9, 7, 8],
                    backgroundColor: 'rgba(230, 126, 34, 0.2)',
                    borderColor: 'rgba(230, 126, 34, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10
                }
            }
        }
    });
}

function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i + 1),
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: generateLearningCurve(50, 0.5, 0.95),
                    borderColor: 'rgba(52, 152, 219, 1)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Validation Accuracy',
                    data: generateLearningCurve(50, 0.45, 0.92),
                    borderColor: 'rgba(46, 204, 113, 1)',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Training Progress (Example)'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.4,
                    max: 1.0
                }
            }
        }
    });
}

function generateLearningCurve(epochs, start, end) {
    const curve = [];
    for (let i = 0; i < epochs; i++) {
        const progress = i / epochs;
        const value = start + (end - start) * (1 - Math.exp(-5 * progress)) + (Math.random() - 0.5) * 0.02;
        curve.push(Math.min(Math.max(value, start), end));
    }
    return curve;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showError(container, message) {
    container.innerHTML = `
        <div class="result-box error">
            <h4>❌ Error</h4>
            <p>${message}</p>
        </div>
    `;
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Add file input change listeners for preview
    const imageInput = document.getElementById('imageInput');
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            document.getElementById('result').innerHTML = '';
        });
    }
});