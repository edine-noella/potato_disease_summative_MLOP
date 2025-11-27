"""
Flask API for Potato Disease Classification
Provides REST endpoints for predictions, training, and monitoring
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from datetime import datetime
import threading
import time
import psutil

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /opt/render/project/src/src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # /opt/render/project/src
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from prediction import DiseasePredictor, explain_prediction
    from train import TrainingPipeline, retrain_model
    from preprocessing import analyze_dataset
except ImportError:
    # Try relative imports
    from .prediction import DiseasePredictor, explain_prediction
    from .train import TrainingPipeline, retrain_model
    from .preprocessing import analyze_dataset

# Initialize Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'data/upload'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "training.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Global variables
predictor = None
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Idle',
    'start_time': None
}
model_uptime_start = datetime.now()
prediction_count = 0

# Initialize predictor
try:
    # Check if model exists, create placeholder if not
    if not os.path.exists('models/potato_disease_model.h5'):
        logger.warning("Model not found. Creating placeholder model...")
        import download_model
        download_model.create_simple_model()
    
    predictor = DiseasePredictor()
    logger.info("Predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ============================================================================
# WEB PAGES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')


@app.route('/visualizations')
def visualizations():
    """Visualizations page"""
    return render_template('visualizations.html')


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    global prediction_count
    
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predictor.predict_single(filepath, return_confidence=True)
        interpretation = predictor.get_interpretation(
            result['predicted_class'],
            result['confidence']
        )
        
        prediction_count += 1
        
        response = {
            'success': True,
            'prediction': result['predicted_class'],
            'confidence': result['confidence'],
            'all_confidences': result['all_confidences'],
            'inference_time_ms': result['inference_time_ms'],
            'interpretation': interpretation,
            'filename': filename
        }
        
        logger.info(f"Prediction made: {result['predicted_class']} ({result['confidence']:.2%})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict diseases for multiple images"""
    global prediction_count
    
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        saved_paths = []
        
        # Save all files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_paths.append(filepath)
        
        # Make batch predictions
        results = predictor.predict_batch(saved_paths)
        prediction_count += len(results)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'total_images': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@app.route('/api/train/status', methods=['GET'])
def training_status_endpoint():
    """Get current training status"""
    return jsonify(training_status), 200


def run_training_thread(data_dir, epochs):
    """Background thread for training"""
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Preparing data...'
        training_status['start_time'] = datetime.now().isoformat()
        
        pipeline = TrainingPipeline(data_dir=data_dir, model_type='cnn')
        
        training_status['message'] = 'Loading data...'
        pipeline.prepare_data()
        training_status['progress'] = 20
        
        training_status['message'] = 'Building model...'
        pipeline.build_model()
        training_status['progress'] = 30
        
        training_status['message'] = f'Training for {epochs} epochs...'
        history = pipeline.train_model(epochs=epochs)
        training_status['progress'] = 80
        
        training_status['message'] = 'Evaluating model...'
        eval_results = pipeline.evaluate_model()
        training_status['progress'] = 100
        
        training_status['is_training'] = False
        training_status['message'] = f'Training completed! Final accuracy: {eval_results["test_accuracy"]:.4f}'
        
        # Reload predictor with new model
        global predictor
        predictor = DiseasePredictor()
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        training_status['is_training'] = False
        training_status['message'] = f'Training failed: {str(e)}'
        logger.error(f"Training error: {e}")


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start model training"""
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.get_json()
    data_dir = data.get('data_dir', 'data/train')
    epochs = data.get('epochs', 50)
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_thread, args=(data_dir, epochs))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started'
    }), 200


@app.route('/api/train/retrain', methods=['POST'])
def start_retraining():
    """Start model retraining with new data"""
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.get_json()
    data_dir = data.get('data_dir', 'data/upload')
    epochs = data.get('epochs', 20)
    
    def retrain_thread():
        global training_status, predictor
        try:
            training_status['is_training'] = True
            training_status['message'] = 'Retraining model...'
            
            history = retrain_model(data_dir, epochs=epochs)
            
            training_status['is_training'] = False
            training_status['message'] = 'Retraining completed!'
            
            # Reload predictor
            predictor = DiseasePredictor()
            
        except Exception as e:
            training_status['is_training'] = False
            training_status['message'] = f'Retraining failed: {str(e)}'
    
    thread = threading.Thread(target=retrain_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Retraining started'
    }), 200


# ============================================================================
# DATA MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/data/upload', methods=['POST'])
def upload_training_data():
    """Upload new training data"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    class_label = request.form.get('class_label')
    
    if not class_label:
        return jsonify({'error': 'No class label provided'}), 400
    
    # Create class directory
    class_dir = os.path.join('data/upload', class_label)
    os.makedirs(class_dir, exist_ok=True)
    
    uploaded_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(class_dir, filename)
            file.save(filepath)
            uploaded_count += 1
    
    return jsonify({
        'success': True,
        'uploaded_files': uploaded_count,
        'class_label': class_label
    }), 200


@app.route('/api/data/stats', methods=['GET'])
def get_data_stats():
    """Get dataset statistics"""
    try:
        train_stats = analyze_dataset('data/train')
        
        return jsonify({
            'success': True,
            'statistics': train_stats
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.route('/api/monitor/uptime', methods=['GET'])
def get_uptime():
    """Get model uptime"""
    uptime = datetime.now() - model_uptime_start
    
    return jsonify({
        'uptime_seconds': uptime.total_seconds(),
        'uptime_formatted': str(uptime).split('.')[0],
        'start_time': model_uptime_start.isoformat()
    }), 200


@app.route('/api/monitor/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics"""
    if predictor:
        pred_stats = predictor.get_prediction_stats()
    else:
        pred_stats = {}
    
    # System stats
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return jsonify({
        'prediction_stats': pred_stats,
        'total_predictions': prediction_count,
        'system_stats': {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024)
        }
    }), 200


@app.route('/api/monitor/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


# ============================================================================
# VISUALIZATION ENDPOINTS
# ============================================================================

@app.route('/api/visualizations/data', methods=['GET'])
def get_visualization_data():
    """Get data for visualizations"""
    try:
        stats = analyze_dataset('data/train')
        
        # Prepare data for charts
        class_distribution = {
            'labels': list(stats['classes'].keys()),
            'values': list(stats['classes'].values())
        }
        
        return jsonify({
            'success': True,
            'class_distribution': class_distribution,
            'total_images': stats['total_images']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Potato Disease Classification API")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    # Create necessary directories
    os.makedirs('data/upload', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=False)