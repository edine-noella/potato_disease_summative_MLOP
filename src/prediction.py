"""
Prediction Module for Potato Disease Classification
Handles single and batch predictions
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import logging
from PIL import Image
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseasePredictor:
    """Handles predictions for potato disease classification"""
    
    def __init__(
        self,
        models_root: str | None = None,
        model_name: str = "potato_mobilenetv2",
        img_size: tuple = (256, 256),
    ):
        """
        Initialize predictor
        
        Args:
        models_root: Optional path to models directory.
                     If None, resolves to project/models/
        model_name: Name prefix of model folder
        img_size: Expected image input size
        """
        if models_root is None:
            # Directory of this file: project/src
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Move up to project root: project/
            project_root = os.path.abspath(os.path.join(current_dir, ".."))

            # Models directory: project/models
            self.models_root = os.path.join(project_root, "models")
        else:
             self.models_root = models_root

        self.model_name = model_name
        self.img_size = img_size

        self.model_dir = None
        self.model_path = None
        self.class_names_path = None

        self.model = None
        self.class_names = None
        self.prediction_log = []

        self._resolve_latest_model_dir()
        self.load_model()
        self.load_class_names()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_latest_model_dir(self):
        """
        Find the latest model directory under models_root matching
        pattern: {model_name}_YYYYMMDD_HHMMSS
        """
        pattern = os.path.join(self.models_root, f"{self.model_name}_*")
        candidate_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

        if not candidate_dirs:
            raise FileNotFoundError(
                f"No model directory found matching pattern:\n  {pattern}\n"
                "Make sure you have trained and saved a model using the notebook."
            )

        # Sort by modification time (or name, both work since timestamp is in name)
        candidate_dirs.sort(key=os.path.getmtime, reverse=True)
        self.model_dir = candidate_dirs[0]

        self.model_path = os.path.join(self.model_dir, f"{self.model_name}.h5")
        self.class_names_path = os.path.join(self.model_dir, "class_names.json")

        logger.info(f"Using model directory: {self.model_dir}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Class names path: {self.class_names_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at:\n  {self.model_path}\n"
                "Check that your training notebook saved the .h5 file correctly."
            )

        if not os.path.exists(self.class_names_path):
            raise FileNotFoundError(
                f"class_names.json not found at:\n  {self.class_names_path}\n"
                "Check that your training notebook saved class_names.json correctly."
            )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_class_names(self):
        """Load class names from JSON"""
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = json.load(f)
            logger.info(f"Class names loaded: {self.class_names}")
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            raise

    # ------------------------------------------------------------------
    # Preprocessing & prediction
    # ------------------------------------------------------------------
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            preprocessed_array: Numpy array ready for prediction
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_single(self, image_path, return_confidence=True):
        """
        Predict disease for a single image
        
        Args:
            image_path: Path to image file
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        inference_time = time.time() - start_time
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_confidence:
            result['all_confidences'] = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        
        # Log prediction
        self.prediction_log.append(result)
        
        logger.info(
            f"Prediction: {predicted_class} ({confidence:.2%}) "
            f"in {inference_time*1000:.2f}ms"
        )
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict diseases for multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Predicting for {len(image_paths)} images")
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # Preprocess batch
            for path in batch_paths:
                img_array = self.preprocess_image(path)
                batch_images.append(img_array[0])
            
            batch_array = np.array(batch_images)
            
            # Predict
            start_time = time.time()
            predictions = self.model.predict(batch_array, verbose=0)
            inference_time = time.time() - start_time
            
            # Process results
            for j, pred in enumerate(predictions):
                predicted_class_idx = np.argmax(pred)
                predicted_class = self.class_names[predicted_class_idx]
                confidence = float(pred[predicted_class_idx])
                
                result = {
                    'image_path': batch_paths[j],
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'inference_time_ms': round(
                        (inference_time / len(batch_paths)) * 1000, 2
                    )
                }
                
                results.append(result)
        
        logger.info(f"Batch prediction completed for {len(results)} images")
        
        return results
    
    # ------------------------------------------------------------------
    # Interpretation & stats
    # ------------------------------------------------------------------
    def get_interpretation(self, predicted_class, confidence):
        """
        Get human-readable interpretation of prediction
        
        Args:
            predicted_class: Predicted disease class
            confidence: Prediction confidence
            
        Returns:
            Interpretation string
        """
        interpretations = {
            'Potato___Early_blight': {
                'disease': 'Early Blight',
                'description': 'A fungal disease causing dark spots with concentric rings on leaves.',
                'recommendation': 'Remove affected leaves, apply fungicide, and ensure proper spacing for air circulation.'
            },
            'Potato___Late_blight': {
                'disease': 'Late Blight',
                'description': 'A serious fungal disease that can destroy entire crops rapidly.',
                'recommendation': 'Remove and destroy infected plants immediately. Apply copper-based fungicides preventatively.'
            },
            'Potato___healthy': {
                'disease': 'Healthy',
                'description': 'The plant appears healthy with no visible disease symptoms.',
                'recommendation': 'Continue regular monitoring and maintain good agricultural practices.'
            }
        }
        
        interpretation = interpretations.get(predicted_class, {
            'disease': 'Unknown',
            'description': 'Disease not recognized.',
            'recommendation': 'Consult with an agricultural expert.'
        })
        
        confidence_level = (
            'High' if confidence > 0.9
            else 'Medium' if confidence > 0.7
            else 'Low'
        )
        
        interpretation['confidence_level'] = confidence_level
        interpretation['confidence_score'] = confidence
        
        return interpretation
    
    def get_prediction_stats(self):
        """Get statistics from prediction log"""
        if not self.prediction_log:
            return {'message': 'No predictions made yet'}
        
        confidences = [p['confidence'] for p in self.prediction_log]
        inference_times = [p['inference_time_ms'] for p in self.prediction_log]
        
        stats = {
            'total_predictions': len(self.prediction_log),
            'average_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'average_inference_time_ms': float(np.mean(inference_times)),
            'min_inference_time_ms': float(np.min(inference_times)),
            'max_inference_time_ms': float(np.max(inference_times))
        }
        
        return stats
    
    def clear_prediction_log(self):
        """Clear prediction log"""
        self.prediction_log = []
        logger.info("Prediction log cleared")


def explain_prediction(predictor, image_path):
    """
    Generate detailed explanation for a prediction
    
    Args:
        predictor: DiseasePredictor instance
        image_path: Path to image
        
    Returns:
        Dictionary with detailed explanation
    """
    result = predictor.predict_single(image_path)
    interpretation = predictor.get_interpretation(
        result['predicted_class'], 
        result['confidence']
    )
    
    explanation = {
        **result,
        **interpretation
    }
    
    return explanation


if __name__ == "__main__":
    # Example usage
    try:
        predictor = DiseasePredictor()
        
        # Example:
        # result = predictor.predict_single('path/to/test/image.jpg')
        # print(json.dumps(result, indent=2))
        
        stats = predictor.get_prediction_stats()
        print("\nPrediction Statistics:")
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        logger.error(f"Error in prediction demo: {e}")
