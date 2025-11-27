"""
Training Pipeline for Potato Disease Classification
Handles initial training and retraining workflows
"""

import os
import json
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import DataPreprocessor
from model import PotatoDiseaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, data_dir='data/train', model_type='cnn'):
        """
        Initialize training pipeline
        
        Args:
            data_dir: Path to training data
            model_type: Type of model ('cnn' or 'transfer')
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.preprocessor = DataPreprocessor(img_size=(256, 256), batch_size=32)
        self.model_builder = None
        self.train_generator = None
        self.val_generator = None
        self.class_names = None
        
    def prepare_data(self, validation_split=0.2):
        """Prepare training and validation data"""
        logger.info("="*80)
        logger.info("PREPARING DATA")
        logger.info("="*80)
        
        self.train_generator, self.val_generator, self.class_names = \
            self.preprocessor.load_data_from_directory(
                self.data_dir,
                validation_split=validation_split
            )
        
        # Save class names
        self.preprocessor.save_class_names()
        
        logger.info(f"Data preparation complete")
        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.val_generator.samples}")
        logger.info(f"Classes: {self.class_names}")
        
        return self.train_generator, self.val_generator
    
    def build_model(self, base_model_name='MobileNetV2'):
        """Build and compile model"""
        logger.info("="*80)
        logger.info("BUILDING MODEL")
        logger.info("="*80)
        
        self.model_builder = PotatoDiseaseModel(
            img_size=(256, 256),
            num_classes=len(self.class_names)
        )
        
        if self.model_type == 'cnn':
            self.model_builder.build_cnn_model()
        elif self.model_type == 'transfer':
            self.model_builder.build_transfer_learning_model(base_model_name)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model_builder.compile_model(learning_rate=0.001)
        
        logger.info("Model built and compiled successfully")
        logger.info(f"\n{self.model_builder.get_model_summary()}")
        
        return self.model_builder.model
    
    def train_model(self, epochs=50, model_save_path='models/potato_disease_model.h5'):
        """Train the model"""
        logger.info("="*80)
        logger.info("TRAINING MODEL")
        logger.info("="*80)
        
        if self.train_generator is None or self.val_generator is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if self.model_builder is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        history = self.model_builder.train(
            self.train_generator,
            self.val_generator,
            epochs=epochs,
            model_path=model_save_path
        )
        
        logger.info("Training completed successfully")
        
        # Save training metadata
        self.save_training_metadata(model_save_path, history)
        
        return history
    
    def save_training_metadata(self, model_path, history):
        """Save training metadata"""
        metadata = {
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'training_samples': self.train_generator.samples,
            'validation_samples': self.val_generator.samples,
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def evaluate_model(self, test_dir='data/test'):
        """Evaluate model on test data"""
        logger.info("="*80)
        logger.info("EVALUATING MODEL")
        logger.info("="*80)
        
        test_generator = self.preprocessor.load_test_data(test_dir)
        
        # Evaluate
        results = self.model_builder.model.evaluate(test_generator, verbose=1)
        
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]:.4f}")
        logger.info(f"Test Precision: {results[2]:.4f}")
        logger.info(f"Test Recall: {results[3]:.4f}")
        
        # Get predictions for confusion matrix
        predictions = self.model_builder.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, self.class_names)
        
        # Save evaluation results
        eval_results = {
            'test_loss': float(results[0]),
            'test_accuracy': float(results[1]),
            'test_precision': float(results[2]),
            'test_recall': float(results[3]),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open('logs/evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info("Evaluation results saved to logs/evaluation_results.json")
        
        return eval_results
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('logs/confusion_matrix.png')
        logger.info("Confusion matrix saved to logs/confusion_matrix.png")
        plt.close()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/training_history.png')
        logger.info("Training history plot saved to logs/training_history.png")
        plt.close()
    
    def run_full_pipeline(self, epochs=50):
        """Run complete training pipeline"""
        logger.info("="*80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Build model
        self.build_model()
        
        # Step 3: Train model
        history = self.train_model(epochs=epochs)
        
        # Step 4: Plot training history
        self.plot_training_history(history)
        
        # Step 5: Evaluate model
        eval_results = self.evaluate_model()
        
        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return history, eval_results


def retrain_model(new_data_dir, existing_model_path='models/potato_disease_model.h5',
                  epochs=20):
    """
    Retrain existing model with new data
    
    Args:
        new_data_dir: Path to new training data
        existing_model_path: Path to existing model
        epochs: Number of retraining epochs
    """
    logger.info("="*80)
    logger.info("STARTING MODEL RETRAINING")
    logger.info("="*80)
    
    # Load existing model
    from tensorflow import keras
    model = keras.models.load_model(existing_model_path)
    logger.info(f"Loaded existing model from {existing_model_path}")
    
    # Prepare new data
    preprocessor = DataPreprocessor()
    train_gen, val_gen, class_names = preprocessor.load_data_from_directory(new_data_dir)
    
    # Create model builder with loaded model
    model_builder = PotatoDiseaseModel(img_size=(256, 256), num_classes=len(class_names))
    model_builder.model = model
    model_builder.compile_model(learning_rate=0.0001)  # Lower learning rate for retraining
    
    # Retrain
    history = model_builder.train(
        train_gen, val_gen,
        epochs=epochs,
        model_path=existing_model_path.replace('.h5', '_retrained.h5')
    )
    
    logger.info("Retraining completed")
    
    return history


if __name__ == "__main__":
    # Run full training pipeline
    pipeline = TrainingPipeline(data_dir='data/train', model_type='cnn')
    history, eval_results = pipeline.run_full_pipeline(epochs=50)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {eval_results['test_accuracy']:.4f}")