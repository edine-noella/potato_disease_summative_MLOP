"""
Model Architecture Module for Potato Disease Classification
Defines CNN and Transfer Learning models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard
)
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PotatoDiseaseModel:
    """Model builder for potato disease classification"""
    
    def __init__(self, img_size=(256, 256), num_classes=3):
        """
        Initialize model builder
        
        Args:
            img_size: Input image dimensions
            num_classes: Number of disease classes
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_cnn_model(self):
        """
        Build a custom CNN model
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building custom CNN model")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        logger.info(f"CNN Model built with {model.count_params():,} parameters")
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='MobileNetV2'):
        """
        Build transfer learning model
        
        Args:
            base_model_name: Name of pretrained model 
                           ('MobileNetV2', 'ResNet50', 'EfficientNetB0')
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building transfer learning model with {base_model_name}")
        
        # Select base model
        if base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        logger.info(f"Transfer Learning Model built with {model.count_params():,} parameters")
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        
        Args:
            learning_rate: Initial learning rate
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_cnn_model() or build_transfer_learning_model() first.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        logger.info("Model compiled successfully")
    
    def get_callbacks(self, model_path='models/potato_disease_model.h5'):
        """
        Get training callbacks
        
        Args:
            model_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'logs/tensorboard/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=50, 
              model_path='models/potato_disease_model.h5'):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            model_path: Path to save model
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built and compiled")
        
        logger.info(f"Starting training for {epochs} epochs")
        
        callbacks = self.get_callbacks(model_path)
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        
        # Save training history
        history_path = model_path.replace('.h5', '_history.json')
        self.save_history(history_path)
        
        return self.history
    
    def save_history(self, filepath):
        """Save training history to JSON"""
        if self.history is not None:
            history_dict = {k: [float(v) for v in values] 
                          for k, values in self.history.history.items()}
            with open(filepath, 'w') as f:
                json.dump(history_dict, f, indent=2)
            logger.info(f"Training history saved to {filepath}")
    
    def load_model(self, model_path='models/potato_disease_model.h5'):
        """Load a saved model"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def get_model_summary(self):
        """Get model summary as string"""
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def fine_tune(self, train_generator, val_generator, 
                  base_layers_to_unfreeze=20, epochs=20,
                  learning_rate=1e-5):
        """
        Fine-tune a transfer learning model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            base_layers_to_unfreeze: Number of base model layers to unfreeze
            epochs: Number of fine-tuning epochs
            learning_rate: Fine-tuning learning rate
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        logger.info(f"Fine-tuning model: unfreezing last {base_layers_to_unfreeze} layers")
        
        # Unfreeze the base model layers
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the last ones
        for layer in base_model.layers[:-base_layers_to_unfreeze]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=learning_rate)
        
        # Continue training
        history = self.train(train_generator, val_generator, epochs=epochs)
        
        return history


if __name__ == "__main__":
    # Example usage
    model_builder = PotatoDiseaseModel(img_size=(256, 256), num_classes=3)
    
    # Build and compile CNN
    model_builder.build_cnn_model()
    model_builder.compile_model()
    
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(model_builder.get_model_summary())