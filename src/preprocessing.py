"""
Data Preprocessing Module for Potato Disease Classification
Handles data loading, augmentation, and preprocessing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, img_size=(256, 256), batch_size=32):
        """
        Initialize preprocessor
        
        Args:
            img_size: Target image dimensions (height, width)
            batch_size: Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None
        self.num_classes = None
        
    def load_data_from_directory(self, data_dir, validation_split=0.2):
        """
        Load data from directory structure
        
        Args:
            data_dir: Path to data directory with class subdirectories
            validation_split: Fraction of data to use for validation
            
        Returns:
            train_generator, val_generator, class_names
        """
        logger.info(f"Loading data from {data_dir}")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        logger.info(f"Found {train_generator.samples} training images")
        logger.info(f"Found {val_generator.samples} validation images")
        logger.info(f"Classes: {self.class_names}")
        
        return train_generator, val_generator, self.class_names
    
    def load_test_data(self, test_dir):
        """
        Load test data
        
        Args:
            test_dir: Path to test data directory
            
        Returns:
            test_generator
        """
        logger.info(f"Loading test data from {test_dir}")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        logger.info(f"Found {test_generator.samples} test images")
        
        return test_generator
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            preprocessed_image: Numpy array ready for prediction
            original_image: PIL Image object
        """
        # Load image
        img = Image.open(image_path)
        original_img = img.copy()
        
        # Resize
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_img
    
    def save_class_names(self, filepath='models/class_names.json'):
        """Save class names to file"""
        if self.class_names is not None:
            with open(filepath, 'w') as f:
                json.dump(self.class_names, f)
            logger.info(f"Class names saved to {filepath}")
    
    def load_class_names(self, filepath='models/class_names.json'):
        """Load class names from file"""
        with open(filepath, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        logger.info(f"Loaded class names: {self.class_names}")
        return self.class_names
    
    def visualize_augmentation(self, data_dir, num_samples=9):
        """
        Visualize data augmentation effects
        
        Args:
            data_dir: Path to data directory
            num_samples: Number of augmented samples to display
        """
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Get first image from first class
        first_class = os.listdir(data_dir)[0]
        first_image_path = os.path.join(data_dir, first_class, 
                                       os.listdir(os.path.join(data_dir, first_class))[0])
        
        img = Image.open(first_image_path)
        img_array = np.array(img)
        img_array = img_array.reshape((1,) + img_array.shape)
        
        # Generate augmented images
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        i = 0
        for batch in datagen.flow(img_array, batch_size=1):
            axes[i].imshow(batch[0].astype('uint8'))
            axes[i].axis('off')
            axes[i].set_title(f'Augmented {i+1}')
            i += 1
            if i >= num_samples:
                break
        
        plt.tight_layout()
        plt.savefig('logs/augmentation_examples.png')
        logger.info("Augmentation visualization saved to logs/augmentation_examples.png")
        
        return fig


def analyze_dataset(data_dir):
    """
    Analyze dataset statistics
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        dict with dataset statistics
    """
    stats = {
        'classes': {},
        'total_images': 0
    }
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            stats['classes'][class_name] = num_images
            stats['total_images'] += num_images
    
    logger.info(f"Dataset Analysis: {stats}")
    return stats


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Analyze dataset
    stats = analyze_dataset('data/train')
    print(f"\nDataset Statistics: {json.dumps(stats, indent=2)}")
    
    # Load data
    train_gen, val_gen, classes = preprocessor.load_data_from_directory('data/train')
    print(f"\nClasses: {classes}")
    
    # Save class names
    preprocessor.save_class_names()