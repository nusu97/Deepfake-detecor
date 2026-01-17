"""
Deepfake Detection using CNN
Based on TechVidvan tutorial approach for deepfake detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class DeepfakeDetector:
    """
    CNN-based deepfake detector for classifying real vs fake images
    """
    
    def __init__(self, img_size=128):
        """
        Initialize the deepfake detector
        
        Args:
            img_size (int): Size to resize images to (default: 128x128)
        """
        self.img_size = img_size
        self.model = None
        
    def build_model(self):
        """
        Build CNN model architecture for deepfake detection
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification: 0=Real, 1=Fake
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_data(self, dataset_path):
        """
        Load and preprocess images from dataset
        
        Args:
            dataset_path (str): Path to dataset directory with 'Real' and 'Fake' subdirectories
            
        Returns:
            X, y: Images and labels
        """
        X = []
        y = []
        
        dataset_path = Path(dataset_path)
        
        # Load Real images (label 0)
        real_path = dataset_path / 'Real'
        if real_path.exists():
            print(f"Loading real images from {real_path}...")
            for img_file in real_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        X.append(img)
                        y.append(0)  # Real
        
        # Load Fake images (label 1)
        fake_path = dataset_path / 'Fake'
        if fake_path.exists():
            print(f"Loading fake images from {fake_path}...")
            for img_file in fake_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        X.append(img)
                        y.append(1)  # Fake
        
        X = np.array(X, dtype='float32') / 255.0  # Normalize
        y = np.array(y)
        
        print(f"Loaded {len(X)} images: {np.sum(y==0)} real, {np.sum(y==1)} fake")
        
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, save_path='deepfake_model.h5'):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            save_path (str): Path to save the best model
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        return history
    
    def predict(self, image_path):
        """
        Predict if an image is real or fake
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            prediction (str): 'Real' or 'Fake'
            confidence (float): Confidence score
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = self.model.predict(img, verbose=0)[0][0]
        
        if prediction > 0.5:
            label = 'Fake'
            confidence = prediction
        else:
            label = 'Real'
            confidence = 1 - prediction
        
        return label, confidence
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            loss, accuracy: Test loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def save_model(self, filepath='deepfake_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='deepfake_model.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_history(self, history):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
        """
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
        plt.close()


def main():
    """
    Main function to demonstrate usage
    """
    # Initialize detector
    detector = DeepfakeDetector(img_size=128)
    
    # Build model
    print("Building CNN model...")
    model = detector.build_model()
    model.summary()
    
    # Check if dataset exists
    dataset_path = 'Dataset'
    if not os.path.exists(dataset_path):
        print(f"\nDataset not found at '{dataset_path}'")
        print("\nTo use this deepfake detector:")
        print("1. Create a 'Dataset' directory")
        print("2. Inside 'Dataset', create two subdirectories: 'Real' and 'Fake'")
        print("3. Place real face images in 'Dataset/Real/'")
        print("4. Place fake/deepfake face images in 'Dataset/Fake/'")
        print("5. Run this script again to train the model")
        return
    
    # Load data
    print("\nLoading dataset...")
    X, y = detector.load_data(dataset_path)
    
    if len(X) == 0:
        print("No images found in dataset. Please add images to 'Dataset/Real' and 'Dataset/Fake' directories.")
        return
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Testing: {len(X_test)} images")
    
    # Train model
    print("\nTraining model...")
    history = detector.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    
    # Plot training history
    detector.plot_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    detector.evaluate(X_test, y_test)
    
    print("\nTraining complete! Model saved as 'deepfake_model.h5'")
    print("\nTo use the model for prediction:")
    print("  detector = DeepfakeDetector()")
    print("  detector.load_model('deepfake_model.h5')")
    print("  label, confidence = detector.predict('path/to/image.jpg')")
    print(f"  print(f'Prediction: {label} (Confidence: {confidence:.2%})')")


if __name__ == '__main__':
    main()
