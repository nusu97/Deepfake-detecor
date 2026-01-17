"""
Prediction script for deepfake detection
Use this to test individual images with a trained model
"""

import sys
import argparse
from deepfake_detector import DeepfakeDetector


def predict_image(image_path, model_path='deepfake_model.h5'):
    """
    Predict if an image is real or fake
    
    Args:
        image_path (str): Path to the image to test
        model_path (str): Path to the trained model
    """
    # Initialize detector and load model
    detector = DeepfakeDetector(img_size=128)
    
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model file exists at '{model_path}'")
        print("Train the model first by running: python deepfake_detector.py")
        return
    
    # Make prediction
    try:
        label, confidence = detector.predict(image_path)
        print(f"\n{'='*50}")
        print(f"Image: {image_path}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"Error making prediction: {e}")


def main():
    parser = argparse.ArgumentParser(description='Predict if an image is real or deepfake')
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='deepfake_model.h5', 
                       help='Path to the trained model (default: deepfake_model.h5)')
    
    args = parser.parse_args()
    
    predict_image(args.image, args.model)


if __name__ == '__main__':
    main()
