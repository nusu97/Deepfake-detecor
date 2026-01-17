"""
Example usage script for the deepfake detector
Demonstrates various ways to use the DeepfakeDetector class
"""

from deepfake_detector import DeepfakeDetector
import os


def example_training():
    """
    Example: Train a model from scratch
    """
    print("\n" + "="*50)
    print("Example 1: Training a Model")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists('Dataset'):
        print("Dataset directory not found. Please run setup_dataset.py first.")
        return
    
    # Initialize detector
    detector = DeepfakeDetector(img_size=128)
    
    # Build model
    print("\nBuilding model...")
    detector.build_model()
    
    # Load data
    print("\nLoading dataset...")
    X, y = detector.load_data('Dataset')
    
    if len(X) == 0:
        print("No images found. Please add images to Dataset/Real and Dataset/Fake")
        return
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train
    print("\nTraining model (this may take a while)...")
    history = detector.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    # Evaluate
    print("\nEvaluating on test set...")
    detector.evaluate(X_test, y_test)
    
    print("\nModel trained successfully!")


def example_prediction():
    """
    Example: Load a trained model and make predictions
    """
    print("\n" + "="*50)
    print("Example 2: Making Predictions")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists('deepfake_model.h5'):
        print("Trained model not found. Please train the model first.")
        print("Run: python deepfake_detector.py")
        return
    
    # Initialize and load model
    detector = DeepfakeDetector(img_size=128)
    detector.load_model('deepfake_model.h5')
    
    # Example predictions (replace with actual image paths)
    test_images = [
        'Dataset/Real/example1.jpg',
        'Dataset/Fake/example2.jpg',
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            label, confidence = detector.predict(img_path)
            print(f"\nImage: {img_path}")
            print(f"Prediction: {label} (Confidence: {confidence:.2%})")
        else:
            print(f"\nImage not found: {img_path}")


def example_batch_prediction():
    """
    Example: Predict multiple images in batch
    """
    print("\n" + "="*50)
    print("Example 3: Batch Prediction")
    print("="*50)
    
    if not os.path.exists('deepfake_model.h5'):
        print("Trained model not found. Please train the model first.")
        return
    
    # Initialize and load model
    detector = DeepfakeDetector(img_size=128)
    detector.load_model('deepfake_model.h5')
    
    # Get all images from a directory
    import glob
    test_dir = 'Dataset/Real'  # Change this to test other directories
    
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        return
    
    image_files = glob.glob(f"{test_dir}/*.jpg") + glob.glob(f"{test_dir}/*.png")
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} images from {test_dir}...\n")
    
    results = {'Real': 0, 'Fake': 0}
    
    for img_path in image_files[:10]:  # Process first 10 images
        try:
            label, confidence = detector.predict(img_path)
            results[label] += 1
            print(f"{os.path.basename(img_path)}: {label} ({confidence:.2%})")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nResults Summary:")
    print(f"Real: {results['Real']}")
    print(f"Fake: {results['Fake']}")


def example_model_info():
    """
    Example: Display model information
    """
    print("\n" + "="*50)
    print("Example 4: Model Information")
    print("="*50)
    
    # Initialize detector and build model
    detector = DeepfakeDetector(img_size=128)
    detector.build_model()
    
    # Display model summary
    print("\nModel Architecture:")
    detector.model.summary()
    
    # Count parameters
    total_params = detector.model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


def main():
    """
    Main function to run examples
    """
    print("\nDeepfake Detector - Example Usage")
    print("="*50)
    print("\nThis script demonstrates various ways to use the deepfake detector.")
    print("\nAvailable examples:")
    print("1. Training a model")
    print("2. Making predictions")
    print("3. Batch prediction")
    print("4. Model information")
    
    # Run model info example (doesn't require dataset)
    example_model_info()
    
    # Uncomment to run other examples:
    # example_training()
    # example_prediction()
    # example_batch_prediction()
    
    print("\n" + "="*50)
    print("To run other examples, uncomment them in the main() function.")
    print("="*50)


if __name__ == '__main__':
    main()
