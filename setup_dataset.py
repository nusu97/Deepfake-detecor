"""
Dataset preparation helper script
Creates the necessary directory structure for the deepfake detector
"""

import os
from pathlib import Path


def create_dataset_structure():
    """
    Create the directory structure for the dataset
    """
    base_dir = Path('Dataset')
    real_dir = base_dir / 'Real'
    fake_dir = base_dir / 'Fake'
    
    # Create directories
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print("Dataset directory structure created successfully!")
    print(f"\nCreated directories:")
    print(f"  {real_dir}  - Place real/authentic face images here")
    print(f"  {fake_dir}  - Place fake/deepfake face images here")
    print(f"\nSupported formats: .jpg, .jpeg, .png")
    print(f"\nNext steps:")
    print("1. Add real face images to Dataset/Real/")
    print("2. Add fake face images to Dataset/Fake/")
    print("3. Run 'python deepfake_detector.py' to train the model")


def check_dataset():
    """
    Check the dataset and report statistics
    """
    base_dir = Path('Dataset')
    real_dir = base_dir / 'Real'
    fake_dir = base_dir / 'Fake'
    
    if not base_dir.exists():
        print("Dataset directory does not exist.")
        print("Run this script to create it.")
        return
    
    # Count images
    real_count = 0
    fake_count = 0
    
    if real_dir.exists():
        real_count = len([f for f in real_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if fake_dir.exists():
        fake_count = len([f for f in fake_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    print("\nDataset Statistics:")
    print(f"{'='*50}")
    print(f"Real images: {real_count}")
    print(f"Fake images: {fake_count}")
    print(f"Total images: {real_count + fake_count}")
    print(f"{'='*50}")
    
    if real_count == 0 or fake_count == 0:
        print("\nWarning: Dataset is incomplete!")
        print("Add images to both 'Dataset/Real' and 'Dataset/Fake' directories.")
    elif real_count + fake_count < 100:
        print("\nWarning: Dataset is small!")
        print("For better results, use at least 100-1000 images per class.")
    else:
        print("\nDataset looks good! Ready to train.")
        print("Run: python deepfake_detector.py")


def main():
    """
    Main function
    """
    print("Deepfake Detector - Dataset Setup Helper")
    print("=" * 50)
    
    # Create dataset structure
    create_dataset_structure()
    
    # Check dataset
    check_dataset()


if __name__ == '__main__':
    main()
