# Deepfake Detector using CNN

A deep learning-based deepfake detection system using Convolutional Neural Networks (CNN) to classify images as real or fake. This project implements a binary classifier that can identify deepfake/manipulated faces from authentic ones.

## Overview

This deepfake detector uses a CNN architecture with multiple convolutional layers, batch normalization, and dropout for robust feature extraction and classification. The model is trained to distinguish between real and fake (deepfake) facial images.

## Features

- **CNN-based Architecture**: Multi-layer convolutional neural network optimized for image classification
- **Data Augmentation**: Automatic data augmentation to improve model generalization
- **Binary Classification**: Classifies images as Real (0) or Fake (1)
- **Easy Training**: Simple interface for training on custom datasets
- **Prediction Script**: Command-line tool for testing individual images
- **Model Checkpointing**: Automatic saving of best model during training
- **Training Visualization**: Plots of accuracy and loss during training

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nusu97/Deepfake-detecor.git
cd Deepfake-detecor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Create a `Dataset` directory in the project root:
```bash
mkdir Dataset
mkdir Dataset/Real
mkdir Dataset/Fake
```

2. Organize your data:
   - Place **real/authentic** face images in `Dataset/Real/`
   - Place **fake/deepfake** face images in `Dataset/Fake/`

3. Supported image formats: `.jpg`, `.jpeg`, `.png`

### Dataset Sources

You can use publicly available datasets such as:
- **FaceForensics++**: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- **Celeb-DF**: [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- **DFDC (Deepfake Detection Challenge)**: [https://ai.facebook.com/datasets/dfdc/](https://ai.facebook.com/datasets/dfdc/)

## Usage

### Training the Model

To train the deepfake detection model:

```bash
python deepfake_detector.py
```

This will:
1. Load images from the `Dataset/Real` and `Dataset/Fake` directories
2. Split data into training (70%), validation (15%), and test (15%) sets
3. Train the CNN model with data augmentation
4. Save the best model as `deepfake_model.h5`
5. Generate a training history plot as `training_history.png`
6. Evaluate the model on the test set

### Model Architecture

The CNN model consists of:
- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization after each convolutional layer
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- 2 Dense layers (512, 256 units) before the output
- Sigmoid activation for binary classification

### Predicting Single Images

After training, you can test individual images:

```bash
python predict.py path/to/image.jpg
```

Or specify a custom model path:

```bash
python predict.py path/to/image.jpg --model custom_model.h5
```

### Using the Detector in Your Code

```python
from deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(img_size=128)

# Load trained model
detector.load_model('deepfake_model.h5')

# Predict an image
label, confidence = detector.predict('path/to/test_image.jpg')
print(f'Prediction: {label} (Confidence: {confidence:.2%})')
```

## Training Parameters

You can customize training parameters by modifying the `main()` function in `deepfake_detector.py`:

- `img_size`: Image size for training (default: 128x128)
- `epochs`: Number of training epochs (default: 30)
- `batch_size`: Batch size for training (default: 32)
- `test_size`: Proportion of data for validation and testing

## Model Performance

The model's performance depends on:
- Quality and quantity of training data
- Diversity of deepfake techniques in the training set
- Image resolution and preprocessing

Expected metrics with a good dataset:
- Training Accuracy: 85-95%
- Validation Accuracy: 80-90%
- Test Accuracy: 80-90%

## Project Structure

```
Deepfake-detecor/
├── deepfake_detector.py    # Main detector class and training script
├── predict.py               # Prediction script for single images
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore file
├── Dataset/                # Dataset directory (not tracked)
│   ├── Real/              # Real face images
│   └── Fake/              # Fake face images
├── deepfake_model.h5       # Trained model (generated after training)
└── training_history.png    # Training plots (generated after training)
```

## How It Works

1. **Data Loading**: Images are loaded from the Real and Fake directories, resized to 128x128 pixels, and normalized to [0, 1] range.

2. **Data Augmentation**: Training images are augmented with random rotations, shifts, flips, zoom, and shear to improve model generalization.

3. **CNN Training**: The model learns to extract features from images through convolutional layers and classify them as real or fake.

4. **Prediction**: For new images, the model outputs a probability score where values close to 0 indicate real images and values close to 1 indicate fake images.

## Limitations

- The model's accuracy depends heavily on the quality and diversity of the training dataset
- May not generalize well to deepfake techniques not present in the training data
- Requires a balanced dataset of real and fake images for optimal performance
- Performance may vary with different image qualities and resolutions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Based on CNN architecture principles for image classification
- Inspired by various deepfake detection research and tutorials
- TechVidvan tutorial approach for deepfake detection

## References

- [Deep Learning for Deepfake Detection](https://arxiv.org/abs/2004.11138)
- [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
- [The Eyes Tell All: Detecting Political Orientation from Eye Movement Data](https://arxiv.org/abs/1901.05533)

## Disclaimer

This tool is for educational and research purposes only. Always verify important information through multiple sources and use deepfake detection responsibly.