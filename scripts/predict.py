from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepfake_detector.config import IMG_SIZE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict real vs deepfake for an image.")
    parser.add_argument("--model", default="models/cnn_deepfake.keras")
    parser.add_argument("image", help="Path to an image file")
    return parser.parse_args()


def load_and_preprocess_image(path: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(str(path), target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model)

    image_path = Path(args.image)
    x = load_and_preprocess_image(image_path)

    prob = float(model.predict(x, verbose=0)[0][0])

    # By convention: 0 -> class0, 1 -> class1; tutorial uses binary sigmoid.
    label = "deepfake" if prob >= 0.5 else "real"
    print(f"prediction={label} prob_deepfake={prob:.4f} file={image_path}")


if __name__ == "__main__":
    main()
