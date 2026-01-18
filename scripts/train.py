from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepfake_detector.config import DEFAULT_EPOCHS  # noqa: E402
from deepfake_detector.data import create_generators  # noqa: E402
from deepfake_detector.model import build_cnn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN deepfake detector (tutorial-style).")
    parser.add_argument("--train-dir", default="train", help="Folder with class subfolders.")
    parser.add_argument("--test-dir", default="test", help="Folder with class subfolders.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--out", default="models/cnn_deepfake.keras", help="Output model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loaders = create_generators(args.train_dir, args.test_dir)

    model = build_cnn()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out,
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ]

    model.fit(
        x=loaders.train,
        validation_data=loaders.test,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Ensure a final save even if checkpoint didn't trigger
    model.save(args.out)


if __name__ == "__main__":
    main()
