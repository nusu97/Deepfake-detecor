from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import IMG_SIZE, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE


@dataclass(frozen=True)
class DataLoaders:
    train: Any
    test: Any


def create_generators(train_dir: str | Path, test_dir: str | Path) -> DataLoaders:
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    training_set = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMG_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        class_mode="binary",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_set = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=IMG_SIZE,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        class_mode="binary",
    )

    return DataLoaders(train=training_set, test=test_set)
