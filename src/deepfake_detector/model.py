from __future__ import annotations

import tensorflow as tf

from .config import CHANNELS, CONV_FILTERS, DENSE_UNITS, DROPOUT, IMG_SIZE, KERNEL_SIZE


def build_cnn() -> tf.keras.Model:
    """CNN mirroring the TechVidvan tutorial architecture."""

    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            filters=CONV_FILTERS,
            kernel_size=KERNEL_SIZE,
            activation="relu",
            input_shape=[IMG_SIZE[0], IMG_SIZE[1], CHANNELS],
        )
    )
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(
        tf.keras.layers.Conv2D(
            filters=CONV_FILTERS,
            kernel_size=KERNEL_SIZE,
            activation="relu",
        )
    )
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=DENSE_UNITS, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    return model
