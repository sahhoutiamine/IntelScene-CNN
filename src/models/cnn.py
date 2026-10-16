
import tensorflow as tf

from src.data.preprocessing import (
    IMAGE_SIZE,
    CLASSES,
    create_data_augmentation
)


def build_cnn():

    model = tf.keras.Sequential([

        tf.keras.Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        ),

        create_data_augmentation(),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            len(CLASSES),
            activation="softmax"
        )

    ], name="intel_scene_cnn")

    return model


if __name__ == "__main__":

    tf.keras.utils.set_random_seed(42)

    model = build_cnn()

    model.summary()
