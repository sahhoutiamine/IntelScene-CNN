"""Step 3 - CNN architecture (configurable for the Step 6 experiments)."""
from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src.data.augmentation import build_augmentation


def build_cnn(input_shape=config.INPUT_SHAPE,
              num_classes: int = config.NUM_CLASSES,
              filters=(32, 64, 128),
              convs_per_block: int = 1,
              batch_norm: bool = False,
              pooling: str = "gap",            # "gap" (GlobalAveragePooling2D) or "flatten"
              dense_units: int = 128,
              dropout: float = 0.3,
              use_augmentation: bool = True,
              aug_params: dict | None = None,
              name: str = "intel_cnn") -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs

    # Augmentation: active ONLY during training (no-op in evaluate / predict)
    if use_augmentation:
        x = build_augmentation(**(aug_params or {}))(x)

    # Normalisation inside the model -> identical preprocessing at prediction time
    x = layers.Rescaling(1.0 / 255, name="rescaling")(x)

    # Feature extractor: [Conv -> (BN) -> ReLU] x N -> MaxPool, repeated per block
    for i, f in enumerate(filters, start=1):
        for _ in range(convs_per_block):
            x = layers.Conv2D(f, 3, padding="same", use_bias=not batch_norm)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(name=f"pool_{i}")(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x) if pooling == "gap" else layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    return keras.Model(inputs, outputs, name=name)
