"""Step 2 - tf.data pipelines. Normalisation + augmentation live INSIDE the model,
so the exact same preprocessing is applied at prediction time and augmentation
is only active during training (Keras random layers are no-ops at inference)."""
import tensorflow as tf
from tensorflow import keras

from src import config


def load_split(split: str, batch_size: int = config.BATCH_SIZE, img_size=config.IMG_SIZE,
               shuffle: bool = False, data_dir=config.DATA_PROCESSED):
    """Raw dataset for one split (keeps the `.file_paths` attribute, useful for error analysis)."""
    return keras.utils.image_dataset_from_directory(
        str(data_dir / split),
        labels="inferred",
        label_mode="int",
        class_names=config.CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=config.SEED,
    )


def load_datasets(batch_size: int = config.BATCH_SIZE, img_size=config.IMG_SIZE):
    """Return (train_ds, val_ds, test_ds) ready for model.fit / evaluate."""
    auto = tf.data.AUTOTUNE
    train = load_split("train", batch_size, img_size, shuffle=True).prefetch(auto)
    val = load_split("val", batch_size, img_size).cache().prefetch(auto)
    test = load_split("test", batch_size, img_size).cache().prefetch(auto)
    return train, val, test
