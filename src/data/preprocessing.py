
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


SPLITS_DIR = Path("data/splits")
FIGURES_DIR = Path("reports/figures")

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
RANDOM_STATE = 42

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}


def load_dataset_csv(csv_path):

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    paths = df["filepath"].tolist()
    labels = df["class"].map(CLASS_TO_INDEX)

    if labels.isna().any():
        raise ValueError("Unknown class found in dataset.")

    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Image not found: {path}")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels.astype("int32").tolist())
    )

    return dataset


def decode_and_resize(image_path, label):

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    return image, label


def normalize_image(image, label):

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def prepare_pipeline(dataset, training=False):

    if training:
        dataset = dataset.shuffle(
            buffer_size=1000,
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_data_augmentation():

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(
            0.10
        ),
        tf.keras.layers.RandomZoom(
            0.10
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),
        tf.keras.layers.RandomContrast(
            0.10
        )
    ], name="data_augmentation")


def visualize_augmentation(train_dataset, data_augmentation):

    for images, labels in train_dataset.take(1):

        original_image = images[0]

        fig, axes = plt.subplots(
            2, 3,
            figsize=(12, 8)
        )

        axes = axes.flatten()

        axes[0].imshow(original_image.numpy())
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        for i in range(1, 6):

            augmented_image = data_augmentation(
                tf.expand_dims(original_image, axis=0),
                training=True
            )

            augmented_image = tf.clip_by_value(
                augmented_image[0],
                0.0,
                1.0
            )

            axes[i].imshow(augmented_image.numpy())
            axes[i].set_title(f"Augmented Version {i}")
            axes[i].axis("off")

        plt.tight_layout()

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = FIGURES_DIR / "data_augmentation.png"

        plt.savefig(output_path)
        plt.close()

        print(f"\nVisualization saved to: {output_path}")


def verify_pipeline(dataset, name):

    print(f"\nChecking {name} dataset")
    print("-------------------------")

    for images, labels in dataset.take(1):

        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
        print("Image dtype:", images.dtype)
        print("Label dtype:", labels.dtype)

        min_value = float(tf.reduce_min(images))
        max_value = float(tf.reduce_max(images))

        print("Minimum pixel:", min_value)
        print("Maximum pixel:", max_value)

        print("Labels:", labels.numpy()[:10])

        assert images.shape[1:] == (150, 150, 3)
        assert images.shape[0] <= BATCH_SIZE
        assert images.dtype == tf.float32
        assert labels.dtype == tf.int32

        assert min_value >= -1e-6
        assert max_value <= 1.0 + 1e-6

        assert bool(tf.reduce_all(labels >= 0))
        assert bool(tf.reduce_all(labels < len(CLASSES)))

        print("PASS: Pipeline checks completed")


def main():

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    train_raw = load_dataset_csv(
        SPLITS_DIR / "train.csv"
    )

    val_raw = load_dataset_csv(
        SPLITS_DIR / "val.csv"
    )

    test_raw = load_dataset_csv(
        SPLITS_DIR / "test.csv"
    )

    print("\nDataset Sizes")
    print("Train:", train_raw.cardinality().numpy())
    print("Validation:", val_raw.cardinality().numpy())
    print("Test:", test_raw.cardinality().numpy())

    train_dataset = prepare_pipeline(
        train_raw,
        training=True
    )

    val_dataset = prepare_pipeline(
        val_raw,
        training=False
    )

    test_dataset = prepare_pipeline(
        test_raw,
        training=False
    )

    verify_pipeline(train_dataset, "Training")
    verify_pipeline(val_dataset, "Validation")
    verify_pipeline(test_dataset, "Test")

    data_augmentation = create_data_augmentation()

    visualize_augmentation(
        train_dataset,
        data_augmentation
    )

    print("\nStep 2 completed successfully!")


if __name__ == "__main__":
    main()
