"""Step 2 - Data Augmentation (rotation, flip, zoom, translation, contrast)."""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_augmentation(flip: bool = True, rotation: float = 0.10, zoom: float = 0.10,
                       translation: float = 0.10, contrast: float = 0.10,
                       name: str = "augmentation") -> keras.Sequential:
    """Factors are fractions (rotation 0.1 = +/- 10% of 2*pi = +/- 36 degrees)."""
    aug = []
    if flip:
        aug.append(layers.RandomFlip("horizontal"))
    if rotation:
        aug.append(layers.RandomRotation(rotation))
    if zoom:
        aug.append(layers.RandomZoom(zoom))
    if translation:
        aug.append(layers.RandomTranslation(translation, translation))
    if contrast:
        aug.append(layers.RandomContrast(contrast))
    return keras.Sequential(aug, name=name)


def visualize_augmentation(dataset, augmentation: keras.Sequential, n_images: int = 3, n_aug: int = 4,
                           save_path=None):
    """Show each original image followed by `n_aug` augmented versions."""
    images, labels = next(iter(dataset))
    fig, axes = plt.subplots(n_images, n_aug + 1, figsize=(3 * (n_aug + 1), 3 * n_images))
    for i in range(n_images):
        axes[i, 0].imshow(images[i].numpy().astype("uint8"))
        axes[i, 0].set_title(f"original ({config.CLASS_NAMES[int(labels[i])]})", fontsize=9)
        for j in range(n_aug):
            out = augmentation(tf.expand_dims(images[i], 0), training=True)[0]
            axes[i, j + 1].imshow(tf.clip_by_value(out, 0, 255).numpy().astype("uint8"))
            axes[i, j + 1].set_title(f"augmented {j + 1}", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110)
