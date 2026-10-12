"""Step 1 helpers - used by notebooks/01_exploration.ipynb."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src import config


def load_inventory() -> pd.DataFrame:
    return pd.read_csv(config.REPORTS_DIR / "dataset_inventory.csv")


def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    counts = df[df["status"] == "ok"]["label"].value_counts().reindex(config.CLASS_NAMES)
    ax = counts.plot(kind="bar", figsize=(8, 4), color="steelblue")
    ax.set_title("Images per class")
    ax.set_ylabel("count")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    if save:
        plt.savefig(config.FIGURES_DIR / "class_distribution.png", dpi=120)
    return counts


def show_samples(root: Path = config.DATA_PROCESSED / "train", n_per_class: int = 4, save: bool = True):
    fig, axes = plt.subplots(config.NUM_CLASSES, n_per_class, figsize=(3 * n_per_class, 3 * config.NUM_CLASSES))
    for i, cls in enumerate(config.CLASS_NAMES):
        files = random.sample(list((root / cls).glob("*")), n_per_class)
        for j, f in enumerate(files):
            axes[i, j].imshow(Image.open(f))
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(cls, loc="left", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(config.FIGURES_DIR / "samples.png", dpi=100)


def size_report(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    return (ok.groupby(["width", "height"]).size().sort_values(ascending=False)
              .rename("n_images").reset_index())
