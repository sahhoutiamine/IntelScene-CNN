"""Step 5 - Evaluate a saved model on the TEST set.

Usage:
    python -m src.evaluation.evaluate --model models/baseline.keras
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow import keras

from src import config
from src.data.loaders import load_split
from src.utils.plots import plot_confusion_matrix


def predict_test(model):
    ds = load_split("test", shuffle=False)          # shuffle=False keeps files aligned with labels
    files = list(ds.file_paths)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=1)
    return files, y_true, probs.argmax(axis=1), probs


def show_misclassified(files, y_true, y_pred, probs, n: int = 12, save_path=None):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("No misclassified images.")
        return
    # most confident mistakes first
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]
    cols = 4
    rows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), wrong):
        ax.imshow(Image.open(files[i]))
        ax.set_title(f"true: {config.CLASS_NAMES[y_true[i]]}\n"
                     f"pred: {config.CLASS_NAMES[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})",
                     fontsize=10, color="crimson")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110)


def evaluate(model_path: str):
    name = Path(model_path).stem
    model = keras.models.load_model(model_path)
    files, y_true, y_pred, probs = predict_test(model)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nTest accuracy: {acc:.4f} | macro precision {p:.4f} | recall {r:.4f} | F1 {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4))

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}_test_metrics.json").write_text(
        json.dumps({"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
                    "per_class": report}, indent=2))
    pd.DataFrame(cm, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).to_csv(
        config.REPORTS_DIR / f"{name}_confusion_matrix.csv")
    plot_confusion_matrix(cm, config.CLASS_NAMES, config.FIGURES_DIR / f"{name}_cm.png", title=f"{name}")
    plot_confusion_matrix(cm, config.CLASS_NAMES, config.FIGURES_DIR / f"{name}_cm_norm.png", normalize=True)
    show_misclassified(files, y_true, y_pred, probs, save_path=config.FIGURES_DIR / f"{name}_errors.png")
    return acc, cm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.FINAL_MODEL_PATH))
    evaluate(ap.parse_args().model)
