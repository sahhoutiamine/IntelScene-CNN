"""Step 4 & 6 - Train one experiment defined in configs/experiments.yaml.

Usage:
    python -m src.training.train --exp baseline
    python -m src.training.train --exp wider_bn_dropout --epochs 60
    python -m src.training.train --exp baseline --final     # also save as models/final_model.keras
"""
import argparse
import json
import shutil
import time

import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras

from src import config
from src.data.loaders import load_datasets
from src.models.cnn import build_cnn
from src.utils.plots import plot_history


def load_experiment(name: str) -> dict:
    cfg = yaml.safe_load(config.EXPERIMENTS_FILE.read_text())
    if name not in cfg["experiments"]:
        raise SystemExit(f"Unknown experiment '{name}'. Available: {list(cfg['experiments'])}")
    return {**cfg["defaults"], **cfg["experiments"][name]}


def run(name: str, epochs: int | None = None, final: bool = False) -> dict:
    keras.utils.set_random_seed(config.SEED)
    cfg = load_experiment(name)
    epochs = epochs or cfg["epochs"]

    train_ds, val_ds, _ = load_datasets(batch_size=cfg["batch_size"])

    aug = dict(cfg["augmentation"])
    use_aug = aug.pop("enabled", True)
    model = build_cnn(use_augmentation=use_aug, aug_params=aug, name=name, **cfg["model"])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",   # integer labels, 6 classes, softmax output
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg["reduce_lr_factor"],
                                          patience=cfg["reduce_lr_patience"], min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    minutes = (time.time() - t0) / 60

    # Save artefacts
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"{name}.keras"
    model.save(model_path)
    if final:
        shutil.copy(model_path, config.FINAL_MODEL_PATH)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (config.REPORTS_DIR / f"{name}_history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, config.FIGURES_DIR / f"{name}_curves.png", title=name)

    best = int(min(range(len(hist["val_loss"])), key=hist["val_loss"].__getitem__))
    row = {
        "experiment": name, "epochs_run": len(hist["loss"]), "best_epoch": best + 1,
        "train_acc": hist["accuracy"][best], "val_acc": hist["val_accuracy"][best],
        "train_loss": hist["loss"][best], "val_loss": hist["val_loss"][best],
        "params": model.count_params(), "minutes": round(minutes, 1),
    }
    summary_path = config.REPORTS_DIR / "experiments_summary.csv"
    new = pd.DataFrame([row])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        df = pd.concat([old[old["experiment"] != name], new], ignore_index=True)
    else:
        df = new
    df.to_csv(summary_path, index=False)
    print("\n", row)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="also save as models/final_model.keras")
    a = ap.parse_args()
    run(a.exp, a.epochs, a.final)
