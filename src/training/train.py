
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data.preprocessing import (
    SPLITS_DIR,
    load_dataset_csv,
    prepare_pipeline
)

from src.models.cnn import build_cnn


MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")

EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_STATE = 42


def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


def create_callbacks():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_cnn.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    return callbacks


def plot_training_history(history):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("CNN - Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("CNN - Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_accuracy.png")
    plt.close()


def analyze_training(history):

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    train_accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    best_epoch = (
        history.history["val_loss"].index(
            min(history.history["val_loss"])
        ) + 1
    )

    print("\nTraining Results")
    print("----------------")

    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Best epoch: {best_epoch}")

    print(f"Final training loss: {train_loss:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")

    print(f"Final training accuracy: {train_accuracy:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")

    print(f"Accuracy gap: {train_accuracy - val_accuracy:.4f}")


def train_model():

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    train_raw = load_dataset_csv(
        SPLITS_DIR / "train.csv"
    )

    val_raw = load_dataset_csv(
        SPLITS_DIR / "val.csv"
    )

    train_dataset = prepare_pipeline(
        train_raw,
        training=True
    )

    val_dataset = prepare_pipeline(
        val_raw,
        training=False
    )

    model = build_cnn()

    model = compile_model(model)

    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=create_callbacks()
    )

    plot_training_history(history)

    analyze_training(history)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        METRICS_DIR / "training_history.csv",
        index=False
    )

    best_model = tf.keras.models.load_model(
        MODELS_DIR / "best_cnn.keras"
    )

    print("\nBest model loaded successfully!")

    print("\nTraining completed!")
    print("Best model: models/best_cnn.keras")
    print("Loss curve: reports/figures/training_loss.png")
    print("Accuracy curve: reports/figures/training_accuracy.png")

    return best_model, history


if __name__ == "__main__":
    train_model()
