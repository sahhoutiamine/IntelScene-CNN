"""Step 7 - Predict the category of a new image.

CLI:     python -m src.predict path/to/image.jpg [--model models/final_model.keras]
Python:  from src.predict import predict_image; predict_image("img.jpg")

Preprocessing = load as RGB, resize to 150x150. Rescaling (/255) is a layer INSIDE the
saved model, so training and inference preprocessing are guaranteed identical.
"""
import argparse
from functools import lru_cache

import numpy as np
from tensorflow import keras

from src import config


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    return keras.models.load_model(model_path)


def predict_image(image_path: str, model_path: str = str(config.FINAL_MODEL_PATH),
                  show: bool = False) -> dict:
    img = keras.utils.load_img(image_path, target_size=config.IMG_SIZE, color_mode="rgb")
    batch = np.expand_dims(keras.utils.img_to_array(img), axis=0)      # (1, 150, 150, 3), 0-255
    probs = _load_model(str(model_path)).predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    result = {
        "class": config.CLASS_NAMES[idx],
        "probability": float(probs[idx]),
        "all_probabilities": {c: float(p) for c, p in zip(config.CLASS_NAMES, probs)},
    }
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{result['class']} ({result['probability']:.1%})")
        plt.show()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=str(config.FINAL_MODEL_PATH))
    a = ap.parse_args()
    res = predict_image(a.image, a.model)
    print(f"Predicted class : {res['class']}")
    print(f"Probability     : {res['probability']:.2%}")
    for c, p in sorted(res["all_probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<10} {p:.2%}")
