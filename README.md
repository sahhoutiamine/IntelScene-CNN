# Intel Image Classification - CNN with TensorFlow/Keras

Classify scene images into 6 classes: `buildings, forest, glacier, mountain, sea, street`.

## Quick start

```bash
# 1. Put the dataset in data/raw/  (e.g. data/raw/seg_train/seg_train/forest/*.jpg ...)
# 2. Build & start JupyterLab (http://localhost:8888)
docker compose build
docker compose up -d jupyter

# 3. Build train/val/test splits (+ data quality report)
docker compose run --rm trainer python -m src.data.prepare --force

# 4. Train experiments
docker compose run --rm trainer python -m src.training.train --exp baseline
docker compose run --rm trainer python -m src.training.train --exp wider_bn_dropout
docker compose run --rm trainer python -m src.training.train --exp no_augmentation
docker compose run --rm trainer python -m src.training.train --exp lr_batch_tuned

# 5. Evaluate / choose the final model / predict
docker compose run --rm trainer python -m src.evaluation.evaluate --model models/baseline.keras
docker compose run --rm trainer python -m src.training.train --exp <best> --final
docker compose run --rm trainer python -m src.predict data/processed/test/forest/xxx.jpg
```
(`make help`-style shortcuts are in the `Makefile`.)

## Structure

| Path | Role |
|---|---|
| `src/config.py` | paths, classes, image size, seed, split ratios |
| `src/data/prepare.py` | Step 1: scan, detect corrupted/duplicate files, stratified split |
| `src/data/explore.py` | Step 1: distribution, samples, size report |
| `src/data/loaders.py` | Step 2: `tf.data` pipelines |
| `src/data/augmentation.py` | Step 2: augmentation layers + before/after visualisation |
| `src/models/cnn.py` | Step 3: configurable CNN |
| `src/training/train.py` | Steps 4 & 6: training with EarlyStopping / ReduceLROnPlateau |
| `src/evaluation/evaluate.py` | Step 5: metrics, confusion matrix, misclassified images |
| `src/predict.py` | Step 7: single-image prediction |
| `configs/experiments.yaml` | Step 6: experiment definitions |
| `notebooks/` | one notebook per step (analysis + written explanations) |
| `reports/` | CSV/JSON metrics and `figures/` (generated) |

## Design notes
- **Split**: all labeled images are pooled, cleaned (corrupted + exact duplicates removed to avoid
  leakage), then split 70/15/15 with stratification. Train fits weights, validation tunes
  hyperparameters / early stopping, test is used once for the final unbiased estimate.
- **Augmentation** is a layer inside the model: active only in `fit()` on the training set,
  automatically disabled for validation, test and prediction.
- **Normalisation** (`Rescaling(1/255)`) is also inside the model -> same preprocessing at prediction.
- Runs on CPU in Docker. For GPU, run natively (or use a `tensorflow/tensorflow:*-gpu` base image on Linux + NVIDIA toolkit).
