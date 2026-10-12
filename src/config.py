"""Central configuration shared by scripts and notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_FILE = ROOT / "configs" / "experiments.yaml"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

# Alphabetical order = label order used by Keras image_dataset_from_directory
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (150, 150)          # Intel images are 150x150 in most cases
INPUT_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 32
SEED = 42

# train / validation / test ratios
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
