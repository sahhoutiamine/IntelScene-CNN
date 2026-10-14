
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define the dataset path
DATASET_PATH = Path("data/intel_image_classification")
TRAIN_PATH = DATASET_PATH / "seg_train"

# 2. Define expected classes
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def explore_dataset():

    if not TRAIN_PATH.exists():
        print("Dataset folder not found!")
        return

    print("Dataset structure:")

    for folder in DATASET_PATH.iterdir():
        if folder.is_dir():
            print(f"- {folder.name}")

    print("\nDetected classes:")

    for folder in sorted(TRAIN_PATH.iterdir()):
        if folder.is_dir():
            print(f"- {folder.name}")

    detected_classes = [
        folder.name
        for folder in TRAIN_PATH.iterdir()
        if folder.is_dir()
    ]

    if set(detected_classes) == set(CLASSES):
        print("\nAll 6 expected classes are present!")
    else:
        print("\nWarning: Classes do not match expectations!")



def analyze_class_distribution():
    pass







if __name__ == "__main__":
    explore_dataset()
    
    
    

