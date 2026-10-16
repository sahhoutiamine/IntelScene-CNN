
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image


SPLITS_DIR = Path("data/splits")
REPORTS_DIR = Path("reports/metrics")

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

EXPECTED_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}


def calculate_perceptual_hash(image_path):

    with Image.open(image_path) as img:
        image = img.convert("L").resize((9, 8))
        pixels = np.asarray(image, dtype=np.int16)

    differences = pixels[:, 1:] > pixels[:, :-1]

    return "".join(
        "1" if value else "0"
        for value in differences.flatten()
    )


def hamming_distance(hash1, hash2):

    return sum(
        a != b
        for a, b in zip(hash1, hash2)
    )


def verify_splits():

    splits = {}

    for split_name in EXPECTED_RATIOS:
        split_path = SPLITS_DIR / f"{split_name}.csv"
        splits[split_name] = pd.read_csv(split_path)

    all_valid = True

    print("\nDataset Split Verification")
    print("--------------------------")

    total_images = sum(len(df) for df in splits.values())

    if total_images == 0:
        raise ValueError("All splits are empty.")

    for split_name, df in splits.items():

        print(f"\n{split_name.upper()}")

        print(f"Images: {len(df)}")

        actual_ratio = len(df) / total_images
        expected_ratio = EXPECTED_RATIOS[split_name]

        print(f"Ratio: {actual_ratio:.2%}")
        print(f"Expected: {expected_ratio:.2%}")

        if abs(actual_ratio - expected_ratio) > 0.01:
            print("WARNING: Unexpected split ratio")
            all_valid = False

        missing_classes = set(CLASSES) - set(df["class"])

        if missing_classes:
            print(f"FAIL: Missing classes: {missing_classes}")
            all_valid = False
        else:
            print("PASS: All classes present")

        print("\nClass distribution:")
        print(df["class"].value_counts().sort_index())

        missing_files = [
            path
            for path in df["filepath"]
            if not Path(path).is_file()
        ]

        if missing_files:
            print(f"FAIL: {len(missing_files)} missing files")
            all_valid = False
        else:
            print("PASS: All image files exist")

        if df["filepath"].duplicated().any():
            print("FAIL: Duplicate paths within split")
            all_valid = False

        if df["hash"].isna().any():
            print("FAIL: Missing SHA-256 hashes")
            all_valid = False

        if df["hash"].duplicated().any():
            print("FAIL: Duplicate hashes within split")
            all_valid = False

    for split_a, split_b in combinations(splits.keys(), 2):

        df_a = splits[split_a]
        df_b = splits[split_b]

        path_overlap = (
            set(df_a["filepath"]) & set(df_b["filepath"])
        )

        hash_overlap = (
            set(df_a["hash"].dropna())
            & set(df_b["hash"].dropna())
        )

        print(f"\n{split_a.upper()} vs {split_b.upper()}")

        if path_overlap:
            print(f"FAIL: {len(path_overlap)} overlapping paths")
            all_valid = False
        else:
            print("PASS: No path overlap")

        if hash_overlap:
            print(f"FAIL: {len(hash_overlap)} overlapping hashes")
            all_valid = False
        else:
            print("PASS: No SHA-256 overlap")

    print("\nChecking perceptual near-duplicates...")

    image_hashes = []

    for split_name, df in splits.items():

        for image_path in df["filepath"]:

            if not Path(image_path).is_file():
                continue

            try:
                perceptual_hash = calculate_perceptual_hash(image_path)

                image_hashes.append({
                    "split": split_name,
                    "filepath": image_path,
                    "perceptual_hash": perceptual_hash
                })

            except Exception as e:
                print(f"Cannot inspect {image_path}: {e}")
                all_valid = False

    near_duplicates = []

    hashes_df = pd.DataFrame(image_hashes)

    if not hashes_df.empty:

        grouped = {
            name: group.to_dict("records")
            for name, group in hashes_df.groupby("split")
        }

        for split_a, split_b in combinations(grouped.keys(), 2):

            for image_a in grouped[split_a]:

                for image_b in grouped[split_b]:

                    distance = hamming_distance(
                        image_a["perceptual_hash"],
                        image_b["perceptual_hash"]
                    )

                    if distance <= 3:

                        near_duplicates.append({
                            "split_a": split_a,
                            "image_a": image_a["filepath"],
                            "split_b": split_b,
                            "image_b": image_b["filepath"],
                            "distance": distance
                        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    near_duplicate_path = REPORTS_DIR / "near_duplicate_candidates.csv"

    pd.DataFrame(
        near_duplicates,
        columns=[
            "split_a",
            "image_a",
            "split_b",
            "image_b",
            "distance"
        ]
    ).to_csv(near_duplicate_path, index=False)

    print(f"Near-duplicate candidates: {len(near_duplicates)}")
    print(f"Report saved to: {near_duplicate_path}")

    print("\nFinal Verification")
    print("------------------")

    if all_valid:
        print("PASS: Structural and exact-duplicate checks passed")
    else:
        print("FAIL: One or more verification checks failed")

    if near_duplicates:
        print("REVIEW REQUIRED: Potential near-duplicates across splits")
        print("Do not finalize the test set before reviewing them.")
    else:
        print("No near-duplicate candidates detected by this method.")

    return all_valid and not near_duplicates


if __name__ == "__main__":
    verify_splits()
