from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from PIL import Image


TRAIN_PATH = Path("data/intel_image_classification/seg_train")

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def calculate_hash(image_path):
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def check_dataset_quality():

    image_data = []
    hashes = defaultdict(list)

    for class_name in CLASSES:

        class_path = TRAIN_PATH / class_name

        for image_path in sorted(class_path.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            status = "valid"
            issues = []
            width = None
            height = None
            color_mode = None
            image_hash = None

            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    img.load()
                    width, height = img.size
                    color_mode = img.mode

                image_hash = calculate_hash(image_path)

                if width < 64 or height < 64:
                    issues.append("small_image")

                if color_mode != "RGB":
                    issues.append("unexpected_color_mode")

                hashes[image_hash].append(
                    (class_name, str(image_path))
                )

            except Exception as e:
                status = "corrupted"
                issues.append(f"corrupted: {e}")

            image_data.append({
                "class": class_name,
                "filename": image_path.name,
                "filepath": str(image_path),
                "width": width,
                "height": height,
                "mode": color_mode,
                "hash": image_hash,
                "status": status,
                "issues": issues
            })

    df = pd.DataFrame(image_data)

    if df.empty:
        print("No images found.")
        return

    duplicate_groups = {
        image_hash: files
        for image_hash, files in hashes.items()
        if len(files) > 1
    }

    conflicting_hashes = {
        image_hash
        for image_hash, files in duplicate_groups.items()
        if len({class_name for class_name, _ in files}) > 1
    }

    df["duplicate"] = df["hash"].apply(
        lambda value: value in duplicate_groups
    )

    df["conflicting_labels"] = df["hash"].apply(
        lambda value: value in conflicting_hashes
    )

    df["issues"] = df["issues"].apply(
        lambda issues: ", ".join(issues)
    )

    print("\nDataset Quality Report")
    print("----------------------")

    print(f"\nTotal images: {len(df)}")

    corrupted = (df["status"] == "corrupted").sum()
    small = df["issues"].str.contains("small_image", regex=False).sum()
    unusual = df["issues"].str.contains("unexpected_color_mode", regex=False).sum()

    print(f"Corrupted images: {corrupted}")
    print(f"Small images: {small}")
    print(f"Unexpected color modes: {unusual}")

    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    print(f"Images in duplicate groups: {df['duplicate'].sum()}")
    print(f"Duplicate groups with conflicting labels: {len(conflicting_hashes)}")

    output_path = Path("reports/metrics/dataset_quality.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    check_dataset_quality()
