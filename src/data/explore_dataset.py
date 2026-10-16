
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
from collections import Counter

DATASET_PATH = Path("data/intel_image_classification")
TRAIN_PATH = DATASET_PATH / "seg_train"

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

    class_counts = {}
    image_extensions = {".jpg", ".jpeg", ".png"}

    for class_name in CLASSES:
        class_path = TRAIN_PATH / class_name
        images = [
            file
            for file in class_path.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        ]
        class_counts[class_name] = len(images)

    df = pd.DataFrame(
        class_counts.items(),
        columns=["Class", "Number of Images"]
    )

    total_images = df["Number of Images"].sum()
    df["Percentage"] = (
        df["Number of Images"] / total_images * 100
    ).round(2)

    print("\nClass Distribution:")
    print(df.to_string(index=False))
    print(f"\nTotal images: {total_images}")

    plt.figure(figsize=(10, 6))
    plt.bar(df["Class"], df["Number of Images"])
    plt.title("Intel Image Classification - Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    output_path = Path("reports/figures/class_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"\nChart saved to: {output_path}")





def visualize_sample_images():

    images_per_class = 3

    random.seed(42)

    fig, axes = plt.subplots(
        nrows=len(CLASSES),
        ncols=images_per_class,
        figsize=(12, 18)
    )

    image_extensions = {".jpg", ".jpeg", ".png"}

    for row, class_name in enumerate(CLASSES):

        class_path = TRAIN_PATH / class_name

        image_paths = sorted([
            file
            for file in class_path.iterdir()
            if file.is_file()
            and file.suffix.lower() in image_extensions
        ])

        selected_images = random.sample(
            image_paths,
            min(images_per_class, len(image_paths))
        )

        for col, image_path in enumerate(selected_images):

            with Image.open(image_path) as img:
                image = img.convert("RGB")

            axes[row, col].imshow(image)

            axes[row, col].set_title(class_name)

            axes[row, col].axis("off")

        for col in range(len(selected_images), images_per_class):
            axes[row, col].axis("off")

    fig.suptitle(
        "Intel Image Classification - Sample Images",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path("reports/figures/sample_images.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)

    plt.close()

    print(f"\nSample images saved to: {output_path}")




def inspect_image_properties():

    image_data = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    for class_name in CLASSES:

        class_path = TRAIN_PATH / class_name

        for image_path in sorted(class_path.iterdir()):

            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue

            try:
                with Image.open(image_path) as img:

                    width, height = img.size
                    image_format = img.format
                    color_mode = img.mode

                    image_array = np.array(img)

                    image_data.append({
                        "class": class_name,
                        "filename": image_path.name,
                        "width": width,
                        "height": height,
                        "format": image_format,
                        "mode": color_mode,
                        "shape": str(image_array.shape),
                        "dtype": str(image_array.dtype),
                        "min_pixel": int(image_array.min()),
                        "max_pixel": int(image_array.max())
                    })

            except Exception as e:
                print(f"Error reading {image_path}: {e}")

    df = pd.DataFrame(image_data)

    if df.empty:
        print("No readable images found.")
        return

    print("\nImage Properties Analysis")
    print("-------------------------")

    print(f"\nTotal inspected images: {len(df)}")

    print("\nImage formats:")
    print(Counter(df["format"]))

    print("\nColor modes:")
    print(Counter(df["mode"]))

    dimensions = list(zip(df["width"], df["height"]))

    print("\nMost common dimensions:")
    for dimension, count in Counter(dimensions).most_common(10):
        print(f"{dimension}: {count}")

    print(f"\nMinimum width: {df['width'].min()}")
    print(f"Maximum width: {df['width'].max()}")

    print(f"\nMinimum height: {df['height'].min()}")
    print(f"Maximum height: {df['height'].max()}")

    print("\nPixel values:")
    print(f"Minimum pixel value: {df['min_pixel'].min()}")
    print(f"Maximum pixel value: {df['max_pixel'].max()}")

    output_path = Path("reports/metrics/image_properties.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nImage properties saved to: {output_path}")





if __name__ == "__main__":
    explore_dataset()
    analyze_class_distribution()
    visualize_sample_images()
    inspect_image_properties()
    

