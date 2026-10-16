
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


QUALITY_REPORT = Path("reports/metrics/dataset_quality.csv")
SPLITS_DIR = Path("data/splits")

RANDOM_STATE = 42


def prepare_dataset():

    df = pd.read_csv(QUALITY_REPORT)

    print(f"Original images: {len(df)}")

    df = df[df["status"] == "valid"].copy()

    print(f"Valid images: {len(df)}")

    df = df.dropna(subset=["hash"])

    conflicting = df[df["conflicting_labels"] == True]

    
    conflicting = df[df["conflicting_labels"] == True].copy()

    print(f"Conflicting images: {len(conflicting)}")

    if not conflicting.empty:

        excluded_path = Path(
            "reports/metrics/excluded_conflicting_images.csv"
        )

        excluded_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        conflicting.to_csv(excluded_path, index=False)

        df = df[df["conflicting_labels"] == False].copy()

        print(f"Excluded {len(conflicting)} conflicting images")
        print(f"Report saved to: {excluded_path}")


    df = df.sort_values("filepath")

    duplicate_count = df.duplicated(subset=["hash"]).sum()

    df = df.drop_duplicates(subset=["hash"], keep="first").copy()

    print(f"Exact duplicates removed: {duplicate_count}")
    print(f"Images after cleaning: {len(df)}")

    df = df[["filepath", "class", "hash"]]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["class"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["class"]
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print("\nDataset Split Results")
    print("---------------------")

    total = len(df)

    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:

        count = len(split_df)
        percentage = count / total * 100

        print(f"\n{name}: {count} images ({percentage:.2f}%)")

        print(split_df["class"].value_counts().sort_index())

    print("\nSplits saved successfully!")


if __name__ == "__main__":
    prepare_dataset()
