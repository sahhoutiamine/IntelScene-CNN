"""Step 1 - Scan the raw dataset, flag problematic files, build train/val/test splits.

Usage:
    python -m src.data.prepare --force

Works whatever the raw layout is (seg_train/seg_train/<class>/..., flat <class>/..., etc.):
every image whose *parent folder name* is one of the 6 classes is used. The unlabeled
`seg_pred` folder is ignored automatically. All labeled images are pooled, cleaned
(corrupted / duplicates removed) and re-split with stratification.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_raw(raw_dir: Path = config.DATA_RAW) -> pd.DataFrame:
    """Return one row per labeled image with size, mode, format, hash and a status."""
    files = [p for p in Path(raw_dir).rglob("*")
             if p.is_file() and p.parent.name in config.CLASS_NAMES]
    if not files:
        raise FileNotFoundError(
            f"No images found in {raw_dir}. Put the Intel dataset there "
            f"(folders named {config.CLASS_NAMES})."
        )

    rows = []
    for p in tqdm(sorted(files), desc="Scanning images"):
        row = dict(path=str(p), label=p.parent.name, width=None, height=None,
                   mode=None, format=None, md5=None, status="ok")
        if p.suffix.lower() not in config.VALID_EXTENSIONS:
            row["status"] = "bad_extension"
        else:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:  # reopen: verify() invalidates the file
                    row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                row["md5"] = _md5(p)
            except Exception:
                row["status"] = "corrupted"
        rows.append(row)

    df = pd.DataFrame(rows)
    dup = df["md5"].notna() & df.duplicated(subset="md5", keep="first")
    df.loc[dup & (df["status"] == "ok"), "status"] = "duplicate"
    return df


def split_dataframe(df_ok: pd.DataFrame) -> pd.DataFrame:
    r = config.SPLIT_RATIOS
    train, temp = train_test_split(
        df_ok, test_size=1 - r["train"], stratify=df_ok["label"], random_state=config.SEED)
    val, test = train_test_split(
        temp, test_size=r["test"] / (r["val"] + r["test"]),
        stratify=temp["label"], random_state=config.SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])


def copy_split(df_split: pd.DataFrame, out_dir: Path = config.DATA_PROCESSED) -> None:
    for _, r in tqdm(df_split.iterrows(), total=len(df_split), desc="Copying files"):
        dst = Path(out_dir) / r["split"] / r["label"] / Path(r["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():  # same filename in two raw folders
            dst = dst.with_name(f"{r['md5'][:8]}_{dst.name}")
        shutil.copy2(r["path"], dst)


def main(force: bool = False) -> None:
    out = config.DATA_PROCESSED
    if any(p for p in out.glob("*") if p.name != ".gitkeep"):
        if not force:
            raise SystemExit(f"{out} is not empty. Re-run with --force to rebuild it.")
        for p in out.glob("*"):
            if p.name != ".gitkeep":
                shutil.rmtree(p) if p.is_dir() else p.unlink()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = scan_raw()
    df.to_csv(config.REPORTS_DIR / "dataset_inventory.csv", index=False)
    print("\nStatus counts:\n", df["status"].value_counts().to_string())

    df_ok = df[df["status"] == "ok"]
    df_split = split_dataframe(df_ok)
    copy_split(df_split)

    summary = (df_split.groupby(["split", "label"]).size().unstack(fill_value=0))
    summary.to_csv(config.REPORTS_DIR / "split_summary.csv")
    (config.REPORTS_DIR / "split_summary.json").write_text(
        json.dumps({"status_counts": df["status"].value_counts().to_dict(),
                    "per_split": summary.to_dict(orient="index")}, indent=2))
    print("\nSplit summary:\n", summary.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild data/processed")
    main(ap.parse_args().force)
