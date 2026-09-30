"""
Build train/val CSVs for fine-tuning fast-plate-ocr on Indian plates.

Test set = the `in_crops` set used by evaluate.py (zenitsu09/indian-number-plate). To keep it
honest, every training sample whose plate text appears in the test set is dropped, and exact
duplicate images are removed. Train/val are split by plate text, so val plates are unseen.

Sources (all permissive licences):
  avinashjadjasadf/indian-vehicle-number-plate-dataset-6   Apache-2.0, ~12k crops
  kp00011/indian-vehical-number-plate-ocr-labeled-dataset  MIT, ~1.7k crops

Optional synthetic data (train split only; val/test stay real):
  abtexp/synthetic-indian-license-plates                   CC0, 18k plates, all 36 states x 5 plate types
  siddheshmm/synthetic-license-plates-indian               licence UNKNOWN, 50k single-line plates
                                                           (--synthetic2 N adds a random N of them)

Usage (from repo root):
  python finetune/build_dataset.py              # real data only
  python finetune/build_dataset.py --synthetic  # + synthetic, written to train_synth.csv
  python finetune/build_dataset.py --synthetic --synthetic2 15000   # written to train_synth2.csv
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KAGGLE = ROOT / "data/kaggle"
OUT = ROOT / "finetune/data"
ALPHABET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_SLOTS = 10
VAL_FRACTION = 0.1


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def load_avinash() -> pd.DataFrame:
    root = KAGGLE / "avinashjadjasadf_indian-vehicle-number-plate-dataset-6/license_plate_dataset"
    df = pd.read_csv(root / "annotations.csv")
    return pd.DataFrame({"path": [root / p for p in df.image_path], "text": df.plate_text, "source": "avinash"})


def load_kp() -> pd.DataFrame:
    root = KAGGLE / "kp00011_indian-vehical-number-plate-ocr-labeled-dataset"
    rows = []
    for split in ("train", "valid", "test"):
        for line in (root / f"{split}_labels.txt").read_text().splitlines():
            name, _, text = line.partition("\t")
            rows.append({"path": root / "images" / split / name, "text": text, "source": "kp00011"})
    return pd.DataFrame(rows)


def load_synthetic() -> pd.DataFrame:
    root = KAGGLE / "abtexp_synthetic-indian-license-plates"
    paths = sorted(root.rglob("*.png"))
    return pd.DataFrame({"path": paths, "text": [p.stem for p in paths], "source": "synthetic"})


def load_synthetic2(n: int) -> pd.DataFrame:
    root = KAGGLE / "siddheshmm_synthetic-license-plates-indian"
    df = pd.read_csv(root / "labels.csv", dtype=str).sample(n=n, random_state=42)
    return pd.DataFrame({"path": [root / f for f in df.filename], "text": df.text, "source": "synthetic2"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="Add synthetic plates to the train split")
    ap.add_argument("--synthetic2", type=int, default=0, help="Add N plates from the second synthetic set")
    args = ap.parse_args()
    from evaluate import load_indian_crops

    test_texts = {s.plates[0][1] for s in load_indian_crops()}

    sources = [load_avinash(), load_kp()] + ([load_synthetic()] if args.synthetic else [])
    sources += [load_synthetic2(args.synthetic2)] if args.synthetic2 else []
    df = pd.concat(sources, ignore_index=True)
    df["text"] = df["text"].map(norm)
    n0 = len(df)
    df = df[df["path"].map(Path.exists)]
    df = df[df["text"].str.len().between(4, MAX_SLOTS) & df["text"].map(lambda t: set(t) <= ALPHABET)]
    n_valid = len(df)
    df = df[~df["text"].isin(test_texts)]
    n_leak = n_valid - len(df)
    df["md5"] = df["path"].map(lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest())
    df = df.drop_duplicates("md5")

    # Split by plate text so the same plate never lands in both train and val.
    # Val plates come from real data only, so the val set is identical with or without --synthetic.
    real = ~df["source"].str.startswith("synthetic")
    plates = sorted(df.loc[real, "text"].unique())
    random.Random(42).shuffle(plates)
    val_plates = set(plates[: int(len(plates) * VAL_FRACTION)])
    df = df[real | ~df["text"].isin(val_plates)]
    df["split"] = df["text"].map(lambda t: "val" if t in val_plates else "train")
    df.loc[~real, "split"] = "train"

    OUT.mkdir(parents=True, exist_ok=True)
    train_name = "train_synth2" if args.synthetic2 else "train_synth" if args.synthetic else "train"
    names = {"train": train_name, "val": "val"}
    for split in ("train", "val"):
        part = df[df["split"] == split]
        pd.DataFrame(
            {"image_path": [Path(p).resolve().relative_to(ROOT).as_posix() for p in part["path"]],
             "plate_text": part["text"]}
        ).to_csv(OUT / f"{names[split]}.csv", index=False)

    # CSV image paths are resolved relative to the CSV's folder, so prefix with ../../
    for split in ("train", "val"):
        csv = OUT / f"{names[split]}.csv"
        c = pd.read_csv(csv, dtype=str)
        c["image_path"] = "../../" + c["image_path"]
        c.to_csv(csv, index=False)

    print(f"raw samples            {n0}")
    print(f"after format filter    {n_valid}")
    print(f"dropped (text in test) {n_leak}")
    print(f"after image dedupe     {len(df)}")
    print(df.groupby(["split", "source"]).size().unstack(fill_value=0))
    print(f"unique plates: train {df[df.split=='train'].text.nunique()}  val {len(val_plates)}  test {len(test_texts)}")


if __name__ == "__main__":
    main()
