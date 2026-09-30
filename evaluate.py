"""
Evaluate PlateReader on labelled plate datasets.

Sets:
  eu, br, us     OpenALPR end-to-end benchmark (full car photos, 1 plate each)
                 git clone --depth 1 https://github.com/openalpr/benchmarks.git data/openalpr_benchmarks
  in_full        Indian full photos, Pascal VOC XML with `number_plate_text` (Datacluster Labs sample)
                 hf download Dataclusterlabspvtltd/indian-number-plates-dataset --repo-type dataset --local-dir data/indian_datacluster
  in_crops       Indian plate crops, OCR only (no detection), ~30 states
                 hf download zenitsu09/indian-number-plate --repo-type dataset --local-dir data/indian_zenitsu

Usage:
    python evaluate.py --sets eu br us in_full in_crops
    python evaluate.py --sets in_crops --ocr cct-xs-v2-global-model --failures failures.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from infer import DEFAULT_DETECTOR, DEFAULT_OCR, PlateReader

DATA = Path("data")
Box = tuple[int, int, int, int]


@dataclass
class Sample:
    name: str
    load: Callable[[], np.ndarray | None]
    plates: list[tuple[Box | None, str]]  # (ground-truth box or None for crops, normalised text)


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def o0(s: str) -> str:
    return s.replace("O", "0")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _imread(path: Path) -> Callable[[], np.ndarray | None]:
    return lambda: cv2.imread(str(path))


# ---------------------------------------------------------------- loaders


def load_openalpr(region: str) -> list[Sample]:
    samples = []
    for txt in sorted((DATA / "openalpr_benchmarks/endtoend" / region).glob("*.txt")):
        for line in txt.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            x, y, w, h = map(int, parts[1:5])
            img = txt.with_name(parts[0])
            samples.append(Sample(str(img), _imread(img), [((x, y, x + w, y + h), norm(parts[5]))]))
    return samples


def load_indian_full() -> list[Sample]:
    root = DATA / "indian_datacluster"
    samples = []
    for xml in sorted((root / "Annotations").glob("*.xml")):
        tree = ET.parse(xml).getroot()
        plates = []
        for obj in tree.iter("object"):
            text = next(
                (a.findtext("value") for a in obj.iter("attribute") if a.findtext("name") == "number_plate_text"),
                None,
            )
            if not text:
                continue  # plate boxed but text not labelled
            bb = obj.find("bndbox")
            box = tuple(int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
            plates.append((box, norm(text)))
        img = root / "images" / tree.findtext("filename")
        if plates and img.exists():
            samples.append(Sample(str(img), _imread(img), plates))
    return samples


def load_indian_crops() -> list[Sample]:
    import pandas as pd  # only needed for this set

    df = pd.read_parquet(next((DATA / "indian_zenitsu").rglob("*.parquet")))
    # Roboflow export: each source image appears ~3x with augmentations. Keep one per original.
    df["orig"] = df["orig_filename"].str.split(".rf.").str[0]
    df = df.drop_duplicates("orig")
    samples = []
    for row in df.itertuples():
        buf = np.frombuffer(row.image["bytes"], np.uint8)
        samples.append(
            Sample(row.orig, lambda b=buf: cv2.imdecode(b, cv2.IMREAD_COLOR), [(None, norm(row.plate_text))])
        )
    return samples


LOADERS: dict[str, Callable[[], list[Sample]]] = {
    "eu": lambda: load_openalpr("eu"),
    "br": lambda: load_openalpr("br"),
    "us": lambda: load_openalpr("us"),
    "in_full": load_indian_full,
    "in_crops": load_indian_crops,
}

# ---------------------------------------------------------------- evaluation


def evaluate(reader: PlateReader, set_name: str, failures: list[dict]) -> dict:
    n = found = exact = exact_o0 = err = chars = extra = 0
    lat: list[float] = []
    for s in LOADERS[set_name]():
        img = s.load()
        if img is None:
            continue
        crop_mode = s.plates[0][0] is None
        t = time.perf_counter()
        if crop_mode:
            text, conf = reader.read_crop(img)
            preds = [(None, norm(text), conf)]
        else:
            preds = [(p.bbox, norm(p.text), p.ocr_confidence) for p in reader.read(img)]
        lat.append((time.perf_counter() - t) * 1000)

        matched = set()
        for gt_box, gt_text in s.plates:
            if crop_mode:
                pred, hit = preds[0][1], True
            else:
                best = max(range(len(preds)), key=lambda i: iou(preds[i][0], gt_box), default=None)
                hit = best is not None and iou(preds[best][0], gt_box) >= 0.3
                pred = preds[best][1] if hit else ""
                if hit:
                    matched.add(best)
            n += 1
            found += hit
            exact += pred == gt_text
            exact_o0 += o0(pred) == o0(gt_text)
            err += edit_distance(pred, gt_text)
            chars += len(gt_text)
            if pred != gt_text:
                failures.append({"set": set_name, "sample": s.name, "gt": gt_text, "pred": pred or None,
                                 "all_reads": [(p[1], round(p[2], 3)) for p in preds]})
        if not crop_mode:
            extra += len(preds) - len(matched)
    return {
        "set": set_name,
        "plates": n,
        "found": found / n,
        "exact": exact / n,
        "exact_o0": exact_o0 / n,
        "char_acc": 1 - err / chars,
        "extra": extra,
        "ms_med": statistics.median(lat),
        "ms_p95": statistics.quantiles(lat, n=20)[-1] if len(lat) > 1 else lat[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--detector", default=DEFAULT_DETECTOR)
    ap.add_argument("--ocr", default=DEFAULT_OCR, help="Hub model name, published model (india-v1) or path to a .onnx")
    ap.add_argument("--ocr-config", help="Plate config YAML for a custom --ocr .onnx")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--plate-format", choices=["india", "none"], help="Constrain reads to a plate format")
    ap.add_argument("--failures", help="Write mismatched reads to this JSON file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    reader = PlateReader(args.detector, args.ocr, args.device, ocr_config=args.ocr_config, plate_format=args.plate_format)
    print(f"detector={args.detector}  ocr={args.ocr}  format={args.plate_format}  device={reader.providers[0]}")
    hdr = f"{'set':<10}{'plates':>7}{'found':>8}{'exact':>8}{'O=0':>8}{'chars':>8}{'extra':>7}{'ms_med':>8}{'ms_p95':>8}"
    print(hdr)
    failures: list[dict] = []
    for name in args.sets:
        r = evaluate(reader, name, failures)
        print(
            f"{r['set']:<10}{r['plates']:>7}{r['found']:>8.1%}{r['exact']:>8.1%}{r['exact_o0']:>8.1%}"
            f"{r['char_acc']:>8.1%}{r['extra']:>7}{r['ms_med']:>8.1f}{r['ms_p95']:>8.1f}"
        )
    if args.failures:
        Path(args.failures).write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} failures -> {args.failures}")


if __name__ == "__main__":
    main()
