"""
Evaluate PlateReader against the OpenALPR end-to-end benchmark (https://github.com/openalpr/benchmarks).

Each image has a sidecar .txt: "<file>\t<x>\t<y>\t<w>\t<h>\t<PLATE>".

Usage:
    git clone --depth 1 https://github.com/openalpr/benchmarks.git data/openalpr_benchmarks
    python evaluate.py --sets eu br us
    python evaluate.py --sets eu --detector yolo-v9-t-384-license-plate-end2end --ocr cct-xs-v2-global-model
    python evaluate.py --sets eu br us --failures failures.json   # dump wrong reads for inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from pathlib import Path

import cv2

from infer import DEFAULT_DETECTOR, DEFAULT_OCR, PlateReader

ROOT = Path("data/openalpr_benchmarks/endtoend")


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def load_set(name: str) -> list[tuple[Path, tuple[int, int, int, int], str]]:
    items = []
    for txt in sorted((ROOT / name).glob("*.txt")):
        for line in txt.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            x, y, w, h = map(int, parts[1:5])
            items.append((txt.with_name(parts[0]), (x, y, x + w, y + h), norm(parts[5])))
    return items


def evaluate(reader: PlateReader, name: str, failures: list[dict]) -> dict:
    items = load_set(name)
    n = det_hit = exact = exact_any = 0
    cer_num = cer_den = extra = 0
    lat: list[float] = []
    for img_path, gt_box, gt_text in items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t = time.perf_counter()
        plates = reader.read(img)
        lat.append((time.perf_counter() - t) * 1000)
        n += 1

        # Match the prediction whose box overlaps the ground-truth plate the most.
        best = max(plates, key=lambda p: iou(p.bbox, gt_box), default=None)
        best_iou = iou(best.bbox, gt_box) if best else 0.0
        pred = norm(best.text) if best and best_iou >= 0.3 else ""
        det_hit += best_iou >= 0.3
        exact += pred == gt_text
        exact_any += gt_text in {norm(p.text) for p in plates}
        extra += max(0, len(plates) - 1)
        cer_num += edit_distance(pred, gt_text)
        cer_den += len(gt_text)
        if pred != gt_text:
            failures.append(
                {
                    "set": name,
                    "image": str(img_path),
                    "gt": gt_text,
                    "pred": pred or None,
                    "all_reads": [(p.text, p.ocr_confidence) for p in plates],
                    "iou": round(best_iou, 2),
                }
            )
    return {
        "set": name,
        "images": n,
        "det_recall": det_hit / n,
        "exact_match": exact / n,
        "exact_any": exact_any / n,
        "char_acc": 1 - cer_num / cer_den,
        "extra_plates": extra,
        "ms_median": statistics.median(lat),
        "ms_p95": statistics.quantiles(lat, n=20)[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["eu", "br", "us"])
    ap.add_argument("--detector", default=DEFAULT_DETECTOR)
    ap.add_argument("--ocr", default=DEFAULT_OCR)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--failures", help="Write mismatched reads to this JSON file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    reader = PlateReader(args.detector, args.ocr, args.device)
    print(f"detector={args.detector}  ocr={args.ocr}  device={reader.providers[0]}")
    print(f"{'set':<5}{'imgs':>5}{'det_rec':>9}{'exact':>8}{'any':>7}{'char':>7}{'extra':>7}{'ms_med':>8}{'ms_p95':>8}")
    failures: list[dict] = []
    rows = [evaluate(reader, s, failures) for s in args.sets]
    for r in rows:
        print(
            f"{r['set']:<5}{r['images']:>5}{r['det_recall']:>9.1%}{r['exact_match']:>8.1%}{r['exact_any']:>7.1%}"
            f"{r['char_acc']:>7.1%}{r['extra_plates']:>7}{r['ms_median']:>8.1f}{r['ms_p95']:>8.1f}"
        )
    total = sum(r["images"] for r in rows)
    wavg = lambda k: sum(r[k] * r["images"] for r in rows) / total  # noqa: E731
    print(f"{'ALL':<5}{total:>5}{wavg('det_recall'):>9.1%}{wavg('exact_match'):>8.1%}{wavg('exact_any'):>7.1%}{wavg('char_acc'):>7.1%}")
    if args.failures:
        Path(args.failures).write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} failures -> {args.failures}")


if __name__ == "__main__":
    main()
