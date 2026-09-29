"""
License plate detection + recognition (ALPR) inference.

Pipeline: YOLOv9 plate detector (open-image-models) -> crop -> CCT plate OCR (fast-plate-ocr),
both running on ONNX Runtime via FastALPR. See RESEARCH.md for why this stack was chosen.

Usage:
    python infer.py car.jpg
    python infer.py images/ --out results/ --json results.json
    python infer.py car.jpg --device cpu --detector yolo-v9-t-384-license-plate-end2end

Library usage:
    from infer import PlateReader
    reader = PlateReader()
    plates = reader.read(cv2.imread("car.jpg"))   # BGR ndarray or path
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fast_alpr import ALPR

log = logging.getLogger("alpr")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Accuracy-leaning defaults; still real-time on CPU. Swap to the "t-384" / "cct-xs" variants
# for edge devices (see RESEARCH.md for the full list).
DEFAULT_DETECTOR = "yolo-v9-s-608-license-plate-end2end"
DEFAULT_OCR = "cct-s-v2-global-model"


@dataclass
class Plate:
    text: str
    ocr_confidence: float  # mean per-character confidence, 0-1
    det_confidence: float  # detector box confidence, 0-1
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    region: str | None = None
    region_confidence: float | None = None


def _pick_providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDA requested but not available. ONNX Runtime providers: {available}")
    preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available]


class PlateReader:
    """Loads the models once; call `read()` per frame. Not thread-safe per instance for drawing,
    but `read()` itself is safe to call from one worker at a time."""

    def __init__(
        self,
        detector_model: str = DEFAULT_DETECTOR,
        ocr_model: str = DEFAULT_OCR,
        device: str = "auto",
        det_conf: float = 0.4,
        min_ocr_conf: float = 0.0,
    ) -> None:
        if device in ("auto", "cuda"):
            # onnxruntime-gpu >= 1.21 can load CUDA/cuDNN DLLs shipped via pip (nvidia-* wheels)
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - absent libs just means we fall back to CPU
                pass
        self.min_ocr_conf = min_ocr_conf
        providers = _pick_providers(device)
        self.alpr = self._build(detector_model, ocr_model, det_conf, providers)
        try:
            _self_test(self.alpr)
        except Exception as e:  # noqa: BLE001
            if providers == ["CPUExecutionProvider"] or device == "cuda":
                raise
            # A provider can be "available" yet unusable at run time (e.g. missing cuDNN DLLs).
            # fast-alpr swallows that and returns zero plates, so fail over explicitly.
            log.warning("GPU self-test failed, falling back to CPU: %s", str(e).splitlines()[0])
            providers = ["CPUExecutionProvider"]
            self.alpr = self._build(detector_model, ocr_model, det_conf, providers)
            _self_test(self.alpr)
        self.providers = providers
        log.info("Running on: %s", providers[0])

    @staticmethod
    def _build(detector_model: str, ocr_model: str, det_conf: float, providers: list[str]) -> ALPR:
        return ALPR(
            detector_model=detector_model,
            detector_conf_thresh=det_conf,
            detector_providers=providers,
            ocr_model=ocr_model,
            ocr_providers=providers,
        )

    def read(self, image: np.ndarray | str | Path) -> list[Plate]:
        img = _load(image)
        plates: list[Plate] = []
        for r in self.alpr.predict(img):
            if r.ocr is None or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            ocr_conf = float(statistics.mean(conf) if isinstance(conf, list) else conf)
            if ocr_conf < self.min_ocr_conf:
                continue
            b = r.detection.bounding_box
            plates.append(
                Plate(
                    text=r.ocr.text.replace("_", "").strip(),  # "_" is the model's pad char
                    ocr_confidence=round(ocr_conf, 4),
                    det_confidence=round(float(r.detection.confidence), 4),
                    bbox=(int(b.x1), int(b.y1), int(b.x2), int(b.y2)),
                    region=r.ocr.region,
                    region_confidence=r.ocr.region_confidence,
                )
            )
        return plates


_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}


def _find_sessions(obj: object, depth: int = 4, seen: set[int] | None = None) -> list[ort.InferenceSession]:
    """Find the ONNX sessions wrapped inside fast-alpr's detector/OCR objects."""
    seen = seen if seen is not None else set()
    if isinstance(obj, ort.InferenceSession):
        return [obj]
    if depth == 0 or id(obj) in seen or not hasattr(obj, "__dict__"):
        return []
    seen.add(id(obj))
    return [s for v in vars(obj).values() for s in _find_sessions(v, depth - 1, seen)]


def _self_test(alpr: ALPR) -> None:
    """Run every session directly on dummy input so provider errors raise instead of being
    swallowed. Doubles as warm-up (first call allocates memory / picks kernels)."""
    sessions = _find_sessions(alpr)
    if not sessions:
        raise RuntimeError("No ONNX sessions found inside ALPR; fast-alpr internals changed?")
    for sess in sessions:
        feeds = {
            i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], _ORT_DTYPES.get(i.type, np.float32))
            for i in sess.get_inputs()
        }
        sess.run(None, feeds)
    alpr.predict(np.zeros((640, 640, 3), dtype=np.uint8))


def _load(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image))
    if img is None:
        raise ValueError(f"Could not read image: {image}")
    return img


def draw(img: np.ndarray, plates: list[Plate]) -> np.ndarray:
    out = img.copy()
    scale = min(1.25, max(0.5, out.shape[1] / 1000))
    thick = 1 if scale < 0.75 else 2
    for p in plates:
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (36, 255, 12), 2)
        label = f"{p.text} {p.ocr_confidence * 100:.0f}%"
        (_, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 - 8 if y1 - th - 8 > 0 else y2 + th + 8
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _collect(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            paths.append(p)
        else:
            log.warning("Skipping missing path: %s", p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect and read license plates in images.")
    ap.add_argument("inputs", nargs="+", help="Image files and/or directories")
    ap.add_argument("--out", help="Directory to write annotated images to")
    ap.add_argument("--json", help="Write all results to this JSON file")
    ap.add_argument("--detector", default=DEFAULT_DETECTOR)
    ap.add_argument("--ocr", default=DEFAULT_OCR)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--det-conf", type=float, default=0.4, help="Detector confidence threshold")
    ap.add_argument("--min-ocr-conf", type=float, default=0.0, help="Drop reads below this OCR confidence")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = _collect(args.inputs)
    if not paths:
        log.error("No images found.")
        return 1

    t0 = time.perf_counter()
    reader = PlateReader(args.detector, args.ocr, args.device, args.det_conf, args.min_ocr_conf)
    log.info("Models loaded in %.2fs", time.perf_counter() - t0)

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for path in paths:
        try:
            img = _load(path)
        except ValueError as e:
            log.warning("%s", e)
            continue
        t = time.perf_counter()
        plates = reader.read(img)
        ms = (time.perf_counter() - t) * 1000
        texts = ", ".join(f"{p.text} ({p.ocr_confidence:.2f})" for p in plates) or "no plate"
        print(f"{path}  [{ms:.1f} ms]  {texts}")
        all_results[str(path)] = {"latency_ms": round(ms, 2), "plates": [asdict(p) for p in plates]}
        if out_dir:
            cv2.imwrite(str(out_dir / f"{path.stem}_alpr{path.suffix}"), draw(img, plates))

    if args.json:
        Path(args.json).write_text(json.dumps(all_results, indent=2))
        log.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
