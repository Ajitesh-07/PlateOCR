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
import hashlib
import json
import logging
import os
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fast_alpr import ALPR
from fast_alpr.base import BaseOCR, OcrResult
from fast_plate_ocr.core.process import preprocess_image
from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source

from plate_format import decode_india

log = logging.getLogger("alpr")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Accuracy-leaning defaults; still real-time on CPU. Swap to the "t-384" / "cct-xs" variants
# for edge devices (see RESEARCH.md for the full list).
DEFAULT_DETECTOR = "yolo-v9-s-608-license-plate-end2end"
DEFAULT_OCR = "cct-s-v2-global-model"

# Fine-tuned OCR models published as GitHub Release assets of this repo. `--ocr india-v1` downloads
# them once into PLATE_OCR_CACHE (default ~/.cache/plate-ocr), checks the SHA-256, then reuses them.
# PLATE_OCR_MODEL_URL overrides the download base (mirror, offline server); files are fetched from
# <base>/<tag>/<file>. See MODEL_CARD.md.
RELEASES_URL = "https://github.com/Ajitesh-07/PlateOCR/releases/download"
CUSTOM_OCR_MODELS = {
    "india-v1": {
        "tag": "india-ocr-v1",
        "onnx": ("india_ocr_v1.onnx", "6fbb878e1bec7e318c9b43d58f4093d5bb7cb21bfe9d7d5fc18047f6c007a6f8"),
        "config": ("india_ocr_v1_plate_config.yaml", "3f4718541abd7ba9e7fc050cfbda7463b64b9795d8c99689019c5b58a8e65b41"),
        "plate_format": "india",
    },
}


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_custom_model(name: str) -> tuple[Path, Path]:
    """Return local (onnx, plate_config) paths for a CUSTOM_OCR_MODELS entry, downloading if needed."""
    spec = CUSTOM_OCR_MODELS[name]
    cache = Path(os.environ.get("PLATE_OCR_CACHE", Path.home() / ".cache" / "plate-ocr")) / name
    base = os.environ.get("PLATE_OCR_MODEL_URL", RELEASES_URL).rstrip("/")
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname, sha in (spec["onnx"], spec["config"]):
        dest = cache / fname
        if not (dest.exists() and _sha256(dest) == sha):
            url = f"{base}/{spec['tag']}/{fname}"
            log.info("Downloading %s", url)
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != sha:
                tmp.unlink()
                raise RuntimeError(f"Checksum mismatch for {url}; refusing to use it")
            tmp.replace(dest)
        paths.append(dest)
    return paths[0], paths[1]


class FormatOCR(BaseOCR):
    """Wraps fast-alpr's DefaultOCR and decodes the raw per-slot probabilities under a plate
    grammar (see plate_format.py) instead of taking the argmax per slot."""

    DECODERS = {"india": decode_india}

    def __init__(self, inner: BaseOCR, plate_format: str) -> None:
        self.inner = inner
        self.decode = self.DECODERS[plate_format]
        self.rec = inner.ocr_model  # fast_plate_ocr LicensePlateRecognizer

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        if cropped_plate is None or cropped_plate.size == 0:
            return None
        cfg = self.rec.config
        code = {"grayscale": cv2.COLOR_BGR2GRAY, "rgb": cv2.COLOR_BGR2RGB}[cfg.image_color_mode]
        x = preprocess_image(_load_image_from_source(cv2.cvtColor(cropped_plate, code), cfg))
        out = self.rec.model.run([self.rec.plate_output_name], {"input": x})[0]
        probs = out.reshape(cfg.max_plate_slots, len(cfg.alphabet))
        text, confs = self.decode(probs, cfg.alphabet, cfg.pad_char)
        return OcrResult(text=text, confidence=confs)


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
        ocr_config: str | None = None,
        plate_format: str | None = None,
    ) -> None:
        """`ocr_model` is a hub name or a path to a fine-tuned .onnx. For a path, `ocr_config` is its
        plate config YAML; if omitted, the only *.yaml next to the .onnx is used.
        `ocr_model` can also name a published fine-tuned model (CUSTOM_OCR_MODELS, e.g. "india-v1"); it is
        downloaded on first use and brings its own default `plate_format`.
        `plate_format="india"` only returns plates that match the Indian format (see plate_format.py);
        "none" turns format decoding off, even for models that default to it."""
        if ocr_model in CUSTOM_OCR_MODELS:
            spec = CUSTOM_OCR_MODELS[ocr_model]
            onnx_path, cfg_path = fetch_custom_model(ocr_model)
            ocr_model, ocr_config = str(onnx_path), str(cfg_path)
            plate_format = plate_format or spec["plate_format"]
        if plate_format == "none":
            plate_format = None
        if device in ("auto", "cuda"):
            # onnxruntime-gpu >= 1.21 can load CUDA/cuDNN DLLs shipped via pip (nvidia-* wheels)
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - absent libs just means we fall back to CPU
                pass
        self.min_ocr_conf = min_ocr_conf
        providers = _pick_providers(device)
        self.alpr = self._build(detector_model, ocr_model, ocr_config, det_conf, providers)
        try:
            _self_test(self.alpr)
        except Exception as e:  # noqa: BLE001
            if providers == ["CPUExecutionProvider"] or device == "cuda":
                raise
            # A provider can be "available" yet unusable at run time (e.g. missing cuDNN DLLs).
            # fast-alpr swallows that and returns zero plates, so fail over explicitly.
            log.warning("GPU self-test failed, falling back to CPU: %s", str(e).splitlines()[0])
            providers = ["CPUExecutionProvider"]
            self.alpr = self._build(detector_model, ocr_model, ocr_config, det_conf, providers)
            _self_test(self.alpr)
        self.providers = providers
        if plate_format:
            self.alpr.ocr = FormatOCR(self.alpr.ocr, plate_format)
        log.info("Running on: %s", providers[0])

    @staticmethod
    def _build(
        detector_model: str, ocr_model: str, ocr_config: str | None, det_conf: float, providers: list[str]
    ) -> ALPR:
        ocr_kwargs: dict = {"ocr_model": ocr_model}
        if ocr_model.endswith(".onnx"):
            model_path = Path(ocr_model)
            if ocr_config is None:
                yamls = list(model_path.parent.glob("*.yaml"))
                if len(yamls) != 1:
                    raise ValueError(f"Pass ocr_config: expected one *.yaml next to {model_path}, found {len(yamls)}")
                ocr_config = str(yamls[0])
            ocr_kwargs = {"ocr_model": None, "ocr_model_path": model_path, "ocr_config_path": ocr_config}
        return ALPR(
            detector_model=detector_model,
            detector_conf_thresh=det_conf,
            detector_providers=providers,
            ocr_providers=providers,
            **ocr_kwargs,
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

    def read_crop(self, plate_img: np.ndarray) -> tuple[str, float]:
        """OCR only, for an image that is already a cropped plate. Returns (text, confidence)."""
        r = self.alpr.ocr.predict(plate_img)
        if r is None or not r.text:
            return "", 0.0
        conf = statistics.mean(r.confidence) if isinstance(r.confidence, list) else r.confidence
        return r.text.replace("_", "").strip(), float(conf)


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
    ap.add_argument("--ocr", default=DEFAULT_OCR, help="Hub model name, published model (india-v1) or path to a .onnx")
    ap.add_argument("--ocr-config", help="Plate config YAML for a custom --ocr .onnx")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--plate-format", choices=[*sorted(FormatOCR.DECODERS), "none"],
                    help="Constrain reads to a plate format (published models set a default; 'none' disables)")
    ap.add_argument("--det-conf", type=float, default=0.4, help="Detector confidence threshold")
    ap.add_argument("--min-ocr-conf", type=float, default=0.0, help="Drop reads below this OCR confidence")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = _collect(args.inputs)
    if not paths:
        log.error("No images found.")
        return 1

    t0 = time.perf_counter()
    reader = PlateReader(args.detector, args.ocr, args.device, args.det_conf, args.min_ocr_conf, args.ocr_config, args.plate_format)
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
