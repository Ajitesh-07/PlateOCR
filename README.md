# License Plate Recognition (ALPR)

Finds the license plates on cars in an image and reads their text.

It runs in two stages on ONNX Runtime, using [FastALPR](https://github.com/ankandrew/fast-alpr) (MIT license):

1. **Detect:** a YOLOv9 model finds the plates and returns their boxes.
2. **Read:** each plate is cropped and a CCT OCR model reads the text. It also predicts the plate's country or region.

```
image ──► YOLOv9 plate detector ──► crop ──► CCT plate OCR ──► "5AU5341" (0.99), bbox, region
```

See [RESEARCH.md](RESEARCH.md) for why this stack was chosen, the other models considered, and full benchmark numbers.

## Results

Measured on the [OpenALPR end-to-end benchmark](https://github.com/openalpr/benchmarks): 444 real car photos from the EU, Brazil and the US. The GPU is an RTX 4060 Laptop.

| Models | Plate found | Plate read exactly | Ignoring O vs 0 | Characters correct | Median latency (GPU) |
|---|---|---|---|---|---|
| **Default:** `yolo-v9-s-608` + `cct-s-v2` | 100% | 90.8% | 93.7% | 98.2% | ~22 ms |
| Fast: `yolo-v9-t-384` + `cct-xs-v2` | 99.8% | 89.0% | 91.7% | 97.7% | ~15 ms |

- **By region (default models):** EU 92.6% (99.1% ignoring O vs 0), Brazil 98.2%, US 86.0%.
- **CPU:** about 355 ms per image with the default models.

### India

The global OCR model reads only **35%** of Indian plates exactly, because it was never trained on them. A version fine-tuned on Indian plates, combined with Indian plate-format decoding, reads **72.7%** exactly (92.6% of characters correct) on a held-out set from about 30 states. Full details are in [RESEARCH.md](RESEARCH.md#india-fine-tuning-results).

| Setup | Indian test set (exact) | Latency per plate (RTX 4060 laptop) |
|---|---|---|
| Global `cct-xs-v2` | 35.4% | 6.4 ms |
| India `cct-s-v2` (round 5) + `--plate-format india` | **72.7%** | 8.5 ms |

The India model is **India-only**: it forgets many non-Indian plates. Use the global model for other regions. Its data, metrics and limitations are in [MODEL_CARD.md](MODEL_CARD.md), and usage is under [Using the India model](#using-the-india-model).

## Setup

You need Python 3.10 or newer. For the GPU, you need an NVIDIA driver that supports CUDA 13; check the "CUDA Version" line in `nvidia-smi`. You do **not** need a system-wide CUDA toolkit: the CUDA and cuDNN libraries install as pip packages (about 1.15 GB).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # GPU (CUDA 13 + cuDNN 9 wheels)
# CPU only:  pip install "fast-alpr[onnx]==0.4.0"
```

On first run the model weights (about 33 MB) are downloaded to your user cache folder:

- `~/.cache/open-image-models/yolo-v9-s-608-license-plate-end2end/` (detector)
- `~/.cache/fast-plate-ocr/cct-s-v2-global-model/` (OCR model and its config)

## Usage

### Command line

```bash
python infer.py car.jpg                                    # print plates
python infer.py images/ --out outputs/ --json results.json  # folder, annotated images, JSON
python infer.py car.jpg --device cpu                        # force CPU
python infer.py car.jpg --min-ocr-conf 0.7                  # drop low-confidence reads
python infer.py car.jpg --detector yolo-v9-t-384-license-plate-end2end --ocr cct-xs-v2-global-model  # fast models
python infer.py car.jpg --ocr india-v1.1                    # Indian plates (see below)
```

Example output:

```
samples\test_image.png  [27.0 ms]  5AU5341 (1.00), 44A4 (0.56)
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `auto` tries CUDA, then DirectML, then CPU. `cuda` fails if the GPU can't be used. `cpu` forces CPU. |
| `--det-conf` | `0.4` | Detector confidence threshold. Lower it to find more plates, at the cost of more false boxes. |
| `--min-ocr-conf` | `0.0` | Drop reads whose mean character confidence is below this value. |
| `--detector` / `--ocr` | `s-608` / `cct-s-v2` | Model names; the full list is in [RESEARCH.md](RESEARCH.md#model-options). `--ocr` also accepts a published model name (`india-v1.1`, `india-v1`) or a path to a `.onnx`. |
| `--ocr-config` | next to `.onnx` | Plate config YAML for a custom `--ocr` model. Only needed if there isn't exactly one `*.yaml` next to it. |
| `--plate-format` | none (`india` for `india-v1.x`) | `india` returns only plates that match the Indian format, choosing the most probable valid plate. `none` turns it off. |
| `--out` | none | Folder to write images with the plates boxed and labelled. |
| `--json` | none | File to write all results to as JSON. |

### Python

```python
import cv2
from infer import PlateReader

reader = PlateReader()                       # load once per process (~8 s on GPU incl. warm-up)
plates = reader.read(cv2.imread("car.jpg"))  # BGR ndarray or a file path

for p in plates:
    print(p.text, p.ocr_confidence, p.det_confidence, p.bbox, p.region)
```

Each result is a `Plate` with these fields:

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | The plate text, e.g. `"5AU5341"` |
| `ocr_confidence` | `float` (0–1) | Mean confidence across the plate's characters |
| `det_confidence` | `float` (0–1) | Confidence of the detector's box |
| `bbox` | `(x1, y1, x2, y2)` | Box position in pixels |
| `region` | `str` or `None` | Predicted country or region, e.g. `"Czech Republic"` |
| `region_confidence` | `float` or `None` | Confidence of the region prediction |

If you already have a cropped plate, from your own detector for example, `reader.read_crop(crop)` runs only the OCR and returns `(text, confidence)`.

## Using the India model

The India model is published as GitHub Releases of this repo, not stored in git:

| Name | Release | Notes |
|---|---|---|
| **`india-v1.1`** (recommended) | [`india-ocr-v1.1`](https://github.com/Ajitesh-07/PlateOCR/releases/tag/india-ocr-v1.1) | Same weights and predictions as `india-v1` (72.7% on `in_crops`). Its `Einsum` ops are rewritten as `MatMul`, making it ~2x faster on CPU (9.4 vs 19.3 ms per crop) and steadier on GPU (~4.9 ms vs 5-8 ms). |
| `india-v1` | [`india-ocr-v1`](https://github.com/Ajitesh-07/PlateOCR/releases/tag/india-ocr-v1) | Original export. Kept so pinned pipelines keep working. |

You don't need to download it yourself. Pass `india-v1.1` as the OCR model, and on first use `infer.py`:

1. downloads the model (4.6 MB) from the release;
2. checks its SHA-256 checksum (a corrupted or tampered file is refused);
3. caches it in `~/.cache/plate-ocr/india-v1.1/`;
4. turns on Indian plate-format decoding.

### Command line

```bash
python infer.py car.jpg --ocr india-v1.1                        # detect + read Indian plates
python infer.py frames/ --ocr india-v1.1 --min-ocr-conf 0.5 --json reads.json
python infer.py car.jpg --ocr india-v1.1 --plate-format none    # raw model output, no format rules
```

### Python (in your own pipeline)

```python
import cv2
from infer import PlateReader

reader = PlateReader(ocr_model="india-v1.1", min_ocr_conf=0.5) # load once per process

for p in reader.read(cv2.imread("car.jpg")):                  # full image: detect + OCR
    print(p.text, p.ocr_confidence, p.bbox)

text, conf = reader.read_crop(plate_crop_bgr)                 # you already have a plate crop
```

### Offline servers, mirrors and CI

| Environment variable | Default | Use |
|---|---|---|
| `PLATE_OCR_CACHE` | `~/.cache/plate-ocr` | Where published models are stored. Pre-fill it (or bake it into a Docker image) for servers without internet access. |
| `PLATE_OCR_MODEL_URL` | this repo's releases | Download base for a mirror. Files are fetched from `<base>/<release tag>/<file>`, e.g. `<base>/india-ocr-v1.1/india_ocr_v1_1.onnx`. |

To install it by hand, download the release files and put them in `$PLATE_OCR_CACHE/india-v1.1/`, or pass the file path directly: `--ocr path/to/india_ocr_v1_1.onnx --plate-format india`. `SHA256SUMS.txt` in the release lets you verify them (`sha256sum -c SHA256SUMS.txt`).

### Using the ONNX file directly (no `infer.py`)

For pipelines in other languages or runtimes:

| | |
|---|---|
| Input | `input`: `uint8`, shape `(1, 64, 128, 3)`, **RGB**, channels-last. Resize the plate crop to 128×64 (width × height) with bilinear interpolation, without keeping the aspect ratio. The model normalises pixel values itself. |
| Output | `plate`: `float32`, shape `(1, 10, 37)`: 10 character slots × softmax over `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_`. |
| Decode | Take the most likely character in each slot and strip the trailing `_` padding characters. Confidence is the mean of the chosen probabilities. For Indian format decoding, port `plate_format.decode_india` (about 50 lines). |

### Publishing a new version (maintainers)

1. **Train and export:** fine-tune (see `finetune/`), then export with `finetune/export_onnx.py`. The export rewrites `Einsum` as `MatMul` automatically (`finetune/onnx_matmul.py`, which also works on an existing `.onnx`).
2. **Package:** rename the files to `india_ocr_vN.onnx` and `india_ocr_vN_plate_config.yaml`. Write `SHA256SUMS.txt` and update `MODEL_CARD.md`.
3. **Release:** create a GitHub Release with tag `india-ocr-vN` and upload those files plus `MODEL_CARD.md`.
4. **Register:** add an `india-vN` entry to `CUSTOM_OCR_MODELS` in `infer.py`, using the new tag, file names and SHA-256 values. Keep the old entries, so pipelines pinned to `india-v1` keep working.

## Benchmarking

```bash
git clone --depth 1 https://github.com/openalpr/benchmarks.git data/openalpr_benchmarks
hf download Dataclusterlabspvtltd/indian-number-plates-dataset --repo-type dataset --local-dir data/indian_datacluster
hf download zenitsu09/indian-number-plate --repo-type dataset --local-dir data/indian_zenitsu
pip install pandas pyarrow   # needed for the in_crops set

python evaluate.py --sets eu br us in_full in_crops --failures outputs/failures.json
```

| Set | What it contains |
|---|---|
| `eu`, `br`, `us` | OpenALPR benchmark: full car photos, one plate each |
| `in_full` | Indian full photos, several plates each (Datacluster Labs sample, 25 plates with text labels) |
| `in_crops` | Indian plate crops from about 30 states (1,684 plates). Tests the OCR only; detection is skipped. |

This prints, for each set: the share of plates found, exact-match accuracy (also with O and 0 counted as the same), character accuracy, extra detections and latency (median and 95th percentile). It also writes every misread to the failures file so you can inspect them.

## Deployment notes

- **Load once:** create one `PlateReader` per worker process and reuse it for every request. Its constructor runs a warm-up, so the first request isn't slow.
- **GPU self-test:** on startup the reader runs every ONNX model directly on dummy input. FastALPR catches GPU errors itself (for example, missing cuDNN DLLs) and returns "no plate" instead of failing. The self-test catches that instead. With `--device auto` the reader logs a warning and falls back to CPU; with `--device cuda` it raises an error.
- **Offline or containers:** bake the two cache folders above into the image so nothing downloads at startup.
- **Accuracy:** all benchmark errors come from reading the text, not from finding plates. Most are look-alike characters: O/0, 8/B, 6/G, 1/I. To improve:
  - add plate-format rules (regex) for your country;
  - fine-tune the OCR on crops from your own cameras with [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr);
  - for video, vote across several frames.
- **Privacy:** license plates count as personal data in many places (for example under GDPR). Decide how long results are kept and who can access them.

## Project layout

```
infer.py          inference: PlateReader class + CLI
plate_format.py   format-constrained decoding (Indian plate grammar)
evaluate.py       benchmarks: OpenALPR (eu/br/us) + Indian sets
MODEL_CARD.md     india-v1 / v1.1: training data, metrics, limitations
finetune/         fine-tuning: build_dataset.py, remote_train.sh, export_onnx.py, model configs
RESEARCH.md       model research, alternatives, measured results
requirements.txt  pinned dependencies (GPU)
samples/          test image
outputs/          annotated images, results JSON, failure lists
data/             benchmark dataset (not needed at runtime)
```
