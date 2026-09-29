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
| `--detector` / `--ocr` | `s-608` / `cct-s-v2` | Model names; the full list is in [RESEARCH.md](RESEARCH.md#model-options). |
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

## Benchmarking

```bash
git clone --depth 1 https://github.com/openalpr/benchmarks.git data/openalpr_benchmarks
python evaluate.py --sets eu br us --failures outputs/failures.json
```

This prints plate-found rate, exact-match accuracy, character accuracy and latency (median and 95th percentile) for each set. It also writes every misread to the failures file so you can inspect them.

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
evaluate.py       benchmark on the OpenALPR dataset
RESEARCH.md       model research, alternatives, measured results
requirements.txt  pinned dependencies (GPU)
samples/          test image
outputs/          annotated images, results JSON, failure lists
data/             benchmark dataset (not needed at runtime)
```
