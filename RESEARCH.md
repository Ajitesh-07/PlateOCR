# License plate recognition: model research (Sept 2026)

## Recommendation

Use **[FastALPR](https://github.com/ankandrew/fast-alpr)**, which is a two-stage pipeline:

| Stage | Model | Source |
|---|---|---|
| Plate detection | YOLOv9 (end-to-end, NMS baked in), ONNX | [open-image-models](https://github.com/ankandrew/open-image-models) |
| Plate OCR | CCT (Compact Convolutional Transformer), ONNX, global multi-country + region head | [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) |

Why this stack:
- **Built for plates.** Both models are trained specifically for plates. General OCR engines (Tesseract, EasyOCR, PaddleOCR) are trained for text in general and need extra work to handle plate fonts, 2-line plates and missing dictionaries.
- **Accuracy.** A 2026 comparison study found that YOLO11x + FastPlateOCR had the highest transcription accuracy of the pipelines it tested. It was also about 13× faster than the best VLM (Qwen3-VL). Zero-shot VLMs were only better at counting plates per image. ([MDPI Electronics 15(18):4242](https://www.mdpi.com/2079-9292/15/18/4242))
- **Deployable.** Everything runs on ONNX Runtime (CPU, CUDA, TensorRT, OpenVINO, DirectML, QNN). There is no PyTorch or TensorFlow at runtime, so Docker images stay small.
- **Fast.** OCR takes about 0.5–0.7 ms per plate on a GPU (roughly 1.5–3k plates/s). The detector is a tiny or small YOLOv9.
- **MIT license** for both the code and the weights, so commercial use is fine.
- **Swappable.** You can plug in your own detector or OCR via `BaseDetector` / `BaseOCR`, or fine-tune the OCR on your country's plates with `fast-plate-ocr`'s training CLI.

## Model options

### Detectors

| Model | Input size | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `yolo-v9-s-608-license-plate-end2end` ← **default in infer.py** | 608 | 0.957 | 0.917 | **0.966** | **0.772** |
| `yolo-v9-t-640-license-plate-end2end` | 640 | 0.966 | 0.896 | 0.958 | 0.758 |
| `yolo-v9-t-512-license-plate-end2end` | 512 | 0.955 | 0.901 | 0.948 | 0.724 |
| `yolo-v9-t-416-license-plate-end2end` | 416 | 0.940 | 0.894 | 0.940 | 0.702 |
| `yolo-v9-t-384-license-plate-end2end` ← FastALPR's default | 384 | 0.942 | 0.863 | 0.920 | 0.687 |
| `yolo-v9-t-256-license-plate-end2end` | 256 | 0.937 | 0.797 | 0.858 | 0.606 |

A larger input size helps with small or distant plates. Use `t-384` or `t-256` on edge devices.

### OCR

| Model | GPU latency | Notes |
|---|---|---|
| `cct-s-v2-global-model` ← **default in infer.py** | 0.68 ms | Most accurate; v2 adds region prediction and export-friendly activations |
| `cct-xs-v2-global-model` | 0.47 ms | FastALPR's default |
| `cct-s-v1` / `cct-xs-v1` | 0.59 / 0.32 ms | Older and faster; no region output |
| `european-plates-mobile-vit-v2-model`, `argentinian-plates-*` | — | Region-specific |

Latency numbers are from the fast-plate-ocr README, measured on an RTX 3090.

## Measured results (OpenALPR end-to-end benchmark, RTX 4060 Laptop, CUDA)

The benchmark has 444 images (108 EU, 114 BR, 222 US). "Exact" means the whole plate string matched. Latency covers the whole image (detection + OCR) and is measured by `evaluate.py`.

| Config | Plate found | Exact | Exact (O=0) | Characters correct | EU / BR / US exact | Median ms | p95 ms |
|---|---|---|---|---|---|---|---|
| `s-608` + `cct-s-v2` (default) | 100% | **90.8%** | **93.7%** | 98.2% | 92.6 / 98.2 / 86.0 | ~22 | ~27 |
| `t-384` + `cct-xs-v2` (fast) | 99.8% | 89.0% | 91.7% | 97.7% | 93.5 / 95.6 / 83.3 | ~15 | ~20–33 |

On CPU (laptop), the default config takes about 355 ms per image. Every error comes from OCR, not detection. About a third of the default config's errors are O vs 0 only. The rest are look-alike characters (8/B, 6/G, 1/I, W/K) and vanity plates, and they are concentrated in US plates. Reproduce with `python evaluate.py --sets eu br us`.

### India: out of the box, it does not work well enough

| Set | Plates | Found | Exact | Exact (O=0) | Characters correct |
|---|---|---|---|---|---|
| `in_full`: Datacluster sample, full photos | 25 | 80% | 32% | 32% | 67% |
| `in_crops`: plate crops, ~30 states, OCR only, `cct-s-v2` | 1684 | — | 31.2% | 37.1% | 77.3% |
| `in_crops` with `cct-xs-v2` (best pretrained OCR) | 1684 | — | 35.4% | 40.0% | 81.2% |
| `in_crops` with the other models (`cct-*-v1`, `mobile-vit-v2`) | 1684 | — | 5–8% | — | 51–67% |

Why:
- **Not trained on India.** India is not among the ~65 regions in the OCR model's training set (see `plate_regions` in the model config).
- **Plate length.** Indian plates are usually 10 characters, which is the model's maximum. It often stops a character early: 125 plates were read as exactly their first 9 characters.
- **Hard plates.** Indian plates add non-standard fonts, dots between groups, two-line layouts and yellow commercial plates.

Indian plate-format rules (letter/digit fixes by position) only raise exact accuracy from 31% to about 37%.

**For India, fine-tune the OCR on Indian plates** using `fast-plate-ocr`'s training CLI, starting from `cct-xs-v2`. The detector is less of a concern (80% found on the small full-photo sample), but it should be checked on a larger set.

## Alternatives considered

| Option | Verdict |
|---|---|
| **YOLOv8/11 + PaddleOCR (PP-OCRv5/v6)** | Strong general OCR; PP-OCRv6 (2026) beats large VLMs on general OCR. For plates it is heavier (a Paddle runtime or a conversion step) and needs rules to clean up plate text. A good fallback if you also need non-plate text. |
| **YOLO + EasyOCR / Tesseract** | Common in tutorials. Noticeably worse on plates, and EasyOCR pulls in PyTorch. Not recommended. |
| **VLMs (Qwen2.5/3-VL, Florence-2, GPT/Gemini)** | Work zero-shot and are robust to odd layouts. They are about 10×+ slower, need a GPU or a paid API, can hallucinate characters, and are lower on exact-match accuracy (e.g. Qwen2.5-VL at ~77%). Useful as an offline second opinion for low-confidence reads. |
| **OpenALPR (open-source)** | Legacy and mostly unmaintained; the modern version is commercial (Rekor). |
| **Plate Recognizer / Sighthound (commercial)** | Very accurate, with regional tuning and vehicle make/model/colour. Paid per call or per camera. The benchmark to beat if accuracy matters more than cost. |

## Deployment notes

- **Runtime:** `pip install "fast-alpr[onnx-gpu]"` on an NVIDIA server, `fast-alpr[onnx]` for CPU, and `[onnx-openvino]` / `[onnx-directml]` / `[onnx-qnn]` for other hardware. `infer.py` picks CUDA → DirectML → CPU automatically.
- **Load once:** create `PlateReader()` once per worker process. It warms up at startup, so the first request isn't slow. Then call `.read(frame)` for each image.
- **Thresholds:** use `--det-conf` (default 0.4) to control missed vs. false plates, and `--min-ocr-conf` to drop weak reads. In production, log low-confidence reads for review.
- **Video:** run on every Nth frame and vote on the text across frames for each track. This fixes most single-frame OCR errors.
- **Accuracy uplift:** the biggest gain usually comes from fine-tuning the OCR on a few thousand crops from your own cameras and country, using `fast-plate-ocr`'s training CLI. Add a regex check for your country's plate format.
- **Privacy:** plates are personal data in many regions (e.g. GDPR). Define how long results are kept and who can access them.

## Sources

- FastALPR: https://github.com/ankandrew/fast-alpr
- open-image-models (detector metrics): https://github.com/ankandrew/open-image-models
- fast-plate-ocr (OCR models/latency): https://github.com/ankandrew/fast-plate-ocr
- YOLO-OCR vs VLMs for car plates (2026): https://www.mdpi.com/2079-9292/15/18/4242
- PP-OCRv6: https://arxiv.org/html/2606.13108v1
- PaddleOCR 3.0 tech report: https://arxiv.org/html/2507.05595v1
- EasyOCR vs PaddleOCR vs Tesseract with YOLOv8: https://ieeexplore.ieee.org/iel8/10723818/10723316/10725878.pdf
- Plate Recognizer's 2026 ALPR overview: https://platerecognizer.com/top-10-alpr-software-solutions/
