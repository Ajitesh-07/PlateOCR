# Model card: `india-v1` (Indian license plate OCR)

An OCR model for **Indian** license plates, fine-tuned from `fast-plate-ocr`'s `cct-s-v2-global-model`. It reads the text on a cropped plate image. Use it together with the YOLOv9 plate detector and `--plate-format india`.

| | |
|---|---|
| **Name / version** | `india-v1` (GitHub Release tag `india-ocr-v1`) |
| **Files** | `india_ocr_v1.onnx` (4.6 MB), `india_ocr_v1_plate_config.yaml` |
| **Architecture** | CCT-S (Compact Convolutional Transformer), 10 character slots, alphabet `0-9 A-Z`, pad `_`. No region head. |
| **Base model** | `cct_s_v2_global.keras` from [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) (MIT) |
| **Trained** | 2026-09-30, round 5 in [RESEARCH.md](RESEARCH.md#india-fine-tuning-results) |

## Intended use

- **For:** reading Indian registration plates (`MH12AB1234`, `DL3CAB1234`, `22BH1234AA`) from plate crops produced by a detector.
- **Not for:** non-Indian plates. Fine-tuning made it forget most foreign formats (US exact match dropped from 83% to 26%). Use the global model for other regions.

## Performance

**Test set:** 1,684 real Indian plate crops from about 30 states (46% Maharashtra), from [zenitsu09/indian-number-plate](https://huggingface.co/datasets/zenitsu09/indian-number-plate). No training plate shares its text with a test plate.

| | Exact plate match | Characters correct |
|---|---|---|
| Global `cct-xs-v2` (baseline) | 35.4% | 81.2% |
| **`india-v1`** | 72.0% | 92.6% |
| **`india-v1` + `plate_format="india"`** | **72.7%** | **92.7%** |

- **Margin of error:** about ±2.2 points (95%).
- **Ceiling:** about 6.7% of the test labels don't follow any valid Indian format and look like labelling errors, so the test set can't show much more than about 93%.
- **Validation:** 92.6% exact on held-out plates similar to the training data (mostly Gujarat).
- **Latency:** about 8.5 ms per plate crop through `PlateReader` on an RTX 4060 laptop (onnxruntime-gpu). The model alone is under 1 ms.

## Training data

All training sources allow redistribution with attribution:

| Source | Licence | Used |
|---|---|---|
| [avinashjadjasadf/indian-vehicle-number-plate-dataset-6](https://www.kaggle.com/datasets/avinashjadjasadf/indian-vehicle-number-plate-dataset-6) | Apache-2.0 | 10,653 real crops |
| [kp00011/indian-vehical-number-plate-ocr-labeled-dataset](https://www.kaggle.com/datasets/kp00011/indian-vehical-number-plate-ocr-labeled-dataset) | MIT | 17 real crops (the rest overlapped the test set and were removed) |
| [abtexp/synthetic-indian-license-plates](https://www.kaggle.com/datasets/abtexp/synthetic-indian-license-plates) | CC0-1.0 | 12,026 synthetic plates, all 36 states and UTs |

The data is rebuilt with `finetune/build_dataset.py --synthetic`, and the model is trained with `finetune/remote_train.sh cct_s_v2 data/train_synth.csv <run>`.

## Known limitations

- **Mostly Gujarat plates in training.** Accuracy varies by state, roughly 60–84% on the test set.
- **Older plates with 3 final digits** (e.g. `KL 34 A 465`): with `plate_format="india"`, the decoder forces 4 digits and may add one.
- **Non-standard plates** (temporary, diplomatic, army): not covered by the Indian format decoder.
- **Low-confidence reads are often wrong.** Set `min_ocr_conf` (e.g. 0.5) in production.
- **Not yet tested on the deployment cameras.** Expect different accuracy on your own footage; fine-tuning on it is the recommended next step.

## Privacy

License plates are personal data in many jurisdictions. Set retention and access rules for stored reads.
