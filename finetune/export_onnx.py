"""
Export a fine-tuned fast-plate-ocr .keras model (trained with the torch backend) to ONNX.

`fast-plate-ocr export` fails on the torch backend: Keras falls back to the TorchScript exporter,
which can't handle the blur-pooling conv. This calls torch's dynamo exporter directly and checks
the result against the Keras model.

Output matches what fast-plate-ocr inference expects: uint8 NHWC input named "input", batch 1,
output named "plate". Writes <out>/<name>.onnx and a plate config without regions (the fine-tuned
model has no region head).

The torch export keeps Keras' attention/dense einsums as ONNX `Einsum` nodes, which ONNX Runtime
runs slowly; they are rewritten as `MatMul` (finetune/onnx_matmul.py, ~6x faster on CPU) unless
--keep-einsum is given.

Usage (from repo root):
    set KERAS_BACKEND=torch
    python finetune/export_onnx.py --model finetune/runs/.../best.keras --out models/india_ocr
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # export on CPU; mixed cuda/cpu tensors break torch.export

import keras  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import fast_plate_ocr.train.model.layers  # noqa: E402,F401  (registers custom layers for loading)
from onnx_matmul import einsum_to_matmul  # noqa: E402


class _Wrapper(torch.nn.Module):
    def __init__(self, model: keras.Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, training=False)["plate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--plate-config", type=Path, help="Defaults to plate_config.yaml next to --model")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="india_cct_xs_v2")
    ap.add_argument("--keep-einsum", action="store_true", help="Skip the Einsum -> MatMul rewrite")
    args = ap.parse_args()

    cfg_path = args.plate_config or args.model.with_name("plate_config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.pop("plate_regions", None)
    h, w = cfg["img_height"], cfg["img_width"]
    c = 1 if cfg.get("image_color_mode") == "grayscale" else 3

    model = keras.saving.load_model(args.model, compile=False)
    assert "plate" in model.output, f"expected a 'plate' output, got {list(model.output)}"
    wrapper = _Wrapper(model).eval()

    dummy = torch.randint(0, 256, (1, h, w, c), dtype=torch.uint8)
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / f"{args.name}.onnx"
    with torch.no_grad():
        prog = torch.onnx.export(wrapper, (dummy,), dynamo=True, input_names=["input"], output_names=["plate"])
    prog.optimize()
    prog.save(str(onnx_path))
    if not args.keep_einsum:
        n = einsum_to_matmul(onnx_path, onnx_path)
        print(f"rewrote {n} Einsum nodes as MatMul")

    (args.out / f"{args.name}_plate_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Check ONNX matches Keras on random inputs.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randint(0, 256, (1, h, w, c), dtype=np.uint8)
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(x)).cpu().numpy()
    out = sess.run(["plate"], {"input": x})[0]
    diff = float(np.abs(ref - out).max())
    print(f"saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)  max|keras-onnx| = {diff:.2e}")
    # fp32 kernel differences give ~1e-4..3e-3 on softmax outputs; checked on 400 real crops for
    # the cct_s_v2 export: 99.97% per-char argmax agreement at max|diff| 2.9e-3.
    assert diff < 5e-3, "ONNX output does not match Keras"


if __name__ == "__main__":
    main()
