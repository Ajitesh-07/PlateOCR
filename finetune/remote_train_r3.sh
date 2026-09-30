#!/usr/bin/env bash
# Round 3 fine-tune on a Linux GPU box (vast.ai). Same recipe as rounds 1-2 except:
#   - data: real + abtexp synthetic + 15k siddheshmm synthetic (train_synth2.csv)
#   - checkpoint / early stopping on val plate accuracy (default metric) instead of val_loss
#   - multiprocessing data loading (works on Linux, hangs on Windows)
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/ocr/finetune
export KERAS_BACKEND=torch
# Pretrained weights are not in git (*.keras is ignored); fetch them on a fresh clone.
[ -f cct_xs_v2_global.keras ] || curl -fsSL -o cct_xs_v2_global.keras \
  https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.keras
exec fast-plate-ocr train \
  --model-config-file cct_xs_v2.yaml \
  --plate-config-file cct_xs_v2_global_plate_config.yaml \
  --annotations data/train_synth2.csv \
  --val-annotations data/val.csv \
  --weights-path cct_xs_v2_global.keras \
  --epochs 40 --batch-size 128 --lr 0.0003 \
  --workers 14 --use-multiprocessing --max-queue-size 32 \
  --early-stopping-patience 10 \
  --output-dir runs/india_xs_v2_synth2 --seed 42
