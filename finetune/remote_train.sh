#!/usr/bin/env bash
# Fine-tune fast-plate-ocr on a Linux GPU box (vast.ai), starting from the pretrained global model.
#
#   ./remote_train.sh MODEL TRAIN_CSV RUN_NAME
#   ./remote_train.sh cct_xs_v2 data/train_synth2.csv india_xs_v2_synth2   # round 3
#   ./remote_train.sh cct_s_v2  data/train_synth2.csv india_s_v2_synth2    # round 4
#
# Checkpoint + early stopping on val plate accuracy (default metric). Multiprocessing data loading
# works on Linux (it hangs on Windows). Both cct_*_v2 models share the same plate config.
set -euo pipefail
MODEL=${1:?model, e.g. cct_xs_v2}
TRAIN_CSV=${2:?train csv, e.g. data/train_synth2.csv}
RUN=${3:?run name}
source /venv/main/bin/activate
cd "$(dirname "$0")"
export KERAS_BACKEND=torch
REL=https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates
# Pretrained weights are not in git (*.keras is ignored); fetch them on a fresh clone.
[ -f "${MODEL}_global.keras" ] || curl -fsSL -o "${MODEL}_global.keras" "$REL/${MODEL}_global.keras"
exec fast-plate-ocr train \
  --model-config-file "${MODEL}.yaml" \
  --plate-config-file cct_xs_v2_global_plate_config.yaml \
  --annotations "$TRAIN_CSV" \
  --val-annotations data/val.csv \
  --weights-path "${MODEL}_global.keras" \
  --epochs 40 --batch-size 128 --lr 0.0003 \
  --workers 14 --use-multiprocessing --max-queue-size 32 \
  --early-stopping-patience 10 \
  --output-dir "runs/$RUN" --seed 42
