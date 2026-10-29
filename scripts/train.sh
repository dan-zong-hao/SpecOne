#!/usr/bin/env bash
# Train SpecOne (paper Sec. IV): full fine-tune of a OneJev base model on utility-soft targets with contrast pairs.
#
#   ONEJEV=/path/to/OneJev DATA=/path/to/jampivot RUNS=./runs \
#     scripts/train.sh <base model dir> <run name> [extra train.sft flags]
#
# SpecOne-0.8B:  scripts/train.sh models/OneJev1.6-0.8B specone_0.8b
# SpecOne-4B:    scripts/train.sh models/OneJev1.6-4B specone_4b \
#                  --fwd-split 2 --attn-impl flash_attention_2 --prefetch-workers 2 --tf32
#   (the 4B flags only change speed and memory and need patches/onejev-speed.patch; see README)
# Expects DATA/train_softpairs.jsonl and DATA/dev.jsonl (specone/build_train.py) and the images under DATA/media.
set -euo pipefail
MODEL=$1; NAME=$2; shift 2
: "${ONEJEV:?set ONEJEV to the OneJev checkout}"
: "${DATA:?set DATA to the JamPivot folder}"
RUNS=${RUNS:-runs}
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ONEJEV${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -m train.sft --config "$HERE/configs/specone.yaml" --model "$MODEL" --media-root "$DATA" \
  --runs-dir "$RUNS" --train "$DATA/train_softpairs.jsonl" --name "$NAME" \
  --eval dev="$DATA/dev.jsonl" "$@"
echo "weights: $RUNS/$NAME/final"
