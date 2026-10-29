#!/usr/bin/env bash
# Chained scoring and metrics of a trained SpecOne model on the JamPivot test splits, then the certified gate.
#
#   ONEJEV=/path/to/OneJev DATA=/path/to/jampivot scripts/evaluate.sh <model dir> <name> [preds dir]
set -euo pipefail
MODEL=$1; NAME=$2; PREDS=${3:-preds}
: "${ONEJEV:?set ONEJEV to the OneJev checkout}"
: "${DATA:?set DATA to the JamPivot folder}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ONEJEV${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PREDS"
for S in test_id ood_combo ood_style pair_test; do
  python "$HERE/specone/score.py" --model "$MODEL" --data "$DATA" --split $S --out "$PREDS/${NAME}_$S.jsonl"
  python "$HERE/specone/metrics.py" --data "$DATA" --split $S --preds "$PREDS/${NAME}_$S.jsonl" \
    --json "$PREDS/${NAME}_$S.metrics.json"
done
python "$HERE/specone/selective.py" --data "$DATA" --split test_id --preds "$NAME=$PREDS/${NAME}_test_id.jsonl" \
  --out "$PREDS/${NAME}_selective.json"
python "$HERE/specone/gate.py" --data "$DATA" --preds "$PREDS/${NAME}_test_id.jsonl" \
  --apply ood_combo="$PREDS/${NAME}_ood_combo.jsonl" ood_style="$PREDS/${NAME}_ood_style.jsonl" \
          pair_test="$PREDS/${NAME}_pair_test.jsonl" --out "$PREDS/${NAME}_gate.json"
