#!/usr/bin/env bash
# Gate A/B experiment: original SLSRec gate vs adaptive gates.
#
# Same conventions as scripts/run_sweep.sh (restartable, JOBS at a time, DRY_RUN).
#
#   bash scripts/run_gate_sweep.sh                      # 6 variants x 3 seeds
#   VARIANTS="sum token" SEEDS=42 bash scripts/run_gate_sweep.sh
#   DRY_RUN=1 bash scripts/run_gate_sweep.sh
#
# Every variant is neutral-initialised, so all of them start from *exactly* the
# original SLSRec layer (tests/test_adaptive_gate.py proves this bit-for-bit).
# The only thing that differs is what each gate is allowed to learn.
#
# Read the variants as a ladder:
#   sum      original eq.(3), fixed 50/50                        <- baseline
#   scalar   one learned coefficient per layer, static           <- capacity control
#   channel  one learned coefficient per channel, static         <- capacity control
#   token    coefficient depends on the item at that position    <- adaptive
#   context  coefficient also sees both branch outputs           <- adaptive, branch-aware
#   fusion   adaptive strength of G gating H (eq. 6 instead)     <- the other axis
#
# If `token` beats `sum` but `scalar`/`channel` do not, the gain came from
# adaptivity.  If `scalar` already captures most of it, the gain was just a
# better fixed balance and the adaptive machinery is not earning its keep.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43 44}"
VARIANTS="${VARIANTS:-sum scalar channel token context fusion}"
JOBS="${JOBS:-4}"
DATASET="${DATASET:-ml-1m}"
NUM_WORKERS="${NUM_WORKERS:-2}"    # keep constant across runs: it perturbs the RNG
GRANULARITY="${GRANULARITY:-channel}"
EXTRA="${EXTRA:-}"                 # e.g. EXTRA="--uncertainty_loss=exact"
LOGDIR="${LOGDIR:-runlogs/gates_$DATASET}"
PYTHON="${PYTHON:-python}"   # e.g. PYTHON=/c/Users/.../envs/SemADR/python.exe on Windows

export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"

flags_for() {
    case "$1" in
        sum)     echo "--gate_mode=sum" ;;
        scalar)  echo "--gate_mode=scalar" ;;
        channel) echo "--gate_mode=channel" ;;
        token)   echo "--gate_mode=token   --gate_granularity=$GRANULARITY" ;;
        context) echo "--gate_mode=context --gate_granularity=$GRANULARITY" ;;
        fusion)  echo "--fusion_mode=adaptive --fusion_source=token --gate_granularity=$GRANULARITY" ;;
        both)    echo "--gate_mode=token --fusion_mode=adaptive --gate_granularity=$GRANULARITY" ;;
        *) echo "unknown variant: $1" >&2; return 1 ;;
    esac
}

JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT

for v in $VARIANTS; do
    for s in $SEEDS; do
        exp="gate_${v}_${DATASET}_seed${s}"
        if [ -f "experiments/slsrec/$exp/test_metrics.json" ]; then
            echo "[skip] $exp -- already finished"
            continue
        fi
        echo "$PYTHON train.py --dataset_code=$DATASET --model_code=slsrec --seed=$s $(flags_for "$v") $EXTRA --num_workers=$NUM_WORKERS --exp_name=$exp > $LOGDIR/$exp.log 2>&1" >> "$JOBFILE"
    done
done

n=$(wc -l < "$JOBFILE" | tr -d ' ')
if [ "$n" -eq 0 ]; then echo "[gates] nothing to do"; exit 0; fi

if [ -n "${DRY_RUN:-}" ]; then
    echo "[gates] DRY_RUN -- $n run(s) that would be launched, $JOBS at a time:"
    cat "$JOBFILE"
    exit 0
fi

echo "[gates] warming the $DATASET dataset cache"
"$PYTHON" - "$DATASET" <<'PY'
import sys
dataset = sys.argv[1]
sys.argv = ['warm', '--dataset_code=' + dataset, '--model_code=slsrec', '--num_workers=0']
from config import args, set_template
set_template(args)
from dataloader import dataloader_factory
train, val, test = dataloader_factory(args)
print('[warm] %d train batches, %d items' % (len(train), args.num_items))
PY

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[gates] no GPU detected -- this will be slow"
echo "[gates] $n run(s), $JOBS at a time, logs in $LOGDIR/"
date

if xargs -P "$JOBS" -I CMD bash -c CMD < "$JOBFILE"; then
    status="all $n run(s) finished"
else
    status="some runs FAILED -- grep -l Traceback $LOGDIR/*.log"
fi

date
echo "[gates] $status"
echo "[gates] next: $PYTHON scripts/compare_gates.py"
