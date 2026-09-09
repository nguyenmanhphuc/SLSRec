#!/usr/bin/env bash
# Run the SLSRec ablation x seed sweep, several jobs at a time on ONE GPU.
#
# The model is small (828k parameters, 56 train batches/epoch on ML-1M) and the
# kernels are tiny, so a single process leaves the GPU mostly idle -- packing
# several runs onto the same card is where the throughput comes from.
#
#   bash scripts/run_sweep.sh                       # A-D x 5 seeds, 4 at a time
#   JOBS=6 SEEDS=42 ABLATIONS=D bash scripts/run_sweep.sh
#   DATASET=beauty bash scripts/run_sweep.sh
#   DRY_RUN=1 bash scripts/run_sweep.sh            # print the commands, run nothing
#
# Restartable: a run whose test_metrics.json already exists is skipped, so
# re-running after a crash (or a pod restart) only picks up what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43 44 45 46}"
ABLATIONS="${ABLATIONS:-A B C D}"
JOBS="${JOBS:-4}"                  # concurrent training processes on the GPU
DATASET="${DATASET:-ml-1m}"
NUM_WORKERS="${NUM_WORKERS:-2}"    # keep constant across runs: it perturbs the RNG
LOGDIR="${LOGDIR:-runlogs/$DATASET}"

export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"

flags_for() {
    case "$1" in
        A) echo "--use_sam_gate=false --sam_conv=false --use_perturbation=false --use_uncertainty=false" ;;
        B) echo "--use_perturbation=false --use_uncertainty=false" ;;
        C) echo "--use_perturbation=true --use_uncertainty=false" ;;
        D) echo "" ;;
        *) echo "unknown ablation: $1" >&2; return 1 ;;
    esac
}

name_for() {
    case "$1" in
        A) echo A_bdlru_ce ;;
        B) echo B_sam_gate ;;
        C) echo C_perturbation ;;
        D) echo D_full_slsrec ;;
        *) echo "unknown ablation: $1" >&2; return 1 ;;
    esac
}

JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT

for ab in $ABLATIONS; do
    for s in $SEEDS; do
        exp="$(name_for "$ab")_${DATASET}_seed${s}"
        if [ -f "experiments/slsrec/$exp/test_metrics.json" ]; then
            echo "[skip] $exp -- already finished"
            continue
        fi
        echo "python train.py --dataset_code=$DATASET --model_code=slsrec --seed=$s $(flags_for "$ab") --num_workers=$NUM_WORKERS --exp_name=$exp > $LOGDIR/$exp.log 2>&1" >> "$JOBFILE"
    done
done

n=$(wc -l < "$JOBFILE" | tr -d ' ')
if [ "$n" -eq 0 ]; then
    echo "[sweep] nothing to do"
    exit 0
fi

if [ -n "${DRY_RUN:-}" ]; then
    echo "[sweep] DRY_RUN -- $n run(s) that would be launched, $JOBS at a time:"
    cat "$JOBFILE"
    exit 0
fi

# Warm the preprocessing cache in a single process first.  Without this the
# parallel jobs would race to download and write the same dataset.pkl.
echo "[sweep] warming the $DATASET dataset cache"
python - "$DATASET" <<'PY'
import sys
dataset = sys.argv[1]
sys.argv = ['warm', '--dataset_code=' + dataset, '--model_code=slsrec', '--num_workers=0']
from config import args, set_template
set_template(args)
from dataloader import dataloader_factory
train, val, test = dataloader_factory(args)
print('[warm] %d train batches, %d val batches, %d items'
      % (len(train), len(val), args.num_items))
PY

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[sweep] $n run(s), $JOBS at a time, logs in $LOGDIR/"
date

# xargs keeps the remaining jobs going when one fails, then exits non-zero;
# catch that here so the summary still prints under `set -e`.
if xargs -P "$JOBS" -I CMD bash -c CMD < "$JOBFILE"; then
    status="all $n run(s) finished"
else
    status="some runs FAILED -- grep -l Traceback $LOGDIR/*.log"
fi

date
echo "[sweep] $status"
echo "[sweep] next: python scripts/collect_results.py"
