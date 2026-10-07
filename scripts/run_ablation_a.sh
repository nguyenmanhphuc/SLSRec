#!/usr/bin/env bash
# Ablation A: BD-LRU + CE, i.e. "what does RecBLR get in this harness?"
#
# Same conventions as scripts/run_gate_sweep.sh (restartable, JOBS at a time, DRY_RUN).
#
#   bash scripts/run_ablation_a.sh                       # 3 variants x 3 seeds
#   VARIANTS=recblr SEEDS=42 bash scripts/run_ablation_a.sh
#   DRY_RUN=1 bash scripts/run_ablation_a.sh
#
# Variants (all: no SAM, no Conv1-D1 gate, no perturbation, no uncertainty):
#   slsrec      SLSRec skeleton with RecBLR's SiLU(Linear) gate branch: keeps P in
#               eq.(1), the GeGLU PFFN of eq.(13) and c=8 in eq.(11)
#   recblr      RecBLR (CIKM'24) architecture: no positional embedding (its eq. 1),
#               SiLU MLP feed-forward (its eq. 4), a_t = exp(-softplus(L) r_t) (its eq. 8);
#               trained the way SLSRec says it trained its baselines (AdamW, wd 1e-2,
#               batch 128, validate every 5 epochs on Recall@10)
#   recblr_wd0  as recblr, without weight decay -- RecBLR's own optimiser (Adam, lr 1e-3)
#
# SLSRec Table II lists RecBLR on ML-1M at Recall@10 0.34626 / Recall@20 0.44979,
# which is NOT the number RecBLR itself reports (HR@10 0.3285, RecBole); compare
# `recblr` against that, and against this repo's B runs (SAM gate added).
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43 44}"
VARIANTS="${VARIANTS:-slsrec recblr recblr_wd0}"
JOBS="${JOBS:-3}"
DATASET="${DATASET:-ml-1m}"
NUM_WORKERS="${NUM_WORKERS:-2}"    # keep constant across runs: it perturbs the RNG
EXTRA="${EXTRA:-}"                 # e.g. EXTRA="--exclude_lambda_from_decay=true"
LOGDIR="${LOGDIR:-runlogs/ablation_a_$DATASET}"
PYTHON="${PYTHON:-python}"

export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"

BASE="--use_sam_gate=false --sam_conv=false --use_perturbation=false --use_uncertainty=false"
RECBLR="--slsrec_pos_emb=false --slsrec_ffn=mlp --bdlru_c=1"

flags_for() {
    case "$1" in
        slsrec)     echo "$BASE" ;;
        recblr)     echo "$BASE $RECBLR" ;;
        recblr_wd0) echo "$BASE $RECBLR --weight_decay=0" ;;
        *) echo "unknown variant: $1" >&2; return 1 ;;
    esac
}

JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT

for v in $VARIANTS; do
    for s in $SEEDS; do
        exp="A_${v}_${DATASET}_seed${s}"
        if [ -f "experiments/slsrec/$exp/test_metrics.json" ]; then
            echo "[skip] $exp -- already finished"
            continue
        fi
        echo "$PYTHON train.py --dataset_code=$DATASET --model_code=slsrec --seed=$s $(flags_for "$v") $EXTRA --num_workers=$NUM_WORKERS --exp_name=$exp > $LOGDIR/$exp.log 2>&1" >> "$JOBFILE"
    done
done

n=$(wc -l < "$JOBFILE" | tr -d ' ')
if [ "$n" -eq 0 ]; then echo "[ablation-a] nothing to do"; exit 0; fi

if [ -n "${DRY_RUN:-}" ]; then
    echo "[ablation-a] DRY_RUN -- $n run(s) that would be launched, $JOBS at a time:"
    cat "$JOBFILE"
    exit 0
fi

echo "[ablation-a] warming the $DATASET dataset cache"
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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[ablation-a] no GPU detected -- this will be slow"
echo "[ablation-a] $n run(s), $JOBS at a time, logs in $LOGDIR/"
date

if xargs -P "$JOBS" -I CMD bash -c CMD < "$JOBFILE"; then
    status="all $n run(s) finished"
else
    status="some runs FAILED -- grep -l Traceback $LOGDIR/*.log"
fi

date
echo "[ablation-a] $status"
echo "[ablation-a] next: $PYTHON scripts/collect_hparams.py --model slsrec --dataset $DATASET"
