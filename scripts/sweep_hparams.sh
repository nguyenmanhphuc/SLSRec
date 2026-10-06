#!/usr/bin/env bash
# Hyperparameter sweep that records the configuration of every run.
#
# The SLSRec paper (Sec. IV-A4) fixes lr = 1e-3, weight decay = 1e-2, dim 64,
# batch 128 for every baseline and searches only the dropout rate, in
# {0.2, 0.4, 0.6, 0.8} for the baselines and {0, 0.2, 0.4, 0.6, 0.8} for SLSRec
# on ML-1M / Steam / Video ({0, 0.25, 0.5, 0.75, 0.95} on Beauty / Sports).
# SLSRec additionally searches theta in {0.05, 0.1, 0.15, 0.2, 0.25} and alpha in
# {0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1}.  Those are the defaults below.
#
#   bash scripts/sweep_hparams.sh                       # LRURec dropout grid, ml-1m, seed 42
#   SEEDS="42 43 44" bash scripts/sweep_hparams.sh
#   PROTOCOL=paper bash scripts/sweep_hparams.sh        # val every 5 epochs, max 1000 (paper text)
#   MODEL=slsrec THETAS="0.05 0.1" ALPHAS="0.3" bash scripts/sweep_hparams.sh
#   DRY_RUN=1 bash scripts/sweep_hparams.sh             # print the job list, run nothing
#
# Every run writes run_config.json (the resolved namespace) + run_summary.json
# next to its test_metrics.json, so scripts/collect_hparams.py can rebuild the
# whole grid afterwards without trusting the folder names.  Restartable: a run
# whose test_metrics.json exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-lru}"
DATASET="${DATASET:-ml-1m}"
SEEDS="${SEEDS:-42}"
DROPOUTS="${DROPOUTS:-0.2 0.4 0.6 0.8}"
ATTN_DROPOUTS="${ATTN_DROPOUTS:-tied}"   # 'tied' -> attention dropout follows --bert_dropout
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.01}"
LRS="${LRS:-0.001}"
THETAS="${THETAS:-default}"              # slsrec only; 'default' leaves the flag off
ALPHAS="${ALPHAS:-default}"              # slsrec only
PROTOCOL="${PROTOCOL:-code}"             # code = LRURec repo cadence | paper = Sec. IV-A4 text
JOBS="${JOBS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"          # keep constant across runs: it perturbs the RNG
LOGDIR="${LOGDIR:-runlogs/hparams_${DATASET}}"

export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"

case "$PROTOCOL" in
    code)  PROTOCOL_FLAGS="" ;;
    paper) PROTOCOL_FLAGS="--num_epochs=1000 --val_strategy=epoch --val_epochs=5" ;;
    *)     echo "PROTOCOL must be 'code' or 'paper', got '$PROTOCOL'" >&2; exit 2 ;;
esac

JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT

for do_ in $DROPOUTS; do
  if [ "$ATTN_DROPOUTS" = "tied" ]; then attn_list="$do_"; else attn_list="$ATTN_DROPOUTS"; fi
  for ad in $attn_list; do
    for wd in $WEIGHT_DECAYS; do
      for lr in $LRS; do
        for th in $THETAS; do
          for al in $ALPHAS; do
            for s in $SEEDS; do
                flags="--bert_dropout=$do_ --bert_attn_dropout=$ad --weight_decay=$wd --lr=$lr"
                tag="do${do_}_ad${ad}_wd${wd}_lr${lr}"
                if [ "$MODEL" = "slsrec" ]; then
                    if [ "$th" != "default" ]; then
                        flags="$flags --perturbation_probability=$th"; tag="${tag}_th${th}"
                    fi
                    if [ "$al" != "default" ]; then
                        flags="$flags --uncertainty_alpha=$al"; tag="${tag}_al${al}"
                    fi
                fi
                exp="${MODEL}_${DATASET}_${PROTOCOL}_${tag}_seed${s}"
                if [ -f "experiments/$MODEL/$exp/test_metrics.json" ]; then
                    echo "[skip] $exp -- already finished"
                    continue
                fi
                echo "python train.py --dataset_code=$DATASET --model_code=$MODEL --seed=$s $flags $PROTOCOL_FLAGS --num_workers=$NUM_WORKERS --exp_name=$exp > $LOGDIR/$exp.log 2>&1" >> "$JOBFILE"
            done
          done
        done
      done
    done
  done
done

n=0
[ -s "$JOBFILE" ] && n=$(wc -l < "$JOBFILE" | tr -d ' ')
if [ "$n" -eq 0 ]; then
    echo "[sweep] nothing to do"
    exit 0
fi

if [ -n "${DRY_RUN:-}" ]; then
    echo "[sweep] DRY_RUN -- $n run(s) that would be launched, $JOBS at a time:"
    cat "$JOBFILE"
    exit 0
fi

# Warm the preprocessing cache in one process first, or the parallel jobs race to
# write the same dataset.pkl.
echo "[sweep] warming the $DATASET dataset cache"
python - "$DATASET" "$MODEL" <<'PY'
import sys
dataset, model = sys.argv[1], sys.argv[2]
sys.argv = ['warm', '--dataset_code=' + dataset, '--model_code=' + model, '--num_workers=0']
from config import args, set_template
set_template(args)
from dataloader import dataloader_factory
train, val, test = dataloader_factory(args)
print('[warm] %d train batches, %d val batches, %d items'
      % (len(train), len(val), args.num_items))
PY

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[sweep] model=$MODEL dataset=$DATASET protocol=$PROTOCOL -- $n run(s), $JOBS at a time"
echo "[sweep] logs in $LOGDIR/"
date

if xargs -P "$JOBS" -I CMD bash -c CMD < "$JOBFILE"; then
    status="all $n run(s) finished"
else
    status="some runs FAILED -- grep -l Traceback $LOGDIR/*.log"
fi

date
echo "[sweep] $status"
echo
python scripts/collect_hparams.py --model "$MODEL" --dataset "$DATASET" \
    --csv "$LOGDIR/results.csv"
