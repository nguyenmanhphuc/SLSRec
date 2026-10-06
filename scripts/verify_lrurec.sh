#!/usr/bin/env bash
# Verify the LRURec baseline BEFORE spending GPU hours on SLSRec.
#
# Runs the SAME protocol twice -- once from the unmodified `_ref_LRURec/`
# source, once from this repository's `--model_code=lru` path -- then compares
# both against each other and against the published LRURec numbers.
#
#   bash scripts/verify_lrurec.sh                  # full protocol, ml-1m, seed 42
#   SMOKE=1 bash scripts/verify_lrurec.sh          # 1 epoch, wiring check only
#   DATASET=beauty bash scripts/verify_lrurec.sh
#   PARALLEL=0 bash scripts/verify_lrurec.sh       # one run at a time
#   SKIP_PARITY=1 bash scripts/verify_lrurec.sh    # skip the pre-flight parity check
#
# The two runs share a byte-identical dataset.pkl, model/lru.py and
# trainer/lru.py, so they must agree to within GPU nondeterminism.  If they do
# not, the harness has drifted and no SLSRec number built on it is trustworthy.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${DATASET:-ml-1m}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PARALLEL="${PARALLEL:-1}"
SMOKE="${SMOKE:-}"
LOGDIR="${LOGDIR:-runlogs/verify_lru}"
OURS_EXP="lru_baseline_${DATASET}_seed${SEED}"

export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"

if [ ! -d _ref_LRURec ]; then
    cat >&2 <<'MSG'
_ref_LRURec/ is missing.  It is gitignored, so a clone of this repo does not
carry it.  Fetch the upstream source (it ships the raw data and the same
preprocessed pickles):

    git clone https://github.com/yueqirex/LRURec.git _ref_LRURec
    git -C _ref_LRURec checkout e02a559
MSG
    exit 1
fi

# Cheap gate first: if the two trees already disagree on the very first batch,
# the full runs cannot possibly agree either.
if [ -z "${SKIP_PARITY:-}" ]; then
    python scripts/lru_parity_check.py --dataset "$DATASET" --seed "$SEED"
    echo
fi

EPOCHS=""
if [ -n "$SMOKE" ]; then
    EPOCHS="--num_epochs=1"
    echo "[verify] SMOKE=1 -- one epoch, wiring check only."
    echo "[verify] upstream set_template pins val_iterations=500 for ml-1m and ignores"
    echo "[verify] the CLI, so 56 batches never trigger a validation: the checkpoint"
    echo "[verify] under test stays untrained.  Do NOT compare the metrics from a smoke."
fi

# _ref_LRURec/train.py imports wandb (never called) and pytorch_lightning (only
# for seed_everything).  Stub whichever is missing rather than letting pip
# resolve a different torch onto the pod.  Real packages always win.
SHIMDIR="$(mktemp -d)"
trap 'rm -rf "$SHIMDIR"' EXIT
for mod in pytorch_lightning wandb; do
    if ! python -c "import $mod" >/dev/null 2>&1; then
        cp "scripts/_ref_shims/$mod.py" "$SHIMDIR/"
        echo "[verify] $mod not installed -- using scripts/_ref_shims/$mod.py"
    fi
done

REF_LOG="$LOGDIR/upstream_${DATASET}_seed${SEED}.log"
OURS_LOG="$LOGDIR/ours_${DATASET}_seed${SEED}.log"
REF_METRICS="_ref_LRURec/experiments/lru/${DATASET}_0.01_0.2_0.2/test_metrics.json"
OURS_METRICS="experiments/lru/${OURS_EXP}/test_metrics.json"

run_upstream() {
    ( cd _ref_LRURec && PYTHONPATH="$SHIMDIR" python train.py \
        --dataset_code="$DATASET" --seed="$SEED" --num_workers="$NUM_WORKERS" $EPOCHS )
}

run_ours() {
    python train.py --dataset_code="$DATASET" --model_code=lru \
        --seed="$SEED" --num_workers="$NUM_WORKERS" $EPOCHS --exp_name="$OURS_EXP"
}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[verify] dataset=$DATASET seed=$SEED  logs in $LOGDIR/"
date

ref_status=0
ours_status=0
if [ "$PARALLEL" = "1" ]; then
    run_upstream > "$REF_LOG" 2>&1 &
    ref_pid=$!
    run_ours > "$OURS_LOG" 2>&1 &
    ours_pid=$!
    wait "$ref_pid" || ref_status=$?
    wait "$ours_pid" || ours_status=$?
else
    run_upstream > "$REF_LOG" 2>&1 || ref_status=$?
    run_ours > "$OURS_LOG" 2>&1 || ours_status=$?
fi

date
[ "$ref_status" -eq 0 ]  || echo "[verify] upstream run FAILED (exit $ref_status) -- see $REF_LOG"
[ "$ours_status" -eq 0 ] || echo "[verify] this repo's run FAILED (exit $ours_status) -- see $OURS_LOG"
echo

python scripts/compare_lru.py --dataset "$DATASET" \
    --ref "$REF_METRICS" --ours "$OURS_METRICS"
