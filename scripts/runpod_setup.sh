#!/usr/bin/env bash
# One-shot environment preflight for a RunPod pod (official PyTorch image).
#
#   bash scripts/runpod_setup.sh
#
# Installs only what the RunPod PyTorch image is missing, then proves the
# environment with the repo's own test suite + config audit.  If both pass, the
# pod is ready for scripts/run_sweep.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
    echo "no nvidia-smi -- this pod has no GPU" >&2; exit 1; }

echo "===== torch ====="
python - <<'PY'
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| available', torch.cuda.is_available())
PY

echo "===== deps ====="
# torch comes with the image; these do not.
pip install --no-cache-dir --quiet pandas tqdm tensorboard pytest

# torch < 2.3 was built against the numpy 1.x ABI and segfaults / raises on
# numpy 2.x.  requirements-minimal.txt pins numpy<2 for exactly this reason.
if ! python - <<'PY'
import sys, numpy, torch
nv = int(numpy.__version__.split('.')[0])
tv = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
sys.exit(1 if (nv >= 2 and tv < (2, 3)) else 0)
PY
then
    echo "[setup] numpy 2.x with torch < 2.3 -- downgrading numpy"
    pip install --no-cache-dir --quiet "numpy<2"
fi

echo "===== tests ====="
python -m pytest tests/ -q

echo "===== config audit ====="
python scripts/verify_config.py --dataset_code=ml-1m --model_code=slsrec \
    --seed=42 --num_workers=0

echo
echo "Environment OK.  Next: bash scripts/run_sweep.sh"
