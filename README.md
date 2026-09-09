# SLSRec reproduction

Faithful, runnable reproduction of **"Self-Attention Gated Linear Recurrent Units
for Sequential Recommendation" (SLSRec)**, IEEE TCSS, DOI `10.1109/TCSS.2026.3679563`,
built on the unchanged [LRURec](https://github.com/yueqirex/LRURec) harness, with
[RecBLR](https://github.com/chengkai-liu/RecBLR) used as the implementation
reference for the behaviour-dependent LRU.

See [`reproduction_report.md`](reproduction_report.md) for the equation mapping,
ambiguity decisions, measured results and the full command list.

## Install

```bash
pip install -r requirements-minimal.txt
```

## Run

```bash
# original LRURec baseline (unmodified)
python train.py --dataset_code=ml-1m --num_workers=0

# full SLSRec (ablation D)
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 --exp_name=D_full_slsrec
```

## Ablations

| | Model | Flags |
|---|---|---|
| A | BD-LRU + CE | `--use_sam_gate=false --sam_conv=false --use_perturbation=false --use_uncertainty=false` |
| B | + self-attention gate | `--use_perturbation=false --use_uncertainty=false` |
| C | + uniform perturbation | `--use_uncertainty=false` |
| D | + Gibbs uncertainty (full SLSRec) | *(defaults)* |

## Tests

```bash
python -m pytest tests/ -q
```

## Layout

```
model/lru.py        original LRURec model (unchanged baseline)
model/bdlru.py      BD-LRU recurrence, scans, causal depth-wise conv
model/slsrec.py     SG-LRU, self-attention gate, PFFN, uncertainty estimator
model/triton_scan.py  RecBLR's Triton scan (optional, --scan_impl=triton)
trainer/lru.py      original LRURec trainer (unchanged)
trainer/slsrec.py   SLSRec objective (L_rec / L_perturb / L_uncertainty / L_total)
configs/slsrec_ml1m.yaml   resolved ML-1M configuration
scripts/smoke_ablations.py short real-data smoke test for all four ablations
tests/              unit tests (scan, causality, padding, perturbation, losses, overfit)
```
