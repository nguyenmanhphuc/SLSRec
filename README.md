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
| A′ | RecBLR (CIKM'24) in this harness | A + `--slsrec_pos_emb=false --slsrec_ffn=mlp --bdlru_c=1` |
| B | + self-attention gate | `--use_perturbation=false --use_uncertainty=false` |
| C | + uniform perturbation | `--use_uncertainty=false` |
| D | + Gibbs uncertainty (full SLSRec) | *(defaults)* |

A keeps the SLSRec skeleton (positional embedding, GeGLU PFFN, `c = 8`). A′ is
RecBLR as its own paper specifies it: no positional embedding (its eq. 1), a
SiLU MLP feed-forward (its eq. 4) and `a_t = exp(-softplus(Λ) r_t)` (its eq. 8,
which is eq. 11 with `c = 1`). SLSRec Table II puts RecBLR at ML-1M Recall@10
0.34626, which is not RecBLR's own number (HR@10 0.3285, RecBole); A′ measures
what RecBLR actually gets here.

```bash
bash scripts/run_ablation_a.sh                  # A, A', A' without weight decay x 3 seeds
VARIANTS=recblr SEEDS=42 bash scripts/run_ablation_a.sh
```

## Hyperparameter sweeps

Every run writes `run_config.json` (the full namespace *after* `set_template`,
plus the commit, GPU and torch build it ran on) and `run_summary.json`
(wall-clock, epochs, early stopping, best validation) next to its
`test_metrics.json` -- so a finished sweep can be reconstructed without trusting
folder names.

```bash
# LRURec baseline over the paper's dropout grid {0.2, 0.4, 0.6, 0.8}, ML-1M
bash scripts/sweep_hparams.sh

# SLSRec: dropout x perturbation rate theta x uncertainty rate alpha
MODEL=slsrec DROPOUTS="0 0.2 0.4" THETAS="0.05 0.1" ALPHAS="0.3" bash scripts/sweep_hparams.sh

# every finished run, only the hyperparameters that differ, next to the published number
python scripts/collect_hparams.py --model lru --dataset ml-1m --csv results.csv
```

`PROTOCOL=paper` switches from LRURec's own cadence (validate every
`--val_iterations` steps, at most 500 epochs) to the one the SLSRec paper
describes in Sec. IV-A4 (validate every 5 epochs, at most 1000) -- the ambiguity
is discussed in `reproduction_report.md` §8.1.  `collect_hparams.py --include-ref`
also picks up runs made from an unmodified `_ref_LRURec/` checkout.

## Adaptive gate (research extension)

Not part of the SLSRec paper. SLSRec fuses its branches with two *fixed* rules:
`G = SAM + Conv` (eq. 3, a hard 50/50) and `L3 = (G ⊙ H) W3` (eq. 6, G always
gates H at full strength). Neither can vary with the input, even though the
paper's own motivation is that some positions need global context and others
need strict ordering. Two flags make each rule adaptive.

| axis | flag | values |
|---|---|---|
| how `G` is composed (eq. 3) | `--gate_mode` | `sum` (original), `scalar`, `channel`, `token`, `context` |
| how strongly `G` gates `H` (eq. 6) | `--fusion_mode` | `mul` (original), `adaptive` |

Both are **neutral-initialised**: `g = 0.5` reproduces eq. (3) and `a = 1`
reproduces eq. (6), so every variant starts bit-for-bit identical to the
original layer (`tests/test_adaptive_gate.py` asserts exactly that). Any
measured difference is therefore attributable to what the gate learns, not to a
change of scale or initialisation.

`scalar` and `channel` are **capacity controls**: they add a learned but static
coefficient. If `token` beats `sum` while `scalar` does not, the gain came from
adaptivity; if `scalar` already captures it, a better fixed balance was all that
was missing.

Linux / macOS / Git Bash:

```bash
bash scripts/run_gate_sweep.sh                    # 6 variants x 3 seeds
DRY_RUN=1 bash scripts/run_gate_sweep.sh          # print the commands only
VARIANTS="sum token" SEEDS=42 bash scripts/run_gate_sweep.sh
```

Windows PowerShell:

```powershell
./scripts/run_gate_sweep.ps1 -DryRun
./scripts/run_gate_sweep.ps1 -Variants sum,scalar,token -Seeds 42 -Jobs 2
```

```bash
python scripts/compare_gates.py --gates
```

`compare_gates.py` reports test metrics over seeds, the **validation plateau**
mean (the last k rounds, not the single best draw — the plateau band on ML-1M is
~1.6% wide, the same order as the effects being measured), the delta against
`sum`, and what each gate actually learned. A `~` marks a delta inside the noise
band.

## Tests

```bash
python -m pytest tests/ -q
```

## Layout

```
model/lru.py        original LRURec model (unchanged baseline)
model/bdlru.py      BD-LRU recurrence, scans, causal depth-wise conv
model/slsrec.py     SG-LRU, self-attention gate, PFFN, uncertainty estimator
model/adaptive_gate.py  adaptive gating (research extension, off by default)
model/triton_scan.py  RecBLR's Triton scan (optional, --scan_impl=triton)
trainer/lru.py      original LRURec trainer (unchanged)
trainer/slsrec.py   SLSRec objective (L_rec / L_perturb / L_uncertainty / L_total)
configs/slsrec_ml1m.yaml   resolved ML-1M configuration
scripts/smoke_ablations.py short real-data smoke test for all four ablations
scripts/sweep_hparams.sh   hyperparameter grid runner (restartable, parallel)
scripts/collect_hparams.py runs + their hyperparameters vs the published tables
scripts/verify_lrurec.sh   LRURec baseline: this repo vs unmodified upstream
scripts/run_ablation_a.sh  ablation A: SLSRec skeleton vs RecBLR proper, 3 seeds
scripts/run_gate_sweep.sh  original gate vs adaptive gate A/B experiment
scripts/run_gate_sweep.ps1 the same experiment on Windows PowerShell
scripts/compare_gates.py   gate comparison table + what each gate learned
tests/              unit tests (scan, causality, padding, perturbation, losses, overfit)
```
