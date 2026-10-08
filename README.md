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

## RecBLR baseline and one-command variant comparison

`--model_code=recblr` trains the official RecBLR architecture
(`model/recblr.py`, a line-for-line port of chengkai-liu/RecBLR `RecBLR.py`,
numerically identical to it) inside this harness, on the same schedule as SLSRec
-- which is how the SLSRec paper produced its RecBLR column (Table II).

```bash
python train.py --dataset_code=ml-1m --model_code=recblr --seed=42
```

`scripts/run_variants.py` trains every variant for each seed and prints one table:
test metrics next to the paper's Table II/III number, the change relative to full
SLSRec next to the paper's own "drop percentage", and training diagnostics that
show whether perturbation (`L_per-L_rec`) and uncertainty (`a*L_unc/L_rec`) are
doing anything.

```bash
python scripts/run_variants.py                        # recblr + Table III ablations + slsrec, seed 42
python scripts/run_variants.py --variants recblr      # RecBLR only
python scripts/run_variants.py --seeds 42 43 44
python scripts/run_variants.py --quick                # 10-epoch trial of the whole pipeline
python scripts/run_variants.py --list                 # all variants and their flags
python scripts/run_variants.py --collect_only --include_legacy   # table from runs already on disk
python scripts/run_variants.py --variants wo_pu p_layer --reuse_legacy   # train only what no older run covers
python scripts/run_variants.py -- --num_workers=2     # flags after -- go to every train.py
```

Finished runs are skipped, so an interrupted sweep resumes. Tables are written to
`experiments/compare/<prefix>_<dataset>.{md,csv,json}`. Why the current results
fall short of the paper: [`gap_analysis_vi.md`](gap_analysis_vi.md).

### Alternative readings of perturbation (P) and uncertainty (U)

On ML-1M both terms are inert as currently read (see the gap analysis). The paper
leaves room for other readings; each is a flag whose default is the original
behaviour (bit-for-bit, checked against the previous commit):

| flag | values (default first) | reading |
|---|---|---|
| `--perturbation_site` | `bdlru`, `bdlru_conv`, `layer`, `embedding` | where eq. (17) replaces x_t: BD-LRU input / L2 before Conv1-D2 / each layer input X^(l-1) (gate, recurrence and residual) / X^(0) once |
| `--perturbation_noise` | `uniform`, `shuffle` | U(low, high) / the vector of a random real position in another sequence |
| `--perturbation_granularity` | `position`, `element` | whole vector per step / each component independently |
| `--perturbation_gates` | `perturbed`, `clean` | r_t, i_t of eq. (9)/(10) see the replaced vector / see the clean x_t, so the noise cannot be gated out (site `bdlru`) |
| `--uncertainty_granularity` | `position`, `sequence`, `global` | sigma per prefix / one sigma_u per sequence from its last row (Sec. III-C2 wording) / one global sigma ([45]) |
| `--uncertainty_weight_norm` | `none`, `batch` | batch: 1/sigma^2 normalised to mean 1, so U cannot switch itself off (control) |
| `--uncertainty_beta` | `0`, any > 0 | the Log(sigma) penalty of eq. (22) the paper drops |

```bash
python scripts/run_variants.py --variants @p   # each P reading with U off, vs wo_pu and wo_u
python scripts/run_variants.py --variants @u   # each U reading with P off, vs wo_pu and wo_p
```

`scripts/diagnose_pu.py` measures on trained checkpoints *why* a reading acts or
not: whether the BD-LRU gates learned to recognise and skip the replacement,
the CE cost of the perturbed pass, whether grad L_per differs from grad L_rec
more than a second dropout draw does, where sigma^2 sits and whether it tracks
hard samples, and (`--item_noise 0 0.1 0.2`) the paper's Fig. 4 robustness test.

```bash
python scripts/diagnose_pu.py experiments/slsrec/RUN_A experiments/slsrec/RUN_B --item_noise 0 0.1 0.2
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
model/recblr.py     official RecBLR architecture, ported to the harness (--model_code=recblr)
model/adaptive_gate.py  adaptive gating (research extension, off by default)
model/triton_scan.py  RecBLR's Triton scan (optional, --scan_impl=triton)
trainer/lru.py      original LRURec trainer (unchanged)
trainer/slsrec.py   SLSRec objective (L_rec / L_perturb / L_uncertainty / L_total)
trainer/recblr.py   RecBLR: plain per-position CE, SLSRec's evaluation
configs/slsrec_ml1m.yaml   resolved ML-1M configuration
scripts/smoke_ablations.py short real-data smoke test for all four ablations
scripts/sweep_hparams.sh   hyperparameter grid runner (restartable, parallel)
scripts/collect_hparams.py runs + their hyperparameters vs the published tables
scripts/verify_lrurec.sh   LRURec baseline: this repo vs unmodified upstream
scripts/run_gate_sweep.sh  original gate vs adaptive gate A/B experiment
scripts/run_gate_sweep.ps1 the same experiment on Windows PowerShell
scripts/compare_gates.py   gate comparison table + what each gate learned
scripts/run_variants.py    RecBLR / ablations / SLSRec per seed, one comparison table vs the paper
tests/              unit tests (scan, causality, padding, perturbation, losses, overfit)
```
