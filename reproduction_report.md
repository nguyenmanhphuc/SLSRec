# SLSRec reproduction report

Reproduction of **"Self-Attention Gated Linear Recurrent Units for Sequential
Recommendation" (SLSRec)** — Jian Yang, Tengfei Bi, Jiajin Huang, Mi Li,
*IEEE Transactions on Computational Social Systems*, DOI `10.1109/TCSS.2026.3679563`.

The paper is the authoritative source for architecture, losses, perturbation,
uncertainty estimation and hyperparameters. The **LRURec** repository
(<https://github.com/yueqirex/LRURec>) supplies the data, training, evaluation and
checkpointing harness *unchanged*. The **RecBLR** repository
(<https://github.com/chengkai-liu/RecBLR>) is used only as an implementation
reference for the behaviour-dependent LRU, its gates, the stable recurrence
parameterisation, the causal depth-wise convolution, the parallel scan, and the
feed-forward layer.

Status: **implemented, tested and run**. 45/45 unit tests pass. On ML-1M with
seed 42 the full model trains end-to-end on CPU with finite, decreasing losses;
validation Recall@10 rises from 0.0042 (untrained) to 0.1552 in three epochs,
checkpoint selection and the reload-and-test path both work. No converged run,
five-seed sweep or multi-dataset reproduction has been attempted (see §10), so
**no claim of numerical reproduction of the paper's Table II is made**.

---

## 1. Implementation-to-equation mapping

| Paper | Meaning | Implemented by |
|---|---|---|
| (1) | `X⁰ = LayerNorm(Dropout(E + P))` | `model/slsrec.py::SLSRecEmbedding.forward` |
| (2) | `L1 = X W1 + b1` | `SGLRULayer.w1` |
| (3) | `G = SAM(L1) + Conv1-D1(L1)` | `SGLRULayer.forward` (gate branch) |
| (4) | `L2 = X W2 + b2` | `SGLRULayer.w2` |
| (5) | `H = BD-LRU(Conv1-D2(L2))` | `SGLRULayer.conv2` → `model/bdlru.py::BDLRU` |
| (6) | `L3 = (G ⊙ H) W3 + b3` | `SGLRULayer.w3` |
| (7) | `O = LayerNorm(Dropout(L3) + X⁽ˡ⁻¹⁾)` | `SGLRULayer.forward` (return) |
| (8) | `SAM(L1) = Softmax(QKᵀ/√d + S)V` | `model/slsrec.py::CausalSelfAttention` |
| (9) | `r_t = Sigmoid(W4 x_t + b4)` | `BDLRU.gates` (first half) |
| (10) | `i_t = Sigmoid(W5 x_t + b5)` | `BDLRU.gates` (second half) |
| (11) | `a_t = Exp(c·Log(Sigmoid(λ)) ⊙ r_t)`, `c = 8` | `BDLRU.forward` (`self.c * logsigmoid(Lambda) * r_t`) |
| (12) | `h_t = a_t ⊙ h_{t-1} + √(1−a_t²) ⊙ (i_t ⊙ x_t)` | `BDLRU.forward` → `parallel_scan` |
| (13) | `PFFN(O) = (GeLU(O W6) ⊙ (O W7)) W8` | `model/slsrec.py::PFFN` |
| (14) | `X⁽ˡ⁾ = LayerNorm(Dropout(PFFN(O)) + O)` | `PFFNBlock` |
| (15) | `y = Softmax(M × x⁽ᴸ⁾)` | `SLSRecModel.logits_from_hidden` (+ softmax inside CE) |
| (16) | `L_Rec = −Log(y_ui)` | `trainer/slsrec.py::slsrec_losses` (`l_rec`) |
| (17) | `x̄_t = U(0,1)` w.p. `θ`, else `x_t` | `BDLRU._perturb` |
| (18) | `L_Per = −Log(ȳ_ui)` | `slsrec_losses` (`l_per`) |
| (19) | `ỹ = Softmax((1/σ²)(M × x⁽ᴸ⁾))` | `slsrec_losses`, `uncertainty_loss='exact'` |
| (20) | `Log(σ²) = Mc[i,:]ᵀ LayerNorm(Dropout(PFFN(xᶜ) + xᶜ))` | `model/slsrec.py::UncertaintyEstimator` → `UncertaintyPFFNBlock` |
| (21) | `L_Unc = −Log(ỹ_ui)` | `slsrec_losses` (`l_unc`, `exact`) |
| (22) | `L_Unc ≈ (1/σ²)L_Rec + β·Log(σ)` | `slsrec_losses` (`l_unc`, `approx`, **default**) |
| (23)/(24) | `L = (1 + α/σ²)L_Rec + L_Per` | `slsrec_losses` (`l_total`) |
| Sec. III-B4 | prediction from the last position, full sort | `SLSRecTrainer.calculate_metrics` |
| Sec. IV-A1 | ML-1M, 5-core, leave-one-out, `N=200` | LRURec `datasets/`, `dataloader/` (unchanged) |
| Sec. IV-A4 | AdamW, lr 1e-3, wd 1e-2, `D=64`, batch 128, Recall@10 selection | `config.py::set_template` (unchanged) |
| Sec. IV-E | `dropout=0.2`, `θ=0.1`, `α=0.3` for ML-1M | `config.py::set_slsrec_template` |

Auxiliary components:

| Component | File |
|---|---|
| Hillis–Steele associative scan (default, portable) | `model/bdlru.py::hillis_steele_scan` |
| Sequential reference scan (tests) | `model/bdlru.py::sequential_scan` |
| RecBLR Triton scan (optional, CUDA) | `model/triton_scan.py` (verbatim copy) |
| Causal depth-wise conv | `model/bdlru.py::CausalDepthwiseConv1d` |
| Multi-term loss logging | `trainer/base.py::_log_train_losses` |

---

## 2. Environment

| | |
|---|---|
| OS | Windows 11 Pro (10.0.22631), 16 CPU cores |
| Python | 3.10.18 (conda env `SemADR`) |
| PyTorch | 2.1.0+cu118 |
| CUDA available | **No** — all results below are CPU |
| numpy / pandas | 1.26.4 / 2.2.3 |
| Dependencies | `requirements-minimal.txt` (torch, numpy, pandas, tqdm, tensorboard, pytest) |

`pytorch-lightning` was the only LRURec import not needed for anything but
`seed_everything`; it was replaced by `trainer/utils.py::seed_everything` so the
minimal environment stays small.

---

## 3. Dataset protocol

Taken verbatim from LRURec (`datasets/ml_1m.py`, `dataloader/sas.py`); the SLSRec
paper states it uses exactly these splits ("we closely follow the configurations
outlined in the LRURec work and request experimental datasets directly from the
authors").

* MovieLens-1M, `ratings.dat`; immediate repeats removed; users and items with
  fewer than 5 interactions dropped (`min_uc = min_sc = 5`, `min_rating = 0`).
* Leave-one-out: last item → test, second-to-last → validation, rest → train.
  At test time the validation item is appended to the input.
* Maximum sequence length 200, **left** padding with item id 0.
* Training target: per-position next-item prediction (`SASTrainDataset`),
  cross-entropy with `ignore_index=0`.
* Evaluation: full sort over all 3417 logits; already-seen items and the padding
  item 0 are set to `-1e9` before ranking (`SLSRecTrainer.calculate_metrics`,
  identical to `LRUTrainer`).

Preprocessed statistics (measured, `data/preprocessed/ml-1m_*/dataset.pkl`):

| | users | items | interactions | avg. length |
|---|---|---|---|---|
| measured | 6040 | 3416 | 999,611 | 165.5 |
| paper Table I | 6040 | 3416 | 1.0 M | 165.5 |

---

## 4. Hyperparameters (ML-1M, seed 42)

| Parameter | Value | Source |
|---|---|---|
| seed | 42 | task |
| embedding dim `D` | 64 | paper Sec. IV-A4 |
| latent dim `d` | 128 (`= 2D`) | paper Sec. III-B2 |
| encoder blocks `L` | 2 | LRURec / task |
| max sequence length | 200 | paper Sec. IV-A1 |
| optimizer | AdamW, `eps=1e-9` | paper Sec. IV-A4 |
| learning rate | 1e-3, constant (no warmup, no decay) | paper Sec. IV-A4 |
| weight decay | 1e-2 | paper Sec. IV-A4 |
| max grad norm | 5.0 | LRURec |
| batch size | 128 (val/test 256) | paper Sec. IV-A4 |
| dropout | 0.2 | paper Sec. IV-E1 (ML-1M optimum) |
| conv kernel `d_conv` | 4 | RecBLR `config.yaml` |
| recurrence constant `c` | 8 | paper eq. (11) |
| `a_t` init range | [0.9, 0.999] | paper Sec. III-B2 / RecBLR |
| perturbation rate `θ` | 0.1 | paper Sec. IV-E2 |
| perturbation range | `U(0,1)` | paper eq. (17) |
| uncertainty rate `α` | 0.3 | paper Sec. IV-E3 (ML-1M optimum) |
| uncertainty `β` | 0 | paper Sec. III-C3 |
| checkpoint metric | Recall@10 | task / LRURec |
| evaluation | full sort, Recall / NDCG / MRR @ {1,5,10,20,50} | task / LRURec |
| validation cadence | every 500 iterations | task / LRURec (see §8.1) |
| early stopping patience | 10 validations | LRURec |

### 4.1 All five datasets

Resolved by `config.py::set_slsrec_template`; audited by `scripts/verify_hparams.py`
(exit 0 = every explicitly published value matches).

| Parameter | ML-1M | Beauty | Video | Sports | Steam | Source |
|---|---|---|---|---|---|---|
| max sequence length | 200 | 50 | 50 | 50 | 50 | Sec. IV-A1, explicit |
| dropout | 0.2 | 0.75 | **0.6** | 0.75 | 0.2 | Figs. 6 & 7 (see below) |
| perturbation rate `θ` | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | Sec. IV-E2, explicit |
| uncertainty rate `α` | 0.3 | 0.01 | 0.3 | 0.3 | 0.3 | Sec. IV-E3, explicit |
| dropout search grid | {0,.2,.4,.6,.8} | {0,.25,.5,.75,.95} | {0,.2,.4,.6,.8} | {0,.25,.5,.75,.95} | {0,.2,.4,.6,.8} | Sec. IV-A4 |

Everything else is dataset independent: `D`=64, `d`=2D=128, 2 blocks, batch 128,
AdamW lr 1e-3, weight decay 1e-2, `c`=8, `a_t` init [0.9, 0.999], `β`=0,
Recall@10 selection, patience 10 validation rounds.

### 4.2 Where the per-dataset dropout comes from

Sec. IV-E1 names only two values in prose — 0.2 for "dense datasets, such as
ML-1M" and 0.75 for "sparse datasets, such as Sports" — so Beauty, Video and
Steam look unspecified. They are not: **Figs. 6 and 7 plot Recall@20 and
NDCG@20 against dropout for all five datasets**, and each curve's peak
reproduces that dataset's Table II entry:

| | code | Fig. 6 peak | value there | Table II Recall@20 | gap |
|---|---|---|---|---|---|
| ML-1M | 0.2 | 0.2 | 0.468 | 0.46732 | 0.0007 |
| Beauty | 0.75 | 0.75 | 0.075 | 0.07727 | 0.0023 |
| Video | 0.6 | 0.6 | 0.173 | 0.17353 | 0.0005 |
| Sports | 0.75 | 0.75 | 0.048 | 0.04768 | 0.0002 |
| Steam | 0.2 | 0.2 | 0.210 | 0.20697 | 0.0030 |

Every gap is within the precision of reading a printed plot (±0.005 on the left
panel, where ML-1M, Video and Steam share a 0.1–0.5 axis), so the check is on
which grid point is the argmax — legible from the figure — rather than on
absolute values.

**Video was initially wrong in this reproduction.** Sec. IV-A4 groups Video with
ML-1M and Steam by *search grid* (`{0, 0.2, 0.4, 0.6, 0.8}`), which reads as
"dense → 0.2", and that is what `set_slsrec_template` used. The figure shows the
optimum is **0.6**: at 0.2 Fig. 6 gives ≈0.158 Recall@20, against the 0.17353
that Table II reports. Fixed; ML-1M results are unaffected.

### 4.3 Values the paper does not pin down

None, for the five published datasets. `xlong` is not in the SLSRec paper and
keeps the 0.2 default.

---

## 5. Tensor shapes (ML-1M, batch 4)

```
input  seqs        (4, 200)            long, left-padded, 0 = pad
mask               (4, 200)            bool, seqs > 0
X⁰ (embedding)     (4, 200, 64)        D = 64
L1 / L2            (4, 200, 128)       d = 2D = 128
SAM(L1)            (4, 200, 128)
Conv1-D1(L1)       (4, 200, 128)       depth-wise, kernel 4, causal
G = SAM + Conv     (4, 200, 128)
Conv1-D2(L2)       (4, 200, 128)
H = BD-LRU(...)    (4, 200, 128)
L3 = (G⊙H)W3       (4, 200, 64)
X⁽ᴸ⁾ (hidden)      (4, 200, 64)
logits             (4, 200, 3417)      |I| + 1 (padding row kept, masked at eval)
Log(σ²)            (4, 200)            per position
W1 / W2            (128, 64)   W3      (64, 128)
Conv kernel                    (128, 1, 4)
λ (Lambda)                     (128,)
```

---

## 6. Parameter counts (ML-1M, `|I| = 3416`)

| Ablation | Total | Encoder (non-embedding) | Embeddings |
|---|---|---|---|
| A — BD-LRU + CE | 447,808 | 216,192 | 231,616 |
| B — + SAM gate | 547,392 | 315,776 | 231,616 |
| C — + perturbation | 547,392 | 315,776 | 231,616 |
| D — full SLSRec | 828,288 | 365,056 | 463,232 |

Reference: the original LRURec model on the same dataset has 322,841 total /
104,025 encoder parameters. The SAM gate adds 99,584 parameters (3 × 128×128 per
layer × 2 layers plus the gate conv); the uncertainty branch adds a *second*
embedding table `Mc` (3417×64) and position table `Pc` (200×64) plus its own PFFN,
which is why D is markedly larger — this is exactly what eq. (20) prescribes.

---

## 7. Test results

Command:

```bash
python -m pytest tests/ -q
```

Result: **45 passed** (`tests/test_scan.py` 13, `tests/test_causality.py` 3,
`tests/test_padding.py` 4, `tests/test_perturbation.py` 8,
`tests/test_losses.py` 12, `tests/test_tiny_overfit.py` 5).

| File | What it checks |
|---|---|
| `test_scan.py` | parallel vs. sequential scan agree to `< 1e-4` for `L ∈ {1,2,3,8,17,64,200}`; the `BDLRU` module agrees across `--scan_impl torch/sequential`; forward/backward finite; gradients reach item embeddings, position embeddings, `λ`, the input+recurrence gates, both convolutions, all three SAM projections, the output projection, the PFFN and the uncertainty branch; toggling `--use_sam_gate` / `--sam_conv` / `--sam_conv_activation` / `--bdlru_conv_activation` / `--use_bdlru` changes the output; disabling the SAM gate removes the SAM parameters entirely; `a_t ∈ (0,1]` and its initialisation spans exactly `[0.9, 0.999]` |
| `test_causality.py` | changing one future item leaves every earlier hidden state unchanged (`< 1e-6`) — checked at a single position, at *every* position, and with perturbation active |
| `test_padding.py` | the same items right-aligned in tensors of length 10 / 15 / 24 give identical hidden states and identical scores; garbage in the padded region cannot reach the BD-LRU state; attention ignores padded keys |
| `test_perturbation.py` | perturbation off ⇒ deterministic; on ⇒ recurrence input changes; replacement values are `U(0,1)` (mean 0.5, var 1/12); padded positions are never perturbed; empirical rate matches `θ` within 0.01; identical seeds reproduce identical masks, different seeds do not; each layer draws its own mask; `θ=0` is a no-op |
| `test_losses.py` | each term finite; `L_total = L_rec + w_per·L_per + α·L_unc` exactly (eq. 23); `L_unc` equals the inverse-variance-weighted CE (eq. 22 with `β=0`); `α=0` and `w_per=0` remove their contributions; disabled branches give exactly 0; eval mode skips the perturbed pass; both `approx` and `exact` uncertainty forms run; `β` adds `β/2·Log(σ²)`; all enabled branches get gradient from `L_total`; the uncertainty branch gets none when `α=0` |
| `test_tiny_overfit.py` | 8 synthetic sequences are memorised by **all four ablations** (final CE < 20 % of the initial `L_rec`, train accuracy > 0.9); each ablation is reachable purely through configuration and wires exactly the intended parameters and loss terms |

---

## 8. Known ambiguities and the decisions taken

Where the paper is ambiguous, the most literal reading is implemented and the
alternative is exposed as a flag.

### 8.1 Validation cadence and epoch budget
Paper Sec. IV-A4: "training is limited to a maximum of 1000 epochs. Validation is
conducted every 5 epochs, and an early stopping mechanism is activated if there is
no improvement in the validation Recall@10 metric for 10 consecutive validation
epochs." LRURec's harness instead validates every `val_iterations = 500` steps
(≈ 8.9 epochs on ML-1M, which has 56 training batches per epoch) with a 500-epoch
cap. **Decision:** follow the task instruction and keep LRURec's
iteration-based cadence as the default. The paper-literal setting is reachable
with `--val_strategy=epoch --val_epochs=5 --num_epochs=1000` (the `--val_epochs`
flag was added for this purpose, and `tests/test_schedule.py` verifies that it
fires after exactly epochs 5, 10, ... while leaving the iteration path
untouched).

Practical significance of the two halves differs:

* **1000 vs 500 epochs: no effect.** Early stopping fires first -- the ML-1M
  run stopped at epoch 206 -- so the cap is never reached.
* **Cadence: it matters.** 25 validations over 206 epochs instead of ~41 means
  coarser model selection on a plateau whose band is ±1.6%. Measured cost on
  the ML-1M run: the last checkpoint beat the Recall@10-selected one by 0.71%
  on test (§9.3). Upstream `set_template` unconditionally
overwrote `--val_iterations` and `--bert_max_len` with its per-dataset values, so
those flags silently did nothing; they are now only applied when the user did not
pass them (defaults are unchanged).

### 8.2 Per-position vs. last-position training
Eq. (15)/(16) define the prediction and the loss for the **last** position only.
The LRURec harness that the paper says it builds on trains with **per-position**
next-item prediction, and the task instruction requires it. **Decision:**
per-position CE, ignoring padding. This also propagates to eq. (20): `Log(σ²)`
is computed per position, using that position's ground-truth item for `Mc[i,:]`.
The uncertainty branch is purely position-wise (embedding → PFFN → LayerNorm),
so this generalisation introduces no information leakage across time. Evaluation
still uses only the last position, exactly as eq. (15) prescribes.

### 8.3 `L_Unc`: exact eq. (21) vs. the eq. (22) approximation
Eq. (21) defines `L_Unc = −Log(ỹ_ui)` with `ỹ` from the temperature-scaled softmax
(19); eq. (22) then *approximates* it as `(1/σ²)L_Rec + β·Log(σ)`, and eq. (23)/(24)
— the loss the paper actually optimises — uses the approximation. **Decision:**
`--uncertainty_loss=approx` is the default (matches eq. 24); `--uncertainty_loss=exact`
implements eq. (19)/(21) literally.

### 8.4 `β = 0` makes `σ` monotonically drift upward
With `β = 0` (the paper's choice, Sec. III-C3) the only gradient on `Log(σ²)`
comes from `−α·e^{−Log σ²}·L_Rec`, which is strictly negative — i.e. it always
pushes `Log(σ²)` up, so `α/σ² → 0` and the uncertainty term decays towards zero
during training. The paper explicitly accepts this ("`L_Rec` retains its
effectiveness even as `σ²` tends to +∞"). This is visible in the smoke run
(§9), where `L_uncertainty` falls much faster than `L_rec`. **This weakness was
not "fixed"** — it is reproduced as specified. `--uncertainty_beta` is available
for anyone who wants the penalty term back. A numerical safety clamp
`--uncertainty_logvar_clamp` (default `|Log σ²| ≤ 8`) prevents overflow; the paper
specifies none.

### 8.5 Activation after the two convolutions
Eq. (3) writes `G = SAM(L1) + Conv1-D1(L1)` with no activation, and eq. (5)
writes `H = BD-LRU(Conv1-D2(L2))`. RecBLR — cited by the paper as [17] for both
convolutions — always applies SiLU after its causal conv. **Decision:**
`--sam_conv_activation=none` (literal eq. 3) and `--bdlru_conv_activation=silu`
(RecBLR reference, since eq. (5)'s conv is the RecBLR one). Both are flags;
`test_scan.py` shows toggling either changes the output.

### 8.6 Causality and padding of the convolutions
The paper does not state that the convolutions are causal, but a non-causal conv
would leak future items into earlier states and contradict the causal mask `S` in
eq. (8). **Decision:** both convolutions are causal (left-padded, right-truncated)
depth-wise convolutions, as in RecBLR. Padded positions are zeroed before every
convolution and before the attention, and padded steps contribute nothing to the
recurrence, so representations at real positions are invariant to the amount of
left padding (`test_padding.py`).

### 8.7 Positional embedding indexing under left padding
Eq. (1) adds a learnable `P ∈ R^{N×D}`, but the LRURec dataloader left-pads.
**Decision:** positions are anchored to the right end (the last item always gets
`P[N-1]`). For the fixed-length batches the harness produces — every sequence is
padded to exactly 200 — this is *identical* to plain left-to-right indexing, and
it additionally makes the model padding-length invariant.

### 8.8 `S` in eq. (8): single-head attention, no output projection
Eq. (8) specifies one `W_Q/W_K/W_V ∈ R^{d×d}` and no output projection or
attention dropout. **Decision:** single head (`--sam_num_heads=1`), no output
projection, `--sam_attn_dropout=0.0`. Both are flags.

### 8.9 Perturbation granularity
Eq. (17) samples a scalar `p` per step and replaces the whole vector `x_t`.
**Decision:** one Bernoulli(`θ`) draw per `(sequence, position)`, replacing the
full `d`-dimensional vector with i.i.d. `U(0,1)` components; an independent draw
per layer ("randomly replacing an item representation **in each layer**"); padded
positions are excluded (the paper is silent, but perturbing padding would
contradict the harness's masking).

### 8.10 Item bias in eq. (15)
Eq. (15) has no per-item bias; LRURec's prediction head does. **Decision:**
follow the paper (`--use_item_bias=false`); the LRURec behaviour is available via
`--use_item_bias=true`.

### 8.11 Parenthesis typo in eq. (14) and uncertainty residual dropout
Eq. (14) is printed as `LayerNorm(Dropout(PFFN(O_u) + O_u)` — unbalanced.
Eq. (7) uses the pattern `LayerNorm(Dropout(·) + residual)`. **Decision:** the
eq. (7) pattern, i.e. `LayerNorm(Dropout(PFFN(O)) + O)`, consistent with Fig. 1.

Eq. (20), however, explicitly places the uncertainty residual **inside** dropout:
`LayerNorm(Dropout(PFFN(xc) + xc))`. Fig. 2 depicts the residual bypassing dropout,
so its diagram conflicts with the printed equation. **Decision (2026-10-06):**
follow the printed eq. (20) for the uncertainty branch, as requested. A separate
`UncertaintyPFFNBlock` applies dropout after adding `xc`. Its parameter names and
shapes are unchanged, so earlier checkpoints load strictly and retain their
evaluation outputs, but further training uses the corrected dropout placement.

The experiments reported here were trained before this correction. Their metrics
remain results of the former implementation; they do not measure retraining with
the corrected equation. Regression tests in `tests/test_uncertainty.py` cover
nonzero training dropout, complete residual dropout and old checkpoint layout.

### 8.12 Table cross-references V/VI are swapped
Sec. IV-E2 says "As shown in Table V" for the perturbation rate `θ`, but Table V
is captioned "EFFECT OF UNCERTAINTY RATE α" and holds the `α` grid
`{0.01 … 1}`, while Table VI is captioned "EFFECT OF PERTURBATION RATE θ" and
holds the `θ` grid `{0.05 … 0.25}`. Read by content (which matches the stated
grids in Sec. IV-A4), the ML-1M optima are `θ = 0.1` and `α = 0.3`, which is what
Sec. IV-E2/E3 conclude in prose. Those are the defaults used here.

### 8.13 RecBLR's parameterisation vs. eq. (11)
RecBLR computes `α_t = exp(−softplus(Λ)·sigmoid(rec))`; SLSRec eq. (11) computes
`a_t = exp(c·Log(Sigmoid(λ))·r_t)` with `c = 8`. Since
`Log(Sigmoid(λ)) = −softplus(−λ)`, the two are the same family; SLSRec's fixed
`c = 8` is the difference. **Decision:** implement eq. (11) exactly, and choose
`λ_j = logit(a_j^{1/c})` so that, with `r_t = 1`, `a` spans `[0.9, 0.999]` —
the same calibration convention RecBLR uses. `--bdlru_lambda_init` switches
between `linspace` (RecBLR, default) and `uniform` (LRURec-style random).

### 8.14 Weight decay on the recurrence parameter `λ`
LRURec's optimizer excludes only names containing `bias` or `layer_norm` from
weight decay, so its recurrence parameterisation *is* decayed; RecBLR trains with
`weight_decay: 0.0` throughout. The paper says its parameters are "configured
identically to those in LRURec" with weight decay 1e-2. **Decision:** follow the
literal reading — `λ` is decayed like every other weight. At `lr·wd = 1e-5` per
step this shrinks `λ` by roughly a factor `e^{-0.5}` over 50k steps, which lowers
the effective `a_t` range; `--exclude_lambda_from_decay=true` reproduces RecBLR's
convention instead.

---

## 9. Measured results

All runs below were executed on **CPU** (no GPU on this machine), ML-1M, seed 42.
They are smoke runs, not converged models: the paper trains for up to 1000 epochs,
these ran for 1–4. **No claim of numerical reproduction of Table II is made here.**

### 9.1 Stage 1 — original LRURec baseline (unmodified)

```bash
python train.py --dataset_code=ml-1m --num_epochs=1 --num_workers=0
```

* Preprocessing found the cached `ml-1m_min_rating0-min_uc5-min_sc5-leave_one_out`
  dataset (6040 users, 3416 items).
* `Total parameters: 322841`, `Encoder parameters: 104025`.
* Forward/backward fine; loss finite and monotonically decreasing over the epoch:
  `8.148 → 7.650`.
* Validation ran, `Update Best Recall@10 Model at 1` written, `test()` reloaded
  `best_acc_model.pth` and produced `test_metrics.json`.
* Padding item 0 excluded from ranking (`scores[:, 0] = -1e9`).

### 9.2 Stage 6 — all four ablations on real ML-1M data

```bash
python scripts/smoke_ablations.py --dataset_code=ml-1m --num_workers=0 \
    --smoke_iters=10 --smoke_val_batches=4
```

10 optimisation steps + 4 full-sort validation batches per ablation. The point of
this run is that each ablation is *selectable by configuration* and that exactly
the intended loss terms are active — not the metric values, which are meaningless
after 10 steps.

| Ablation | Params | `L_rec` (step 1 → 10) | `L_perturb` | `L_uncertainty` | `L_total` (step 10) | s/iter |
|---|---|---|---|---|---|---|
| A — BD-LRU + CE | 447,808 | 8.147 → 8.086 | 0 (off) | 0 (off) | 8.086 | 1.79 |
| B — + SAM gate | 547,392 | 8.147 → 8.093 | 0 (off) | 0 (off) | 8.093 | 1.92 |
| C — + perturbation | 547,392 | 8.147 → 8.089 | 8.146 → 8.086 | 0 (off) | 16.175 | 3.82 |
| D — full SLSRec | 828,288 | 8.145 → 8.088 | 8.146 → 8.084 | 8.227 → 7.807 | 18.514 | 3.85 |

`L_total` for D at step 1 is `8.1451 + 8.1463 + 0.3 × 8.2269 = 18.7595`, exactly
eq. (23). The perturbed pass roughly doubles the step time, as expected.

Cross-check of the two scan implementations on real data
(`--scan_impl=sequential`, 2 steps): step-1 `L_total` values are **identical** to
the parallel path — `8.1470 / 8.1467 / 16.2928 / 18.7595` for A/B/C/D — at ~3.5×
the wall-clock cost.

### 9.3 Stage 6 — full SLSRec, multi-epoch ML-1M smoke run

Two runs were executed. First, the one-epoch run the task asks for (with the
protocol's `--val_iterations=500`, which one epoch of 56 batches never reaches, so
only the initial validation fires):

```bash
python train.py --dataset_code=ml-1m --model_code=slsrec --num_epochs=1 --num_workers=0
```

Then a three-epoch run with a shortened validation cadence so that model
selection, checkpointing and test-time reload are all exercised on a *trained*
model:

```bash
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 \
    --num_epochs=3 --val_iterations=80 --num_workers=0 --exp_name=smoke3ep_D_full_slsrec
```

`Total parameters: 828288`, `Encoder parameters: 365056`.

Per-epoch training averages of the four logged terms
(`experiments/slsrec/smoke3ep_D_full_slsrec/train_losses.jsonl`):

| Epoch | iters | `L_rec` | `L_perturb` | `L_uncertainty` | `L_total` |
|---|---|---|---|---|---|
| 1 | 56 | 7.7280 | 7.7257 | 4.4296 | 16.7826 |
| 2 | 112 | 7.2744 | 7.2783 | 0.6549 | 14.7492 |
| 3 | 168 | 6.7422 | 6.7413 | 0.2565 | 13.5604 |

Every term is finite and decreasing, and `L_total` equals
`L_rec + L_perturb + 0.3 × L_uncertainty` at every epoch (e.g. epoch 3:
`6.7422 + 6.7413 + 0.3 × 0.2565 = 13.5604`).

Validation (full sort, 6040 users) and checkpoint selection:

| accum_iter | val Recall@10 | val NDCG@10 | checkpoint |
|---|---|---|---|
| 0 (untrained) | 0.0042 | 0.0019 | `Update Best Recall@10 Model at 1` |
| 80 | 0.0539 | 0.0278 | `Update Best Recall@10 Model at 2` |
| 160 | 0.1552 | 0.0858 | `Update Best Recall@10 Model at 3` |

Test metrics after reloading `best_acc_model.pth`
(`experiments/slsrec/smoke3ep_D_full_slsrec/test_metrics.json`):

| Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | MRR@10 |
|---|---|---|---|---|---|
| 0.14972 | 0.21572 | 0.34247 | 0.08192 | 0.09857 | 0.06135 |

For orientation only: the paper's converged ML-1M numbers are Recall@10 0.35595,
Recall@20 0.46732, NDCG@10 0.21332, NDCG@20 0.24141 (Table II). **These are not
comparable** — this run saw 168 of a planned ~28,000 optimisation steps. What the
run does establish is that the pipeline learns (val Recall@10 improves 36× over
the untrained model in three epochs), that checkpoint selection tracks the metric,
and that the reload-and-test path works.

**Observation (not a bug — see §8.4).** `L_uncertainty` collapses ~17× in three
epochs while `L_rec` falls only ~13 %. That is the predicted behaviour of eq. (24)
with `β = 0`: the only gradient on `Log(σ²)` is `−α·e^{−Log σ²}·L_Rec < 0`, so
`σ²` grows monotonically and the `α/σ²` weight on `L_Rec` decays towards zero.
The uncertainty branch therefore acts as a decaying per-sample re-weighting of the
recommendation loss rather than a persistent regulariser. This is reproduced as
specified rather than "fixed"; `--uncertainty_beta` restores the penalty term for
anyone who wants to investigate it.

### 9.4 Checkpointing and evaluation

Both verified working for `lru` and `slsrec`:
`experiments/<model_code>/<exp_name>/models/{best_acc_model.pth,checkpoint-recent.pth}`,
`logs/events.out.tfevents.*` (TensorBoard), `test_metrics.json`, and
`train_losses.jsonl`. `BaseTrainer.test()` reloads `best_acc_model.pth` via
`load_state_dict` and re-evaluates — no key mismatches for either model.

---

## 10. Remaining steps for a full five-seed reproduction

1. Run each of the four ablations to convergence on a GPU (§11 commands),
   seed 42 first.
2. Repeat for seeds `{42, 43, 44, 45, 46}` and report mean ± std of
   Recall@{10,20}, NDCG@{10,20}, MRR@{10,20}.
3. Extend to Beauty, Video, Sports and Steam (`--dataset_code`), remembering the
   paper's per-dataset hyperparameters: `max_len = 50`, `dropout ∈ {0,0.25,0.5,0.75,0.95}`
   for Beauty/Sports and `{0,0.2,0.4,0.6,0.8}` for the rest, `α = 0.01` for Beauty
   and `0.3` elsewhere (already wired into `config.py::set_slsrec_template`).
4. Optionally re-run with `--val_strategy=epoch --val_epochs=5 --num_epochs=1000`
   to match the paper's stated schedule exactly (§8.1) and compare.
5. Optionally enable `--scan_impl=triton` on CUDA and confirm it matches the
   PyTorch scan to `< 1e-4` (the test harness already parameterises over scan
   implementations; only the CUDA path is untested here).

---

## 11. Commands

### Full runs (GPU)

`args.device` is set to `cuda` automatically whenever `torch.cuda.is_available()`,
so no device flag is needed. Peak memory is dominated by the `[128, 200, 3417]`
logits tensor, which the perturbed pass duplicates — budget ~8 GB; halve
`--train_batch_size` if that is tight.

```bash
# A -- BD-LRU + CE
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 \
    --use_sam_gate=false --sam_conv=false \
    --use_perturbation=false --use_uncertainty=false \
    --exp_name=A_bdlru_ce
```

```bash
# B -- BD-LRU + self-attention gate + CE
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 \
    --use_perturbation=false --use_uncertainty=false \
    --exp_name=B_sam_gate
```

```bash
# C -- B + uniform perturbation injection
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 \
    --use_perturbation=true --use_uncertainty=false \
    --exp_name=C_perturbation
```

```bash
# D -- full SLSRec (C + Gibbs uncertainty scaling)
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 \
    --exp_name=D_full_slsrec
```

Paper-literal schedule variant (§8.1):

```bash
python train.py --dataset_code=ml-1m --model_code=slsrec --seed=42 \
    --val_strategy=epoch --val_epochs=5 --num_epochs=1000 \
    --exp_name=D_full_slsrec_paper_schedule
```

Five-seed sweep:

```bash
for s in 42 43 44 45 46; do python train.py --dataset_code=ml-1m --model_code=slsrec --seed=$s --exp_name=D_seed$s; done
```

### Reference / baseline

```bash
python train.py --dataset_code=ml-1m --num_workers=0
```

### Tests and smoke checks

```bash
python -m pytest tests/ -q
```

```bash
python scripts/smoke_ablations.py --dataset_code=ml-1m --num_workers=0 --smoke_iters=10 --smoke_val_batches=4
```

---

## 12. Files

### New

| File | Purpose |
|---|---|
| `model/bdlru.py` | BD-LRU (eq. 9–12), perturbation (eq. 17), causal depth-wise conv, three scan implementations |
| `model/slsrec.py` | Embedding (eq. 1), SG-LRU (eq. 2–8), PFFN (eq. 13–14), prediction head (eq. 15), uncertainty estimator (eq. 20) |
| `model/triton_scan.py` | RecBLR's Triton scan, copied verbatim (optional, `--scan_impl=triton`) |
| `trainer/slsrec.py` | `slsrec_losses` (eq. 16, 18, 19–24) + `SLSRecTrainer` |
| `tests/conftest.py` | test fixtures / synthetic left-padded batches |
| `tests/test_scan.py`, `test_causality.py`, `test_padding.py`, `test_perturbation.py`, `test_losses.py`, `test_tiny_overfit.py` | 45 unit tests |
| `configs/slsrec_ml1m.yaml` | resolved ML-1M configuration |
| `scripts/smoke_ablations.py` | short real-data smoke test over all four ablations |
| `requirements-minimal.txt`, `README.md`, `reproduction_report.md`, `pytest.ini`, `.gitignore` | packaging / docs |

### Modified (from upstream LRURec)

| File | Change |
|---|---|
| `config.py` | constants moved above the package imports (fixes a latent circular import); `str2bool`; `--model_code` selectable; `--bert_max_len` / `--val_iterations` no longer clobber explicit CLI values; `set_slsrec_template`; ~25 SLSRec flags; `parse_known_args` so the module is importable under pytest |
| `train.py` | dropped the `pytorch_lightning` / `wandb` imports; `model_factory` / `trainer_factory`; `--exp_name` |
| `trainer/base.py` | logs decomposed loss terms (`self.loss_parts`) to tqdm + `train_losses.jsonl`; `--val_epochs`; `--exclude_lambda_from_decay` |
| `trainer/utils.py` | appended a local `seed_everything` (replaces pytorch-lightning) |
| `model/__init__.py`, `trainer/__init__.py` | factories + exports |

### Unchanged (byte-identical to upstream LRURec)

`model/lru.py`, `trainer/lru.py`, `trainer/loggers.py`, `dataloader/base.py`,
`dataloader/sas.py`, `dataloader/__init__.py`, `datasets/base.py`,
`datasets/ml_1m.py`, `datasets/utils.py`, `datasets/__init__.py` — verified with
`diff` against a fresh clone. The original LRURec baseline therefore still runs
exactly as published (`python train.py --dataset_code=ml-1m`).
