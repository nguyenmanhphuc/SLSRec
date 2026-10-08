### ml-1m -- test metrics (mean over seeds; std in parentheses)

| variant | seeds | R@10 | R@20 | N@10 | N@20 | paper R@20 | gap vs paper | R@20 vs slsrec | paper | N@20 vs slsrec | paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lru | 42,43,44 | 0.3272 (0.0020) | 0.4322 (0.0060) | 0.1952 (0.0011) | 0.2217 (0.0022) | 0.4331 | -0.23% | -4.30% | -7.32% | -3.96% | -8.92% |
| recblr | - | - | - | - | - | 0.4498 | - | - | -3.75% | - | -3.13% |
| bdlru | - | - | - | - | - | - | - | - | - | - | - |
| wo_sam | - | - | - | - | - | 0.4612 | - | - | -1.30% | - | -1.15% |
| wo_bdlru | - | - | - | - | - | 0.4582 | - | - | -1.96% | - | -2.24% |
| wo_p | - | - | - | - | - | 0.4580 | - | - | -1.99% | - | -2.64% |
| wo_u* | 42 | 0.3441 | 0.4531 | 0.2046 | 0.2321 | 0.4599 | -1.47% | +0.34% | -1.59% | +0.54% | -1.84% |
| wo_pu | 42,43,44 | 0.3444 (0.0014) | 0.4529 (0.0020) | 0.2044 (0.0012) | 0.2318 (0.0013) | 0.4485 | +0.98% | +0.30% | -4.02% | +0.40% | -4.81% |
| slsrec* | 42 | 0.3511 | 0.4516 | 0.2055 | 0.2309 | 0.4673 | -3.37% | +0.00% | +0.00% | +0.00% | +0.00% |
| slsrec_exact* | 42,43,44 | 0.3435 (0.0028) | 0.4524 (0.0023) | 0.2040 (0.0016) | 0.2315 (0.0008) | - | - | +0.17% | - | +0.28% | - |

### ml-1m -- training diagnostics (mean over seeds)

| variant | best ep | epochs | minutes | val R@20 | val-test R@20 | val CE | train L_rec | L_per-L_rec | a*L_unc/L_rec | note |
|---|---|---|---|---|---|---|---|---|---|---|
| lru | 289 | 378 | 42.4 | 0.4659 | +0.0337 | - | 5.188 | - | - | LRURec baseline, LRURec protocol (500 ep, val every 500 it) |
| recblr | - | - | - | - | - | - | - | - | - | official RecBLR (model/recblr.py), paper protocol |
| bdlru | - | - | - | - | - | - | - | - | - | SLSRec block with BD-LRU only, gate SiLU(L1) -- RecBLR-like, not RecBLR |
| wo_sam | - | - | - | - | - | - | - | - | - | G = Conv1-D1(L1) |
| wo_bdlru | - | - | - | - | - | - | - | - | - | H = Conv1-D2(L2); P lives in the BD-LRU, so L_per only sees new dropout |
| wo_p | - | - | - | - | - | - | - | - | - |  |
| wo_u | 250 | 300 | 18.8 | 0.4787 | +0.0256 | 5.429 | 4.981 | +0.0058 | - |  |
| wo_pu | 228 | 278 | 45.8 | 0.4801 | +0.0272 | 5.443 | 4.990 | - | - | SAM + BD-LRU, plain CE |
| slsrec | 150 | 200 | 65.6 | 0.4814 | +0.0298 | 5.439 | 5.010 | +0.0071 | 0.00010 | full SLSRec, eq. (24) approx L_unc |
| slsrec_exact | 137 | 187 | 56.5 | 0.4814 | +0.0290 | 5.444 | 5.019 | +0.0071 | 0.28654 | full SLSRec, exact eq. (19)/(21) L_unc |

`*` includes a run older than some flag(s) (perturbation_granularity, perturbation_noise, perturbation_site, uncertainty_granularity, uncertainty_weight_norm); taken as the default.
Includes runs from outside this sweep (--include_legacy), matched by run_config.json.

`L_per-L_rec`: last-epoch cost of the perturbed pass; ~0 = perturbation is inert. `a*L_unc/L_rec`: last-epoch extra weight of the uncertainty term; ~0 = sigma^2 sits at the clamp and U is off.  Seed-to-seed std of test R@20 on ML-1M measured here is 0.002 (w/o P&U) to 0.006 (LRURec), so a single-seed difference below ~1% is not a finding.
