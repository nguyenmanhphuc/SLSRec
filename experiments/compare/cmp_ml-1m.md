### ml-1m -- test metrics (mean over seeds; std in parentheses)

| variant | seeds | R@10 | R@20 | N@10 | N@20 | paper R@20 | gap vs paper |
|---|---|---|---|---|---|---|---|
| recblr | 42 | 0.3452 | 0.4529 | 0.2078 | 0.2349 | 0.4498 | +0.69% |

### ml-1m -- training diagnostics (mean over seeds)

| variant | best ep | epochs | minutes | val R@20 | val-test R@20 | val CE | train L_rec | L_per-L_rec | a*L_unc/L_rec | note |
|---|---|---|---|---|---|---|---|---|---|---|
| recblr | 365 | 415 | 92.2 | 0.4870 | +0.0341 | 5.424 | 5.044 | - | - | official RecBLR (model/recblr.py), paper protocol |

`L_per-L_rec`: last-epoch cost of the perturbed pass; ~0 = perturbation is inert. `a*L_unc/L_rec`: last-epoch extra weight of the uncertainty term; ~0 = sigma^2 sits at the clamp and U is off.  Seed-to-seed std of test R@20 on ML-1M measured here is 0.002 (w/o P&U) to 0.006 (LRURec), so a single-seed difference below ~1% is not a finding.
