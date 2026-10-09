### ml-1m -- test metrics (mean over seeds; std in parentheses)

| variant | seeds | R@10 | R@20 | N@10 | N@20 | paper R@20 | gap vs paper |
|---|---|---|---|---|---|---|---|
| wo_u | 42 | 0.3167 | 0.4227 | 0.1866 | 0.2133 | 0.4599 | -8.09% |
| p_clean_gates | 42 | 0.3123 | 0.4163 | 0.1826 | 0.2088 | - | - |
| p_shuffle | 42 | 0.3139 | 0.4213 | 0.1840 | 0.2111 | - | - |
| p_layer | 42 | 0.3133 | 0.4190 | 0.1843 | 0.2110 | - | - |

### ml-1m -- training diagnostics (mean over seeds)

| variant | best ep | epochs | minutes | val R@20 | val-test R@20 | val CE | train L_rec | L_per-L_rec | a*L_unc/L_rec | note |
|---|---|---|---|---|---|---|---|---|---|---|
| wo_u | 20 | 20 | 9.3 | 0.4520 | +0.0293 | 5.569 | 5.356 | +0.0025 | - |  |
| p_clean_gates | 20 | 20 | 9.1 | 0.4492 | +0.0328 | 5.581 | 5.353 | +0.0038 | - | P only in the input term of eq. (12); r_t, i_t from the clean x_t |
| p_shuffle | 20 | 20 | 9.3 | 0.4463 | +0.0250 | 5.586 | 5.361 | +0.0034 | - | P at the BD-LRU input, vector borrowed from another sequence |
| p_layer | 20 | 20 | 15.9 | 0.4456 | +0.0267 | 5.589 | 5.373 | +0.1145 | - | P on each layer input X^(l-1): gate branch, BD-LRU and residual |

`L_per-L_rec`: last-epoch cost of the perturbed pass; ~0 = perturbation is inert. `a*L_unc/L_rec`: last-epoch extra weight of the uncertainty term; ~0 = sigma^2 sits at the clamp and U is off.  Seed-to-seed std of test R@20 on ML-1M measured here is 0.002 (w/o P&U) to 0.006 (LRURec), so a single-seed difference below ~1% is not a finding.
