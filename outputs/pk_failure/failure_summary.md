# PK Failure Benchmark Summary

Run timestamp (UTC): `2026-03-26T19:14:12.387663+00:00`
Identification margin: `0.05`
Goodness threshold (norm resid): `0.25`
CARTOGRAPH sequence: `['E1', 'E2', 'E4', 'E5', 'E3']`

## Results

| Scenario | Type | Final Resid | Norm Resid | Gap | Fit OK | Final ID | Revoked? |
|---|---|---:|---:|---:|---|---|---|
| timevarying_strong | failure | 1.8643 | 0.3404 | 0.5626 | **NO** | no | REVOKED |
| saturable_elimination | failure | 1.4244 | 0.2601 | 0.7739 | **NO** | no | REVOKED |
| enterohepatic_recirculation | failure | 1.7780 | 0.3246 | 0.6057 | **NO** | no | REVOKED |
| control_in_library | control | 0.7361 | 0.1344 | 1.3003 | yes | YES |  |

## Interpretation

**All failure truths are correctly unidentified at the final round, while the control truth remains identified.** This demonstrates principled refusal to resolve when the library is genuinely insufficient.

Notably, ['timevarying_strong', 'saturable_elimination', 'enterohepatic_recirculation'] were *tentatively identified at early rounds but identification was revoked* as additional experiments exposed the structural misfit. This is the desired behavior: more data should increase confidence for well-specified models and *decrease* confidence for misspecified ones.

## Per-Round Details

### timevarying_strong (failure)

Description: Strong time-varying clearance (100% at t=0, tau=8h)

| Round | Best Model | Best Resid | Norm Resid | Gap | Gap OK | Fit OK | Identified |
|---:|---|---:|---:|---:|---|---|---|
| 0 | C | 0.3477 | 0.1555 | 0.2426 | yes | yes | YES |
| 1 | C | 0.5107 | 0.1615 | 0.5431 | yes | yes | YES |
| 2 | A | 1.3740 | 0.3548 | 0.9984 | yes | **NO** | no |
| 3 | A | 1.5538 | 0.3474 | 0.8359 | yes | **NO** | no |
| 4 | A | 1.7156 | 0.3431 | 0.6928 | yes | **NO** | no |
| 5 | A | 1.8643 | 0.3404 | 0.5626 | yes | **NO** | no |

### saturable_elimination (failure)

Description: Michaelis-Menten elimination (V_max=1.5, K_m=5)

| Round | Best Model | Best Resid | Norm Resid | Gap | Gap OK | Fit OK | Identified |
|---:|---|---:|---:|---:|---|---|---|
| 0 | A | 0.3969 | 0.1775 | 0.1676 | yes | yes | YES |
| 1 | B | 0.7401 | 0.2340 | 1.3236 | yes | yes | YES |
| 2 | B | 0.8825 | 0.2279 | 1.2688 | yes | yes | YES |
| 3 | B | 1.0908 | 0.2439 | 1.0756 | yes | yes | YES |
| 4 | B | 1.2689 | 0.2538 | 0.9136 | yes | **NO** | no |
| 5 | B | 1.4244 | 0.2601 | 0.7739 | yes | **NO** | no |

### enterohepatic_recirculation (failure)

Description: Bile recirculation loop (k_bile=0.12, k_reabs=0.20)

| Round | Best Model | Best Resid | Norm Resid | Gap | Gap OK | Fit OK | Identified |
|---:|---|---:|---:|---:|---|---|---|
| 0 | C | 0.3087 | 0.1381 | 0.2465 | yes | yes | YES |
| 1 | C | 0.4596 | 0.1453 | 0.5484 | yes | yes | YES |
| 2 | A | 1.3104 | 0.3383 | 1.0280 | yes | **NO** | no |
| 3 | A | 1.4811 | 0.3312 | 0.8710 | yes | **NO** | no |
| 4 | A | 1.6360 | 0.3272 | 0.7319 | yes | **NO** | no |
| 5 | A | 1.7780 | 0.3246 | 0.6057 | yes | **NO** | no |

### control_in_library (control)

Description: Perturbed B-family (should be identified)

| Round | Best Model | Best Resid | Norm Resid | Gap | Gap OK | Fit OK | Identified |
|---:|---|---:|---:|---:|---|---|---|
| 0 | A | 0.1287 | 0.0576 | 0.1706 | yes | yes | YES |
| 1 | B | 0.3456 | 0.1093 | 1.6603 | yes | yes | YES |
| 2 | B | 0.3863 | 0.0997 | 1.6423 | yes | yes | YES |
| 3 | B | 0.5293 | 0.1183 | 1.5019 | yes | yes | YES |
| 4 | B | 0.6411 | 0.1282 | 1.3927 | yes | yes | YES |
| 5 | B | 0.7361 | 0.1344 | 1.3003 | yes | yes | YES |

## Goodness-Threshold Sensitivity

Identification margin fixed at `0.05`. Each cell shows whether the scenario is identified (ID) or not (no) at the final round under the given threshold.

| Threshold | timevarying_strong | saturable_elimination | enterohepatic_recirculation | control_in_library |
|---:|---|---|---|---|
| 0.20 | no | no | no | **ID** |
| 0.25 | no | no | no | **ID** |
| 0.30 | no | **ID** | no | **ID** |
| 0.35 | **ID** | **ID** | **ID** | **ID** |

## Per-Feature Residual Breakdown

Shows which features drive the misfit for each scenario. Higher values indicate the feature detects the out-of-library mechanism.

| Feature | timevarying_strong | saturable_elimination | enterohepatic_recirculation | control_in_library |
|---|---:|---:|---:|---:|
| T_max | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C_max | 1.8339 | 1.3556 | 1.7364 | 0.7317 |
| AUC_frac | 0.0625 | 0.0779 | 0.1100 | 0.0313 |
| terminal_slope | 0.2204 | 0.3769 | 0.2914 | 0.0061 |
| loglin_RMSE | 0.2452 | 0.2076 | 0.2218 | 0.0734 |

## Artifact Paths

- `figure_1_failure_residual_trajectories`: `outputs/pk_failure/figure_1_failure_residual_trajectories.png`
- `figure_2_final_residual_comparison`: `outputs/pk_failure/figure_2_final_residual_comparison.png`
- `figure_3_failure_concentration_profiles`: `outputs/pk_failure/figure_3_failure_concentration_profiles.png`
- `failure_results_json`: `outputs/pk_failure/failure_results.json`
- `failure_summary_md`: `outputs/pk_failure/failure_summary.md`
