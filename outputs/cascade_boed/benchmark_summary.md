# Structured Cascade Benchmark Summary

Run timestamp (UTC): `2026-03-28T06:50:33.746772+00:00`

This benchmark uses a shared nonlinear cascade ODE family with explicit
mechanism coordinates, local Jacobian blocks from ODE sensitivities, and
nonlinear truth evaluation through posterior-mean recovery.

## Primary Table

| d | Init unresolved dim | CART final MSE | BOED final MSE | Disagreement final MSE | Random E[MSE] | CART hidden-best | BOED hidden-best | Disagreement hidden-best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.0464 | 0.0464 | 0.0464 | 0.0474 | 0.33 | 0.33 | 0.33 |
| 4 | 3 | 0.3311 | 0.3311 | 0.3311 | 0.2008 | 0.00 | 0.33 | 0.00 |
| 8 | 6 | 0.1479 | 0.1479 | 0.1479 | 6.1008 | 0.17 | 0.67 | 0.17 |
| 16 | 12 | 80.1184 | 83.9876 | 80.1184 | 92331.0912 | 0.17 | 0.50 | 0.17 |

## Method Sequences

### d=2
- Initial unresolved dimension: `1`
- CARTOGRAPH: `['E6', 'E4', 'E5', 'E8', 'E2', 'E7', 'E3', 'E1']`
- BOED A-opt: `['E6', 'E4', 'E5', 'E8', 'E2', 'E7', 'E3', 'E1']`
- Disagreement: `['E6', 'E5', 'E4', 'E8', 'E2', 'E7', 'E3', 'E1']`

### d=4
- Initial unresolved dimension: `3`
- CARTOGRAPH: `['E6', 'E5', 'E8', 'E7', 'E4', 'E3', 'E2', 'E1']`
- BOED A-opt: `['E8', 'E6', 'E5', 'E7', 'E4', 'E3', 'E2', 'E1']`
- Disagreement: `['E6', 'E5', 'E8', 'E4', 'E7', 'E3', 'E2', 'E1']`

### d=8
- Initial unresolved dimension: `6`
- CARTOGRAPH: `['E6', 'E5', 'E8', 'E7', 'E4', 'E3', 'E2', 'E1']`
- BOED A-opt: `['E8', 'E5', 'E6', 'E7', 'E4', 'E3', 'E2', 'E1']`
- Disagreement: `['E6', 'E5', 'E8', 'E7', 'E4', 'E3', 'E2', 'E1']`

### d=16
- Initial unresolved dimension: `12`
- CARTOGRAPH: `['E6', 'E8', 'E5', 'E7', 'E4', 'E3', 'E2', 'E1']`
- BOED A-opt: `['E8', 'E6', 'E7', 'E5', 'E4', 'E3', 'E2', 'E1']`
- Disagreement: `['E6', 'E8', 'E5', 'E7', 'E4', 'E3', 'E2', 'E1']`

## Interpretation

- `BOED A-opt` is the exact unresolved posterior-trace reduction baseline on the current unresolved basis.
- `CARTOGRAPH` uses the unresolved projection score `||H_e U_tau||_F^2`.
- `Disagreement` uses total sensitivity `||H_e||_F^2`.
- The key question is whether CARTOGRAPH tracks BOED more closely than disagreement as `d` grows.

## Artifact Paths

- `figure_1_performance_vs_dimension`: `outputs/cascade_boed/figure_1_performance_vs_dimension.png`
- `figure_2_sequence_comparison`: `outputs/cascade_boed/figure_2_sequence_comparison.png`
- `benchmark_results_json`: `outputs/cascade_boed/benchmark_results.json`