# Structured Cascade Benchmark: Final And Latest Results

Run timestamp (UTC): `2026-03-28T06:50:33.746772+00:00`
Script: `cascade_boed_benchmark.py`

## What This Benchmark Tests

- Shared nonlinear cascade ODE family with explicit mechanism coordinates
- Local Jacobian blocks from ODE sensitivities around a common reference model
- Nonlinear truth evaluation via posterior-mean recovery
- Methods:
  - `CARTOGRAPH` = raw unresolved projection score
  - `BOED A-opt` = exact unresolved posterior-trace reduction
  - `Disagreement` = total sensitivity
  - `Random`

## Main Finding

The high-dimensional result is more nuanced than the original hope:

- Raw `CARTOGRAPH` still behaves very similarly to `Disagreement` on this first structured benchmark.
- Exact unresolved `BOED A-opt` is the method that clearly improves the first decision as dimension grows.
- At `d=2`, all methods are effectively near-tied.
- At `d=4, 8, 16`, `BOED A-opt` has much lower one-step regret and higher hidden-best match rate.

This means the BOED bridge is not just explanatory. It points to a real algorithmic upgrade.

## Primary Numbers

| d | CART hidden-best | BOED hidden-best | Disagreement hidden-best | CART regret | BOED regret | Disagreement regret |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.33 | 0.33 | 0.33 | 0.0527 | 0.0527 | 0.0527 |
| 4 | 0.00 | 0.33 | 0.00 | 0.3127 | 0.0204 | 0.3127 |
| 8 | 0.17 | 0.67 | 0.17 | 19.4321 | 0.0052 | 19.4321 |
| 16 | 0.17 | 0.50 | 0.17 | 2.8786 | 0.0145 | 2.8786 |

## Interpretation

- The current raw score is best viewed as a first-order unresolved-information surrogate.
- Exact unresolved `A-opt` is a stronger acquisition criterion on the same unresolved subspace.
- The benchmark supports moving the paper toward:
  - BOED bridge theory,
  - exact unresolved A-opt as the strengthened rule,
  - raw CARTOGRAPH as the simple special case / first-order approximation.

## Artifacts

- Summary: `outputs/cascade_boed/benchmark_summary.md`
- Machine-readable results: `outputs/cascade_boed/benchmark_results.json`
- Figure 1: `outputs/cascade_boed/figure_1_performance_vs_dimension.png`
- Figure 2: `outputs/cascade_boed/figure_2_sequence_comparison.png`
