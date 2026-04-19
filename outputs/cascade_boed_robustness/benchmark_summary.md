# Cascade BOED Robustness Summary

Run timestamp (UTC): `2026-03-28T07:09:50.353856+00:00`
Dimensions: `[2, 4, 8, 16]`
Noise seeds per dimension: `24`
Rounds evaluated: `3`

This benchmark repeats the structured cascade experiment over many noise seeds
to test whether the exact unresolved A-opt upgrade is robust rather than a
single-seed artifact.

## Primary Table

| d | Trials | Init unresolved dim | Raw hidden-best | A-opt hidden-best | Disagreement hidden-best | Raw regret | A-opt regret | Disagreement regret |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 144 | 1 | 0.44 | 0.44 | 0.44 | 0.0518 | 0.0518 | 0.0518 |
| 4 | 144 | 3 | 0.00 | 0.09 | 0.00 | 0.3119 | 0.4182 | 0.3119 |
| 8 | 144 | 6 | 0.02 | 0.65 | 0.02 | 19.9404 | 0.0102 | 19.9404 |
| 16 | 144 | 12 | 0.07 | 0.72 | 0.07 | 2.8322 | 0.0142 | 2.8322 |

## Pairwise Trial Counts

| d | A-opt vs Raw | A-opt vs Disagreement |
|---:|---|---|
| 2 | 0W / 144T / 0L | 0W / 144T / 0L |
| 4 | 73W / 0T / 71L | 73W / 0T / 71L |
| 8 | 129W / 0T / 15L | 129W / 0T / 15L |
| 16 | 120W / 0T / 24L | 120W / 0T / 24L |

## Runtime Table

| d | Raw seq ms | A-opt seq ms | Disagreement seq ms | Raw score ms | A-opt score ms | Disagreement score ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.2001 | 1.2320 | 0.0091 | 0.0056 | 0.0342 | 0.0003 |
| 4 | 0.1990 | 1.3937 | 0.0149 | 0.0055 | 0.0387 | 0.0004 |
| 8 | 0.5556 | 9.6442 | 0.0136 | 0.0154 | 0.2679 | 0.0004 |
| 16 | 1.1473 | 46.7718 | 0.0194 | 0.0319 | 1.2992 | 0.0005 |

## Method Sequences

### d=2
- Raw CARTOGRAPH: `['E6', 'E4', 'E5', 'E8', 'E2', 'E7', 'E3', 'E1']`
- Exact unresolved A-opt: `['E6', 'E4', 'E5', 'E8', 'E2', 'E7', 'E3', 'E1']`
- Disagreement: `['E6', 'E5', 'E4', 'E8', 'E2', 'E7', 'E3', 'E1']`

### d=4
- Raw CARTOGRAPH: `['E6', 'E5', 'E8', 'E7', 'E4', 'E3', 'E2', 'E1']`
- Exact unresolved A-opt: `['E5', 'E6', 'E7', 'E8', 'E4', 'E3', 'E2', 'E1']`
- Disagreement: `['E6', 'E5', 'E8', 'E4', 'E7', 'E3', 'E2', 'E1']`

### d=8
- Raw CARTOGRAPH: `['E6', 'E5', 'E8', 'E7', 'E4', 'E3', 'E2', 'E1']`
- Exact unresolved A-opt: `['E8', 'E5', 'E6', 'E7', 'E4', 'E3', 'E2', 'E1']`
- Disagreement: `['E6', 'E5', 'E8', 'E7', 'E4', 'E3', 'E2', 'E1']`

### d=16
- Raw CARTOGRAPH: `['E6', 'E8', 'E5', 'E7', 'E4', 'E3', 'E2', 'E1']`
- Exact unresolved A-opt: `['E8', 'E6', 'E7', 'E5', 'E4', 'E3', 'E2', 'E1']`
- Disagreement: `['E6', 'E8', 'E5', 'E7', 'E4', 'E3', 'E2', 'E1']`

## Interpretation

- `A-opt` uses exact posterior-trace reduction on the current unresolved basis.
- `Raw CARTOGRAPH` uses the unresolved projection score without posterior weighting.
- `Disagreement` ignores the unresolved basis and scores total sensitivity.
- The main robustness question is whether the exact A-opt advantage persists over many noise seeds in the structured nonlinear benchmark.

## Artifact Paths

- `figure_1_robust_performance`: `outputs/cascade_boed_robustness/figure_1_robust_performance.png`
- `figure_2_runtime`: `outputs/cascade_boed_robustness/figure_2_runtime.png`
- `benchmark_results_json`: `outputs/cascade_boed_robustness/benchmark_results.json`