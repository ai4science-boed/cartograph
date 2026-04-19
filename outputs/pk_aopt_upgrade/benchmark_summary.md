# PK A-opt Upgrade Benchmark Summary

Run timestamp (UTC): `2026-03-28T07:01:29.031292+00:00`

Identification margin: `0.05`
Prior variance: `1.00`
Noise variance: `1.00`
Fallback to weakest direction when unresolved is empty: `False`

## Method Sequences

- **Raw CARTOGRAPH**: `['E1', 'E2', 'E4', 'E5', 'E3', 'E7', 'E6']`
- **Exact unresolved A-opt**: `['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']`
- **Disagreement**: `['E2', 'E1', 'E6', 'E7', 'E5', 'E3', 'E4']`

## Primary Table

| Truth | Oracle | Raw CART | A-opt | Disagreement | Random E[round] |
|---|---|---:|---:|---:|---:|
| absorption_variant | B | 1 | 1 | 2 | 4.00 |
| absorption_variant_slow | B | 1 | 1 | 1 | 1.14 |
| distribution_variant_easy | C | 0 | 0 | 0 | 0.00 |
| distribution_variant_hard | C | 1 | 1 | 1 | 1.60 |
| distribution_variant_subtle | C | 1 | 1 | 1 | 2.40 |
| mixed_absorption | A | 2 | 2 | 2 | 5.33 |
| mixed_balanced | A | 2 | 2 | 2 | 5.33 |

**A-opt vs Disagreement**: 1W / 6T / 0L

**A-opt vs Raw CARTOGRAPH**: 0W / 7T / 0L

## Artifact Paths

- `figure_1_rounds_heatmap`: `outputs/pk_aopt_upgrade/figure_1_rounds_heatmap.png`
- `figure_2_sequence_comparison`: `outputs/pk_aopt_upgrade/figure_2_sequence_comparison.png`
- `figure_3_gap_progress`: `outputs/pk_aopt_upgrade/figure_3_gap_progress.png`
- `benchmark_results_json`: `outputs/pk_aopt_upgrade/benchmark_results.json`
- `benchmark_summary_md`: `outputs/pk_aopt_upgrade/benchmark_summary.md`