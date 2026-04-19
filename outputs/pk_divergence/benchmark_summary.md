# PK Divergence Benchmark Summary

Run timestamp (UTC): `2026-03-27T19:17:45.524258+00:00`

Primary identification margin: `0.05`
Random unresolved round value: `8`
Follow-up experiments: `['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']`

## Method Sequences

- **CARTOGRAPH**: `['E1', 'E2', 'E4', 'E5', 'E3', 'E7', 'E6']`
- **Disagreement**: `['E2', 'E1', 'E6', 'E7', 'E5', 'E3', 'E4']`

**Sequences diverge at round 1**: CARTOGRAPH picks `E1`, Disagreement picks `E2`.

## Primary Table

| Truth | Family | Oracle | CARTOGRAPH | Disagreement | Random E[round] |
|---|---|---|---:|---:|---:|
| absorption_variant | B-family perturbed truth | B | 1 | 2 | 4.00 |
| absorption_variant_slow | B-family slow transit | B | 1 | 1 | 1.14 |
| distribution_variant_easy | C-family perturbed truth | C | 0 | 0 | 0.00 |
| distribution_variant_hard | C-family perturbed truth | C | 1 | 1 | 1.60 |
| distribution_variant_subtle | C-family subtle distribution | C | 1 | 1 | 2.40 |
| mixed_absorption | D-family absorption-leaning mix | A | 2 | 2 | 5.33 |
| mixed_balanced | D-family balanced mix | A | 2 | 2 | 5.33 |

**CARTOGRAPH vs Disagreement**: 1W / 6T / 0L

## Identification-Margin Sensitivity (CARTOGRAPH)

| Truth | Margin 0.03 | Margin 0.05 | Margin 0.07 |
|---|---:|---:|---:|
| absorption_variant | 1 | 1 | 1 |
| absorption_variant_slow | 1 | 1 | 1 |
| distribution_variant_easy | 0 | 0 | 0 |
| distribution_variant_hard | 1 | 1 | 1 |
| distribution_variant_subtle | 1 | 1 | 1 |
| mixed_absorption | 2 | 2 | 2 |
| mixed_balanced | 2 | 2 | 2 |

## Experiment Menu Details

| Key | Name | Route | Times |
|---|---|---|---|
| e0 | Sparse Oral | oral | `[0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 24.0]` |
| E1 | Absorption-Peak Oral | oral | `[0.1, 0.33, 0.75, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0, 24.0]` |
| E2 | Late-Spread Oral | oral | `[0.2, 0.33, 0.5, 1.25, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]` |
| E3 | Mid-Window Oral | oral | `[0.1, 0.33, 0.5, 0.75, 1.25, 2.5, 10.0, 12.0, 20.0, 24.0]` |
| E4 | Early-Cluster Oral | oral | `[0.2, 0.33, 0.75, 1.0, 1.25, 1.5, 2.5, 10.0, 16.0, 24.0]` |
| E5 | Tail-Emphasis Oral | oral | `[0.1, 0.75, 1.25, 1.5, 2.5, 8.0, 10.0, 16.0, 20.0, 24.0]` |
| E6 | IV Dense Early-Mid | iv | `[0.08, 0.17, 0.33, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 24.0]` |
| E7 | IV Spread | iv | `[0.08, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0]` |

## Artifact Paths

- `figure_1_rounds_to_identification`: `outputs/pk_divergence/figure_1_rounds_to_identification.png`
- `figure_2_sequence_comparison`: `outputs/pk_divergence/figure_2_sequence_comparison.png`
- `figure_3_per_round_residuals`: `outputs/pk_divergence/figure_3_per_round_residuals.png`
- `benchmark_results_json`: `outputs/pk_divergence/benchmark_results.json`
- `benchmark_summary_md`: `outputs/pk_divergence/benchmark_summary.md`
