# Real-Data One-Step Retrospective Summary

Run timestamp (UTC): `2026-03-27T21:00:23.262772+00:00`
Dataset: `data/cvtdb_v2_0_0_no_audit.sqlite`
Identification margin (BIC gap): `2.00`

## Summary Table

| Series | Oracle | Initial Margin | CART Pick | CART Margin | Disagreement Pick | Disagreement Margin | Hidden Best Block | Hidden Best Margin | Random E[margin] |
|---|---|---:|---|---:|---|---:|---|---:|---:|
| 1,2-Dichloroethane oral | A | -78.091 | E1 | -3.678 | E1 | -3.678 | E3 | 3.308 | -3.981 |
| Dichloromethane oral | A | -71.065 | E1 | -4.637 | E1 | -4.637 | E3 | 3.357 | -3.834 |
| Trichloroethylene oral | C | 32.083 | E1 | 55.558 | E1 | 55.558 | E2 | 59.276 | 41.466 |
| Chloroform oral | A | -14.379 | E1 | -0.933 | E1 | -0.933 | E1 | -0.933 | -11.977 |

**CARTOGRAPH vs Disagreement (one-step oracle margin)**: 0W / 4T / 0L

## Per-Series Candidate Details

### 1,2-Dichloroethane oral
- Oracle full-data model: `A`
- Initial sparse times: `[0.0, 0.16667, 0.31667, 1.0, 3.0, 5.0]`
- Initial oracle margin: `-78.091`
- Hidden best one-step block: `E3` with oracle margin `3.308`

| Block | Times | CART Score | Disagreement Score | Oracle Margin | Identified? | Best Model |
|---|---|---:|---:|---:|---|---|
| E1 | `[0.03333, 0.06667, 0.1, 0.19215, 0.21667, 0.26667]` | 0.000 | 1.925 | -3.678 | no | C |
| E2 | `[0.36667, 0.41667, 0.5, 0.7778, 1.5]` | 0.000 | 0.062 | -11.572 | no | C |
| E3 | `[2.0, 2.5, 3.5, 4.0, 4.5]` | 0.000 | 0.072 | 3.308 | yes | A |

### Dichloromethane oral
- Oracle full-data model: `A`
- Initial sparse times: `[0.0, 0.14384, 0.31667, 0.76712, 2.5, 5.0]`
- Initial oracle margin: `-71.065`
- Hidden best one-step block: `E3` with oracle margin `3.357`

| Block | Times | CART Score | Disagreement Score | Oracle Margin | Identified? | Best Model |
|---|---|---:|---:|---:|---|---|
| E1 | `[0.03333, 0.06667, 0.1, 0.16667, 0.21667]` | 0.000 | 1.487 | -4.637 | no | C |
| E2 | `[0.26667, 0.36667, 0.41667, 0.5, 1.0]` | 0.000 | 0.030 | -10.223 | no | C |
| E3 | `[1.5, 2.0, 3.0, 3.5, 4.0]` | 0.000 | 0.034 | 3.357 | yes | A |

### Trichloroethylene oral
- Oracle full-data model: `C`
- Initial sparse times: `[0.0, 0.14476, 0.31667, 0.5, 2.0, 4.0]`
- Initial oracle margin: `32.083`
- Hidden best one-step block: `E2` with oracle margin `59.276`

| Block | Times | CART Score | Disagreement Score | Oracle Margin | Identified? | Best Model |
|---|---|---:|---:|---:|---|---|
| E1 | `[0.03333, 0.06667, 0.1, 0.16667, 0.21667]` | 1.133 | 9.704 | 55.558 | yes | C |
| E2 | `[0.26667, 0.36667, 0.41667, 0.76935, 1.0]` | 0.004 | 1.291 | 59.276 | yes | C |
| E3 | `[1.5, 2.5, 3.0, 3.5]` | 0.000 | 0.261 | 9.564 | yes | C |

### Chloroform oral
- Oracle full-data model: `A`
- Initial sparse times: `[0.0, 0.1, 0.21667, 0.41667, 1.0, 2.5]`
- Initial oracle margin: `-14.379`
- Hidden best one-step block: `E1` with oracle margin `-0.933`

| Block | Times | CART Score | Disagreement Score | Oracle Margin | Identified? | Best Model |
|---|---|---:|---:|---:|---|---|
| E1 | `[0.03333, 0.06667, 0.13697, 0.16667]` | 0.271 | 2.224 | -0.933 | no | B |
| E2 | `[0.26667, 0.31667, 0.36667, 0.5]` | 0.001 | 0.199 | -15.792 | no | C |
| E3 | `[0.74989, 1.5, 2.0]` | 0.000 | 0.047 | -19.206 | no | C |

## Artifact Paths

- `figure_1_oracle_margins`: `outputs/real_data_one_step/figure_1_oracle_margins.png`
- `one_step_results_json`: `outputs/real_data_one_step/one_step_results.json`
- `one_step_summary_md`: `outputs/real_data_one_step/one_step_summary.md`