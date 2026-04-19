# Real-Data Validation: Final And Latest Results

Run timestamp (UTC): `2026-03-27T21:00:23.262772+00:00`
Dataset: `data/cvtdb_v2_0_0_no_audit.sqlite`
Protocol: one-step retrospective on dense oral concentration-time curves from EPA CvTdb, using sparse initial points plus three held-out candidate time blocks.

## Evaluated Series

- `122` — `1,2-Dichloroethane oral`
- `120` — `Dichloromethane oral`
- `123` — `Trichloroethylene oral`
- `65744` — `Chloroform oral`

## Top-Line Result

- `CARTOGRAPH vs Disagreement (one-step oracle margin)`: `0W / 4T / 0L`
- `CARTOGRAPH matched the hidden best one-step block`: `1 / 4`
- The real-data study currently supports the **low-dimensional near-tie** story more than a clear real-data superiority story.

## Per-Series Outcome

| Series | Full-data oracle | CART pick | CART oracle margin | Hidden best block | Hidden best margin |
|---|---|---|---:|---|---:|
| 1,2-Dichloroethane oral | A | E1 | -3.678 | E3 | 3.308 |
| Dichloromethane oral | A | E1 | -4.637 | E3 | 3.357 |
| Trichloroethylene oral | C | E1 | 55.558 | E2 | 59.276 |
| Chloroform oral | A | E1 | -0.933 | E1 | -0.933 |

## Honest Takeaway

- The real-data validation is **complete and runnable**, but it is not a new empirical win over disagreement.
- On these four EPA oral series, both methods choose the same first held-out block.
- In three of four cases, the chosen block is **not** the hidden best block for improving the full-data oracle-model BIC margin.
- This makes the real-data result useful as a realism check and limitation statement, not as a new headline benchmark.

## Artifacts

- Summary: `outputs/real_data_one_step/one_step_summary.md`
- Machine-readable results: `outputs/real_data_one_step/one_step_results.json`
- Figure: `outputs/real_data_one_step/figure_1_oracle_margins.png`
