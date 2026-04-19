# A-opt Pivot Validation Summary

Run timestamp (UTC): `2026-03-28T07:13:38.817657+00:00`

This artifact consolidates the upgraded method story after the unresolved
A-opt pivot: structured high-dimensional gain, preserved low-dimensional
boundary behavior, real-data realism, and refusal-case stability.

## Cascade Robustness

| d | Trials | Raw hidden-best | A-opt hidden-best | Disagreement hidden-best | Raw regret | A-opt regret | Disagreement regret | A-opt vs Raw |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 144 | 0.44 | 0.44 | 0.44 | 0.0518 | 0.0518 | 0.0518 | 0W / 144T / 0L |
| 4 | 144 | 0.00 | 0.09 | 0.00 | 0.3119 | 0.4182 | 0.3119 | 73W / 0T / 71L |
| 8 | 144 | 0.02 | 0.65 | 0.02 | 19.9404 | 0.0102 | 19.9404 | 129W / 0T / 15L |
| 16 | 144 | 0.07 | 0.72 | 0.07 | 2.8322 | 0.0142 | 2.8322 | 120W / 0T / 24L |

## PK Boundary Case

- A-opt vs disagreement: `1W / 6T / 0L`
- A-opt vs raw CARTOGRAPH: `0W / 7T / 0L`
- Mean rounds: raw `1.14`, A-opt `1.14`, disagreement `1.29`

## EPA Real-Data Check

- Dataset: `data/cvtdb_v2_0_0_no_audit.sqlite`
- Degenerate initial unresolved space on `2 / 4` series
- A-opt vs raw CARTOGRAPH: `0W / 4T / 0L`
- A-opt vs disagreement: `0W / 4T / 0L`
- Mean oracle margin: raw `11.578`, A-opt `11.578`, disagreement `11.578`

| Series | Unresolved dim | Raw pick | A-opt pick | Disagreement pick | Hidden best |
|---|---:|---|---|---|---|
| 1,2-Dichloroethane oral | 0 | E1 | E1 | E1 | E3 |
| Dichloromethane oral | 0 | E1 | E1 | E1 | E3 |
| Trichloroethylene oral | 1 | E1 | E1 | E1 | E2 |
| Chloroform oral | 1 | E1 | E1 | E1 | E1 |

## Failure Benchmark Preservation

- Raw CARTOGRAPH sequence: `['E1', 'E2', 'E4', 'E5', 'E3']`
- A-opt sequence: `['E1', 'E2', 'E3', 'E4', 'E5']`
- Raw refusal/control status: failures refused=`True`, control identified=`True`
- A-opt refusal/control status: failures refused=`True`, control identified=`True`

| Scenario | Type | Raw final ID | A-opt final ID | Raw best | A-opt best |
|---|---|---|---|---|---|
| timevarying_strong | failure | no | no | A | A |
| saturable_elimination | failure | no | no | B | B |
| enterohepatic_recirculation | failure | no | no | A | A |
| control_in_library | control | YES | YES | B | B |

## Interpretation

- The upgraded unresolved A-opt rule is meaningfully stronger on the structured high-dimensional cascade benchmark, especially from `d=8` onward.
- PK and EPA remain low-dimensional boundary cases where A-opt changes little or nothing, which is consistent with the earlier scaling story.
- The refusal benchmark survives the pivot: the stronger selector does not collapse the failure-case honesty result.

## Artifact Paths

- `figure_1_cross_benchmark`: `outputs/aopt_pivot/figure_1_cross_benchmark.png`
- `pivot_results_json`: `outputs/aopt_pivot/pivot_results.json`
- `pivot_summary_md`: `outputs/aopt_pivot/pivot_summary.md`