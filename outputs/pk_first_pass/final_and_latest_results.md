# PK First-Pass Final and Latest Results

Run timestamp (UTC): `2026-03-26T10:02:14.137115+00:00`

## Latest Run

- Initial experiment: `e0`
- Tau ratio: `0.20`
- Initial singular values: `[0.47978891635571547, 0.07917922208291155]`
- Initial sigma ratio: `0.165029`
- Initial unresolved dimension (dynamic tau): `1`
- Initial unresolved dimension (fixed tau): `1`
- Selected experiment by unresolved-subspace score: `A`
- Selected experiment by disagreement magnitude: `A`

### Parameter Snapshot

- `A`: `{'k_a': 0.8, 'k_e': 0.18, 'V': 20.0}`
- `B`: `{'k_tr': 1.2, 'k_a': 0.8, 'k_e': 0.18, 'V': 20.0}`
- `C`: `{'k_a': 0.7, 'k_10': 0.16, 'k_12': 0.2, 'k_21': 0.25, 'V_c': 18.0}`
- `truth`: `{'k_tr': 1.6, 'k_a': 1.0, 'k_10': 0.16, 'k_12': 0.2, 'k_21': 0.25, 'V_c': 18.5}`

### Initial Ambiguity Check

- Sparse-oral feature distance between `B` and `C`: `0.265816`
- Sparse-oral sampled-curve distance between `B` and `C`: `1.547359`
- Truth residual vs `A` under `e0`: `0.312396`
- Truth residual vs `B` under `e0`: `0.141630`
- Truth residual vs `C` under `e0`: `0.141326`

### Candidate Scores

- `A`: unresolved_score=`0.844017`, sigma_min=`0.918704`, disagreement_mag=`2.386225`
- `C`: unresolved_score=`0.061383`, sigma_min=`0.247756`, disagreement_mag=`1.223778`
- `B`: unresolved_score=`0.006167`, sigma_min=`0.078533`, disagreement_mag=`0.916631`

### Threshold Sensitivity

- tau_ratio=`0.18`: unresolved_dim=`1`, winner=`A`
- tau_ratio=`0.20`: unresolved_dim=`1`, winner=`A`
- tau_ratio=`0.22`: unresolved_dim=`1`, winner=`A`

### One-Step Update Metrics

- `A`: sigma_min=`0.534092`, sigma_ratio=`0.504267`, sigma_ratio_gain=`0.339238`, unresolved_dim_dynamic=`0`, unresolved_dim_fixed=`0`
- `B`: sigma_min=`0.111509`, sigma_ratio=`0.164094`, sigma_ratio_gain=`-0.000935`, unresolved_dim_dynamic=`1`, unresolved_dim_fixed=`0`
- `C`: sigma_min=`0.173779`, sigma_ratio=`0.228050`, sigma_ratio_gain=`0.063021`, unresolved_dim_dynamic=`0`, unresolved_dim_fixed=`0`

### Residual Norms Under Selected Experiment

- `A`: truth_resid=`1.052456`, failure_resid=`1.102875`, failure_gap=`0.050419`
- `B`: truth_resid=`0.133244`, failure_resid=`0.259030`, failure_gap=`0.125786`
- `C`: truth_resid=`1.000972`, failure_resid=`1.008093`, failure_gap=`0.007121`

## Current Final Takeaways

- The primary truth is an out-of-library combined transit-plus-distribution model, so no library model has zero residual by construction.
- The sparse-oral setup makes `B` and `C` genuinely close in feature space (`0.265816`), which is the intended ambiguity regime.
- At the observed sparse samples, `B` and `C` remain close (`1.547359`), which is why the feature-space ambiguity is not just an artifact of full continuous trajectories.
- The unresolved-subspace rule and raw disagreement magnitude both select `A` in this tuned pass.
- The added value of the structured rule here is the singular-value-based justification and the clearer separation between genuinely useful and weak follow-up experiments.
- The best one-step update improves sigma ratio from `0.165029` to `0.504267`.
- Dynamic threshold collapse is still conservative, so the continuous singular-value improvement is the more informative ambiguity metric in this pass.

## Artifact Paths

- `figure_1_initial_curves`: `outputs/pk_first_pass/figure_1_initial_curves.png`
- `figure_2_acquisition_scores`: `outputs/pk_first_pass/figure_2_acquisition_scores.png`
- `figure_3_update_metrics`: `outputs/pk_first_pass/figure_3_update_metrics.png`
- `figure_4_selected_experiment_curves`: `outputs/pk_first_pass/figure_4_selected_experiment_curves.png`
- `figure_5_feature_space_summary`: `outputs/pk_first_pass/figure_5_feature_space_summary.png`
- `figure_6_failure_residuals`: `outputs/pk_first_pass/figure_6_failure_residuals.png`
- `latest_results_json`: `outputs/pk_first_pass/latest_results.json`
- `final_and_latest_results_md`: `outputs/pk_first_pass/final_and_latest_results.md`
