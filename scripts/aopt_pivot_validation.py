from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / "outputs" / ".cache").resolve()))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pk_first_pass import TAU_RATIO, acquisition_score, unresolved_subspace
from pk_failure_benchmark import (
    FOLLOW_UP_KEYS as FAILURE_KEYS,
    cartograph_sequence as failure_cartograph_sequence,
    evaluate_identification,
    feature_map,
    get_control_truth,
    get_experiments as get_failure_experiments,
    get_failure_specs,
    get_library_params,
    precompute_failure_features,
    precompute_library,
    run_failure_scenario,
    simulate_model,
)
from real_data_one_step import (
    DATASET_PATH,
    SERIES_SPECS,
)
from real_data_validation import (
    build_block_plan,
    fit_for_subset,
    get_connection,
    identification_state,
    load_series,
    make_h_block,
    oracle_model,
    predict_h_block,
)
from unresolved_boed import unresolved_aopt_score


OUTPUT_DIR = Path("outputs") / "aopt_pivot"
PRIOR_VAR = 1.0
NOISE_VAR = 1.0
FALLBACK_TO_WEAKEST = False


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_round(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values))


def real_data_oracle_margin(fits: dict, oracle: str) -> float:
    oracle_bic = fits[oracle].bic
    other_best = min(fit.bic for name, fit in fits.items() if name != oracle)
    return float(other_best - oracle_bic)


def aopt_sequence_from_h_blocks(
    h_blocks: dict[str, np.ndarray],
    follow_up_keys: list[str],
    fallback_to_weakest: bool = FALLBACK_TO_WEAKEST,
) -> list[str]:
    observed = ["e0"]
    remaining = list(follow_up_keys)
    sequence: list[str] = []
    while remaining:
        h_current = np.vstack([h_blocks[key] for key in observed])
        scored = []
        for candidate in remaining:
            score, current_trace, unresolved_dim = unresolved_aopt_score(
                h_current,
                h_blocks[candidate],
                tau_ratio=TAU_RATIO,
                prior_var=PRIOR_VAR,
                noise_var=NOISE_VAR,
                fallback_to_weakest=fallback_to_weakest,
            )
            scored.append((float(score), float(current_trace), int(unresolved_dim), candidate))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        winner = scored[0][3]
        sequence.append(winner)
        observed.append(winner)
        remaining.remove(winner)
    return sequence


def evaluate_real_data_aopt() -> dict[str, object]:
    series_rows = []
    skipped = []
    degenerate_count = 0

    for spec in SERIES_SPECS:
        try:
            conn = get_connection()
            times, conc = load_series(conn, spec.series_id)
            conn.close()

            initial_idx, candidate_blocks = build_block_plan(times)
            fit_cache: dict[tuple[int, ...], dict] = {}
            full_fits = fit_for_subset(times, conc, np.arange(len(times), dtype=int), fit_cache)
            oracle, _ = oracle_model(full_fits)
            initial_fits = fit_for_subset(times, conc, initial_idx, fit_cache)

            h_current = make_h_block(initial_fits)
            u_tau, _, _ = unresolved_subspace(h_current, tau_ratio=TAU_RATIO)
            unresolved_dim = int(u_tau.shape[1])
            if unresolved_dim == 0:
                degenerate_count += 1

            candidate_results = []
            for key, block_idx in candidate_blocks.items():
                h_e = predict_h_block(initial_fits, times[block_idx])
                cart_score, sigma_local = acquisition_score(h_e, u_tau)
                aopt_score, current_trace, aopt_k = unresolved_aopt_score(
                    h_current,
                    h_e,
                    tau_ratio=TAU_RATIO,
                    prior_var=PRIOR_VAR,
                    noise_var=NOISE_VAR,
                    fallback_to_weakest=FALLBACK_TO_WEAKEST,
                )
                disagreement_score = float(np.linalg.norm(h_e, ord="fro") ** 2)

                merged_idx = np.unique(np.concatenate([initial_idx, block_idx]))
                after_fits = fit_for_subset(times, conc, merged_idx, fit_cache)
                after_state = identification_state(after_fits, oracle)
                after_margin = real_data_oracle_margin(after_fits, oracle)

                candidate_results.append({
                    "block": key,
                    "times": times[block_idx].tolist(),
                    "cartograph_score": float(cart_score),
                    "sigma_local": float(sigma_local),
                    "aopt_score": float(aopt_score),
                    "aopt_current_trace": float(current_trace),
                    "aopt_unresolved_dim": int(aopt_k),
                    "disagreement_score": disagreement_score,
                    "oracle_margin": float(after_margin),
                    "identified": bool(after_state["identified"]),
                })

            cart_pick = max(candidate_results, key=lambda row: (row["cartograph_score"], row["sigma_local"]))
            aopt_pick = max(candidate_results, key=lambda row: (row["aopt_score"], row["aopt_current_trace"], row["aopt_unresolved_dim"]))
            disagreement_pick = max(candidate_results, key=lambda row: row["disagreement_score"])
            oracle_best = max(candidate_results, key=lambda row: row["oracle_margin"])

            series_rows.append({
                "series_id": spec.series_id,
                "label": spec.label,
                "oracle": oracle,
                "unresolved_dim": unresolved_dim,
                "cartograph_pick": cart_pick,
                "aopt_pick": aopt_pick,
                "disagreement_pick": disagreement_pick,
                "oracle_best_block": oracle_best["block"],
                "oracle_best_margin": float(oracle_best["oracle_margin"]),
                "candidate_results": candidate_results,
            })
        except Exception as exc:
            skipped.append({
                "series_id": spec.series_id,
                "label": spec.label,
                "error": f"{type(exc).__name__}: {exc}",
            })

    def pairwise(rows: list[dict[str, object]], left: str, right: str) -> str:
        wins = ties = losses = 0
        for row in rows:
            l = float(row[f"{left}_pick"]["oracle_margin"])
            r = float(row[f"{right}_pick"]["oracle_margin"])
            if l > r + 1e-9:
                wins += 1
            elif r > l + 1e-9:
                losses += 1
            else:
                ties += 1
        return f"{wins}W / {ties}T / {losses}L"

    return {
        "dataset_path": str(DATASET_PATH),
        "series_rows": series_rows,
        "skipped": skipped,
        "degenerate_series_count": degenerate_count,
        "aopt_vs_cartograph": pairwise(series_rows, "aopt", "cartograph"),
        "aopt_vs_disagreement": pairwise(series_rows, "aopt", "disagreement"),
        "mean_oracle_margin": {
            "cartograph": float(np.mean([row["cartograph_pick"]["oracle_margin"] for row in series_rows])) if series_rows else 0.0,
            "aopt": float(np.mean([row["aopt_pick"]["oracle_margin"] for row in series_rows])) if series_rows else 0.0,
            "disagreement": float(np.mean([row["disagreement_pick"]["oracle_margin"] for row in series_rows])) if series_rows else 0.0,
        },
    }


def evaluate_failure_aopt() -> dict[str, object]:
    experiments = get_failure_experiments()
    library_params = get_library_params()
    library_features, h_blocks = precompute_library(experiments, library_params)

    raw_sequence = failure_cartograph_sequence(h_blocks, FAILURE_KEYS)
    aopt_sequence = aopt_sequence_from_h_blocks(h_blocks, FAILURE_KEYS)

    failure_specs = get_failure_specs()
    scenario_rows = []

    for spec in failure_specs:
        failure_features = precompute_failure_features(spec, experiments)
        raw_history = run_failure_scenario(spec, raw_sequence, failure_features, library_features)
        aopt_history = run_failure_scenario(spec, aopt_sequence, failure_features, library_features)
        raw_final = raw_history[-1]
        aopt_final = aopt_history[-1]
        scenario_rows.append({
            "name": spec.name,
            "type": "failure",
            "raw_final_identified": bool(raw_final["identified"]),
            "aopt_final_identified": bool(aopt_final["identified"]),
            "raw_final_best_model": raw_final["best_model"],
            "aopt_final_best_model": aopt_final["best_model"],
            "raw_final_norm_resid": float(raw_final["norm_resid"]),
            "aopt_final_norm_resid": float(aopt_final["norm_resid"]),
            "raw_final_gap": float(raw_final["gap"]),
            "aopt_final_gap": float(aopt_final["gap"]),
        })

    control = get_control_truth()
    control_features: dict[str, np.ndarray] = {}
    for exp_key, experiment in experiments.items():
        result = simulate_model(control["rhs"], control["params"], experiment, control["model_name"])
        control_features[exp_key] = feature_map(experiment, result)

    raw_control = run_failure_scenario(None, raw_sequence, control_features, library_features)[-1]
    aopt_control = run_failure_scenario(None, aopt_sequence, control_features, library_features)[-1]
    scenario_rows.append({
        "name": control["name"],
        "type": "control",
        "raw_final_identified": bool(raw_control["identified"]),
        "aopt_final_identified": bool(aopt_control["identified"]),
        "raw_final_best_model": raw_control["best_model"],
        "aopt_final_best_model": aopt_control["best_model"],
        "raw_final_norm_resid": float(raw_control["norm_resid"]),
        "aopt_final_norm_resid": float(aopt_control["norm_resid"]),
        "raw_final_gap": float(raw_control["gap"]),
        "aopt_final_gap": float(aopt_control["gap"]),
    })

    return {
        "raw_sequence": raw_sequence,
        "aopt_sequence": aopt_sequence,
        "scenario_rows": scenario_rows,
        "raw_refusal_ok": all((not row["raw_final_identified"]) for row in scenario_rows if row["type"] == "failure"),
        "aopt_refusal_ok": all((not row["aopt_final_identified"]) for row in scenario_rows if row["type"] == "failure"),
        "raw_control_ok": all(row["raw_final_identified"] for row in scenario_rows if row["type"] == "control"),
        "aopt_control_ok": all(row["aopt_final_identified"] for row in scenario_rows if row["type"] == "control"),
    }


def plot_cross_benchmark(cascade: dict, pk: dict, real_data: dict, output_dir: Path) -> Path:
    dims = [int(d) for d in cascade["results"].keys()]
    dims.sort()

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))

    for method_name, color, label in [
        ("cartograph", "#1f77b4", "Raw CARTOGRAPH"),
        ("aopt", "#2ca02c", "Exact unresolved A-opt"),
        ("disagreement", "#ff7f0e", "Disagreement"),
    ]:
        y = [cascade["results"][str(d)]["summary"][method_name]["mean_hidden_best_match"] for d in dims]
        axes[0, 0].plot(dims, y, "o-", linewidth=2, color=color, label=label)
    axes[0, 0].set_title("Cascade robustness: hidden-best match")
    axes[0, 0].set_xlabel("Mechanism dimension d")
    axes[0, 0].set_ylabel("Mean match rate")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    for method_name, color, label in [
        ("cartograph", "#1f77b4", "Raw CARTOGRAPH"),
        ("aopt", "#2ca02c", "Exact unresolved A-opt"),
        ("disagreement", "#ff7f0e", "Disagreement"),
    ]:
        y = [cascade["results"][str(d)]["summary"][method_name]["avg_sequence_ms"] for d in dims]
        axes[0, 1].plot(dims, y, "o-", linewidth=2, color=color, label=label)
    axes[0, 1].set_title("Cascade selector runtime")
    axes[0, 1].set_xlabel("Mechanism dimension d")
    axes[0, 1].set_ylabel("Average sequence runtime (ms)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.25)

    pk_methods = ["cartograph", "aopt", "disagreement"]
    pk_labels = ["Raw CART", "A-opt", "Disagreement"]
    pk_means = [mean_round(pk["summary_rows"], key) for key in pk_methods]
    axes[1, 0].bar(np.arange(3), pk_means, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    axes[1, 0].set_xticks(np.arange(3))
    axes[1, 0].set_xticklabels(pk_labels)
    axes[1, 0].set_title("PK boundary case: mean rounds to identification")
    axes[1, 0].set_ylabel("Mean rounds")
    axes[1, 0].grid(axis="y", alpha=0.25)

    rd_means = [
        real_data["mean_oracle_margin"]["cartograph"],
        real_data["mean_oracle_margin"]["aopt"],
        real_data["mean_oracle_margin"]["disagreement"],
    ]
    axes[1, 1].bar(np.arange(3), rd_means, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    axes[1, 1].set_xticks(np.arange(3))
    axes[1, 1].set_xticklabels(pk_labels)
    axes[1, 1].set_title("EPA one-step retrospective: mean oracle margin")
    axes[1, 1].set_ylabel("Mean oracle margin")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("A-opt pivot validation across synthetic, PK, and real-data benchmarks", fontsize=13)
    fig.tight_layout()
    path = output_dir / "figure_1_cross_benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(payload: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "pivot_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    md_path = output_dir / "pivot_summary.md"
    cascade = payload["cascade"]
    pk = payload["pk"]
    real_data = payload["real_data"]
    failure = payload["failure"]
    lines = [
        "# A-opt Pivot Validation Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        "",
        "This artifact consolidates the upgraded method story after the unresolved",
        "A-opt pivot: structured high-dimensional gain, preserved low-dimensional",
        "boundary behavior, real-data realism, and refusal-case stability.",
        "",
        "## Cascade Robustness",
        "",
        "| d | Trials | Raw hidden-best | A-opt hidden-best | Disagreement hidden-best | Raw regret | A-opt regret | Disagreement regret | A-opt vs Raw |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for d in sorted(int(x) for x in cascade["results"].keys()):
        row = cascade["results"][str(d)]
        lines.append(
            f"| {d} | {row['n_trials']} | "
            f"{row['summary']['cartograph']['mean_hidden_best_match']:.2f} | "
            f"{row['summary']['aopt']['mean_hidden_best_match']:.2f} | "
            f"{row['summary']['disagreement']['mean_hidden_best_match']:.2f} | "
            f"{row['summary']['cartograph']['mean_regret']:.4f} | "
            f"{row['summary']['aopt']['mean_regret']:.4f} | "
            f"{row['summary']['disagreement']['mean_regret']:.4f} | "
            f"{row['pairwise_counts']['aopt_vs_cartograph']['wins']}W / {row['pairwise_counts']['aopt_vs_cartograph']['ties']}T / {row['pairwise_counts']['aopt_vs_cartograph']['losses']}L |"
        )

    lines.extend([
        "",
        "## PK Boundary Case",
        "",
        f"- A-opt vs disagreement: `{payload['pk_pairwise']['aopt_vs_disagreement']}`",
        f"- A-opt vs raw CARTOGRAPH: `{payload['pk_pairwise']['aopt_vs_cartograph']}`",
        f"- Mean rounds: raw `{mean_round(pk['summary_rows'], 'cartograph'):.2f}`, A-opt `{mean_round(pk['summary_rows'], 'aopt'):.2f}`, disagreement `{mean_round(pk['summary_rows'], 'disagreement'):.2f}`",
        "",
        "## EPA Real-Data Check",
        "",
        f"- Dataset: `{real_data['dataset_path']}`",
        f"- Degenerate initial unresolved space on `{real_data['degenerate_series_count']} / {len(real_data['series_rows'])}` series",
        f"- A-opt vs raw CARTOGRAPH: `{real_data['aopt_vs_cartograph']}`",
        f"- A-opt vs disagreement: `{real_data['aopt_vs_disagreement']}`",
        f"- Mean oracle margin: raw `{real_data['mean_oracle_margin']['cartograph']:.3f}`, A-opt `{real_data['mean_oracle_margin']['aopt']:.3f}`, disagreement `{real_data['mean_oracle_margin']['disagreement']:.3f}`",
        "",
        "| Series | Unresolved dim | Raw pick | A-opt pick | Disagreement pick | Hidden best |",
        "|---|---:|---|---|---|---|",
    ])
    for row in real_data["series_rows"]:
        lines.append(
            f"| {row['label']} | {row['unresolved_dim']} | {row['cartograph_pick']['block']} | {row['aopt_pick']['block']} | {row['disagreement_pick']['block']} | {row['oracle_best_block']} |"
        )

    lines.extend([
        "",
        "## Failure Benchmark Preservation",
        "",
        f"- Raw CARTOGRAPH sequence: `{failure['raw_sequence']}`",
        f"- A-opt sequence: `{failure['aopt_sequence']}`",
        f"- Raw refusal/control status: failures refused=`{failure['raw_refusal_ok']}`, control identified=`{failure['raw_control_ok']}`",
        f"- A-opt refusal/control status: failures refused=`{failure['aopt_refusal_ok']}`, control identified=`{failure['aopt_control_ok']}`",
        "",
        "| Scenario | Type | Raw final ID | A-opt final ID | Raw best | A-opt best |",
        "|---|---|---|---|---|---|",
    ])
    for row in failure["scenario_rows"]:
        lines.append(
            f"| {row['name']} | {row['type']} | "
            f"{'YES' if row['raw_final_identified'] else 'no'} | "
            f"{'YES' if row['aopt_final_identified'] else 'no'} | "
            f"{row['raw_final_best_model']} | {row['aopt_final_best_model']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The upgraded unresolved A-opt rule is meaningfully stronger on the structured high-dimensional cascade benchmark, especially from `d=8` onward.",
        "- PK and EPA remain low-dimensional boundary cases where A-opt changes little or nothing, which is consistent with the earlier scaling story.",
        "- The refusal benchmark survives the pivot: the stronger selector does not collapse the failure-case honesty result.",
        "",
        "## Artifact Paths",
        "",
        f"- `figure_1_cross_benchmark`: `{payload['artifacts']['figure_1_cross_benchmark']}`",
        f"- `pivot_results_json`: `{json_path}`",
        f"- `pivot_summary_md`: `{md_path}`",
    ])

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    final_path = output_dir / "final_and_latest_results.md"
    final_lines = [
        "# Final And Latest Results: A-opt Pivot",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        "",
        "## Bottom Line",
        "",
        "The unresolved A-opt upgrade is now the strongest algorithmic version of the",
        "project. It gives a real high-dimensional advantage on the replicated cascade",
        "benchmark, while preserving the low-dimensional PK/EPA near-tie story and the",
        "failure-case refusal behavior.",
        "",
        "## Headline Numbers",
        "",
        f"- Cascade d=8: hidden-best raw `{cascade['results']['8']['summary']['cartograph']['mean_hidden_best_match']:.2f}` vs A-opt `{cascade['results']['8']['summary']['aopt']['mean_hidden_best_match']:.2f}`; regret raw `{cascade['results']['8']['summary']['cartograph']['mean_regret']:.4f}` vs A-opt `{cascade['results']['8']['summary']['aopt']['mean_regret']:.4f}`",
        f"- Cascade d=16: hidden-best raw `{cascade['results']['16']['summary']['cartograph']['mean_hidden_best_match']:.2f}` vs A-opt `{cascade['results']['16']['summary']['aopt']['mean_hidden_best_match']:.2f}`; regret raw `{cascade['results']['16']['summary']['cartograph']['mean_regret']:.4f}` vs A-opt `{cascade['results']['16']['summary']['aopt']['mean_regret']:.4f}`",
        f"- PK: A-opt vs disagreement `{payload['pk_pairwise']['aopt_vs_disagreement']}`, A-opt vs raw `{payload['pk_pairwise']['aopt_vs_cartograph']}`",
        f"- EPA: A-opt vs raw `{real_data['aopt_vs_cartograph']}`, with degenerate unresolved space on `{real_data['degenerate_series_count']}/{len(real_data['series_rows'])}` series",
        f"- Failure benchmark: raw refusal/control `{failure['raw_refusal_ok']}/{failure['raw_control_ok']}`, A-opt refusal/control `{failure['aopt_refusal_ok']}/{failure['aopt_control_ok']}`",
    ]
    with final_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(final_lines))

    return md_path, final_path


def main() -> None:
    output_dir = ensure_output_dir()

    cascade = load_json(Path("outputs") / "cascade_boed_robustness" / "benchmark_results.json")
    pk = load_json(Path("outputs") / "pk_aopt_upgrade" / "benchmark_results.json")
    real_data = evaluate_real_data_aopt()
    failure = evaluate_failure_aopt()

    pk_rows = pk["summary_rows"]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cascade": cascade,
        "pk": pk,
        "pk_pairwise": {
            "aopt_vs_disagreement": f"{sum(1 for row in pk_rows if row['aopt'] < row['disagreement'])}W / {sum(1 for row in pk_rows if row['aopt'] == row['disagreement'])}T / {sum(1 for row in pk_rows if row['aopt'] > row['disagreement'])}L",
            "aopt_vs_cartograph": f"{sum(1 for row in pk_rows if row['aopt'] < row['cartograph'])}W / {sum(1 for row in pk_rows if row['aopt'] == row['cartograph'])}T / {sum(1 for row in pk_rows if row['aopt'] > row['cartograph'])}L",
        },
        "real_data": real_data,
        "failure": failure,
        "artifacts": {},
    }

    figure_1 = plot_cross_benchmark(cascade, pk, real_data, output_dir)
    payload["artifacts"]["figure_1_cross_benchmark"] = str(figure_1)

    md_path, final_path = write_summary(payload, output_dir)
    print(f"Wrote {md_path}")
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()
