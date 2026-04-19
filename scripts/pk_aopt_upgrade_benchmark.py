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

from pk_divergence_benchmark import (
    FOLLOW_UP_KEYS,
    IDENTIFICATION_MARGIN,
    RANDOM_UNRESOLVED_ROUND,
    disagreement_sequence,
    evaluate_identification,
    format_round,
    get_experiments,
    get_library_params,
    get_truth_specs,
    mean_random_round,
    oracle_library_model,
    precompute_library,
    precompute_truth_features,
    random_sequences,
    round_to_identification,
)
from pk_divergence_benchmark import cartograph_sequence as raw_cartograph_sequence
from unresolved_boed import unresolved_aopt_score


OUTPUT_DIR = Path("outputs") / "pk_aopt_upgrade"
PRIOR_VAR = 1.0
NOISE_VAR = 1.0
FALLBACK_TO_WEAKEST = False


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def aopt_sequence(h_blocks: dict[str, np.ndarray], follow_up_keys: list[str]) -> list[str]:
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
                prior_var=PRIOR_VAR,
                noise_var=NOISE_VAR,
                fallback_to_weakest=FALLBACK_TO_WEAKEST,
            )
            scored.append((float(score), float(current_trace), int(unresolved_dim), candidate))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        winner = scored[0][3]
        sequence.append(winner)
        observed.append(winner)
        remaining.remove(winner)
    return sequence


def plot_rounds_heatmap(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    method_order = ["cartograph", "aopt", "disagreement", "random_expected"]
    matrix = np.array(
        [
            [RANDOM_UNRESOLVED_ROUND if row[m] is None else row[m] for m in method_order]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=float(RANDOM_UNRESOLVED_ROUND))
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(["Raw CART", "A-opt", "Disagreement", "Random E[round]"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["truth"] for row in rows], fontsize=8)
    ax.set_title("PK Upgrade Benchmark: Rounds To Oracle Identification")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = rows[i][method_order[j]]
            label = f"{value:.2f}" if isinstance(value, float) else format_round(value)
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#102030")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / "figure_1_rounds_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sequence_comparison(output_dir: Path, sequences: dict[str, list[str]]) -> Path:
    methods = [("cartograph", "Raw CART", "#1f77b4"), ("aopt", "A-opt", "#2ca02c"), ("disagreement", "Disagreement", "#ff7f0e")]
    all_keys = sorted(set().union(*[set(seq) for seq in sequences.values()]))
    x = np.arange(len(all_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (key, label, color) in zip([-width, 0.0, width], methods):
        ranks = {exp_key: i + 1 for i, exp_key in enumerate(sequences[key])}
        ax.bar(x + offset, [ranks[k] for k in all_keys], width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys)
    ax.set_ylabel("Selection rank (1 = first)")
    ax.set_title("Experiment Selection Order: Raw CART vs A-opt vs Disagreement")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "figure_2_sequence_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gap_progress(output_dir: Path, scenario_details: dict[str, dict[str, object]]) -> Path:
    truth_names = sorted(scenario_details)
    fig, axes = plt.subplots(1, len(truth_names), figsize=(4 * len(truth_names), 4), squeeze=False)
    axes = axes[0]
    for idx, truth_name in enumerate(truth_names):
        ax = axes[idx]
        detail = scenario_details[truth_name]
        for method_name, color, label in [
            ("cartograph", "#1f77b4", "Raw CART"),
            ("aopt", "#2ca02c", "A-opt"),
            ("disagreement", "#ff7f0e", "Disagreement"),
        ]:
            history = detail[f"{method_name}_history"]
            rounds = [h["round"] for h in history]
            gaps = [h["gap"] for h in history]
            ax.plot(rounds, gaps, marker="o", linewidth=2, markersize=4, color=color, label=label)
        ax.axhline(IDENTIFICATION_MARGIN, color="#cc3333", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(truth_name, fontsize=8)
        ax.set_xlabel("Round")
        ax.set_ylabel("Residual gap")
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Residual Gap Progression Under The A-opt Upgrade", fontsize=11)
    fig.tight_layout()
    path = output_dir / "figure_3_gap_progress.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary(output_dir: Path, payload: dict[str, object]) -> Path:
    path = output_dir / "benchmark_summary.md"
    lines = [
        "# PK A-opt Upgrade Benchmark Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        "",
        f"Identification margin: `{payload['identification_margin']:.2f}`",
        f"Prior variance: `{payload['prior_var']:.2f}`",
        f"Noise variance: `{payload['noise_var']:.2f}`",
        f"Fallback to weakest direction when unresolved is empty: `{payload['fallback_to_weakest']}`",
        "",
        "## Method Sequences",
        "",
        f"- **Raw CARTOGRAPH**: `{payload['method_sequences']['cartograph']}`",
        f"- **Exact unresolved A-opt**: `{payload['method_sequences']['aopt']}`",
        f"- **Disagreement**: `{payload['method_sequences']['disagreement']}`",
        "",
        "## Primary Table",
        "",
        "| Truth | Oracle | Raw CART | A-opt | Disagreement | Random E[round] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['truth']} | {row['oracle']} | {format_round(row['cartograph'])} | "
            f"{format_round(row['aopt'])} | {format_round(row['disagreement'])} | {row['random_expected']:.2f} |"
        )

    for left, right, label in [
        ("aopt", "disagreement", "A-opt vs Disagreement"),
        ("aopt", "cartograph", "A-opt vs Raw CARTOGRAPH"),
    ]:
        wins = ties = losses = 0
        for row in payload["summary_rows"]:
            l = row[left] if row[left] is not None else RANDOM_UNRESOLVED_ROUND
            r = row[right] if row[right] is not None else RANDOM_UNRESOLVED_ROUND
            if l < r:
                wins += 1
            elif l == r:
                ties += 1
            else:
                losses += 1
        lines.extend(["", f"**{label}**: {wins}W / {ties}T / {losses}L"])

    lines.extend([
        "",
        "## Artifact Paths",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    output_dir = ensure_output_dir()
    experiments = get_experiments()
    params_by_model = get_library_params()
    truths = get_truth_specs()
    library_features, h_blocks = precompute_library(experiments, params_by_model)

    method_sequences = {
        "cartograph": raw_cartograph_sequence(h_blocks, FOLLOW_UP_KEYS),
        "aopt": aopt_sequence(h_blocks, FOLLOW_UP_KEYS),
        "disagreement": disagreement_sequence(library_features, FOLLOW_UP_KEYS),
    }
    all_random = random_sequences(FOLLOW_UP_KEYS)

    print("Method sequences:")
    for name, seq in method_sequences.items():
        print(f"  {name}: {seq}")

    summary_rows: list[dict[str, object]] = []
    scenario_details: dict[str, dict[str, object]] = {}
    all_exp_keys = ["e0"] + FOLLOW_UP_KEYS

    for truth_spec in truths:
        truth_features = precompute_truth_features(truth_spec, experiments)
        oracle_model, full_residuals = oracle_library_model(truth_features, library_features, all_exp_keys)

        primary_rounds: dict[str, int | None | float] = {}
        method_histories: dict[str, list[dict[str, object]]] = {}
        for method_name, sequence in method_sequences.items():
            round_value, history = round_to_identification(
                sequence,
                truth_features,
                library_features,
                oracle_model,
                IDENTIFICATION_MARGIN,
            )
            primary_rounds[method_name] = round_value
            method_histories[method_name] = history

        random_rounds = []
        for seq in all_random:
            rnd, _ = round_to_identification(seq, truth_features, library_features, oracle_model, IDENTIFICATION_MARGIN)
            random_rounds.append(rnd)
        random_expected = mean_random_round(random_rounds)

        summary_rows.append({
            "truth": truth_spec.name,
            "family": truth_spec.family,
            "oracle": oracle_model,
            "cartograph": primary_rounds["cartograph"],
            "aopt": primary_rounds["aopt"],
            "disagreement": primary_rounds["disagreement"],
            "random_expected": random_expected,
        })
        scenario_details[truth_spec.name] = {
            "family": truth_spec.family,
            "truth_params": truth_spec.params,
            "oracle_model": oracle_model,
            "full_residuals": full_residuals,
            "cartograph_history": method_histories["cartograph"],
            "aopt_history": method_histories["aopt"],
            "disagreement_history": method_histories["disagreement"],
        }

    summary_rows.sort(key=lambda row: row["truth"])

    artifacts: dict[str, str] = {}
    artifacts["figure_1_rounds_heatmap"] = str(plot_rounds_heatmap(output_dir, summary_rows))
    artifacts["figure_2_sequence_comparison"] = str(plot_sequence_comparison(output_dir, method_sequences))
    artifacts["figure_3_gap_progress"] = str(plot_gap_progress(output_dir, scenario_details))
    artifacts["benchmark_results_json"] = str(output_dir / "benchmark_results.json")
    artifacts["benchmark_summary_md"] = str(output_dir / "benchmark_summary.md")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identification_margin": IDENTIFICATION_MARGIN,
        "prior_var": PRIOR_VAR,
        "noise_var": NOISE_VAR,
        "fallback_to_weakest": FALLBACK_TO_WEAKEST,
        "method_sequences": method_sequences,
        "summary_rows": summary_rows,
        "scenario_details": scenario_details,
        "artifacts": artifacts,
    }

    with (output_dir / "benchmark_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_summary(output_dir, payload)

    print("\nPrimary summary:")
    for row in summary_rows:
        print(
            f"  {row['truth']:28s} oracle={row['oracle']} "
            f"RC={format_round(row['cartograph'])} "
            f"AO={format_round(row['aopt'])} "
            f"D={format_round(row['disagreement'])} "
            f"R={row['random_expected']:.2f}"
        )
    print("\nSaved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
