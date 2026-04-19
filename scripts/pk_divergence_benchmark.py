"""Divergence benchmark: expanded experiment menu where CARTOGRAPH and
disagreement-magnitude select different first-round experiments.

This promotes the best search-derived candidates from pk_candidate_search
into named experiments and reruns the full multi-round benchmark to verify
that the *sequence* diverges (not just the first pick).
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / "outputs" / ".cache").resolve()))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pk_first_pass import (
    TAU_RATIO,
    Experiment,
    acquisition_score,
    build_h_block,
    disagreement_magnitude_score,
    feature_map,
    model_a_rhs,
    model_b_rhs,
    model_c_rhs,
    model_d_rhs,
    sampled_curve_distance,
    simulate_model,
    singular_metrics,
    unresolved_subspace,
)


OUTPUT_DIR = Path("outputs") / "pk_divergence"
IDENTIFICATION_MARGIN = 0.05
IDENTIFICATION_MARGIN_ABLATION = [0.03, 0.05, 0.07]
RANDOM_UNRESOLVED_ROUND = 8  # higher ceiling because 7 follow-up experiments


@dataclass(frozen=True)
class TruthSpec:
    name: str
    family: str
    truth_model_name: str
    rhs: Callable[[float, np.ndarray, dict[str, float]], np.ndarray]
    params: dict[str, float]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Expanded experiment menu
# ---------------------------------------------------------------------------

def get_experiments() -> dict[str, Experiment]:
    """Return e0 (initial) plus 5 follow-up experiments from the divergence menu."""
    return {
        "e0": Experiment(
            "Sparse Oral", "oral",
            np.array([0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 24.0]),
        ),
        # CARTOGRAPH's top pick (high unresolved-subspace score)
        "E1": Experiment(
            "Absorption-Peak Oral", "oral",
            np.array([0.1, 0.33, 0.75, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0, 24.0]),
        ),
        # Disagreement-magnitude's top pick (high pairwise feature spread)
        "E2": Experiment(
            "Late-Spread Oral", "oral",
            np.array([0.2, 0.33, 0.5, 1.25, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]),
        ),
        # Filler candidates with moderate scores
        "E3": Experiment(
            "Mid-Window Oral", "oral",
            np.array([0.1, 0.33, 0.5, 0.75, 1.25, 2.5, 10.0, 12.0, 20.0, 24.0]),
        ),
        "E4": Experiment(
            "Early-Cluster Oral", "oral",
            np.array([0.2, 0.33, 0.75, 1.0, 1.25, 1.5, 2.5, 10.0, 16.0, 24.0]),
        ),
        "E5": Experiment(
            "Tail-Emphasis Oral", "oral",
            np.array([0.1, 0.75, 1.25, 1.5, 2.5, 8.0, 10.0, 16.0, 20.0, 24.0]),
        ),
        # IV experiments — bypass absorption, directly probe distribution
        "E6": Experiment(
            "IV Dense Early-Mid", "iv",
            np.array([0.08, 0.17, 0.33, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 24.0]),
        ),
        "E7": Experiment(
            "IV Spread", "iv",
            np.array([0.08, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0]),
        ),
    }


FOLLOW_UP_KEYS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]


def get_library_params() -> dict[str, dict[str, float]]:
    return {
        "A": {"k_a": 0.80, "k_e": 0.18, "V": 20.0},
        "B": {"k_tr": 1.20, "k_a": 0.80, "k_e": 0.18, "V": 20.0},
        "C": {"k_a": 0.70, "k_10": 0.16, "k_12": 0.20, "k_21": 0.25, "V_c": 18.0},
    }


def get_truth_specs() -> list[TruthSpec]:
    return [
        TruthSpec(
            name="absorption_variant",
            family="B-family perturbed truth",
            truth_model_name="B",
            rhs=model_b_rhs,
            params={"k_tr": 1.45, "k_a": 0.88, "k_e": 0.18, "V": 19.2},
        ),
        TruthSpec(
            name="distribution_variant_easy",
            family="C-family perturbed truth",
            truth_model_name="C",
            rhs=model_c_rhs,
            params={"k_a": 0.68, "k_10": 0.16, "k_12": 0.24, "k_21": 0.30, "V_c": 18.4},
        ),
        TruthSpec(
            name="distribution_variant_hard",
            family="C-family perturbed truth",
            truth_model_name="C",
            rhs=model_c_rhs,
            params={"k_a": 0.75, "k_10": 0.16, "k_12": 0.24, "k_21": 0.28, "V_c": 18.2},
        ),
        TruthSpec(
            name="mixed_balanced",
            family="D-family balanced mix",
            truth_model_name="truth",
            rhs=model_d_rhs,
            params={"k_tr": 1.60, "k_a": 1.00, "k_10": 0.16, "k_12": 0.20, "k_21": 0.25, "V_c": 18.5},
        ),
        TruthSpec(
            name="mixed_absorption",
            family="D-family absorption-leaning mix",
            truth_model_name="truth",
            rhs=model_d_rhs,
            params={"k_tr": 1.80, "k_a": 1.00, "k_10": 0.16, "k_12": 0.15, "k_21": 0.25, "V_c": 19.5},
        ),
        TruthSpec(
            name="absorption_variant_slow",
            family="B-family slow transit",
            truth_model_name="B",
            rhs=model_b_rhs,
            params={"k_tr": 0.90, "k_a": 0.95, "k_e": 0.19, "V": 20.5},
        ),
        TruthSpec(
            name="distribution_variant_subtle",
            family="C-family subtle distribution",
            truth_model_name="C",
            rhs=model_c_rhs,
            params={"k_a": 0.78, "k_10": 0.17, "k_12": 0.18, "k_21": 0.22, "V_c": 19.0},
        ),
    ]


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def precompute_library(
    experiments: dict[str, Experiment],
    params_by_model: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    library_features: dict[str, dict[str, np.ndarray]] = {}
    h_blocks: dict[str, np.ndarray] = {}
    for exp_key, experiment in experiments.items():
        features: dict[str, np.ndarray] = {}
        for model_name, rhs in [("A", model_a_rhs), ("B", model_b_rhs), ("C", model_c_rhs)]:
            result = simulate_model(rhs, params_by_model[model_name], experiment, model_name)
            features[model_name] = feature_map(experiment, result)
        library_features[exp_key] = features
        h_blocks[exp_key] = build_h_block(experiment, features, normalize=False)
    return library_features, h_blocks


def precompute_truth_features(
    truth_spec: TruthSpec,
    experiments: dict[str, Experiment],
) -> dict[str, np.ndarray]:
    truth_features: dict[str, np.ndarray] = {}
    for exp_key, experiment in experiments.items():
        result = simulate_model(truth_spec.rhs, truth_spec.params, experiment, truth_spec.truth_model_name)
        truth_features[exp_key] = feature_map(experiment, result)
    return truth_features


# ---------------------------------------------------------------------------
# Identification logic
# ---------------------------------------------------------------------------

def stacked_residuals(
    observed_keys: list[str],
    truth_features: dict[str, np.ndarray],
    library_features: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for model_name in ["A", "B", "C"]:
        truth_stack = np.concatenate([truth_features[key] for key in observed_keys])
        model_stack = np.concatenate([library_features[key][model_name] for key in observed_keys])
        residuals[model_name] = float(np.linalg.norm(truth_stack - model_stack))
    return residuals


def oracle_library_model(
    truth_features: dict[str, np.ndarray],
    library_features: dict[str, dict[str, np.ndarray]],
    all_exp_keys: list[str],
) -> tuple[str, dict[str, float]]:
    residuals = stacked_residuals(all_exp_keys, truth_features, library_features)
    best = min(residuals, key=residuals.get)
    return best, residuals


def evaluate_identification(
    observed_keys: list[str],
    truth_features: dict[str, np.ndarray],
    library_features: dict[str, dict[str, np.ndarray]],
    oracle_model: str,
    id_margin: float,
) -> dict[str, object]:
    residuals = stacked_residuals(observed_keys, truth_features, library_features)
    ordered = sorted(residuals.items(), key=lambda item: item[1])
    best_model, best_resid = ordered[0]
    second_resid = ordered[1][1]
    gap = second_resid - best_resid
    return {
        "best_model": best_model,
        "residuals": residuals,
        "gap": float(gap),
        "identified": bool(best_model == oracle_model and gap >= id_margin),
    }


# ---------------------------------------------------------------------------
# Sequencing methods
# ---------------------------------------------------------------------------

def cartograph_sequence(
    h_blocks: dict[str, np.ndarray],
    follow_up_keys: list[str],
) -> list[str]:
    """Greedy CARTOGRAPH: pick the experiment maximizing unresolved-subspace acquisition."""
    observed = ["e0"]
    remaining = list(follow_up_keys)
    sequence: list[str] = []
    while remaining:
        h_current = np.vstack([h_blocks[key] for key in observed])
        current_metrics = singular_metrics(h_current, tau_ratio=TAU_RATIO)
        u_tau, _, _ = unresolved_subspace(h_current, tau_ratio=TAU_RATIO)
        scored = []
        for candidate in remaining:
            score, sigma_local = acquisition_score(h_blocks[candidate], u_tau)
            h_plus = np.vstack([h_current, h_blocks[candidate]])
            plus_metrics = singular_metrics(
                h_plus, tau_ratio=TAU_RATIO,
                tau_reference=float(current_metrics["tau"]),
            )
            scored.append((
                float(score),
                float(plus_metrics["sigma_ratio"]),
                float(sigma_local),
                candidate,
            ))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        winner = scored[0][3]
        sequence.append(winner)
        observed.append(winner)
        remaining.remove(winner)
    return sequence


def disagreement_sequence(
    library_features: dict[str, dict[str, np.ndarray]],
    follow_up_keys: list[str],
) -> list[str]:
    """Static sort by disagreement magnitude (one-shot, not adaptive)."""
    scored = []
    for candidate in follow_up_keys:
        mag = disagreement_magnitude_score(library_features[candidate])
        scored.append((float(mag), candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored]


def random_sequences(follow_up_keys: list[str]) -> list[list[str]]:
    return [list(perm) for perm in itertools.permutations(follow_up_keys)]


# ---------------------------------------------------------------------------
# Round-to-identification
# ---------------------------------------------------------------------------

def round_to_identification(
    sequence: list[str],
    truth_features: dict[str, np.ndarray],
    library_features: dict[str, dict[str, np.ndarray]],
    oracle_model: str,
    id_margin: float,
) -> tuple[int | None, list[dict[str, object]]]:
    history: list[dict[str, object]] = []
    observed = ["e0"]
    state = evaluate_identification(observed, truth_features, library_features, oracle_model, id_margin)
    state["round"] = 0
    state["observed"] = observed.copy()
    history.append(state)
    if state["identified"]:
        return 0, history
    for round_idx, exp_key in enumerate(sequence, start=1):
        observed.append(exp_key)
        state = evaluate_identification(observed, truth_features, library_features, oracle_model, id_margin)
        state["round"] = round_idx
        state["observed"] = observed.copy()
        history.append(state)
        if state["identified"]:
            return round_idx, history
    return None, history


def format_round(round_value: int | None) -> str:
    return "NR" if round_value is None else str(round_value)


def mean_random_round(rounds: list[int | None]) -> float:
    mapped = [RANDOM_UNRESOLVED_ROUND if v is None else v for v in rounds]
    return float(np.mean(mapped))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rounds_heatmap(
    output_dir: Path,
    truth_names: list[str],
    rows: list[dict[str, object]],
) -> Path:
    method_order = ["cartograph", "disagreement", "random_expected"]
    matrix = np.array(
        [
            [
                RANDOM_UNRESOLVED_ROUND if row[m] is None else row[m]
                for m in method_order
            ]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=float(RANDOM_UNRESOLVED_ROUND))
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(["CARTOGRAPH", "Disagreement", "Random E[round]"])
    ax.set_yticks(np.arange(len(truth_names)))
    ax.set_yticklabels(truth_names, fontsize=8)
    ax.set_title("Rounds To Oracle-Library Identification (Expanded Menu)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = rows[i][method_order[j]]
            label = f"{value:.2f}" if isinstance(value, float) else format_round(value)
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#102030")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / "figure_1_rounds_to_identification.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sequence_comparison(
    output_dir: Path,
    cartograph_seq: list[str],
    disagreement_seq: list[str],
) -> Path:
    """Bar chart showing the rank each method assigns to each experiment."""
    all_keys = sorted(set(cartograph_seq))
    c_rank = {k: i + 1 for i, k in enumerate(cartograph_seq)}
    d_rank = {k: i + 1 for i, k in enumerate(disagreement_seq)}
    x = np.arange(len(all_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [c_rank[k] for k in all_keys], width=width, color="#1f77b4", label="CARTOGRAPH")
    ax.bar(x + width / 2, [d_rank[k] for k in all_keys], width=width, color="#ff7f0e", label="Disagreement")
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys)
    ax.set_ylabel("Selection rank (1 = first)")
    ax.set_title("Experiment Selection Order: CARTOGRAPH vs Disagreement")
    ax.legend(frameon=False)
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "figure_2_sequence_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_per_round_residuals(
    output_dir: Path,
    scenario_details: dict[str, dict],
) -> Path:
    """For each truth scenario, plot residual gap across rounds for both methods."""
    n_truths = len(scenario_details)
    fig, axes = plt.subplots(1, n_truths, figsize=(4 * n_truths, 4), squeeze=False)
    axes = axes[0]
    for idx, (truth_name, detail) in enumerate(sorted(scenario_details.items())):
        ax = axes[idx]
        for method_name, color, label in [
            ("cartograph", "#1f77b4", "CARTOGRAPH"),
            ("disagreement", "#ff7f0e", "Disagreement"),
        ]:
            history = detail[f"{method_name}_history"]
            rounds = [h["round"] for h in history]
            gaps = [h["gap"] for h in history]
            ax.plot(rounds, gaps, marker="o", color=color, linewidth=2, markersize=5, label=label)
        ax.axhline(IDENTIFICATION_MARGIN, color="#cc3333", linewidth=1, linestyle="--", alpha=0.7, label=f"margin={IDENTIFICATION_MARGIN}")
        ax.set_title(truth_name, fontsize=9)
        ax.set_xlabel("Round")
        ax.set_ylabel("Residual gap")
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Residual Gap Progression Per Truth Scenario", fontsize=11)
    fig.tight_layout()
    path = output_dir / "figure_3_per_round_residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results_json(output_dir: Path, payload: dict) -> Path:
    path = output_dir / "benchmark_results.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def write_summary_markdown(output_dir: Path, payload: dict) -> Path:
    path = output_dir / "benchmark_summary.md"
    lines = [
        "# PK Divergence Benchmark Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        "",
        f"Primary identification margin: `{payload['identification_margin']:.2f}`",
        f"Random unresolved round value: `{payload['random_unresolved_round']}`",
        f"Follow-up experiments: `{payload['follow_up_keys']}`",
        "",
        "## Method Sequences",
        "",
        f"- **CARTOGRAPH**: `{payload['method_sequences']['cartograph']}`",
        f"- **Disagreement**: `{payload['method_sequences']['disagreement']}`",
        "",
    ]

    # Divergence check
    c_seq = payload["method_sequences"]["cartograph"]
    d_seq = payload["method_sequences"]["disagreement"]
    if c_seq == d_seq:
        lines.append("**WARNING: sequences are identical — no divergence on this menu.**")
    else:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(c_seq, d_seq)) if a != b),
            len(c_seq),
        )
        lines.append(f"**Sequences diverge at round {first_diff + 1}**: CARTOGRAPH picks `{c_seq[first_diff]}`, Disagreement picks `{d_seq[first_diff]}`.")
    lines.append("")

    lines.extend([
        "## Primary Table",
        "",
        "| Truth | Family | Oracle | CARTOGRAPH | Disagreement | Random E[round] |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['truth']} | {row['family']} | {row['oracle']} | "
            f"{format_round(row['cartograph'])} | {format_round(row['disagreement'])} | {row['random_expected']:.2f} |"
        )

    # Win/tie/loss summary
    wins = ties = losses = 0
    for row in payload["summary_rows"]:
        c = row["cartograph"] if row["cartograph"] is not None else RANDOM_UNRESOLVED_ROUND
        d = row["disagreement"] if row["disagreement"] is not None else RANDOM_UNRESOLVED_ROUND
        if c < d:
            wins += 1
        elif c == d:
            ties += 1
        else:
            losses += 1
    lines.extend([
        "",
        f"**CARTOGRAPH vs Disagreement**: {wins}W / {ties}T / {losses}L",
        "",
    ])

    lines.extend([
        "## Identification-Margin Sensitivity (CARTOGRAPH)",
        "",
        "| Truth | Margin 0.03 | Margin 0.05 | Margin 0.07 |",
        "|---|---:|---:|---:|",
    ])
    for row in payload["margin_sensitivity_rows"]:
        lines.append(
            f"| {row['truth']} | {format_round(row['0.03'])} | {format_round(row['0.05'])} | {format_round(row['0.07'])} |"
        )

    lines.extend([
        "",
        "## Experiment Menu Details",
        "",
        "| Key | Name | Route | Times |",
        "|---|---|---|---|",
    ])
    for key, exp in payload["experiment_details"].items():
        lines.append(f"| {key} | {exp['name']} | {exp['route']} | `{exp['times']}` |")

    lines.extend([
        "",
        "## Artifact Paths",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = ensure_output_dir()
    experiments = get_experiments()
    params_by_model = get_library_params()
    truths = get_truth_specs()
    library_features, h_blocks = precompute_library(experiments, params_by_model)

    method_sequences = {
        "cartograph": cartograph_sequence(h_blocks, FOLLOW_UP_KEYS),
        "disagreement": disagreement_sequence(library_features, FOLLOW_UP_KEYS),
    }
    all_random = random_sequences(FOLLOW_UP_KEYS)

    print("Method sequences:")
    for method_name, seq in method_sequences.items():
        print(f"  {method_name}: {seq}")

    diverged = method_sequences["cartograph"] != method_sequences["disagreement"]
    print(f"\nSequences diverge: {diverged}")
    if diverged:
        for i, (c, d) in enumerate(zip(
            method_sequences["cartograph"],
            method_sequences["disagreement"],
        )):
            if c != d:
                print(f"  First divergence at round {i + 1}: CARTOGRAPH={c}, Disagreement={d}")
                break

    summary_rows: list[dict[str, object]] = []
    margin_sensitivity_rows: list[dict[str, object]] = []
    scenario_details: dict[str, dict[str, object]] = {}

    all_exp_keys = ["e0"] + FOLLOW_UP_KEYS

    for truth_spec in truths:
        truth_features = precompute_truth_features(truth_spec, experiments)
        oracle_model, full_residuals = oracle_library_model(
            truth_features, library_features, all_exp_keys,
        )

        method_histories: dict[str, list[dict[str, object]]] = {}
        primary_rounds: dict[str, int | None | float] = {}
        for method_name, sequence in method_sequences.items():
            round_value, history = round_to_identification(
                sequence, truth_features, library_features, oracle_model, IDENTIFICATION_MARGIN,
            )
            primary_rounds[method_name] = round_value
            method_histories[method_name] = history

        random_rounds: list[int | None] = []
        for sequence in all_random:
            round_value, _ = round_to_identification(
                sequence, truth_features, library_features, oracle_model, IDENTIFICATION_MARGIN,
            )
            random_rounds.append(round_value)
        random_expected = mean_random_round(random_rounds)

        margin_row: dict[str, object] = {"truth": truth_spec.name}
        for margin in IDENTIFICATION_MARGIN_ABLATION:
            round_value, _ = round_to_identification(
                method_sequences["cartograph"],
                truth_features, library_features, oracle_model, margin,
            )
            margin_row[f"{margin:.2f}"] = round_value

        summary_rows.append({
            "truth": truth_spec.name,
            "family": truth_spec.family,
            "oracle": oracle_model,
            "cartograph": primary_rounds["cartograph"],
            "disagreement": primary_rounds["disagreement"],
            "random_expected": random_expected,
        })
        margin_sensitivity_rows.append(margin_row)
        scenario_details[truth_spec.name] = {
            "family": truth_spec.family,
            "truth_params": truth_spec.params,
            "oracle_model": oracle_model,
            "full_residuals": full_residuals,
            "cartograph_history": method_histories["cartograph"],
            "disagreement_history": method_histories["disagreement"],
            "random_rounds": random_rounds,
        }

    summary_rows.sort(key=lambda row: row["truth"])
    margin_sensitivity_rows.sort(key=lambda row: row["truth"])

    # Plots
    artifacts: dict[str, str] = {}
    artifacts["figure_1_rounds_to_identification"] = str(
        plot_rounds_heatmap(output_dir, [row["truth"] for row in summary_rows], summary_rows)
    )
    artifacts["figure_2_sequence_comparison"] = str(
        plot_sequence_comparison(
            output_dir,
            method_sequences["cartograph"],
            method_sequences["disagreement"],
        )
    )
    artifacts["figure_3_per_round_residuals"] = str(
        plot_per_round_residuals(output_dir, scenario_details)
    )
    artifacts["benchmark_results_json"] = str(output_dir / "benchmark_results.json")
    artifacts["benchmark_summary_md"] = str(output_dir / "benchmark_summary.md")

    experiment_details = {
        key: {"name": exp.name, "route": exp.route, "times": exp.times.tolist()}
        for key, exp in experiments.items()
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identification_margin": IDENTIFICATION_MARGIN,
        "random_unresolved_round": RANDOM_UNRESOLVED_ROUND,
        "follow_up_keys": FOLLOW_UP_KEYS,
        "method_sequences": method_sequences,
        "summary_rows": summary_rows,
        "margin_sensitivity_rows": margin_sensitivity_rows,
        "scenario_details": scenario_details,
        "experiment_details": experiment_details,
        "artifacts": artifacts,
    }
    write_results_json(output_dir, payload)
    write_summary_markdown(output_dir, payload)

    print("\nPrimary summary:")
    for row in summary_rows:
        c = format_round(row["cartograph"])
        d = format_round(row["disagreement"])
        r = f"{row['random_expected']:.2f}"
        print(f"  {row['truth']:30s} oracle={row['oracle']}  C={c}  D={d}  R={r}")

    print("\nSaved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
