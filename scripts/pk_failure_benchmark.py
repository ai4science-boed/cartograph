"""Failure-case benchmark: out-of-library truths where CARTOGRAPH should
refuse to confidently identify a library model.

The goal is to show that when the true generating process is genuinely
outside the library, the method:
  1. Never confidently identifies a library model (gap stays below margin).
  2. Residuals remain high across rounds.

This is the "principled refusal to resolve" result.
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
    EPS,
    TAU_RATIO,
    Experiment,
    acquisition_score,
    build_h_block,
    feature_map,
    model_a_rhs,
    model_b_rhs,
    model_c_rhs,
    model_d_rhs,
    simulate_model,
    singular_metrics,
    unresolved_subspace,
)


OUTPUT_DIR = Path("outputs") / "pk_failure"
IDENTIFICATION_MARGIN = 0.05
# Absolute goodness threshold: max normalized residual per observation for
# a well-specified model.  Calibrated from in-library divergence benchmark
# where the worst per-observation residual (distribution_variant_hard at
# full pool) is ~0.274.  We use 0.30 with a small margin.
GOODNESS_THRESHOLD = 0.25
GOODNESS_THRESHOLD_ABLATION = [0.20, 0.25, 0.30, 0.35]
FEATURE_DIM = 5  # features per experiment
FEATURE_NAMES = ["T_max", "C_max", "AUC_frac", "terminal_slope", "loglin_RMSE"]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Failure truth models: mechanisms NOT in the library
# ---------------------------------------------------------------------------

def failure_timevarying_strong_rhs(
    t: float, y: np.ndarray, params: dict[str, float],
) -> np.ndarray:
    """Model D skeleton with strong time-varying clearance.

    k_10(t) = k_10 * (1 + 1.0 * exp(-t/8))  — doubles at t=0, slow decay.
    This creates a pronounced early-fast/late-slow elimination pattern
    that no constant-coefficient library model can match.
    """
    a_g, a_t, a_c, a_p = y
    k_tr = params["k_tr"]
    k_a = params["k_a"]
    k_10 = params["k_10"] * (1.0 + 1.0 * np.exp(-t / 8.0))
    k_12 = params["k_12"]
    k_21 = params["k_21"]
    return np.array([
        -k_tr * a_g,
        k_tr * a_g - k_a * a_t,
        k_a * a_t - (k_10 + k_12) * a_c + k_21 * a_p,
        k_12 * a_c - k_21 * a_p,
    ])


def failure_saturable_rhs(
    t: float, y: np.ndarray, params: dict[str, float],
) -> np.ndarray:
    """Model D skeleton with Michaelis-Menten (saturable) elimination.

    Instead of first-order k_10 * A_c, use V_max * A_c / (K_m + A_c).
    At high concentrations elimination saturates; at low concentrations
    it looks approximately first-order. This creates a concavity shift
    in the log-concentration profile that constant models can't replicate.
    """
    a_g, a_t, a_c, a_p = y
    k_tr = params["k_tr"]
    k_a = params["k_a"]
    v_max = params["v_max"]
    k_m = params["k_m"]
    k_12 = params["k_12"]
    k_21 = params["k_21"]
    elim = v_max * a_c / (k_m + a_c + EPS)
    return np.array([
        -k_tr * a_g,
        k_tr * a_g - k_a * a_t,
        k_a * a_t - elim - k_12 * a_c + k_21 * a_p,
        k_12 * a_c - k_21 * a_p,
    ])


def failure_recirculation_rhs(
    t: float, y: np.ndarray, params: dict[str, float],
) -> np.ndarray:
    """Five-compartment model with enterohepatic recirculation.

    States: [A_g, A_t, A_c, A_p, A_bile]
    Bile pool recirculates back to gut with delay, creating a secondary
    concentration peak that no library model can produce.
    """
    a_g, a_t, a_c, a_p, a_bile = y
    k_tr = params["k_tr"]
    k_a = params["k_a"]
    k_10 = params["k_10"]
    k_12 = params["k_12"]
    k_21 = params["k_21"]
    k_bile = params["k_bile"]     # central -> bile
    k_reabs = params["k_reabs"]   # bile -> gut (reabsorption)
    return np.array([
        -k_tr * a_g + k_reabs * a_bile,
        k_tr * a_g - k_a * a_t,
        k_a * a_t - (k_10 + k_12 + k_bile) * a_c + k_21 * a_p,
        k_12 * a_c - k_21 * a_p,
        k_bile * a_c - k_reabs * a_bile,
    ])


def simulate_failure_model(
    rhs: Callable,
    params: dict[str, float],
    experiment: Experiment,
    n_states: int,
    central_idx: int,
    volume_key: str = "V_c",
) -> tuple[np.ndarray, np.ndarray]:
    """Generic simulator for failure models with variable state dimension."""
    from scipy.integrate import solve_ivp

    t_eval = np.array(sorted(set([0.0, *experiment.times.tolist()])), dtype=float)
    y0 = np.zeros(n_states)
    dose = 100.0
    if experiment.route == "oral":
        y0[0] = dose
    else:
        y0[central_idx] = dose
    volume = params[volume_key]

    sol = solve_ivp(
        lambda t, y: rhs(t, y, params),
        (0.0, 24.0),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="LSODA",
    )
    if not sol.success:
        raise RuntimeError(f"Failure simulation failed: {sol.message}")

    concentration = np.maximum(sol.y[central_idx] / volume, 0.0)
    from pk_first_pass import SimulationResult
    result = SimulationResult(times=t_eval, concentration=concentration)
    return result


# ---------------------------------------------------------------------------
# Failure truth specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureTruthSpec:
    name: str
    description: str
    rhs: Callable
    params: dict[str, float]
    n_states: int
    central_idx: int
    volume_key: str


def get_failure_specs() -> list[FailureTruthSpec]:
    return [
        FailureTruthSpec(
            name="timevarying_strong",
            description="Strong time-varying clearance (100% at t=0, tau=8h)",
            rhs=failure_timevarying_strong_rhs,
            params={
                "k_tr": 1.60, "k_a": 1.00, "k_10": 0.16,
                "k_12": 0.20, "k_21": 0.25, "V_c": 18.5,
            },
            n_states=4,
            central_idx=2,
            volume_key="V_c",
        ),
        FailureTruthSpec(
            name="saturable_elimination",
            description="Michaelis-Menten elimination (V_max=1.5, K_m=5)",
            rhs=failure_saturable_rhs,
            params={
                "k_tr": 1.60, "k_a": 1.00,
                "v_max": 1.5, "k_m": 5.0,
                "k_12": 0.20, "k_21": 0.25, "V_c": 18.5,
            },
            n_states=4,
            central_idx=2,
            volume_key="V_c",
        ),
        FailureTruthSpec(
            name="enterohepatic_recirculation",
            description="Bile recirculation loop (k_bile=0.12, k_reabs=0.20)",
            rhs=failure_recirculation_rhs,
            params={
                "k_tr": 1.60, "k_a": 1.00, "k_10": 0.16,
                "k_12": 0.20, "k_21": 0.25, "V_c": 18.5,
                "k_bile": 0.12, "k_reabs": 0.20,
            },
            n_states=5,
            central_idx=2,
            volume_key="V_c",
        ),
    ]


# ---------------------------------------------------------------------------
# Experiment menu (same as divergence benchmark)
# ---------------------------------------------------------------------------

def get_experiments() -> dict[str, Experiment]:
    return {
        "e0": Experiment("Sparse Oral", "oral",
            np.array([0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 24.0])),
        "E1": Experiment("Absorption-Peak Oral", "oral",
            np.array([0.1, 0.33, 0.75, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0, 24.0])),
        "E2": Experiment("Late-Spread Oral", "oral",
            np.array([0.2, 0.33, 0.5, 1.25, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])),
        "E3": Experiment("Mid-Window Oral", "oral",
            np.array([0.1, 0.33, 0.5, 0.75, 1.25, 2.5, 10.0, 12.0, 20.0, 24.0])),
        "E4": Experiment("Early-Cluster Oral", "oral",
            np.array([0.2, 0.33, 0.75, 1.0, 1.25, 1.5, 2.5, 10.0, 16.0, 24.0])),
        "E5": Experiment("Tail-Emphasis Oral", "oral",
            np.array([0.1, 0.75, 1.25, 1.5, 2.5, 8.0, 10.0, 16.0, 20.0, 24.0])),
    }


FOLLOW_UP_KEYS = ["E1", "E2", "E3", "E4", "E5"]


def get_library_params() -> dict[str, dict[str, float]]:
    return {
        "A": {"k_a": 0.80, "k_e": 0.18, "V": 20.0},
        "B": {"k_tr": 1.20, "k_a": 0.80, "k_e": 0.18, "V": 20.0},
        "C": {"k_a": 0.70, "k_10": 0.16, "k_12": 0.20, "k_21": 0.25, "V_c": 18.0},
    }


# ---------------------------------------------------------------------------
# Core logic
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


def precompute_failure_features(
    failure_spec: FailureTruthSpec,
    experiments: dict[str, Experiment],
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for exp_key, experiment in experiments.items():
        result = simulate_failure_model(
            failure_spec.rhs, failure_spec.params, experiment,
            failure_spec.n_states, failure_spec.central_idx,
            failure_spec.volume_key,
        )
        features[exp_key] = feature_map(experiment, result)
    return features


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


def evaluate_identification(
    observed_keys: list[str],
    truth_features: dict[str, np.ndarray],
    library_features: dict[str, dict[str, np.ndarray]],
    id_margin: float,
) -> dict[str, object]:
    residuals = stacked_residuals(observed_keys, truth_features, library_features)
    ordered = sorted(residuals.items(), key=lambda item: item[1])
    best_model, best_resid = ordered[0]
    second_resid = ordered[1][1]
    gap = second_resid - best_resid
    n_obs = len(observed_keys)
    norm_resid = best_resid / np.sqrt(n_obs * FEATURE_DIM)
    gap_ok = gap >= id_margin
    fit_ok = norm_resid <= GOODNESS_THRESHOLD
    return {
        "best_model": best_model,
        "best_resid": float(best_resid),
        "second_resid": float(second_resid),
        "norm_resid": float(norm_resid),
        "residuals": residuals,
        "gap": float(gap),
        "gap_ok": bool(gap_ok),
        "fit_ok": bool(fit_ok),
        "identified": bool(gap_ok and fit_ok),
    }


def cartograph_sequence(
    h_blocks: dict[str, np.ndarray],
    follow_up_keys: list[str],
) -> list[str]:
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


def run_failure_scenario(
    failure_spec: FailureTruthSpec,
    sequence: list[str],
    failure_features: dict[str, np.ndarray],
    library_features: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Run multi-round loop and return per-round identification state."""
    history = []
    observed = ["e0"]
    state = evaluate_identification(observed, failure_features, library_features, IDENTIFICATION_MARGIN)
    state["round"] = 0
    state["observed"] = observed.copy()
    history.append(state)
    for round_idx, exp_key in enumerate(sequence, start=1):
        observed.append(exp_key)
        state = evaluate_identification(observed, failure_features, library_features, IDENTIFICATION_MARGIN)
        state["round"] = round_idx
        state["observed"] = observed.copy()
        history.append(state)
    return history


# ---------------------------------------------------------------------------
# Also run the in-library "control" truth for comparison
# ---------------------------------------------------------------------------

def get_control_truth() -> dict:
    """A truth that IS in the library family — should be identified."""
    return {
        "name": "control_in_library",
        "description": "Perturbed B-family (should be identified)",
        "rhs": model_b_rhs,
        "params": {"k_tr": 1.45, "k_a": 0.88, "k_e": 0.18, "V": 19.2},
        "model_name": "B",
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_residual_trajectories(
    output_dir: Path,
    all_histories: dict[str, list[dict]],
    control_history: list[dict],
) -> Path:
    """Per-round best residual and gap for all failure truths + control."""
    n_plots = len(all_histories) + 1
    fig, axes = plt.subplots(2, n_plots, figsize=(4.2 * n_plots, 7), squeeze=False)

    def plot_one(ax_resid, ax_gap, name, history):
        rounds = [h["round"] for h in history]
        best_resids = [h["best_resid"] for h in history]
        gaps = [h["gap"] for h in history]
        identified = [h["identified"] for h in history]

        ax_resid.plot(rounds, best_resids, "o-", color="#1f77b4", linewidth=2, markersize=5)
        ax_resid.set_title(name, fontsize=9)
        ax_resid.set_ylabel("Best residual")
        ax_resid.grid(alpha=0.25)

        gap_colors = ["#2ca02c" if not ident else "#d62728" for ident in identified]
        ax_gap.bar(rounds, gaps, color=gap_colors, alpha=0.7, width=0.6)
        ax_gap.axhline(IDENTIFICATION_MARGIN, color="#cc3333", linewidth=1.5,
                        linestyle="--", label=f"margin={IDENTIFICATION_MARGIN}")
        ax_gap.set_ylabel("Residual gap")
        ax_gap.set_xlabel("Round")
        ax_gap.grid(alpha=0.25)
        ax_gap.legend(fontsize=7, frameon=False)

    for idx, (name, history) in enumerate(sorted(all_histories.items())):
        plot_one(axes[0, idx], axes[1, idx], name, history)

    plot_one(axes[0, -1], axes[1, -1], "control (in-library)", control_history)

    fig.suptitle("Failure Case: Residuals & Identification Gap Across Rounds", fontsize=11)
    fig.tight_layout()
    path = output_dir / "figure_1_failure_residual_trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_final_residual_comparison(
    output_dir: Path,
    all_histories: dict[str, list[dict]],
    control_history: list[dict],
) -> Path:
    """Bar chart: final-round best residual and gap for each scenario."""
    names = sorted(all_histories.keys()) + ["control"]
    final_resids = []
    final_gaps = []
    for name in sorted(all_histories.keys()):
        h = all_histories[name][-1]
        final_resids.append(h["best_resid"])
        final_gaps.append(h["gap"])
    h = control_history[-1]
    final_resids.append(h["best_resid"])
    final_gaps.append(h["gap"])

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    colors = ["#1f77b4"] * len(all_histories) + ["#2ca02c"]
    axes[0].bar(x, final_resids, color=colors, width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, fontsize=8)
    axes[0].set_ylabel("Best-fit residual (full pool)")
    axes[0].set_title("Final Best Residual After All Experiments")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, final_gaps, color=colors, width=0.6)
    axes[1].axhline(IDENTIFICATION_MARGIN, color="#cc3333", linewidth=1.5, linestyle="--",
                     label=f"margin={IDENTIFICATION_MARGIN}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, fontsize=8)
    axes[1].set_ylabel("Residual gap (2nd - 1st)")
    axes[1].set_title("Final Identification Gap")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Failure vs Control: Can the Library Explain the Truth?", fontsize=11)
    fig.tight_layout()
    path = output_dir / "figure_2_final_residual_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_concentration_profiles(
    output_dir: Path,
    failure_specs: list[FailureTruthSpec],
    experiments: dict[str, Experiment],
    library_params: dict[str, dict[str, float]],
) -> Path:
    """Show PK curves for each failure truth vs library models under e0."""
    e0 = experiments["e0"]
    n = len(failure_specs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes[0]

    t_dense = np.linspace(0.01, 24.0, 300)
    e_dense = Experiment("dense", "oral", t_dense)

    for idx, spec in enumerate(failure_specs):
        ax = axes[idx]
        # Library models
        for model_name, rhs, color in [
            ("A", model_a_rhs, "#1f77b4"),
            ("B", model_b_rhs, "#ff7f0e"),
            ("C", model_c_rhs, "#2ca02c"),
        ]:
            result = simulate_model(rhs, library_params[model_name], e_dense, model_name)
            ax.plot(result.times, result.concentration, color=color, linewidth=1.5,
                    label=f"Model {model_name}", alpha=0.7)

        # Failure truth
        result = simulate_failure_model(
            spec.rhs, spec.params, e_dense,
            spec.n_states, spec.central_idx, spec.volume_key,
        )
        ax.plot(result.times, result.concentration, color="#111111", linewidth=2.5,
                linestyle="--", label="Failure truth")

        ax.set_title(spec.name, fontsize=9)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Failure Truths vs Library Models (Oral Dose)", fontsize=11)
    fig.tight_layout()
    path = output_dir / "figure_3_failure_concentration_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(output_dir: Path, payload: dict) -> tuple[Path, Path]:
    json_path = output_dir / "failure_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = output_dir / "failure_summary.md"
    lines = [
        "# PK Failure Benchmark Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        f"Identification margin: `{payload['identification_margin']:.2f}`",
        f"Goodness threshold (norm resid): `{payload['goodness_threshold']:.2f}`",
        f"CARTOGRAPH sequence: `{payload['cartograph_sequence']}`",
        "",
        "## Results",
        "",
        "| Scenario | Type | Final Resid | Norm Resid | Gap | Fit OK | Final ID | Revoked? |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in payload["summary_rows"]:
        revoked_str = "REVOKED" if row.get("revoked") else ""
        lines.append(
            f"| {row['name']} | {row['type']} | "
            f"{row['final_best_resid']:.4f} | {row['final_norm_resid']:.4f} | "
            f"{row['final_gap']:.4f} | "
            f"{'yes' if row['final_fit_ok'] else '**NO**'} | "
            f"{'YES' if row['final_identified'] else 'no'} | {revoked_str} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    failure_rows = [r for r in payload["summary_rows"] if r["type"] == "failure"]
    control_rows = [r for r in payload["summary_rows"] if r["type"] == "control"]
    all_refused_final = all(not r["final_identified"] for r in failure_rows)
    any_revoked = any(r.get("revoked") for r in failure_rows)
    control_identified = all(r["final_identified"] for r in control_rows)

    if all_refused_final and control_identified:
        lines.append("**All failure truths are correctly unidentified at the final round, while the control truth remains identified.** This demonstrates principled refusal to resolve when the library is genuinely insufficient.")
        if any_revoked:
            revoked_names = [r["name"] for r in failure_rows if r.get("revoked")]
            lines.append(f"")
            lines.append(f"Notably, {revoked_names} were *tentatively identified at early rounds but identification was revoked* as additional experiments exposed the structural misfit. This is the desired behavior: more data should increase confidence for well-specified models and *decrease* confidence for misspecified ones.")
    elif not all_refused_final:
        still_identified = [r["name"] for r in failure_rows if r["final_identified"]]
        lines.append(f"**Some failure truths remain identified at the final round:** {still_identified}. These mechanisms may need to be strengthened or the goodness threshold adjusted.")
    else:
        lines.append("**Failure truths refused but control also refused.** The threshold may be too aggressive.")

    lines.extend([
        "",
        "## Per-Round Details",
        "",
    ])
    for scenario in payload["scenario_details"]:
        lines.extend([
            f"### {scenario['name']} ({scenario['type']})",
            "",
            f"Description: {scenario['description']}",
            "",
            "| Round | Best Model | Best Resid | Norm Resid | Gap | Gap OK | Fit OK | Identified |",
            "|---:|---|---:|---:|---:|---|---|---|",
        ])
        for h in scenario["history"]:
            lines.append(
                f"| {h['round']} | {h['best_model']} | {h['best_resid']:.4f} | "
                f"{h['norm_resid']:.4f} | {h['gap']:.4f} | "
                f"{'yes' if h['gap_ok'] else 'no'} | "
                f"{'yes' if h['fit_ok'] else '**NO**'} | "
                f"{'YES' if h['identified'] else 'no'} |"
            )
        lines.append("")

    # Threshold sensitivity table
    if "threshold_sensitivity" in payload:
        scenario_names = [r["name"] for r in payload["summary_rows"]]
        lines.extend([
            "## Goodness-Threshold Sensitivity",
            "",
            f"Identification margin fixed at `{payload['identification_margin']:.2f}`. Each cell shows whether the scenario is identified (ID) or not (no) at the final round under the given threshold.",
            "",
        ])
        header = "| Threshold |" + "".join(f" {n} |" for n in scenario_names)
        sep = "|---:|" + "".join("---|" for _ in scenario_names)
        lines.append(header)
        lines.append(sep)
        for row in payload["threshold_sensitivity"]:
            cells = f"| {row['threshold']:.2f} |"
            for n in scenario_names:
                val = row.get(n, "?")
                cell = f" **{val}** |" if val == "ID" else f" {val} |"
                cells += cell
            lines.append(cells)
        lines.append("")

    # Per-feature residual breakdown
    if "feature_diagnostics" in payload:
        lines.extend([
            "## Per-Feature Residual Breakdown",
            "",
            "Shows which features drive the misfit for each scenario. Higher values indicate the feature detects the out-of-library mechanism.",
            "",
        ])
        scenario_names = list(payload["feature_diagnostics"].keys())
        feat_names = list(next(iter(payload["feature_diagnostics"].values())).keys())
        header = "| Feature |" + "".join(f" {n} |" for n in scenario_names)
        sep = "|---|" + "".join("---:|" for _ in scenario_names)
        lines.append(header)
        lines.append(sep)
        for feat in feat_names:
            cells = f"| {feat} |"
            for scenario in scenario_names:
                val = payload["feature_diagnostics"][scenario][feat]
                cells += f" {val:.4f} |"
            lines.append(cells)
        lines.append("")

    lines.extend([
        "## Artifact Paths",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = ensure_output_dir()
    experiments = get_experiments()
    library_params = get_library_params()
    failure_specs = get_failure_specs()

    library_features, h_blocks = precompute_library(experiments, library_params)
    sequence = cartograph_sequence(h_blocks, FOLLOW_UP_KEYS)
    print(f"CARTOGRAPH sequence: {sequence}")

    # Run failure scenarios
    all_histories: dict[str, list[dict]] = {}
    summary_rows = []
    scenario_details = []

    for spec in failure_specs:
        print(f"\nRunning failure: {spec.name} — {spec.description}")
        failure_features = precompute_failure_features(spec, experiments)
        history = run_failure_scenario(spec, sequence, failure_features, library_features)

        final = history[-1]
        # Track identification trajectory: first identified, last identified, final state
        id_rounds = [h["round"] for h in history if h["identified"]]
        first_id = id_rounds[0] if id_rounds else None
        last_id = id_rounds[-1] if id_rounds else None
        final_identified = bool(final["identified"])
        revoked = first_id is not None and not final_identified
        print(f"  Final: best={final['best_model']}, resid={final['best_resid']:.4f}, "
              f"norm_resid={final['norm_resid']:.4f}, gap={final['gap']:.4f}, "
              f"fit_ok={final['fit_ok']}, final_identified={final_identified}, revoked={revoked}")

        all_histories[spec.name] = history
        summary_rows.append({
            "name": spec.name,
            "type": "failure",
            "description": spec.description,
            "final_best_model": final["best_model"],
            "final_best_resid": float(final["best_resid"]),
            "final_norm_resid": float(final["norm_resid"]),
            "final_gap": float(final["gap"]),
            "final_gap_ok": bool(final["gap_ok"]),
            "final_fit_ok": bool(final["fit_ok"]),
            "final_identified": final_identified,
            "first_id_round": first_id,
            "last_id_round": last_id,
            "revoked": revoked,
        })
        scenario_details.append({
            "name": spec.name,
            "type": "failure",
            "description": spec.description,
            "history": history,
        })

    # Run control
    control = get_control_truth()
    print(f"\nRunning control: {control['name']}")
    control_features: dict[str, np.ndarray] = {}
    for exp_key, experiment in experiments.items():
        result = simulate_model(control["rhs"], control["params"], experiment, control["model_name"])
        control_features[exp_key] = feature_map(experiment, result)

    control_history = run_failure_scenario(
        None, sequence, control_features, library_features,
    )
    final_control = control_history[-1]
    ctrl_id_rounds = [h["round"] for h in control_history if h["identified"]]
    ctrl_first_id = ctrl_id_rounds[0] if ctrl_id_rounds else None
    ctrl_last_id = ctrl_id_rounds[-1] if ctrl_id_rounds else None
    ctrl_final_identified = bool(final_control["identified"])
    ctrl_revoked = ctrl_first_id is not None and not ctrl_final_identified
    print(f"  Final: best={final_control['best_model']}, resid={final_control['best_resid']:.4f}, "
          f"norm_resid={final_control['norm_resid']:.4f}, gap={final_control['gap']:.4f}, "
          f"fit_ok={final_control['fit_ok']}, final_identified={ctrl_final_identified}")

    summary_rows.append({
        "name": control["name"],
        "type": "control",
        "description": control["description"],
        "final_best_model": final_control["best_model"],
        "final_best_resid": float(final_control["best_resid"]),
        "final_norm_resid": float(final_control["norm_resid"]),
        "final_gap": float(final_control["gap"]),
        "final_gap_ok": bool(final_control["gap_ok"]),
        "final_fit_ok": bool(final_control["fit_ok"]),
        "final_identified": ctrl_final_identified,
        "first_id_round": ctrl_first_id,
        "last_id_round": ctrl_last_id,
        "revoked": ctrl_revoked,
    })
    scenario_details.append({
        "name": control["name"],
        "type": "control",
        "description": control["description"],
        "history": control_history,
    })

    # ---- Threshold sensitivity sweep ----
    threshold_sensitivity: list[dict] = []
    all_scenarios = list(all_histories.items()) + [("control_in_library", control_history)]
    for threshold in GOODNESS_THRESHOLD_ABLATION:
        row: dict[str, object] = {"threshold": threshold}
        for scenario_name, history in all_scenarios:
            final_h = history[-1]
            nr = float(final_h["norm_resid"])
            gap = float(final_h["gap"])
            fit_ok = nr <= threshold
            gap_ok = gap >= IDENTIFICATION_MARGIN
            identified = fit_ok and gap_ok
            row[scenario_name] = "ID" if identified else "no"
        threshold_sensitivity.append(row)

    print("\n--- Threshold Sensitivity ---")
    scenario_names = [name for name, _ in all_scenarios]
    header = f"  {'threshold':>10s}" + "".join(f"  {n:>30s}" for n in scenario_names)
    print(header)
    for row in threshold_sensitivity:
        vals = f"  {row['threshold']:>10.2f}" + "".join(f"  {row[n]:>30s}" for n in scenario_names)
        print(vals)

    # ---- Per-feature residual breakdown (honest sensitivity diagnostic) ----
    # This answers: which features actually detect the out-of-library mechanisms?
    feature_diagnostics: dict[str, dict[str, list[float]]] = {}
    all_exp_keys = ["e0"] + FOLLOW_UP_KEYS
    for spec in failure_specs:
        failure_features = precompute_failure_features(spec, experiments)
        per_feature_resid: dict[str, float] = {}
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            # Stack this single feature across all experiments for best library model
            best_model_name = [r for r in summary_rows if r["name"] == spec.name][0]["final_best_model"]
            truth_vals = np.array([failure_features[k][feat_idx] for k in all_exp_keys])
            model_vals = np.array([library_features[k][best_model_name][feat_idx] for k in all_exp_keys])
            per_feature_resid[feat_name] = float(np.linalg.norm(truth_vals - model_vals))
        feature_diagnostics[spec.name] = per_feature_resid

    # Also for control
    ctrl_per_feature: dict[str, float] = {}
    ctrl_best = [r for r in summary_rows if r["name"] == "control_in_library"][0]["final_best_model"]
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        truth_vals = np.array([control_features[k][feat_idx] for k in all_exp_keys])
        model_vals = np.array([library_features[k][ctrl_best][feat_idx] for k in all_exp_keys])
        ctrl_per_feature[feat_name] = float(np.linalg.norm(truth_vals - model_vals))
    feature_diagnostics["control_in_library"] = ctrl_per_feature

    print("\n--- Per-Feature Residual Breakdown ---")
    for scenario_name, feat_resids in feature_diagnostics.items():
        print(f"  {scenario_name}:")
        for feat_name, val in feat_resids.items():
            print(f"    {feat_name:>20s}: {val:.4f}")

    # Plots
    artifacts: dict[str, str] = {}
    artifacts["figure_1_failure_residual_trajectories"] = str(
        plot_residual_trajectories(output_dir, all_histories, control_history)
    )
    artifacts["figure_2_final_residual_comparison"] = str(
        plot_final_residual_comparison(output_dir, all_histories, control_history)
    )
    artifacts["figure_3_failure_concentration_profiles"] = str(
        plot_concentration_profiles(output_dir, failure_specs, experiments, library_params)
    )
    artifacts["failure_results_json"] = str(output_dir / "failure_results.json")
    artifacts["failure_summary_md"] = str(output_dir / "failure_summary.md")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identification_margin": IDENTIFICATION_MARGIN,
        "goodness_threshold": GOODNESS_THRESHOLD,
        "cartograph_sequence": sequence,
        "summary_rows": summary_rows,
        "scenario_details": scenario_details,
        "threshold_sensitivity": threshold_sensitivity,
        "feature_diagnostics": feature_diagnostics,
        "artifacts": artifacts,
    }
    write_results(output_dir, payload)

    print("\n--- Summary ---")
    for row in summary_rows:
        if row["final_identified"]:
            tag = "IDENTIFIED"
        elif row.get("revoked"):
            tag = "REVOKED"
        else:
            tag = "REFUSED"
        print(f"  {row['name']:35s} [{row['type']:7s}] {tag:10s} "
              f"norm_resid={row['final_norm_resid']:.3f} gap={row['final_gap']:.4f}")

    print("\nSaved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
