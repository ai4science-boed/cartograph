from __future__ import annotations

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
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOSE = 100.0
T_END = 24.0
EPS = 1e-8
TAU_RATIO = 0.20
TAU_ABLATION_RATIOS = [0.18, 0.20, 0.22]
OUTPUT_DIR = Path("outputs") / "pk_first_pass"


@dataclass(frozen=True)
class Experiment:
    name: str
    route: str
    times: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    times: np.ndarray
    concentration: np.ndarray


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["XDG_CACHE_HOME"])).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def model_a_rhs(_t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
    a_g, a_c = y
    k_a = params["k_a"]
    k_e = params["k_e"]
    return np.array([
        -k_a * a_g,
        k_a * a_g - k_e * a_c,
    ])


def model_b_rhs(_t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
    a_g, a_t, a_c = y
    k_tr = params["k_tr"]
    k_a = params["k_a"]
    k_e = params["k_e"]
    return np.array([
        -k_tr * a_g,
        k_tr * a_g - k_a * a_t,
        k_a * a_t - k_e * a_c,
    ])


def model_c_rhs(_t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
    a_g, a_c, a_p = y
    k_a = params["k_a"]
    k_10 = params["k_10"]
    k_12 = params["k_12"]
    k_21 = params["k_21"]
    return np.array([
        -k_a * a_g,
        k_a * a_g - (k_10 + k_12) * a_c + k_21 * a_p,
        k_12 * a_c - k_21 * a_p,
    ])


def model_d_rhs(_t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
    a_g, a_t, a_c, a_p = y
    k_tr = params["k_tr"]
    k_a = params["k_a"]
    k_10 = params["k_10"]
    k_12 = params["k_12"]
    k_21 = params["k_21"]
    return np.array([
        -k_tr * a_g,
        k_tr * a_g - k_a * a_t,
        k_a * a_t - (k_10 + k_12) * a_c + k_21 * a_p,
        k_12 * a_c - k_21 * a_p,
    ])


def failure_truth_rhs(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
    a_g, a_t, a_c, a_p = y
    k_tr = params["k_tr"]
    k_a = params["k_a"]
    k_10 = params["k_10"] * (1.0 + 0.35 * np.exp(-t / 4.0))
    k_12 = params["k_12"]
    k_21 = params["k_21"]
    return np.array([
        -k_tr * a_g,
        k_tr * a_g - k_a * a_t,
        k_a * a_t - (k_10 + k_12) * a_c + k_21 * a_p,
        k_12 * a_c - k_21 * a_p,
    ])


def simulate_model(
    rhs: Callable[[float, np.ndarray, dict[str, float]], np.ndarray],
    params: dict[str, float],
    experiment: Experiment,
    model_name: str,
) -> SimulationResult:
    t_eval = np.array(sorted(set([0.0, *experiment.times.tolist()])), dtype=float)

    if model_name == "A":
        if experiment.route == "oral":
            y0 = np.array([DOSE, 0.0])
        else:
            y0 = np.array([0.0, DOSE])
        volume = params["V"]
        central_idx = 1
    elif model_name == "B":
        if experiment.route == "oral":
            y0 = np.array([DOSE, 0.0, 0.0])
        else:
            y0 = np.array([0.0, 0.0, DOSE])
        volume = params["V"]
        central_idx = 2
    elif model_name == "C":
        if experiment.route == "oral":
            y0 = np.array([DOSE, 0.0, 0.0])
        else:
            y0 = np.array([0.0, DOSE, 0.0])
        volume = params["V_c"]
        central_idx = 1
    elif model_name in {"truth", "truth_failure"}:
        if experiment.route == "oral":
            y0 = np.array([DOSE, 0.0, 0.0, 0.0])
        else:
            y0 = np.array([0.0, 0.0, DOSE, 0.0])
        volume = params["V_c"]
        central_idx = 2
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    sol = solve_ivp(
        lambda t, y: rhs(t, y, params),
        (0.0, T_END),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="LSODA",
    )
    if not sol.success:
        raise RuntimeError(f"Simulation failed for model {model_name}: {sol.message}")

    central = sol.y[central_idx]
    concentration = np.maximum(central / volume, 0.0)
    return SimulationResult(times=t_eval, concentration=concentration)


def trapezoid_auc(times: np.ndarray, values: np.ndarray, upper: float) -> float:
    mask = times <= upper + EPS
    t = times[mask]
    v = values[mask]
    if t[-1] < upper:
        interp = np.interp(upper, times, values)
        t = np.append(t, upper)
        v = np.append(v, interp)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(v, t))


def terminal_log_slope(times: np.ndarray, concentration: np.ndarray) -> float:
    positive_mask = concentration > 0
    t = times[positive_mask]
    c = concentration[positive_mask]
    if len(c) < 4:
        return 0.0
    t_tail = t[-4:]
    log_c = np.log(c[-4:])
    slope, _ = np.polyfit(t_tail, log_c, 1)
    return float(slope)


def post_peak_loglinearity_rmse(times: np.ndarray, concentration: np.ndarray) -> float:
    positive_mask = concentration > 0
    t = times[positive_mask]
    c = concentration[positive_mask]
    if len(c) < 4:
        return 0.0
    peak_idx = int(np.argmax(c))
    t_seg = t[peak_idx:]
    c_seg = c[peak_idx:]
    if len(c_seg) < 3:
        return 0.0
    log_c = np.log(c_seg)
    slope, intercept = np.polyfit(t_seg, log_c, 1)
    residual = log_c - (slope * t_seg + intercept)
    return float(np.sqrt(np.mean(residual**2)))


def feature_map(experiment: Experiment, result: SimulationResult) -> np.ndarray:
    times = result.times
    conc = result.concentration
    sampled_mask = np.isin(times, experiment.times) | np.isclose(times, 0.0)
    sampled_times = times[sampled_mask]
    sampled_conc = conc[sampled_mask]

    peak_idx = int(np.argmax(sampled_conc))
    t_max = float(sampled_times[peak_idx])
    c_max = float(sampled_conc[peak_idx])

    early_window = 2.0 if experiment.route == "oral" else 1.0
    auc_early = trapezoid_auc(sampled_times, sampled_conc, early_window)
    auc_total = max(trapezoid_auc(sampled_times, sampled_conc, 24.0), EPS)
    early_fraction = auc_early / auc_total

    slope = terminal_log_slope(sampled_times, sampled_conc)
    curvature = post_peak_loglinearity_rmse(sampled_times, sampled_conc)

    return np.array([t_max, c_max, early_fraction, slope, curvature], dtype=float)


def build_h_block(
    experiment: Experiment,
    model_features: dict[str, np.ndarray],
    normalize: bool = False,
) -> np.ndarray:
    h_abs = model_features["B"] - model_features["A"]
    h_dist = model_features["C"] - model_features["A"]
    block = np.column_stack([h_abs, h_dist])
    if not normalize:
        return block
    out = block.astype(float).copy()
    for idx in range(out.shape[1]):
        out[:, idx] /= np.linalg.norm(out[:, idx]) + EPS
    return out


def unresolved_subspace(h_current: np.ndarray, tau_ratio: float = 0.20) -> tuple[np.ndarray, np.ndarray, float]:
    _, singular_values, vt = np.linalg.svd(h_current, full_matrices=False)
    tau = tau_ratio * singular_values[0]
    unresolved_indices = np.where(singular_values <= tau + EPS)[0]
    if len(unresolved_indices) == 0:
        return np.zeros((h_current.shape[1], 0)), singular_values, tau
    basis = vt[unresolved_indices].T
    return basis, singular_values, tau


def acquisition_score(h_e: np.ndarray, u_tau: np.ndarray) -> tuple[float, float]:
    if u_tau.shape[1] == 0:
        return 0.0, 0.0
    projected = h_e @ u_tau
    fro_sq = float(np.linalg.norm(projected, ord="fro") ** 2)
    singular_values = np.linalg.svd(projected, compute_uv=False)
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    return fro_sq, sigma_min


def disagreement_magnitude_score(features: dict[str, np.ndarray]) -> float:
    model_names = sorted(features)
    total = 0.0
    for i, name_i in enumerate(model_names):
        for name_j in model_names[i + 1 :]:
            total += float(np.linalg.norm(features[name_j] - features[name_i]))
    return total


def print_feature_table(experiments: dict[str, Experiment], all_features: dict[str, dict[str, np.ndarray]]) -> None:
    for exp_name, experiment in experiments.items():
        print(f"\nExperiment {exp_name} ({experiment.route}) features:")
        for model_name in ["A", "B", "C", "truth", "truth_failure"]:
            feat = all_features[exp_name][model_name]
            feat_str = ", ".join(f"{x:.4f}" for x in feat)
            print(f"  {model_name:>12}: [{feat_str}]")


def sampled_profile(experiment: Experiment, result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    sampled_times = experiment.times
    sampled_conc = np.interp(sampled_times, result.times, result.concentration)
    return sampled_times, sampled_conc


def sampled_curve_distance(
    experiment: Experiment,
    result_i: SimulationResult,
    result_j: SimulationResult,
) -> float:
    _, conc_i = sampled_profile(experiment, result_i)
    _, conc_j = sampled_profile(experiment, result_j)
    return float(np.linalg.norm(conc_i - conc_j))


def singular_metrics(
    h: np.ndarray,
    tau_ratio: float = TAU_RATIO,
    tau_reference: float | None = None,
) -> dict[str, float | int | list[float]]:
    singular_values = np.linalg.svd(h, compute_uv=False)
    tau = tau_ratio * singular_values[0]
    tau_reference = tau if tau_reference is None else tau_reference
    unresolved_dim = int(np.sum(singular_values <= tau + EPS))
    unresolved_dim_fixed = int(np.sum(singular_values <= tau_reference + EPS))
    sigma_min = float(singular_values[-1])
    sigma_ratio = float(singular_values[-1] / singular_values[0])
    return {
        "singular_values": singular_values.tolist(),
        "tau": float(tau),
        "tau_reference": float(tau_reference),
        "unresolved_dim": unresolved_dim,
        "unresolved_dim_fixed": unresolved_dim_fixed,
        "sigma_min": sigma_min,
        "sigma_ratio": sigma_ratio,
    }


def plot_initial_curves(
    output_dir: Path,
    experiment: Experiment,
    simulations: dict[str, SimulationResult],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = {
        "A": "Model A",
        "B": "Model B",
        "C": "Model C",
        "truth": "Primary Truth",
        "truth_failure": "Failure Truth",
    }
    styles = {
        "A": ("#1f77b4", "-"),
        "B": ("#ff7f0e", "-"),
        "C": ("#2ca02c", "-"),
        "truth": ("#111111", "--"),
        "truth_failure": ("#aa3377", ":"),
    }
    ax_full, ax_obs = axes
    for key in ["A", "B", "C", "truth", "truth_failure"]:
        color, linestyle = styles[key]
        result = simulations[key]
        ax_full.plot(result.times, result.concentration, color=color, linestyle=linestyle, linewidth=2, label=labels[key])
        obs_t, obs_c = sampled_profile(experiment, result)
        ax_full.scatter(obs_t, obs_c, color=color, s=18)
        ax_obs.plot(obs_t, obs_c, color=color, linestyle=linestyle, linewidth=2, marker="o", markersize=4, label=labels[key])
    ax_full.set_title("Underlying Profiles")
    ax_full.set_xlabel("Time (h)")
    ax_full.set_ylabel("Concentration (mg/L)")
    ax_full.grid(alpha=0.25)
    ax_obs.set_title("Observed Sparse Samples")
    ax_obs.set_xlabel("Time (h)")
    ax_obs.set_ylabel("Concentration (mg/L)")
    ax_obs.grid(alpha=0.25)
    ax_obs.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Initial Sparse-Oral Ambiguity With Out-of-Library Truth")
    fig.tight_layout()
    path = output_dir / "figure_1_initial_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_acquisition_scores(output_dir: Path, ranking_rows: list[tuple[str, float, float, float]]) -> Path:
    labels = [row[0] for row in ranking_rows]
    unresolved_scores = [row[1] for row in ranking_rows]
    disagreement_scores = [row[3] for row in ranking_rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, unresolved_scores, width=width, color="#1f77b4", label="Unresolved-subspace score")
    ax.bar(x + width / 2, disagreement_scores, width=width, color="#ff7f0e", label="Disagreement magnitude")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Candidate Experiment Scores")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "figure_2_acquisition_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_update_metrics(
    output_dir: Path,
    base_metrics: dict[str, float | int | list[float]],
    update_metrics: dict[str, dict[str, float | int | list[float]]],
) -> Path:
    labels = ["e0", *update_metrics.keys()]
    sigma_min_values = [base_metrics["sigma_min"], *[update_metrics[label]["sigma_min"] for label in update_metrics]]
    sigma_ratio_values = [base_metrics["sigma_ratio"], *[update_metrics[label]["sigma_ratio"] for label in update_metrics]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x, sigma_min_values, color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Smallest Singular Value")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, sigma_ratio_values, color="#9467bd")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Singular Value Ratio")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Predicted Ambiguity Metrics Before and After One Added Experiment")
    fig.tight_layout()
    path = output_dir / "figure_3_update_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_selected_experiment_curves(
    output_dir: Path,
    experiment_key: str,
    experiment: Experiment,
    simulations: dict[str, SimulationResult],
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {
        "A": "Model A",
        "B": "Model B",
        "C": "Model C",
        "truth": "Primary Truth",
        "truth_failure": "Failure Truth",
    }
    styles = {
        "A": ("#1f77b4", "-"),
        "B": ("#ff7f0e", "-"),
        "C": ("#2ca02c", "-"),
        "truth": ("#111111", "--"),
        "truth_failure": ("#aa3377", ":"),
    }
    for key in ["A", "B", "C", "truth", "truth_failure"]:
        color, linestyle = styles[key]
        result = simulations[key]
        ax.plot(result.times, result.concentration, color=color, linestyle=linestyle, linewidth=2, label=labels[key])
        ax.scatter(experiment.times, np.interp(experiment.times, result.times, result.concentration), color=color, s=18)
    ax.set_title(f"Selected Experiment {experiment_key}: {experiment.name}")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "figure_4_selected_experiment_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_space_summary(
    output_dir: Path,
    experiment_key: str,
    features: dict[str, np.ndarray],
) -> Path:
    model_order = ["A", "B", "C", "truth", "truth_failure"]
    feature_labels = ["Tmax", "Cmax", "AUCfrac", "TailSlope", "LogLinRMSE"]
    label_map = {
        "A": "A",
        "B": "B",
        "C": "C",
        "truth": "Truth",
        "truth_failure": "Fail",
    }
    matrix = np.vstack([features[key] for key in model_order])
    dist = np.zeros((len(model_order), len(model_order)))
    for i, key_i in enumerate(model_order):
        for j, key_j in enumerate(model_order):
            dist[i, j] = float(np.linalg.norm(features[key_i] - features[key_j]))

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    standardized = (matrix - mean) / std

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im = axes[0].imshow(dist, cmap="YlOrRd")
    axes[0].set_xticks(np.arange(len(model_order)))
    axes[0].set_yticks(np.arange(len(model_order)))
    axes[0].set_xticklabels([label_map[key] for key in model_order])
    axes[0].set_yticklabels([label_map[key] for key in model_order])
    axes[0].set_title(f"Feature-Space Distances ({experiment_key})")
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            axes[0].text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#222222")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for key, style in [("A", "#1f77b4"), ("B", "#ff7f0e"), ("C", "#2ca02c"), ("truth", "#111111"), ("truth_failure", "#aa3377")]:
        axes[1].plot(feature_labels, standardized[model_order.index(key)], marker="o", linewidth=2, color=style, label=label_map[key])
    axes[1].axhline(0.0, color="#888888", linewidth=1, alpha=0.6)
    axes[1].set_title(f"Standardized Feature Profiles ({experiment_key})")
    axes[1].set_ylabel("Standardized value")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path = output_dir / "figure_5_feature_space_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_failure_residuals(
    output_dir: Path,
    experiment_key: str,
    selected_residuals: dict[str, dict[str, float]],
) -> Path:
    labels = list(selected_residuals)
    truth_values = [selected_residuals[key]["truth_resid"] for key in labels]
    failure_values = [selected_residuals[key]["failure_resid"] for key in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, truth_values, width=width, color="#111111", label="Primary truth residual")
    ax.bar(x + width / 2, failure_values, width=width, color="#aa3377", label="Failure-truth residual")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Feature residual norm")
    ax.set_title(f"Residuals Under Selected Experiment {experiment_key}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "figure_6_failure_residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_results_json(output_dir: Path, payload: dict) -> Path:
    path = output_dir / "latest_results.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def write_results_markdown(output_dir: Path, payload: dict) -> Path:
    latest = payload["latest_run"]
    same_winner = latest["selected_experiment"] == latest["disagreement_winner"]
    path = output_dir / "final_and_latest_results.md"
    lines = [
        "# PK First-Pass Final and Latest Results",
        "",
        f"Run timestamp (UTC): `{latest['timestamp_utc']}`",
        "",
        "## Latest Run",
        "",
        f"- Initial experiment: `{latest['initial_experiment']}`",
        f"- Tau ratio: `{latest['tau_ratio']:.2f}`",
        f"- Initial singular values: `{latest['initial_metrics']['singular_values']}`",
        f"- Initial sigma ratio: `{latest['initial_metrics']['sigma_ratio']:.6f}`",
        f"- Initial unresolved dimension (dynamic tau): `{latest['initial_metrics']['unresolved_dim']}`",
        f"- Initial unresolved dimension (fixed tau): `{latest['initial_metrics']['unresolved_dim_fixed']}`",
        f"- Selected experiment by unresolved-subspace score: `{latest['selected_experiment']}`",
        f"- Selected experiment by disagreement magnitude: `{latest['disagreement_winner']}`",
        "",
        "### Parameter Snapshot",
        "",
        f"- `A`: `{latest['parameters']['A']}`",
        f"- `B`: `{latest['parameters']['B']}`",
        f"- `C`: `{latest['parameters']['C']}`",
        f"- `truth`: `{latest['parameters']['truth']}`",
        "",
        "### Initial Ambiguity Check",
        "",
        f"- Sparse-oral feature distance between `B` and `C`: `{latest['initial_ambiguity']['bc_feature_distance']:.6f}`",
        f"- Sparse-oral sampled-curve distance between `B` and `C`: `{latest['initial_ambiguity']['bc_sampled_curve_distance']:.6f}`",
        f"- Truth residual vs `A` under `e0`: `{latest['initial_ambiguity']['truth_residuals_e0']['A']:.6f}`",
        f"- Truth residual vs `B` under `e0`: `{latest['initial_ambiguity']['truth_residuals_e0']['B']:.6f}`",
        f"- Truth residual vs `C` under `e0`: `{latest['initial_ambiguity']['truth_residuals_e0']['C']:.6f}`",
        "",
        "### Candidate Scores",
        "",
    ]
    for row in latest["candidate_scores"]:
        lines.append(
            f"- `{row['experiment']}`: unresolved_score=`{row['score']:.6f}`, "
            f"sigma_min=`{row['sigma_min']:.6f}`, disagreement_mag=`{row['disagreement_mag']:.6f}`"
        )
    lines.extend(
        [
            "",
            "### Threshold Sensitivity",
            "",
        ]
    )
    for row in latest["tau_ablation"]:
        lines.append(
            f"- tau_ratio=`{row['tau_ratio']:.2f}`: unresolved_dim=`{row['unresolved_dim']}`, "
            f"winner=`{row['winner']}`"
        )
    lines.extend(
        [
            "",
            "### One-Step Update Metrics",
            "",
        ]
    )
    for key, metrics in latest["update_metrics"].items():
        lines.append(
            f"- `{key}`: sigma_min=`{metrics['sigma_min']:.6f}`, sigma_ratio=`{metrics['sigma_ratio']:.6f}`, "
            f"sigma_ratio_gain=`{metrics['sigma_ratio_gain']:.6f}`, "
            f"unresolved_dim_dynamic=`{metrics['unresolved_dim']}`, "
            f"unresolved_dim_fixed=`{metrics['unresolved_dim_fixed']}`"
        )
    lines.extend(
        [
            "",
            "### Residual Norms Under Selected Experiment",
            "",
        ]
    )
    for key, metrics in latest["selected_residuals"].items():
        lines.append(
            f"- `{key}`: truth_resid=`{metrics['truth_resid']:.6f}`, "
            f"failure_resid=`{metrics['failure_resid']:.6f}`, "
            f"failure_gap=`{metrics['failure_gap']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Current Final Takeaways",
            "",
            "- The primary truth is an out-of-library combined transit-plus-distribution model, so no library model has zero residual by construction.",
            f"- The sparse-oral setup makes `B` and `C` genuinely close in feature space (`{latest['initial_ambiguity']['bc_feature_distance']:.6f}`), which is the intended ambiguity regime.",
            f"- At the observed sparse samples, `B` and `C` remain close (`{latest['initial_ambiguity']['bc_sampled_curve_distance']:.6f}`), which is why the feature-space ambiguity is not just an artifact of full continuous trajectories.",
            f"- The best one-step update improves sigma ratio from `{latest['initial_metrics']['sigma_ratio']:.6f}` to `{latest['update_metrics'][latest['selected_experiment']]['sigma_ratio']:.6f}`.",
            "- Dynamic threshold collapse is still conservative, so the continuous singular-value improvement is the more informative ambiguity metric in this pass.",
            "",
            "## Artifact Paths",
            "",
        ]
    )
    if same_winner:
        lines[lines.index("## Artifact Paths") - 3 : lines.index("## Artifact Paths") - 3] = [
            f"- The unresolved-subspace rule and raw disagreement magnitude both select `{latest['selected_experiment']}` in this tuned pass.",
            "- The added value of the structured rule here is the singular-value-based justification and the clearer separation between genuinely useful and weak follow-up experiments.",
        ]
    else:
        lines[lines.index("## Artifact Paths") - 3 : lines.index("## Artifact Paths") - 3] = [
            f"- The unresolved-subspace rule selects `{latest['selected_experiment']}`, while raw disagreement magnitude selects `{latest['disagreement_winner']}`.",
            "- This shows the structured acquisition rule is not collapsing to the simple disagreement heuristic.",
        ]
    for name, artifact_path in latest["artifacts"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.append("")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    output_dir = ensure_output_dir()
    experiments = {
        "e0": Experiment("Sparse Oral", "oral", np.array([0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 24.0])),
        "A": Experiment("Dense Early Oral", "oral", np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0, 10.0, 24.0])),
        "B": Experiment("Dense Late Oral", "oral", np.array([0.5, 1.5, 3.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0])),
        "C": Experiment("IV Dense Early-Mid", "iv", np.array([0.08, 0.17, 0.33, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 24.0])),
    }

    params_a = {"k_a": 0.80, "k_e": 0.18, "V": 20.0}
    params_b = {"k_tr": 1.20, "k_a": 0.80, "k_e": 0.18, "V": 20.0}
    params_c = {"k_a": 0.70, "k_10": 0.16, "k_12": 0.20, "k_21": 0.25, "V_c": 18.0}
    params_truth = {"k_tr": 1.60, "k_a": 1.00, "k_10": 0.16, "k_12": 0.20, "k_21": 0.25, "V_c": 18.5}

    all_features: dict[str, dict[str, np.ndarray]] = {}
    h_blocks: dict[str, np.ndarray] = {}
    all_simulations: dict[str, dict[str, SimulationResult]] = {}

    for exp_key, experiment in experiments.items():
        result_a = simulate_model(model_a_rhs, params_a, experiment, "A")
        result_b = simulate_model(model_b_rhs, params_b, experiment, "B")
        result_c = simulate_model(model_c_rhs, params_c, experiment, "C")
        result_truth = simulate_model(model_d_rhs, params_truth, experiment, "truth")
        result_failure = simulate_model(failure_truth_rhs, params_truth, experiment, "truth_failure")

        all_simulations[exp_key] = {
            "A": result_a,
            "B": result_b,
            "C": result_c,
            "truth": result_truth,
            "truth_failure": result_failure,
        }
        features = {
            "A": feature_map(experiment, result_a),
            "B": feature_map(experiment, result_b),
            "C": feature_map(experiment, result_c),
            "truth": feature_map(experiment, result_truth),
            "truth_failure": feature_map(experiment, result_failure),
        }
        all_features[exp_key] = features
        h_blocks[exp_key] = build_h_block(experiment, features, normalize=False)

    h_current = h_blocks["e0"]
    u_tau, singular_values, tau = unresolved_subspace(h_current, tau_ratio=TAU_RATIO)
    base_metrics = singular_metrics(h_current, tau_ratio=TAU_RATIO)

    print("Initial sparse-oral H_e0:")
    print(np.array2string(h_current, precision=4, suppress_small=True))
    print(f"Singular values: {np.array2string(singular_values, precision=4, suppress_small=True)}")
    print(f"tau: {tau:.4f}")
    print(f"Unresolved subspace dimension: {u_tau.shape[1]}")
    if u_tau.shape[1] > 0:
        print("U_tau basis:")
        print(np.array2string(u_tau, precision=4, suppress_small=True))

    print_feature_table(experiments, all_features)

    print("\nCandidate acquisition scores:")
    ranking_rows = []
    for exp_key in ["A", "B", "C"]:
        fro_sq, sigma_min = acquisition_score(h_blocks[exp_key], u_tau)
        mag = disagreement_magnitude_score({k: v for k, v in all_features[exp_key].items() if k in {"A", "B", "C"}})
        ranking_rows.append((exp_key, fro_sq, sigma_min, mag))
        print(
            f"  {exp_key}: score={fro_sq:.6f}, sigma_min={sigma_min:.6f}, "
            f"disagreement_mag={mag:.6f}"
        )

    ranking_rows.sort(key=lambda row: row[1], reverse=True)
    artifacts: dict[str, str] = {}
    tau_ablation_rows: list[dict[str, float | int | str]] = []
    update_metrics: dict[str, dict[str, float | int | list[float]]] = {}
    selected_residuals: dict[str, dict[str, float]] = {}
    initial_truth_residuals = {
        model_name: float(np.linalg.norm(all_features["e0"]["truth"] - all_features["e0"][model_name]))
        for model_name in ["A", "B", "C"]
    }
    initial_bc_distance = float(np.linalg.norm(all_features["e0"]["B"] - all_features["e0"]["C"]))
    initial_bc_sampled_curve_distance = sampled_curve_distance(
        experiments["e0"],
        all_simulations["e0"]["B"],
        all_simulations["e0"]["C"],
    )

    if ranking_rows:
        winner = ranking_rows[0][0]
        disagreement_winner = max(ranking_rows, key=lambda row: row[3])[0]
        print(f"\nTop acquisition by unresolved-subspace score: {winner}")
        print(f"Top acquisition by disagreement magnitude: {disagreement_winner}")

        print("\nPredicted post-update singular values by candidate:")
        for exp_key in ["A", "B", "C"]:
            h_plus = np.vstack([h_blocks["e0"], h_blocks[exp_key]])
            metrics = singular_metrics(h_plus, tau_ratio=TAU_RATIO, tau_reference=float(base_metrics["tau"]))
            s_plus = np.array(metrics["singular_values"])
            tau_plus = float(metrics["tau"])
            unresolved_dim_plus = int(metrics["unresolved_dim"])
            metrics["sigma_min_gain"] = float(metrics["sigma_min"] - base_metrics["sigma_min"])
            metrics["sigma_ratio_gain"] = float(metrics["sigma_ratio"] - base_metrics["sigma_ratio"])
            update_metrics[exp_key] = metrics
            print(
                f"  {exp_key}: singular={np.array2string(s_plus, precision=4, suppress_small=True)}, "
                f"tau={tau_plus:.4f}, unresolved_dim={unresolved_dim_plus}, "
                f"sigma_ratio={metrics['sigma_ratio']:.4f}"
            )

        print("\nResidual norms against selected experiment features:")
        selected_truth = all_features[winner]["truth"]
        selected_failure = all_features[winner]["truth_failure"]
        for model_name in ["A", "B", "C"]:
            truth_resid = float(np.linalg.norm(selected_truth - all_features[winner][model_name]))
            failure_resid = float(np.linalg.norm(selected_failure - all_features[winner][model_name]))
            selected_residuals[model_name] = {
                "truth_resid": truth_resid,
                "failure_resid": failure_resid,
                "failure_gap": failure_resid - truth_resid,
            }
            print(
                f"  {model_name}: truth_resid={truth_resid:.6f}, "
                f"failure_resid={failure_resid:.6f}"
            )

        print("\nThreshold sensitivity:")
        _, base_s, base_vt = np.linalg.svd(h_current, full_matrices=False)
        for tau_ratio in TAU_ABLATION_RATIOS:
            tau_i = tau_ratio * base_s[0]
            unresolved_idx = np.where(base_s <= tau_i + EPS)[0]
            u_i = base_vt[unresolved_idx].T if len(unresolved_idx) else np.zeros((h_current.shape[1], 0))
            scores = {}
            for exp_key in ["A", "B", "C"]:
                score_i, _ = acquisition_score(h_blocks[exp_key], u_i)
                scores[exp_key] = score_i
            winner_i = max(scores, key=scores.get) if scores else "A"
            tau_ablation_rows.append(
                {
                    "tau_ratio": tau_ratio,
                    "unresolved_dim": int(u_i.shape[1]),
                    "winner": winner_i,
                }
            )
            print(
                f"  tau_ratio={tau_ratio:.2f}: unresolved_dim={u_i.shape[1]}, "
                f"winner={winner_i}, scores={scores}"
            )

        artifacts["figure_1_initial_curves"] = str(
            plot_initial_curves(output_dir, experiments["e0"], all_simulations["e0"])
        )
        artifacts["figure_2_acquisition_scores"] = str(
            plot_acquisition_scores(output_dir, ranking_rows)
        )
        artifacts["figure_3_update_metrics"] = str(
            plot_update_metrics(output_dir, base_metrics, update_metrics)
        )
        artifacts["figure_4_selected_experiment_curves"] = str(
            plot_selected_experiment_curves(output_dir, winner, experiments[winner], all_simulations[winner])
        )
        artifacts["figure_5_feature_space_summary"] = str(
            plot_feature_space_summary(output_dir, "e0", all_features["e0"])
        )
        artifacts["figure_6_failure_residuals"] = str(
            plot_failure_residuals(output_dir, winner, selected_residuals)
        )
        artifacts["latest_results_json"] = str(output_dir / "latest_results.json")
        artifacts["final_and_latest_results_md"] = str(output_dir / "final_and_latest_results.md")

        payload = {
            "latest_run": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dose": DOSE,
                "t_end": T_END,
                "tau_ratio": TAU_RATIO,
                "initial_experiment": "e0",
                "parameters": {
                    "A": params_a,
                    "B": params_b,
                    "C": params_c,
                    "truth": params_truth,
                },
                "initial_metrics": base_metrics,
                "initial_ambiguity": {
                    "bc_feature_distance": initial_bc_distance,
                    "bc_sampled_curve_distance": initial_bc_sampled_curve_distance,
                    "truth_residuals_e0": initial_truth_residuals,
                },
                "selected_experiment": winner,
                "disagreement_winner": disagreement_winner,
                "candidate_scores": [
                    {
                        "experiment": row[0],
                        "score": row[1],
                        "sigma_min": row[2],
                        "disagreement_mag": row[3],
                    }
                    for row in ranking_rows
                ],
                "tau_ablation": tau_ablation_rows,
                "update_metrics": update_metrics,
                "selected_residuals": selected_residuals,
                "artifacts": artifacts,
            }
        }
        write_results_json(output_dir, payload)
        write_results_markdown(output_dir, payload)

        print("\nSaved artifacts:")
        for name, artifact_path in artifacts.items():
            print(f"  {name}: {artifact_path}")


if __name__ == "__main__":
    main()
