"""
cascade_boed_benchmark.py — Structured high-dimensional benchmark for CARTOGRAPH.

This benchmark is the first implementation of the BOED bridge path:

- a shared nonlinear cascade ODE family with explicit mechanism coordinates,
- local linearization around a common reference point,
- CARTOGRAPH vs disagreement vs exact unresolved A-optimal design,
- and nonlinear-truth evaluation using posterior mean error and hidden-best regret.

The benchmark is intentionally controlled. It is not a new domain claim like PK.
Its purpose is to test whether the scaling advantage survives in a structured,
non-random scientific system with dimension d > 2.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / "outputs" / ".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pk_first_pass import TAU_RATIO, unresolved_subspace


OUTPUT_DIR = Path("outputs") / "cascade_boed"
NOISE_STD = 0.015
PRIOR_VAR = 0.50
RANDOM_SEQUENCE_SAMPLES = 200
SEED = 7


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    name: str
    amplitude: float
    duration: float
    times: np.ndarray
    observed_indices: tuple[int, ...]


@dataclass(frozen=True)
class TruthSpec:
    name: str
    support: tuple[int, ...]
    z_true: np.ndarray


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def time_grid(kind: str) -> np.ndarray:
    if kind == "early":
        return np.array([0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00], dtype=float)
    if kind == "mid":
        return np.array([0.25, 0.75, 1.50, 2.50, 4.00, 6.00, 8.00], dtype=float)
    if kind == "late":
        return np.array([1.00, 2.00, 4.00, 6.00, 8.00, 12.00, 16.00], dtype=float)
    if kind == "broad":
        return np.array([0.10, 0.25, 0.75, 1.50, 3.00, 6.00, 10.00, 16.00], dtype=float)
    raise ValueError(kind)


def observed_subset(kind: str, d: int) -> tuple[int, ...]:
    if d == 1:
        return (0,)
    if kind == "proximal":
        return tuple(sorted(set([0, min(1, d - 1)])))
    if kind == "mid":
        center = d // 2
        return tuple(sorted(set([max(0, center - 1), min(d - 1, center)])))
    if kind == "distal":
        return tuple(sorted(set([max(0, d - 2), d - 1])))
    if kind == "mixed":
        return tuple(sorted(set([0, d // 2, d - 1])))
    if kind == "full":
        return tuple(range(d))
    raise ValueError(kind)


def build_experiments(d: int) -> dict[str, ExperimentSpec]:
    return {
        "e0": ExperimentSpec("e0", "Weak Proximal Baseline", 0.75, 0.50, time_grid("early"), observed_subset("proximal", d)),
        "E1": ExperimentSpec("E1", "Low Early Proximal", 0.50, 0.50, time_grid("early"), observed_subset("proximal", d)),
        "E2": ExperimentSpec("E2", "High Early Proximal", 2.00, 0.50, time_grid("early"), observed_subset("proximal", d)),
        "E3": ExperimentSpec("E3", "Mid Window Mid-Cascade", 1.00, 1.50, time_grid("mid"), observed_subset("mid", d)),
        "E4": ExperimentSpec("E4", "High Mid Mixed", 2.50, 1.50, time_grid("mid"), observed_subset("mixed", d)),
        "E5": ExperimentSpec("E5", "Long Late Distal", 1.50, 4.00, time_grid("late"), observed_subset("distal", d)),
        "E6": ExperimentSpec("E6", "High Long Distal", 3.00, 4.00, time_grid("late"), observed_subset("distal", d)),
        "E7": ExperimentSpec("E7", "Broad Mixed", 1.00, 2.50, time_grid("broad"), observed_subset("mixed", d)),
        "E8": ExperimentSpec("E8", "Broad Full", 1.75, 2.50, time_grid("broad"), observed_subset("full", d)),
    }


FOLLOW_UP_KEYS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]


def cascade_params(d: int) -> dict[str, np.ndarray]:
    idx = np.arange(d, dtype=float)
    gamma = 0.25 + 0.02 * idx
    k_lin = 0.90 + 0.03 * idx
    v_max = 1.35 + 0.04 * idx
    k_m = 0.60 + 0.03 * idx
    return {
        "gamma": gamma,
        "k_lin": k_lin,
        "v_max": v_max,
        "k_m": k_m,
    }


def input_signal(t: float, amplitude: float, duration: float) -> float:
    return amplitude if t <= duration else 0.0


def transfer_fn(x_prev: float, z_i: float, k_lin: float, v_max: float, k_m: float) -> float:
    linear = k_lin * x_prev
    saturating = v_max * x_prev / (k_m + x_prev + 1e-8)
    return (1.0 - z_i) * linear + z_i * saturating


def cascade_rhs(
    t: float,
    x: np.ndarray,
    z: np.ndarray,
    params: dict[str, np.ndarray],
    experiment: ExperimentSpec,
) -> np.ndarray:
    d = len(z)
    dx = np.zeros(d, dtype=float)
    inp = input_signal(t, experiment.amplitude, experiment.duration)
    dx[0] = transfer_fn(inp, z[0], params["k_lin"][0], params["v_max"][0], params["k_m"][0]) - params["gamma"][0] * x[0]
    for i in range(1, d):
        incoming = transfer_fn(x[i - 1], z[i], params["k_lin"][i], params["v_max"][i], params["k_m"][i])
        dx[i] = incoming - params["gamma"][i] * x[i]
    return dx


def simulate_experiment(
    z: np.ndarray,
    params: dict[str, np.ndarray],
    experiment: ExperimentSpec,
    hidden_feedback: float = 0.0,
) -> np.ndarray:
    d = len(z)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        dx = cascade_rhs(t, x, z, params, experiment)
        if hidden_feedback != 0.0:
            dx[0] += hidden_feedback * x[-1] / (0.8 + x[-1] + 1e-8)
        return dx

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(experiment.times[-1])),
        y0=np.zeros(d, dtype=float),
        t_eval=experiment.times,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed for experiment {experiment.key}: {sol.message}")
    observed = sol.y[list(experiment.observed_indices), :]
    return observed.reshape(-1)


def finite_difference_jacobian(
    z_ref: np.ndarray,
    params: dict[str, np.ndarray],
    experiment: ExperimentSpec,
    step: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    y_ref = simulate_experiment(z_ref, params, experiment)
    d = len(z_ref)
    jac = np.zeros((len(y_ref), d), dtype=float)
    for i in range(d):
        z_plus = z_ref.copy()
        z_plus[i] = min(1.0, z_plus[i] + step)
        y_plus = simulate_experiment(z_plus, params, experiment)
        jac[:, i] = (y_plus - y_ref) / step
    return y_ref, jac


def truth_specs(d: int) -> list[TruthSpec]:
    specs = []

    def vec(active: list[tuple[int, float]], name: str) -> TruthSpec:
        z = np.zeros(d, dtype=float)
        for idx, value in active:
            if 0 <= idx < d:
                z[idx] = value
        support = tuple(sorted(idx for idx, value in active if 0 <= idx < d and abs(value) > 1e-9))
        return TruthSpec(name=name, support=support, z_true=z)

    specs.append(vec([(0, 0.90)], "stage_1_only"))
    specs.append(vec([(max(0, d // 2), 0.85)], "mid_stage_only"))
    specs.append(vec([(d - 1, 0.90)], "late_stage_only"))
    specs.append(vec([(0, 0.85), (d - 1, 0.85)], "early_late_pair"))
    specs.append(vec([(max(0, d // 3), 0.75), (min(d - 1, (2 * d) // 3), 0.90)], "spread_pair"))
    specs.append(vec([(0, 0.65), (max(0, d // 2), 0.65), (d - 1, 0.65)], "three_stage_mix"))
    return specs


def prior_precision(d: int, prior_var: float = PRIOR_VAR) -> np.ndarray:
    return np.eye(d, dtype=float) / prior_var


def posterior_from_observations(
    observed_keys: list[str],
    h_blocks: dict[str, np.ndarray],
    y_blocks: dict[str, np.ndarray],
    noise_std: float = NOISE_STD,
    prior_var: float = PRIOR_VAR,
) -> tuple[np.ndarray, np.ndarray]:
    d = next(iter(h_blocks.values())).shape[1]
    precision = prior_precision(d, prior_var)
    rhs = np.zeros(d, dtype=float)
    inv_noise = 1.0 / (noise_std ** 2)
    for key in observed_keys:
        H = h_blocks[key]
        y = y_blocks[key]
        precision += inv_noise * (H.T @ H)
        rhs += inv_noise * (H.T @ y)
    cov = np.linalg.inv(precision)
    mean = cov @ rhs
    return mean, cov


def current_unresolved_basis(observed_keys: list[str], h_blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    stacked = np.vstack([h_blocks[key] for key in observed_keys])
    U_tau, s_vals, tau = unresolved_subspace(stacked, tau_ratio=TAU_RATIO)
    if U_tau.shape[1] == 0:
        _, s_full, vt = np.linalg.svd(stacked, full_matrices=False)
        U_tau = vt[-1:].T
        return U_tau, s_full, tau
    return U_tau, s_vals, tau


def cartograph_score(H: np.ndarray, U_tau: np.ndarray) -> float:
    projected = H @ U_tau
    return float(np.linalg.norm(projected, ord="fro") ** 2)


def disagreement_score(H: np.ndarray) -> float:
    return float(np.linalg.norm(H, ord="fro") ** 2)


def boed_aopt_score(
    cov_current: np.ndarray,
    H: np.ndarray,
    U_tau: np.ndarray,
    noise_std: float = NOISE_STD,
) -> float:
    precision_current = np.linalg.inv(cov_current)
    precision_next = precision_current + (H.T @ H) / (noise_std ** 2)
    cov_next = np.linalg.inv(precision_next)
    current_trace = float(np.trace(U_tau.T @ cov_current @ U_tau))
    next_trace = float(np.trace(U_tau.T @ cov_next @ U_tau))
    return current_trace - next_trace


def build_sequences(
    h_blocks: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    remaining = FOLLOW_UP_KEYS.copy()
    observed_cart = ["e0"]
    observed_boed = ["e0"]

    cart_seq: list[str] = []
    boed_seq: list[str] = []

    while remaining:
        U_tau_cart, _, _ = current_unresolved_basis(observed_cart, h_blocks)
        cart_scores = [(cartograph_score(h_blocks[key], U_tau_cart), key) for key in remaining]
        cart_scores.sort(reverse=True)
        winner_cart = cart_scores[0][1]
        cart_seq.append(winner_cart)
        observed_cart.append(winner_cart)
        remaining.remove(winner_cart)

    remaining = FOLLOW_UP_KEYS.copy()
    while remaining:
        U_tau_boed, _, _ = current_unresolved_basis(observed_boed, h_blocks)
        _, cov_current = posterior_from_observations(observed_boed, h_blocks, {k: np.zeros(h_blocks[k].shape[0]) for k in observed_boed})
        scores = [(boed_aopt_score(cov_current, h_blocks[key], U_tau_boed), key) for key in remaining]
        scores.sort(reverse=True)
        winner_boed = scores[0][1]
        boed_seq.append(winner_boed)
        observed_boed.append(winner_boed)
        remaining.remove(winner_boed)

    disag_scores = [(disagreement_score(h_blocks[key]), key) for key in FOLLOW_UP_KEYS]
    disag_scores.sort(reverse=True)
    disag_seq = [key for _, key in disag_scores]

    return {
        "cartograph": cart_seq,
        "boed_aopt": boed_seq,
        "disagreement": disag_seq,
    }


def evaluate_sequence(
    sequence: list[str],
    truth: TruthSpec,
    h_blocks: dict[str, np.ndarray],
    y_truth: dict[str, np.ndarray],
    rounds: int = 3,
) -> dict[str, object]:
    observed = ["e0"]
    round_records = []
    for round_idx in range(0, rounds + 1):
        mean, cov = posterior_from_observations(observed, h_blocks, y_truth)
        mse = float(np.mean((mean - truth.z_true) ** 2))
        support_size = max(1, len(truth.support))
        top_support = tuple(sorted(np.argsort(np.abs(mean))[-support_size:].tolist()))
        support_match = tuple(sorted(truth.support)) == top_support
        round_records.append({
            "round": round_idx,
            "observed": observed.copy(),
            "mse": mse,
            "support_match": bool(support_match),
            "top_support": top_support,
            "posterior_trace": float(np.trace(cov)),
        })
        if round_idx == rounds:
            break
        observed.append(sequence[round_idx])
    return {
        "truth": truth.name,
        "final_mse": round_records[-1]["mse"],
        "final_support_match": round_records[-1]["support_match"],
        "round_records": round_records,
    }


def hidden_best_regret(
    truth: TruthSpec,
    h_blocks: dict[str, np.ndarray],
    y_truth: dict[str, np.ndarray],
) -> dict[str, object]:
    candidate_losses = {}
    for key in FOLLOW_UP_KEYS:
        observed = ["e0", key]
        mean, _ = posterior_from_observations(observed, h_blocks, y_truth)
        candidate_losses[key] = float(np.mean((mean - truth.z_true) ** 2))
    best_key = min(candidate_losses, key=candidate_losses.get)
    return {
        "best_key": best_key,
        "best_loss": candidate_losses[best_key],
        "candidate_losses": candidate_losses,
    }


def random_baseline_stats(
    truths: list[TruthSpec],
    h_blocks: dict[str, np.ndarray],
    truth_y: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    rounds: int = 3,
) -> dict[str, float]:
    losses = []
    support_matches = []
    for _ in range(RANDOM_SEQUENCE_SAMPLES):
        seq = FOLLOW_UP_KEYS.copy()
        rng.shuffle(seq)
        for truth in truths:
            result = evaluate_sequence(seq, truth, h_blocks, truth_y[truth.name], rounds=rounds)
            losses.append(result["final_mse"])
            support_matches.append(1.0 if result["final_support_match"] else 0.0)
    return {
        "expected_final_mse": float(np.mean(losses)),
        "expected_support_match_rate": float(np.mean(support_matches)),
    }


def run_dimension(
    d: int,
    rng: np.random.Generator,
    rounds: int = 3,
) -> dict[str, object]:
    params = cascade_params(d)
    experiments = build_experiments(d)
    z_ref = np.zeros(d, dtype=float)

    y_ref = {}
    h_blocks = {}
    for key, exp in experiments.items():
        y_base, H = finite_difference_jacobian(z_ref, params, exp)
        y_ref[key] = y_base
        h_blocks[key] = H

    sequences = build_sequences(h_blocks)
    truths = truth_specs(d)

    truth_y: dict[str, dict[str, np.ndarray]] = {}
    for truth in truths:
        y_blocks = {}
        for key, exp in experiments.items():
            nonlinear = simulate_experiment(truth.z_true, params, exp)
            noise = rng.normal(0.0, NOISE_STD, size=nonlinear.shape)
            y_blocks[key] = nonlinear - y_ref[key] + noise
        truth_y[truth.name] = y_blocks

    methods = {
        "cartograph": sequences["cartograph"],
        "boed_aopt": sequences["boed_aopt"],
        "disagreement": sequences["disagreement"],
    }

    method_results: dict[str, list[dict[str, object]]] = {name: [] for name in methods}
    hidden_best_records = []
    for truth in truths:
        hidden = hidden_best_regret(truth, h_blocks, truth_y[truth.name])
        hidden_best_records.append({"truth": truth.name, **hidden})
        for name, sequence in methods.items():
            method_results[name].append(evaluate_sequence(sequence, truth, h_blocks, truth_y[truth.name], rounds=rounds))

    random_stats = random_baseline_stats(truths, h_blocks, truth_y, rng, rounds=rounds)

    summary = {}
    for name, rows in method_results.items():
        final_mses = [row["final_mse"] for row in rows]
        support_rate = np.mean([1.0 if row["final_support_match"] else 0.0 for row in rows])
        regret = []
        hidden_match = 0
        for truth, row in zip(truths, rows):
            hidden = next(item for item in hidden_best_records if item["truth"] == truth.name)
            picked = methods[name][0]
            regret.append(row["round_records"][1]["mse"] - hidden["best_loss"])
            if picked == hidden["best_key"]:
                hidden_match += 1
        summary[name] = {
            "avg_final_mse": float(np.mean(final_mses)),
            "support_match_rate": float(support_rate),
            "avg_one_step_regret": float(np.mean(regret)),
            "hidden_best_match_rate": float(hidden_match / len(truths)),
            "sequence": methods[name],
        }

    initial_observed = ["e0"]
    U_tau, singular_values, tau = current_unresolved_basis(initial_observed, h_blocks)

    return {
        "dimension": d,
        "n_truths": len(truths),
        "initial_unresolved_dim": int(U_tau.shape[1]),
        "initial_singular_values": [float(x) for x in singular_values],
        "initial_tau": float(tau),
        "summary": summary,
        "random": random_stats,
        "hidden_best_records": hidden_best_records,
    }


def plot_performance(results: dict[int, dict[str, object]]) -> Path:
    dims = sorted(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    methods = [("cartograph", "#1f77b4", "CARTOGRAPH"), ("boed_aopt", "#2ca02c", "BOED A-opt"), ("disagreement", "#ff7f0e", "Disagreement")]

    for method_key, color, label in methods:
        axes[0].plot(dims, [results[d]["summary"][method_key]["avg_final_mse"] for d in dims], "o-", color=color, linewidth=2, label=label)
        axes[1].plot(dims, [results[d]["summary"][method_key]["hidden_best_match_rate"] for d in dims], "o-", color=color, linewidth=2, label=label)
        axes[2].plot(dims, [results[d]["summary"][method_key]["avg_one_step_regret"] for d in dims], "o-", color=color, linewidth=2, label=label)

    axes[0].plot(dims, [results[d]["random"]["expected_final_mse"] for d in dims], "s--", color="#7f7f7f", linewidth=1.5, label="Random")
    axes[1].plot(dims, [results[d]["random"]["expected_support_match_rate"] for d in dims], "s--", color="#7f7f7f", linewidth=1.5, label="Random")

    axes[0].set_title("Final posterior-mean MSE\n(after 3 rounds)")
    axes[0].set_xlabel("Mechanism dimension d")
    axes[0].set_ylabel("Average MSE")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Round-1 hidden-best match rate")
    axes[1].set_xlabel("Mechanism dimension d")
    axes[1].set_ylabel("Fraction of truths")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Round-1 regret vs hidden best")
    axes[2].set_xlabel("Mechanism dimension d")
    axes[2].set_ylabel("Average regret")
    axes[2].grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Structured Cascade Benchmark: CARTOGRAPH vs BOED vs Disagreement", fontsize=13, y=1.03)
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_1_performance_vs_dimension.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sequences(results: dict[int, dict[str, object]]) -> Path:
    dims = sorted(results.keys())
    fig, axes = plt.subplots(len(dims), 1, figsize=(11.5, 2.2 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]
    for ax, d in zip(axes, dims):
        summary = results[d]["summary"]
        y_positions = np.array([2, 1, 0], dtype=float)
        labels = ["CARTOGRAPH", "BOED A-opt", "Disagreement"]
        seqs = [
            summary["cartograph"]["sequence"][:4],
            summary["boed_aopt"]["sequence"][:4],
            summary["disagreement"]["sequence"][:4],
        ]
        for y, seq, color in zip(y_positions, seqs, ["#1f77b4", "#2ca02c", "#ff7f0e"]):
            for x, key in enumerate(seq):
                ax.scatter(x, y, s=120, color=color, alpha=0.85)
                ax.text(x, y, key, ha="center", va="center", fontsize=8, color="white")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_title(f"d={d} initial unresolved dim={results[d]['initial_unresolved_dim']}")
        ax.grid(axis="x", alpha=0.2)
    axes[-1].set_xticks([0, 1, 2, 3])
    axes[-1].set_xticklabels(["Round 1", "Round 2", "Round 3", "Round 4"])
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_2_sequence_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(results: dict[int, dict[str, object]], artifacts: dict[str, str]) -> Path:
    path = OUTPUT_DIR / "benchmark_summary.md"
    lines = [
        "# Structured Cascade Benchmark Summary",
        "",
        f"Run timestamp (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "This benchmark uses a shared nonlinear cascade ODE family with explicit",
        "mechanism coordinates, local Jacobian blocks from ODE sensitivities, and",
        "nonlinear truth evaluation through posterior-mean recovery.",
        "",
        "## Primary Table",
        "",
        "| d | Init unresolved dim | CART final MSE | BOED final MSE | Disagreement final MSE | Random E[MSE] | CART hidden-best | BOED hidden-best | Disagreement hidden-best |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(results):
        row = results[d]
        lines.append(
            f"| {d} | {row['initial_unresolved_dim']} | "
            f"{row['summary']['cartograph']['avg_final_mse']:.4f} | "
            f"{row['summary']['boed_aopt']['avg_final_mse']:.4f} | "
            f"{row['summary']['disagreement']['avg_final_mse']:.4f} | "
            f"{row['random']['expected_final_mse']:.4f} | "
            f"{row['summary']['cartograph']['hidden_best_match_rate']:.2f} | "
            f"{row['summary']['boed_aopt']['hidden_best_match_rate']:.2f} | "
            f"{row['summary']['disagreement']['hidden_best_match_rate']:.2f} |"
        )

    lines.extend([
        "",
        "## Method Sequences",
        "",
    ])
    for d in sorted(results):
        row = results[d]
        lines.append(f"### d={d}")
        lines.append(f"- Initial unresolved dimension: `{row['initial_unresolved_dim']}`")
        lines.append(f"- CARTOGRAPH: `{row['summary']['cartograph']['sequence']}`")
        lines.append(f"- BOED A-opt: `{row['summary']['boed_aopt']['sequence']}`")
        lines.append(f"- Disagreement: `{row['summary']['disagreement']['sequence']}`")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- `BOED A-opt` is the exact unresolved posterior-trace reduction baseline on the current unresolved basis.",
        "- `CARTOGRAPH` uses the unresolved projection score `||H_e U_tau||_F^2`.",
        "- `Disagreement` uses total sensitivity `||H_e||_F^2`.",
        "- The key question is whether CARTOGRAPH tracks BOED more closely than disagreement as `d` grows.",
        "",
        "## Artifact Paths",
        "",
    ])
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    ensure_output_dir()
    rng = np.random.default_rng(SEED)
    dimensions = [2, 4, 8, 16]

    print("=" * 72)
    print("Structured Cascade Benchmark — CARTOGRAPH vs BOED vs Disagreement")
    print("=" * 72)
    print(f"Dimensions: {dimensions}")
    print(f"Noise std: {NOISE_STD}")
    print(f"Prior variance: {PRIOR_VAR}")
    print()

    results: dict[int, dict[str, object]] = {}
    for d in dimensions:
        print(f"Running d={d}...", flush=True)
        results[d] = run_dimension(d, rng)
        summary = results[d]["summary"]
        print(
            f"  MSE: CART={summary['cartograph']['avg_final_mse']:.4f}, "
            f"BOED={summary['boed_aopt']['avg_final_mse']:.4f}, "
            f"DISAG={summary['disagreement']['avg_final_mse']:.4f}, "
            f"RAND={results[d]['random']['expected_final_mse']:.4f}",
            flush=True,
        )

    artifacts: dict[str, str] = {}
    artifacts["figure_1_performance_vs_dimension"] = str(plot_performance(results))
    artifacts["figure_2_sequence_comparison"] = str(plot_sequences(results))

    json_path = OUTPUT_DIR / "benchmark_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    artifacts["benchmark_results_json"] = str(json_path)

    artifacts["benchmark_summary_md"] = str(write_summary(results, artifacts))

    print("\nSaved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
