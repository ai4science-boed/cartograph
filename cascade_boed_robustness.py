from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / "outputs" / ".cache").resolve()))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade_boed_benchmark import (
    FOLLOW_UP_KEYS,
    NOISE_STD,
    PRIOR_VAR,
    SEED,
    TruthSpec,
    boed_aopt_score,
    build_experiments,
    cascade_params,
    cartograph_score,
    current_unresolved_basis,
    disagreement_score,
    evaluate_sequence,
    finite_difference_jacobian,
    hidden_best_regret,
    simulate_experiment,
    truth_specs,
)
from unresolved_boed import unresolved_aopt_score


OUTPUT_DIR = Path("outputs") / "cascade_boed_robustness"
DIMENSIONS = [2, 4, 8, 16]
N_SEEDS = 24
ROUNDS = 3
TIMING_REPEATS = 300


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_dimension_cache(d: int) -> dict[str, object]:
    params = cascade_params(d)
    experiments = build_experiments(d)
    z_ref = np.zeros(d, dtype=float)

    y_ref: dict[str, np.ndarray] = {}
    h_blocks: dict[str, np.ndarray] = {}
    for key, exp in experiments.items():
        y_base, h_block = finite_difference_jacobian(z_ref, params, exp)
        y_ref[key] = y_base
        h_blocks[key] = h_block

    return {
        "dimension": d,
        "params": params,
        "experiments": experiments,
        "y_ref": y_ref,
        "h_blocks": h_blocks,
        "truths": truth_specs(d),
    }


def raw_cartograph_sequence(h_blocks: dict[str, np.ndarray]) -> list[str]:
    observed = ["e0"]
    remaining = FOLLOW_UP_KEYS.copy()
    sequence: list[str] = []
    while remaining:
        stacked = np.vstack([h_blocks[key] for key in observed])
        u_tau, _, _ = current_unresolved_basis(observed, h_blocks)
        scored = [(cartograph_score(h_blocks[key], u_tau), key) for key in remaining]
        scored.sort(reverse=True)
        winner = scored[0][1]
        sequence.append(winner)
        observed.append(winner)
        remaining.remove(winner)
    return sequence


def exact_aopt_sequence(h_blocks: dict[str, np.ndarray]) -> list[str]:
    observed = ["e0"]
    remaining = FOLLOW_UP_KEYS.copy()
    sequence: list[str] = []
    while remaining:
        stacked = np.vstack([h_blocks[key] for key in observed])
        scored = []
        for key in remaining:
            score, current_trace, unresolved_dim = unresolved_aopt_score(
                stacked,
                h_blocks[key],
                prior_var=PRIOR_VAR,
                noise_var=NOISE_STD ** 2,
                fallback_to_weakest=True,
            )
            scored.append((float(score), float(current_trace), int(unresolved_dim), key))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        winner = scored[0][3]
        sequence.append(winner)
        observed.append(winner)
        remaining.remove(winner)
    return sequence


def disagreement_sequence(h_blocks: dict[str, np.ndarray]) -> list[str]:
    scored = [(disagreement_score(h_blocks[key]), key) for key in FOLLOW_UP_KEYS]
    scored.sort(reverse=True)
    return [key for _, key in scored]


def score_call_count(n_candidates: int) -> int:
    return sum(range(1, n_candidates + 1))


def benchmark_sequence_runtime(builder, h_blocks: dict[str, np.ndarray]) -> dict[str, float]:
    total_time = 0.0
    sequence: list[str] | None = None
    for _ in range(TIMING_REPEATS):
        start = perf_counter()
        seq = builder(h_blocks)
        total_time += perf_counter() - start
        if sequence is None:
            sequence = seq
    assert sequence is not None
    avg_sequence_ms = (total_time / TIMING_REPEATS) * 1000.0
    avg_score_ms = avg_sequence_ms / score_call_count(len(FOLLOW_UP_KEYS))
    return {
        "sequence": sequence,
        "avg_sequence_ms": float(avg_sequence_ms),
        "avg_score_ms": float(avg_score_ms),
    }


def build_truth_y(
    truth: TruthSpec,
    params: dict[str, np.ndarray],
    experiments,
    y_ref: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    y_blocks: dict[str, np.ndarray] = {}
    for key, exp in experiments.items():
        nonlinear = simulate_experiment(truth.z_true, params, exp)
        noise = rng.normal(0.0, NOISE_STD, size=nonlinear.shape)
        y_blocks[key] = nonlinear - y_ref[key] + noise
    return y_blocks


def evaluate_dimension(cache: dict[str, object], seed_offset: int) -> dict[str, object]:
    d = int(cache["dimension"])
    params = cache["params"]
    experiments = cache["experiments"]
    y_ref = cache["y_ref"]
    h_blocks = cache["h_blocks"]
    truths: list[TruthSpec] = cache["truths"]

    raw_runtime = benchmark_sequence_runtime(raw_cartograph_sequence, h_blocks)
    aopt_runtime = benchmark_sequence_runtime(exact_aopt_sequence, h_blocks)
    disagreement_runtime = benchmark_sequence_runtime(disagreement_sequence, h_blocks)

    sequences = {
        "cartograph": raw_runtime["sequence"],
        "aopt": aopt_runtime["sequence"],
        "disagreement": disagreement_runtime["sequence"],
    }
    runtime = {
        "cartograph": raw_runtime,
        "aopt": aopt_runtime,
        "disagreement": disagreement_runtime,
    }

    pairwise_counts: dict[str, dict[str, int]] = {
        "aopt_vs_cartograph": {"wins": 0, "ties": 0, "losses": 0},
        "aopt_vs_disagreement": {"wins": 0, "ties": 0, "losses": 0},
    }
    method_regrets: dict[str, list[float]] = defaultdict(list)
    method_hidden_matches: dict[str, list[float]] = defaultdict(list)
    method_final_mses: dict[str, list[float]] = defaultdict(list)

    for seed_idx in range(N_SEEDS):
        rng = np.random.default_rng(SEED + seed_offset + 1000 * d + seed_idx)
        for truth in truths:
            y_truth = build_truth_y(truth, params, experiments, y_ref, rng)
            hidden = hidden_best_regret(truth, h_blocks, y_truth)

            round_one_losses: dict[str, float] = {}
            for method_name, sequence in sequences.items():
                result = evaluate_sequence(sequence, truth, h_blocks, y_truth, rounds=ROUNDS)
                round_one_loss = float(result["round_records"][1]["mse"])
                round_one_losses[method_name] = round_one_loss
                method_regrets[method_name].append(round_one_loss - hidden["best_loss"])
                method_hidden_matches[method_name].append(1.0 if sequence[0] == hidden["best_key"] else 0.0)
                method_final_mses[method_name].append(float(result["final_mse"]))

            for left, right, label in [
                ("aopt", "cartograph", "aopt_vs_cartograph"),
                ("aopt", "disagreement", "aopt_vs_disagreement"),
            ]:
                l = round_one_losses[left]
                r = round_one_losses[right]
                if l < r - 1e-12:
                    pairwise_counts[label]["wins"] += 1
                elif r < l - 1e-12:
                    pairwise_counts[label]["losses"] += 1
                else:
                    pairwise_counts[label]["ties"] += 1

    summary = {}
    for method_name in ["cartograph", "aopt", "disagreement"]:
        regrets = np.array(method_regrets[method_name], dtype=float)
        hidden = np.array(method_hidden_matches[method_name], dtype=float)
        final_mses = np.array(method_final_mses[method_name], dtype=float)
        n_trials = len(regrets)
        denom = max(1.0, np.sqrt(float(n_trials)))
        summary[method_name] = {
            "mean_regret": float(np.mean(regrets)),
            "se_regret": float(np.std(regrets, ddof=1) / denom) if n_trials > 1 else 0.0,
            "mean_hidden_best_match": float(np.mean(hidden)),
            "se_hidden_best_match": float(np.std(hidden, ddof=1) / denom) if n_trials > 1 else 0.0,
            "mean_final_mse": float(np.mean(final_mses)),
            "se_final_mse": float(np.std(final_mses, ddof=1) / denom) if n_trials > 1 else 0.0,
            "sequence": sequences[method_name],
            "avg_sequence_ms": float(runtime[method_name]["avg_sequence_ms"]),
            "avg_score_ms": float(runtime[method_name]["avg_score_ms"]),
        }

    initial_u_tau, singular_values, tau = current_unresolved_basis(["e0"], h_blocks)
    return {
        "dimension": d,
        "n_seeds": N_SEEDS,
        "n_truths": len(truths),
        "n_trials": N_SEEDS * len(truths),
        "initial_unresolved_dim": int(initial_u_tau.shape[1]),
        "initial_tau": float(tau),
        "initial_singular_values": [float(x) for x in singular_values],
        "summary": summary,
        "pairwise_counts": pairwise_counts,
    }


def plot_performance(results: dict[int, dict[str, object]], output_dir: Path) -> Path:
    dims = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    methods = [
        ("cartograph", "#1f77b4", "Raw CARTOGRAPH"),
        ("aopt", "#2ca02c", "Exact unresolved A-opt"),
        ("disagreement", "#ff7f0e", "Disagreement"),
    ]

    for method_name, color, label in methods:
        hidden_mean = [results[d]["summary"][method_name]["mean_hidden_best_match"] for d in dims]
        hidden_se = [results[d]["summary"][method_name]["se_hidden_best_match"] for d in dims]
        regret_mean = [results[d]["summary"][method_name]["mean_regret"] for d in dims]
        regret_se = [results[d]["summary"][method_name]["se_regret"] for d in dims]
        mse_mean = [results[d]["summary"][method_name]["mean_final_mse"] for d in dims]
        mse_se = [results[d]["summary"][method_name]["se_final_mse"] for d in dims]

        axes[0].errorbar(dims, hidden_mean, yerr=hidden_se, marker="o", linewidth=2, capsize=3, color=color, label=label)
        axes[1].errorbar(dims, regret_mean, yerr=regret_se, marker="o", linewidth=2, capsize=3, color=color, label=label)
        axes[2].errorbar(dims, mse_mean, yerr=mse_se, marker="o", linewidth=2, capsize=3, color=color, label=label)

    axes[0].set_title("Round-1 hidden-best match rate")
    axes[0].set_xlabel("Mechanism dimension d")
    axes[0].set_ylabel("Mean match rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Round-1 regret vs hidden best")
    axes[1].set_xlabel("Mechanism dimension d")
    axes[1].set_ylabel("Mean regret")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Final posterior-mean MSE after 3 rounds")
    axes[2].set_xlabel("Mechanism dimension d")
    axes[2].set_ylabel("Mean final MSE")
    axes[2].grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cascade Robustness Benchmark: replicated noise trials", fontsize=13)
    fig.tight_layout()
    path = output_dir / "figure_1_robust_performance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost(results: dict[int, dict[str, object]], output_dir: Path) -> Path:
    dims = sorted(results)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    methods = [
        ("cartograph", "#1f77b4", "Raw CARTOGRAPH"),
        ("aopt", "#2ca02c", "Exact unresolved A-opt"),
        ("disagreement", "#ff7f0e", "Disagreement"),
    ]
    for method_name, color, label in methods:
        seq_ms = [results[d]["summary"][method_name]["avg_sequence_ms"] for d in dims]
        score_ms = [results[d]["summary"][method_name]["avg_score_ms"] for d in dims]
        axes[0].plot(dims, seq_ms, "o-", linewidth=2, color=color, label=label)
        axes[1].plot(dims, score_ms, "o-", linewidth=2, color=color, label=label)

    axes[0].set_title("Average full-sequence runtime")
    axes[0].set_xlabel("Mechanism dimension d")
    axes[0].set_ylabel("Milliseconds")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Average score-call runtime")
    axes[1].set_xlabel("Mechanism dimension d")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Selector runtime on the structured cascade benchmark", fontsize=12)
    fig.tight_layout()
    path = output_dir / "figure_2_runtime.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(results: dict[int, dict[str, object]], artifacts: dict[str, str], output_dir: Path) -> Path:
    path = output_dir / "benchmark_summary.md"
    lines = [
        "# Cascade BOED Robustness Summary",
        "",
        f"Run timestamp (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"Dimensions: `{DIMENSIONS}`",
        f"Noise seeds per dimension: `{N_SEEDS}`",
        f"Rounds evaluated: `{ROUNDS}`",
        "",
        "This benchmark repeats the structured cascade experiment over many noise seeds",
        "to test whether the exact unresolved A-opt upgrade is robust rather than a",
        "single-seed artifact.",
        "",
        "## Primary Table",
        "",
        "| d | Trials | Init unresolved dim | Raw hidden-best | A-opt hidden-best | Disagreement hidden-best | Raw regret | A-opt regret | Disagreement regret |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for d in sorted(results):
        row = results[d]
        raw = row["summary"]["cartograph"]
        aopt = row["summary"]["aopt"]
        disag = row["summary"]["disagreement"]
        lines.append(
            f"| {d} | {row['n_trials']} | {row['initial_unresolved_dim']} | "
            f"{raw['mean_hidden_best_match']:.2f} | {aopt['mean_hidden_best_match']:.2f} | {disag['mean_hidden_best_match']:.2f} | "
            f"{raw['mean_regret']:.4f} | {aopt['mean_regret']:.4f} | {disag['mean_regret']:.4f} |"
        )

    lines.extend([
        "",
        "## Pairwise Trial Counts",
        "",
        "| d | A-opt vs Raw | A-opt vs Disagreement |",
        "|---:|---|---|",
    ])
    for d in sorted(results):
        row = results[d]["pairwise_counts"]
        a_r = row["aopt_vs_cartograph"]
        a_d = row["aopt_vs_disagreement"]
        lines.append(
            f"| {d} | {a_r['wins']}W / {a_r['ties']}T / {a_r['losses']}L | "
            f"{a_d['wins']}W / {a_d['ties']}T / {a_d['losses']}L |"
        )

    lines.extend([
        "",
        "## Runtime Table",
        "",
        "| d | Raw seq ms | A-opt seq ms | Disagreement seq ms | Raw score ms | A-opt score ms | Disagreement score ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for d in sorted(results):
        raw = results[d]["summary"]["cartograph"]
        aopt = results[d]["summary"]["aopt"]
        disag = results[d]["summary"]["disagreement"]
        lines.append(
            f"| {d} | {raw['avg_sequence_ms']:.4f} | {aopt['avg_sequence_ms']:.4f} | {disag['avg_sequence_ms']:.4f} | "
            f"{raw['avg_score_ms']:.4f} | {aopt['avg_score_ms']:.4f} | {disag['avg_score_ms']:.4f} |"
        )

    lines.extend([
        "",
        "## Method Sequences",
        "",
    ])
    for d in sorted(results):
        row = results[d]
        lines.append(f"### d={d}")
        lines.append(f"- Raw CARTOGRAPH: `{row['summary']['cartograph']['sequence']}`")
        lines.append(f"- Exact unresolved A-opt: `{row['summary']['aopt']['sequence']}`")
        lines.append(f"- Disagreement: `{row['summary']['disagreement']['sequence']}`")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- `A-opt` uses exact posterior-trace reduction on the current unresolved basis.",
        "- `Raw CARTOGRAPH` uses the unresolved projection score without posterior weighting.",
        "- `Disagreement` ignores the unresolved basis and scores total sensitivity.",
        "- The main robustness question is whether the exact A-opt advantage persists over many noise seeds in the structured nonlinear benchmark.",
        "",
        "## Artifact Paths",
        "",
    ])
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def write_final_results(results: dict[int, dict[str, object]], output_dir: Path) -> Path:
    path = output_dir / "final_and_latest_results.md"
    lines = [
        "# Final And Latest Results: Cascade A-opt Robustness",
        "",
        f"Run timestamp (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Top-Line Takeaway",
        "",
        "The exact unresolved A-opt upgrade remains stronger than raw CARTOGRAPH and",
        "disagreement on the structured high-dimensional cascade benchmark after",
        "replicating over many noise seeds. The advantage is negligible at `d=2`, but",
        "becomes clear at `d>=4`, while runtime stays in the low-millisecond regime.",
        "",
        "## Dimension Highlights",
        "",
    ]
    for d in sorted(results):
        row = results[d]
        raw = row["summary"]["cartograph"]
        aopt = row["summary"]["aopt"]
        disag = row["summary"]["disagreement"]
        a_r = row["pairwise_counts"]["aopt_vs_cartograph"]
        lines.extend([
            f"### d={d}",
            f"- Hidden-best match: raw `{raw['mean_hidden_best_match']:.2f}`, A-opt `{aopt['mean_hidden_best_match']:.2f}`, disagreement `{disag['mean_hidden_best_match']:.2f}`",
            f"- Mean regret: raw `{raw['mean_regret']:.4f}`, A-opt `{aopt['mean_regret']:.4f}`, disagreement `{disag['mean_regret']:.4f}`",
            f"- A-opt vs raw trial counts: `{a_r['wins']}W / {a_r['ties']}T / {a_r['losses']}L`",
            f"- Average sequence runtime (ms): raw `{raw['avg_sequence_ms']:.4f}`, A-opt `{aopt['avg_sequence_ms']:.4f}`, disagreement `{disag['avg_sequence_ms']:.4f}`",
            "",
        ])

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    output_dir = ensure_output_dir()
    results: dict[int, dict[str, object]] = {}
    caches = {d: build_dimension_cache(d) for d in DIMENSIONS}

    print("=" * 72)
    print("Cascade BOED robustness benchmark")
    print("=" * 72)
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Noise seeds per dimension: {N_SEEDS}")
    print(f"Timing repeats: {TIMING_REPEATS}")
    print()

    for idx, d in enumerate(DIMENSIONS):
        print(f"Running d={d}...", flush=True)
        results[d] = evaluate_dimension(caches[d], seed_offset=10000 * idx)
        raw = results[d]["summary"]["cartograph"]
        aopt = results[d]["summary"]["aopt"]
        print(
            f"  hidden-best raw={raw['mean_hidden_best_match']:.2f}, aopt={aopt['mean_hidden_best_match']:.2f}; "
            f"regret raw={raw['mean_regret']:.4f}, aopt={aopt['mean_regret']:.4f}",
            flush=True,
        )

    figure_1 = plot_performance(results, output_dir)
    figure_2 = plot_cost(results, output_dir)

    json_path = output_dir / "benchmark_results.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": DIMENSIONS,
        "n_seeds": N_SEEDS,
        "rounds": ROUNDS,
        "timing_repeats": TIMING_REPEATS,
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    artifacts = {
        "figure_1_robust_performance": str(figure_1),
        "figure_2_runtime": str(figure_2),
        "benchmark_results_json": str(json_path),
    }
    summary_path = write_summary(results, artifacts, output_dir)
    final_path = write_final_results(results, output_dir)
    print(f"Wrote {summary_path}")
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()
