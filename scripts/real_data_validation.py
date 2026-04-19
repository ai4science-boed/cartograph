from __future__ import annotations

import itertools
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / "outputs" / ".cache").resolve()))

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pk_first_pass import (
    TAU_RATIO,
    Experiment,
    acquisition_score,
    model_a_rhs,
    model_b_rhs,
    model_c_rhs,
    simulate_model,
    unresolved_subspace,
)


DATASET_PATH = Path("data") / "cvtdb_v2_0_0_no_audit.sqlite"
OUTPUT_DIR = Path("outputs") / "real_data"
EPS = 1e-8
IDENTIFICATION_MARGIN = 2.0  # BIC gap
N_INITIAL_POINTS = 6
N_CANDIDATE_BLOCKS = 3


@dataclass(frozen=True)
class SeriesSpec:
    series_id: int
    label: str


@dataclass(frozen=True)
class FittedModel:
    name: str
    params: dict[str, float]
    bic: float
    sse: float
    pred_log_conc: np.ndarray


SERIES_SPECS = [
    SeriesSpec(122, "1,2-Dichloroethane oral"),
    SeriesSpec(120, "Dichloromethane oral"),
    SeriesSpec(65743, "Chloroform oral"),
]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["XDG_CACHE_HOME"])).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DATASET_PATH)


def load_series(conn: sqlite3.Connection, series_id: int) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        """
        SELECT time_hr, conc
        FROM conc_time_values
        WHERE fk_series_id = ?
          AND time_hr IS NOT NULL
          AND conc IS NOT NULL
        ORDER BY time_hr
        """,
        (series_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"No data found for series {series_id}")

    times_raw: list[float] = []
    conc_raw: list[float] = []
    for time_hr, conc in rows:
        try:
            t = float(time_hr)
            c = max(float(conc), 0.0)
        except (TypeError, ValueError):
            continue
        if math.isnan(t) or math.isnan(c):
            continue
        times_raw.append(t)
        conc_raw.append(c)

    if not times_raw:
        raise ValueError(f"No numeric data found for series {series_id}")

    grouped: dict[float, list[float]] = {}
    for t, c in zip(times_raw, conc_raw):
        grouped.setdefault(float(t), []).append(float(c))
    times = np.array(sorted(grouped.keys()), dtype=float)
    conc = np.array([np.mean(grouped[t]) for t in times], dtype=float)
    return times, conc


def build_block_plan(times: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(times) < N_INITIAL_POINTS + N_CANDIDATE_BLOCKS:
        raise ValueError(f"Need at least {N_INITIAL_POINTS + N_CANDIDATE_BLOCKS} time points, got {len(times)}")

    initial_idx = np.unique(np.round(np.linspace(0, len(times) - 1, N_INITIAL_POINTS)).astype(int))
    remaining = np.array([idx for idx in range(len(times)) if idx not in set(initial_idx)], dtype=int)
    blocks = np.array_split(remaining, N_CANDIDATE_BLOCKS)
    candidate_blocks = {f"E{i+1}": block for i, block in enumerate(blocks) if len(block) > 0}
    return initial_idx, candidate_blocks


def simulate_oral(model_name: str, params: dict[str, float], times: np.ndarray) -> np.ndarray:
    rhs_lookup: dict[str, Callable[[float, np.ndarray, dict[str, float]], np.ndarray]] = {
        "A": model_a_rhs,
        "B": model_b_rhs,
        "C": model_c_rhs,
    }
    result = simulate_model(rhs_lookup[model_name], params, Experiment("obs", "oral", times), model_name)
    return np.maximum(result.concentration, EPS)


def parameter_spec(model_name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    if model_name == "A":
        names = ["k_a", "k_e", "V"]
        lower = np.array([0.01, 0.005, 0.05], dtype=float)
        upper = np.array([20.0, 10.0, 1e5], dtype=float)
    elif model_name == "B":
        names = ["k_tr", "k_a", "k_e", "V"]
        lower = np.array([0.01, 0.01, 0.005, 0.05], dtype=float)
        upper = np.array([20.0, 20.0, 10.0, 1e5], dtype=float)
    elif model_name == "C":
        names = ["k_a", "k_10", "k_12", "k_21", "V_c"]
        lower = np.array([0.01, 0.001, 0.001, 0.001, 0.05], dtype=float)
        upper = np.array([20.0, 10.0, 10.0, 10.0, 1e5], dtype=float)
    else:
        raise ValueError(model_name)
    return names, lower, upper


def initial_guesses(model_name: str, y_obs: np.ndarray) -> list[np.ndarray]:
    vmax = max(100.0 / max(np.max(y_obs), 1e-3), 0.1)
    if model_name == "A":
        return [
            np.array([1.5, 0.5, vmax], dtype=float),
            np.array([0.5, 0.2, vmax * 2.0], dtype=float),
        ]
    if model_name == "B":
        return [
            np.array([0.8, 1.5, 0.5, vmax], dtype=float),
            np.array([1.5, 0.8, 0.2, vmax * 2.0], dtype=float),
        ]
    return [
        np.array([1.5, 0.3, 0.3, 0.3, vmax], dtype=float),
        np.array([0.8, 0.1, 0.5, 0.2, vmax * 2.0], dtype=float),
    ]


def fit_model(model_name: str, times: np.ndarray, conc: np.ndarray) -> FittedModel:
    param_names, lower, upper = parameter_spec(model_name)
    y_target = np.log(np.maximum(conc, EPS))
    best_cost = float("inf")
    best_params = None
    best_pred = None

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = dict(zip(param_names, theta.tolist()))
        pred = np.log(simulate_oral(model_name, params, times))
        return pred - y_target

    for guess in initial_guesses(model_name, conc):
        guess = np.clip(guess, lower * 1.01, upper * 0.99)
        try:
            result = least_squares(
                residuals,
                guess,
                bounds=(lower, upper),
                max_nfev=150,
            )
        except Exception:
            continue
        if not result.success:
            continue
        cost = float(np.sum(result.fun ** 2))
        if cost < best_cost:
            best_cost = cost
            best_params = result.x.copy()
            best_pred = np.exp(y_target + result.fun)

    if best_params is None:
        raise RuntimeError(f"Failed to fit model {model_name}")

    n = len(times)
    p = len(param_names)
    sse = max(best_cost, EPS)
    bic = n * math.log(sse / n + EPS) + p * math.log(max(n, 2))
    pred_log = np.log(simulate_oral(model_name, dict(zip(param_names, best_params.tolist())), times))
    return FittedModel(
        name=model_name,
        params=dict(zip(param_names, best_params.tolist())),
        bic=float(bic),
        sse=float(sse),
        pred_log_conc=pred_log,
    )


def fit_library(times: np.ndarray, conc: np.ndarray) -> dict[str, FittedModel]:
    return {name: fit_model(name, times, conc) for name in ["A", "B", "C"]}


def oracle_model(full_fits: dict[str, FittedModel]) -> tuple[str, dict[str, float]]:
    bics = {name: fit.bic for name, fit in full_fits.items()}
    best = min(bics, key=bics.get)
    return best, bics


def identification_state(fits: dict[str, FittedModel], oracle: str) -> dict[str, object]:
    ordered = sorted(((name, fit.bic) for name, fit in fits.items()), key=lambda item: item[1])
    best_name, best_bic = ordered[0]
    second_bic = ordered[1][1]
    gap = second_bic - best_bic
    return {
        "best_model": best_name,
        "best_bic": float(best_bic),
        "gap": float(gap),
        "identified": bool(best_name == oracle and gap >= IDENTIFICATION_MARGIN),
        "bic_by_model": {name: float(fit.bic) for name, fit in fits.items()},
    }


def make_h_block_from_predictions(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    pred_c: np.ndarray,
) -> np.ndarray:
    return np.column_stack([pred_b - pred_a, pred_c - pred_a])


def make_h_block(fits: dict[str, FittedModel]) -> np.ndarray:
    return make_h_block_from_predictions(
        fits["A"].pred_log_conc,
        fits["B"].pred_log_conc,
        fits["C"].pred_log_conc,
    )


def predict_h_block(fits: dict[str, FittedModel], times: np.ndarray) -> np.ndarray:
    pred_a = np.log(simulate_oral("A", fits["A"].params, times))
    pred_b = np.log(simulate_oral("B", fits["B"].params, times))
    pred_c = np.log(simulate_oral("C", fits["C"].params, times))
    return make_h_block_from_predictions(pred_a, pred_b, pred_c)


def fit_for_subset(
    all_times: np.ndarray,
    all_conc: np.ndarray,
    idx: np.ndarray,
    cache: dict[tuple[int, ...], dict[str, FittedModel]],
) -> dict[str, FittedModel]:
    key = tuple(sorted(int(v) for v in idx.tolist()))
    if key not in cache:
        times = all_times[list(key)]
        conc = all_conc[list(key)]
        cache[key] = fit_library(times, conc)
    return cache[key]


def cartograph_sequence(
    all_times: np.ndarray,
    all_conc: np.ndarray,
    initial_idx: np.ndarray,
    candidate_blocks: dict[str, np.ndarray],
) -> list[str]:
    fit_cache: dict[tuple[int, ...], dict[str, FittedModel]] = {}
    observed_idx = initial_idx.copy()
    remaining = list(candidate_blocks.keys())
    sequence: list[str] = []

    while remaining:
        current_fits = fit_for_subset(all_times, all_conc, observed_idx, fit_cache)
        h_current = make_h_block(current_fits)
        u_tau, _, _ = unresolved_subspace(h_current, tau_ratio=TAU_RATIO)
        scored: list[tuple[float, float, str]] = []
        for key in remaining:
            block_idx = candidate_blocks[key]
            h_e = predict_h_block(current_fits, all_times[block_idx])
            score, sigma_local = acquisition_score(h_e, u_tau)
            scored.append((float(score), float(sigma_local), key))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        winner = scored[0][2]
        sequence.append(winner)
        observed_idx = np.unique(np.concatenate([observed_idx, candidate_blocks[winner]]))
        remaining.remove(winner)
    return sequence


def disagreement_sequence(
    all_times: np.ndarray,
    all_conc: np.ndarray,
    initial_idx: np.ndarray,
    candidate_blocks: dict[str, np.ndarray],
) -> list[str]:
    fit_cache: dict[tuple[int, ...], dict[str, FittedModel]] = {}
    current_fits = fit_for_subset(all_times, all_conc, initial_idx, fit_cache)
    scored: list[tuple[float, str]] = []
    for key, block_idx in candidate_blocks.items():
        h_e = predict_h_block(current_fits, all_times[block_idx])
        score = float(np.linalg.norm(h_e, ord="fro") ** 2)
        scored.append((score, key))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [key for _, key in scored]


def random_sequences(candidate_keys: list[str]) -> list[list[str]]:
    return [list(perm) for perm in itertools.permutations(candidate_keys)]


def round_to_identification(
    sequence: list[str],
    all_times: np.ndarray,
    all_conc: np.ndarray,
    initial_idx: np.ndarray,
    candidate_blocks: dict[str, np.ndarray],
    oracle: str,
) -> tuple[int | None, list[dict[str, object]]]:
    history: list[dict[str, object]] = []
    fit_cache: dict[tuple[int, ...], dict[str, FittedModel]] = {}
    observed_idx = initial_idx.copy()
    current_fits = fit_for_subset(all_times, all_conc, observed_idx, fit_cache)
    state = identification_state(current_fits, oracle)
    state["round"] = 0
    state["observed_points"] = int(len(observed_idx))
    history.append(state)
    if state["identified"]:
        return 0, history

    for round_idx, key in enumerate(sequence, start=1):
        observed_idx = np.unique(np.concatenate([observed_idx, candidate_blocks[key]]))
        current_fits = fit_for_subset(all_times, all_conc, observed_idx, fit_cache)
        state = identification_state(current_fits, oracle)
        state["round"] = round_idx
        state["observed_points"] = int(len(observed_idx))
        state["block"] = key
        history.append(state)
        if state["identified"]:
            return round_idx, history
    return None, history


def format_round(value: int | None) -> str:
    return "NR" if value is None else str(value)


def mean_random_round(values: list[int | None], unresolved_round: int) -> float:
    mapped = [unresolved_round if v is None else v for v in values]
    return float(np.mean(mapped))


def plot_rounds(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    methods = ["cartograph", "disagreement", "random_expected"]
    matrix = np.array(
        [[float(row[m]) if isinstance(row[m], float) else (len(methods) + 2 if row[m] is None else row[m]) for m in methods] for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(["CARTOGRAPH", "Disagreement", "Random E[round]"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["label"] for row in rows], fontsize=8)
    for i, row in enumerate(rows):
        for j, method in enumerate(methods):
            value = row[method]
            label = f"{value:.2f}" if isinstance(value, float) else format_round(value)
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#102030")
    ax.set_title("Real-Data Retrospective: Rounds To Full-Data Oracle")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / "figure_1_rounds_to_identification.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_example_series(output_dir: Path, series_payload: dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    times = np.array(series_payload["times"], dtype=float)
    conc = np.array(series_payload["conc"], dtype=float)
    e0_idx = np.array(series_payload["initial_idx"], dtype=int)
    candidate_blocks = {key: np.array(val, dtype=int) for key, val in series_payload["candidate_blocks"].items()}

    ax.plot(times, conc, color="#111111", linewidth=1.8, label="Full real curve")
    ax.scatter(times[e0_idx], conc[e0_idx], s=36, color="#1f77b4", label="Initial sparse points", zorder=4)
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for color, (key, idx) in zip(colors, candidate_blocks.items()):
        ax.scatter(times[idx], conc[idx], s=26, color=color, alpha=0.8, label=key)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration")
    ax.set_title(f"Real CvTdb Series: {series_payload['label']}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "figure_2_example_series_blocks.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(output_dir: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    json_path = output_dir / "real_data_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    md_path = output_dir / "real_data_summary.md"
    lines = [
        "# Real-Data Retrospective Validation Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        f"Dataset: `{payload['dataset_path']}`",
        f"Identification margin (BIC gap): `{payload['identification_margin']:.2f}`",
        "",
        "## Summary Table",
        "",
        "| Series | Oracle | CARTOGRAPH | Disagreement | Random E[round] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['label']} | {row['oracle']} | {format_round(row['cartograph'])} | "
            f"{format_round(row['disagreement'])} | {row['random_expected']:.2f} |"
        )

    wins = ties = losses = 0
    unresolved_round = payload["unresolved_round"]
    for row in payload["summary_rows"]:
        c = unresolved_round if row["cartograph"] is None else row["cartograph"]
        d = unresolved_round if row["disagreement"] is None else row["disagreement"]
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
        "## Method Sequences",
        "",
    ])
    for detail in payload["series_details"]:
        lines.append(f"### {detail['label']}")
        lines.append(f"- Oracle full-data model: `{detail['oracle']}`")
        lines.append(f"- CARTOGRAPH sequence: `{detail['cartograph_sequence']}`")
        lines.append(f"- Disagreement sequence: `{detail['disagreement_sequence']}`")
        lines.append(f"- Initial sparse points: `{detail['initial_times']}`")
        lines.append("")

    lines.extend([
        "## Artifact Paths",
        "",
    ])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return json_path, md_path


def main() -> None:
    output_dir = ensure_output_dir()
    conn = get_connection()

    summary_rows: list[dict[str, object]] = []
    series_details: list[dict[str, object]] = []
    example_payload: dict[str, object] | None = None

    for spec in SERIES_SPECS:
        print(f"Running {spec.label}...", flush=True)
        times, conc = load_series(conn, spec.series_id)
        initial_idx, candidate_blocks = build_block_plan(times)

        full_fits = fit_library(times, conc)
        oracle, full_bics = oracle_model(full_fits)

        cart_seq = cartograph_sequence(times, conc, initial_idx, candidate_blocks)
        disag_seq = disagreement_sequence(times, conc, initial_idx, candidate_blocks)
        random_rounds = []
        for seq in random_sequences(list(candidate_blocks.keys())):
            rnd, _ = round_to_identification(seq, times, conc, initial_idx, candidate_blocks, oracle)
            random_rounds.append(rnd)

        unresolved_round = len(candidate_blocks) + 1
        cart_round, cart_hist = round_to_identification(cart_seq, times, conc, initial_idx, candidate_blocks, oracle)
        disag_round, disag_hist = round_to_identification(disag_seq, times, conc, initial_idx, candidate_blocks, oracle)
        rand_expected = mean_random_round(random_rounds, unresolved_round)

        summary_rows.append({
            "label": spec.label,
            "oracle": oracle,
            "cartograph": cart_round,
            "disagreement": disag_round,
            "random_expected": rand_expected,
        })

        detail = {
            "series_id": spec.series_id,
            "label": spec.label,
            "oracle": oracle,
            "full_bics": full_bics,
            "initial_idx": initial_idx.tolist(),
            "initial_times": times[initial_idx].tolist(),
            "candidate_blocks": {key: block.tolist() for key, block in candidate_blocks.items()},
            "cartograph_sequence": cart_seq,
            "disagreement_sequence": disag_seq,
            "cartograph_history": cart_hist,
            "disagreement_history": disag_hist,
            "times": times.tolist(),
            "conc": conc.tolist(),
        }
        series_details.append(detail)
        if example_payload is None:
            example_payload = detail

        print(
            f"{spec.label}: oracle={oracle}, CART={format_round(cart_round)}, "
            f"Disagreement={format_round(disag_round)}, Random={rand_expected:.2f}"
        )

    conn.close()

    artifacts = {
        "figure_1_rounds_to_identification": str(plot_rounds(output_dir, summary_rows)),
        "real_data_results_json": str(output_dir / "real_data_results.json"),
        "real_data_summary_md": str(output_dir / "real_data_summary.md"),
    }
    if example_payload is not None:
        artifacts["figure_2_example_series_blocks"] = str(plot_example_series(output_dir, example_payload))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "identification_margin": IDENTIFICATION_MARGIN,
        "unresolved_round": len(candidate_blocks) + 1 if series_details else 0,
        "summary_rows": summary_rows,
        "series_details": series_details,
        "artifacts": artifacts,
    }
    write_outputs(output_dir, payload)

    print("\nSaved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
