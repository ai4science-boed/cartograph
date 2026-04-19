from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / "outputs" / ".cache").resolve()))

import numpy as np

from pk_divergence_benchmark import (
    FOLLOW_UP_KEYS,
    IDENTIFICATION_MARGIN,
    RANDOM_UNRESOLVED_ROUND,
    cartograph_sequence,
    disagreement_sequence,
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
from unresolved_boed import (
    current_unresolved_basis,
    unresolved_information_matrix,
    unresolved_posterior_covariance,
    unresolved_aopt_score,
)


OUTPUT_DIR = Path("outputs") / "pk_boed_baselines"
PRIOR_VAR = 1.0
NOISE_VAR = 1.0
FALLBACK_TO_WEAKEST = False


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def eig_score(h_current: np.ndarray, h_candidate: np.ndarray) -> tuple[float, float, int]:
    u_tau, _, _ = current_unresolved_basis(
        h_current,
        fallback_to_weakest=FALLBACK_TO_WEAKEST,
    )
    k = u_tau.shape[1]
    if k == 0:
        return 0.0, 0.0, 0
    lambda_cur = unresolved_posterior_covariance(
        h_current,
        u_tau,
        prior_var=PRIOR_VAR,
        noise_var=NOISE_VAR,
    )
    g_e = unresolved_information_matrix(h_candidate, u_tau, noise_var=NOISE_VAR)
    eye = np.eye(g_e.shape[0], dtype=float)
    sign, logdet = np.linalg.slogdet(eye + lambda_cur @ g_e)
    if sign <= 0:
        return 0.0, float(np.trace(lambda_cur)), k
    return float(0.5 * logdet), float(np.trace(lambda_cur)), k


def eig_sequence(h_blocks: dict[str, np.ndarray], follow_up_keys: list[str]) -> list[str]:
    observed = ["e0"]
    remaining = list(follow_up_keys)
    sequence: list[str] = []
    while remaining:
        h_current = np.vstack([h_blocks[key] for key in observed])
        scored = []
        for candidate in remaining:
            score, current_trace, unresolved_dim = eig_score(h_current, h_blocks[candidate])
            scored.append((score, current_trace, unresolved_dim, candidate))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        winner = scored[0][3]
        sequence.append(winner)
        observed.append(winner)
        remaining.remove(winner)
    return sequence


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


def boxhill_score(model_features: dict[str, np.ndarray], noise_var: float = 1.0) -> float:
    names = ["A", "B", "C"]
    total = 0.0
    count = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = model_features[names[i]] - model_features[names[j]]
            total += 0.5 * float(np.dot(diff, diff)) / max(noise_var, 1e-12)
            count += 1
    return total / max(count, 1)


def boxhill_sequence(library_features: dict[str, dict[str, np.ndarray]], follow_up_keys: list[str]) -> list[str]:
    scored = []
    for candidate in follow_up_keys:
        score = boxhill_score(library_features[candidate], noise_var=NOISE_VAR)
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored]


def pairwise_wtl(rows: list[dict[str, object]], left: str, right: str) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for row in rows:
        lv = RANDOM_UNRESOLVED_ROUND if row[left] is None else row[left]
        rv = RANDOM_UNRESOLVED_ROUND if row[right] is None else row[right]
        if lv < rv:
            wins += 1
        elif lv > rv:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def one_sided_sign_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) * (0.5 ** n))


def write_summary(output_dir: Path, payload: dict[str, object]) -> Path:
    path = output_dir / "baseline_summary.md"
    lines = [
        "# PK BOED Baselines Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        f"Identification margin: `{payload['identification_margin']:.2f}`",
        "",
        "## Method Sequences",
        "",
    ]
    for key, seq in payload["method_sequences"].items():
        lines.append(f"- **{key}**: `{seq}`")

    lines.extend([
        "",
        "## Primary Table",
        "",
        "| Truth | Oracle | CART | A-opt | EIG | Box-Hill | Disagreement | Random E[round] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['truth']} | {row['oracle']} | {format_round(row['cartograph'])} | "
            f"{format_round(row['aopt'])} | {format_round(row['eig'])} | "
            f"{format_round(row['boxhill'])} | {format_round(row['disagreement'])} | "
            f"{row['random_expected']:.2f} |"
        )

    lines.extend(["", "## Pairwise Results", ""])
    for label, stats in payload["pairwise"].items():
        lines.append(
            f"- **{label}**: `{stats['wins']}W / {stats['ties']}T / {stats['losses']}L`, "
            f"one-sided sign test p=`{stats['p_value']:.4g}`"
        )

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
        "cartograph": cartograph_sequence(h_blocks, FOLLOW_UP_KEYS),
        "aopt": aopt_sequence(h_blocks, FOLLOW_UP_KEYS),
        "eig": eig_sequence(h_blocks, FOLLOW_UP_KEYS),
        "boxhill": boxhill_sequence(library_features, FOLLOW_UP_KEYS),
        "disagreement": disagreement_sequence(library_features, FOLLOW_UP_KEYS),
    }
    all_random = random_sequences(FOLLOW_UP_KEYS)

    summary_rows: list[dict[str, object]] = []
    for truth_spec in truths:
        truth_features = precompute_truth_features(truth_spec, experiments)
        oracle_model, _ = oracle_library_model(truth_features, library_features, ["e0"] + FOLLOW_UP_KEYS)
        row = {
            "truth": truth_spec.name,
            "family": truth_spec.family,
            "oracle": oracle_model,
        }
        for method_name, sequence in method_sequences.items():
            round_value, _ = round_to_identification(
                sequence,
                truth_features,
                library_features,
                oracle_model,
                IDENTIFICATION_MARGIN,
            )
            row[method_name] = round_value
        random_rounds = []
        for seq in all_random:
            rnd, _ = round_to_identification(
                seq,
                truth_features,
                library_features,
                oracle_model,
                IDENTIFICATION_MARGIN,
            )
            random_rounds.append(rnd)
        row["random_expected"] = mean_random_round(random_rounds)
        summary_rows.append(row)

    pairwise = {}
    for left, right, label in [
        ("cartograph", "disagreement", "CART vs Disagreement"),
        ("aopt", "disagreement", "A-opt vs Disagreement"),
        ("eig", "disagreement", "EIG vs Disagreement"),
        ("boxhill", "disagreement", "Box-Hill vs Disagreement"),
        ("aopt", "cartograph", "A-opt vs CART"),
        ("eig", "cartograph", "EIG vs CART"),
    ]:
        wins, ties, losses = pairwise_wtl(summary_rows, left, right)
        pairwise[label] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "p_value": one_sided_sign_pvalue(wins, losses),
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identification_margin": IDENTIFICATION_MARGIN,
        "method_sequences": method_sequences,
        "summary_rows": summary_rows,
        "pairwise": pairwise,
    }
    json_path = output_dir / "baseline_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    summary_path = write_summary(output_dir, payload)
    print(f"Saved {json_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
