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
from real_data_validation import (
    DATASET_PATH,
    IDENTIFICATION_MARGIN,
    SeriesSpec,
    build_block_plan,
    ensure_output_dir,
    fit_for_subset,
    get_connection,
    identification_state,
    load_series,
    make_h_block,
    oracle_model,
    predict_h_block,
)


ONE_STEP_OUTPUT_DIR = Path("outputs") / "real_data_one_step"
SERIES_SPECS = [
    SeriesSpec(122, "1,2-Dichloroethane oral"),
    SeriesSpec(120, "Dichloromethane oral"),
    SeriesSpec(123, "Trichloroethylene oral"),
    SeriesSpec(65744, "Chloroform oral"),
]


def ensure_one_step_output_dir() -> Path:
    ensure_output_dir()
    ONE_STEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return ONE_STEP_OUTPUT_DIR


def oracle_margin(fits: dict, oracle: str) -> float:
    oracle_bic = fits[oracle].bic
    other_best = min(fit.bic for name, fit in fits.items() if name != oracle)
    return float(other_best - oracle_bic)


def evaluate_series(series_id: int, label: str) -> dict[str, object]:
    conn = get_connection()
    times, conc = load_series(conn, series_id)
    conn.close()

    initial_idx, candidate_blocks = build_block_plan(times)
    fit_cache: dict[tuple[int, ...], dict] = {}

    full_fits = fit_for_subset(times, conc, np.arange(len(times), dtype=int), fit_cache)
    oracle, full_bics = oracle_model(full_fits)

    initial_fits = fit_for_subset(times, conc, initial_idx, fit_cache)
    initial_state = identification_state(initial_fits, oracle)
    initial_margin = oracle_margin(initial_fits, oracle)

    h_current = make_h_block(initial_fits)
    u_tau, _, _ = unresolved_subspace(h_current, tau_ratio=TAU_RATIO)

    candidate_results: list[dict[str, object]] = []
    for key, block_idx in candidate_blocks.items():
        h_e = predict_h_block(initial_fits, times[block_idx])
        cart_score, sigma_local = acquisition_score(h_e, u_tau)
        disag_score = float(np.linalg.norm(h_e, ord="fro") ** 2)

        merged_idx = np.unique(np.concatenate([initial_idx, block_idx]))
        after_fits = fit_for_subset(times, conc, merged_idx, fit_cache)
        after_state = identification_state(after_fits, oracle)
        after_margin = oracle_margin(after_fits, oracle)

        candidate_results.append({
            "block": key,
            "times": times[block_idx].tolist(),
            "cartograph_score": float(cart_score),
            "sigma_local": float(sigma_local),
            "disagreement_score": disag_score,
            "oracle_margin": after_margin,
            "identified": bool(after_state["identified"]),
            "best_model": after_state["best_model"],
            "best_bic": float(after_state["best_bic"]),
            "gap": float(after_state["gap"]),
        })

    cart_pick = max(candidate_results, key=lambda row: (row["cartograph_score"], row["sigma_local"]))
    disag_pick = max(candidate_results, key=lambda row: row["disagreement_score"])
    oracle_best = max(candidate_results, key=lambda row: row["oracle_margin"])
    random_expected_margin = float(np.mean([row["oracle_margin"] for row in candidate_results]))
    random_identification_rate = float(np.mean([1.0 if row["identified"] else 0.0 for row in candidate_results]))

    return {
        "series_id": series_id,
        "label": label,
        "oracle": oracle,
        "full_bics": full_bics,
        "times": times.tolist(),
        "conc": conc.tolist(),
        "initial_idx": initial_idx.tolist(),
        "initial_times": times[initial_idx].tolist(),
        "initial_best_model": initial_state["best_model"],
        "initial_gap": float(initial_state["gap"]),
        "initial_oracle_margin": initial_margin,
        "candidate_results": candidate_results,
        "cartograph_pick": cart_pick,
        "disagreement_pick": disag_pick,
        "oracle_best_block": oracle_best["block"],
        "oracle_best_margin": float(oracle_best["oracle_margin"]),
        "random_expected_oracle_margin": random_expected_margin,
        "random_identification_rate": random_identification_rate,
    }


def plot_oracle_margins(output_dir: Path, results: list[dict[str, object]]) -> Path:
    labels = [row["label"] for row in results]
    initial = [row["initial_oracle_margin"] for row in results]
    cart = [row["cartograph_pick"]["oracle_margin"] for row in results]
    disag = [row["disagreement_pick"]["oracle_margin"] for row in results]
    rand = [row["random_expected_oracle_margin"] for row in results]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 1.5 * width, initial, width=width, color="#7f7f7f", label="Initial sparse")
    ax.bar(x - 0.5 * width, cart, width=width, color="#1f77b4", label="CARTOGRAPH")
    ax.bar(x + 0.5 * width, disag, width=width, color="#ff7f0e", label="Disagreement")
    ax.bar(x + 1.5 * width, rand, width=width, color="#2ca02c", label="Random E[margin]")
    ax.axhline(IDENTIFICATION_MARGIN, color="#cc3333", linestyle="--", linewidth=1.5, label=f"ID margin={IDENTIFICATION_MARGIN}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Oracle-model BIC margin after one step")
    ax.set_title("Real CvTdb One-Step Retrospective")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = output_dir / "figure_1_oracle_margins.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(output_dir: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    json_path = output_dir / "one_step_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    md_path = output_dir / "one_step_summary.md"
    lines = [
        "# Real-Data One-Step Retrospective Summary",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        f"Dataset: `{payload['dataset_path']}`",
        f"Identification margin (BIC gap): `{payload['identification_margin']:.2f}`",
        "",
        "## Summary Table",
        "",
        "| Series | Oracle | Initial Margin | CART Pick | CART Margin | Disagreement Pick | Disagreement Margin | Hidden Best Block | Hidden Best Margin | Random E[margin] |",
        "|---|---|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for row in payload["series_results"]:
        lines.append(
            f"| {row['label']} | {row['oracle']} | {row['initial_oracle_margin']:.3f} | "
            f"{row['cartograph_pick']['block']} | {row['cartograph_pick']['oracle_margin']:.3f} | "
            f"{row['disagreement_pick']['block']} | {row['disagreement_pick']['oracle_margin']:.3f} | "
            f"{row['oracle_best_block']} | {row['oracle_best_margin']:.3f} | "
            f"{row['random_expected_oracle_margin']:.3f} |"
        )

    wins = ties = losses = 0
    for row in payload["series_results"]:
        c = row["cartograph_pick"]["oracle_margin"]
        d = row["disagreement_pick"]["oracle_margin"]
        if c > d + 1e-9:
            wins += 1
        elif d > c + 1e-9:
            losses += 1
        else:
            ties += 1

    lines.extend([
        "",
        f"**CARTOGRAPH vs Disagreement (one-step oracle margin)**: {wins}W / {ties}T / {losses}L",
        "",
    ])

    if payload.get("skipped_series"):
        lines.extend([
            "## Skipped Series",
            "",
        ])
        for row in payload["skipped_series"]:
            lines.append(f"- `{row['label']}` (series `{row['series_id']}`): `{row['error']}`")
        lines.append("")

    lines.extend([
        "## Per-Series Candidate Details",
        "",
    ])

    for row in payload["series_results"]:
        lines.append(f"### {row['label']}")
        lines.append(f"- Oracle full-data model: `{row['oracle']}`")
        lines.append(f"- Initial sparse times: `{row['initial_times']}`")
        lines.append(f"- Initial oracle margin: `{row['initial_oracle_margin']:.3f}`")
        lines.append(f"- Hidden best one-step block: `{row['oracle_best_block']}` with oracle margin `{row['oracle_best_margin']:.3f}`")
        lines.append("")
        lines.append("| Block | Times | CART Score | Disagreement Score | Oracle Margin | Identified? | Best Model |")
        lines.append("|---|---|---:|---:|---:|---|---|")
        for cand in row["candidate_results"]:
            lines.append(
                f"| {cand['block']} | `{cand['times']}` | {cand['cartograph_score']:.3f} | "
                f"{cand['disagreement_score']:.3f} | {cand['oracle_margin']:.3f} | "
                f"{'yes' if cand['identified'] else 'no'} | {cand['best_model']} |"
            )
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
    output_dir = ensure_one_step_output_dir()
    series_results = []
    skipped_series = []
    for spec in SERIES_SPECS:
        print(f"Running one-step retrospective for {spec.label}...", flush=True)
        try:
            result = evaluate_series(spec.series_id, spec.label)
        except Exception as exc:
            skipped_series.append({
                "series_id": spec.series_id,
                "label": spec.label,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  Skipping after fit/evaluation failure: {type(exc).__name__}: {exc}", flush=True)
            continue
        series_results.append(result)
        print(
            f"  CART {result['cartograph_pick']['block']} -> {result['cartograph_pick']['oracle_margin']:.3f}; "
            f"Disagreement {result['disagreement_pick']['block']} -> {result['disagreement_pick']['oracle_margin']:.3f}",
            flush=True,
        )

    if not series_results:
        raise RuntimeError("No real-data series completed successfully.")

    artifacts = {
        "figure_1_oracle_margins": str(plot_oracle_margins(output_dir, series_results)),
        "one_step_results_json": str(output_dir / "one_step_results.json"),
        "one_step_summary_md": str(output_dir / "one_step_summary.md"),
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "identification_margin": IDENTIFICATION_MARGIN,
        "series_results": series_results,
        "skipped_series": skipped_series,
        "artifacts": artifacts,
    }
    write_outputs(output_dir, payload)
    print("\nSaved artifacts:", flush=True)
    for key, value in artifacts.items():
        print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
