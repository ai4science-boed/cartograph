from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
    EPS,
    build_block_plan,
    ensure_output_dir,
    fit_for_subset,
    get_connection,
    identification_state,
    load_series,
    make_h_block,
    oracle_model,
    predict_h_block,
    simulate_oral,
)


OUTPUT_DIR = Path("outputs") / "real_data_topt"
MIN_POINTS = 10
MAX_TIME_HOURS = 24.0
MAX_SUCCESSFUL_SERIES = 18
QUERY_POOL_SIZE = 320


@dataclass(frozen=True)
class CandidateSeries:
    series_id: int
    label: str
    n_points: int
    max_time: float


def ensure_benchmark_output_dir() -> Path:
    ensure_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def fetch_candidate_series(pool_size: int = QUERY_POOL_SIZE) -> list[CandidateSeries]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            s.id AS series_id,
            c.preferred_name AS preferred_name,
            COUNT(v.time_hr) AS n_points,
            MAX(v.time_hr) AS max_time
        FROM series s
        JOIN studies st ON s.fk_study_id = st.id
        JOIN administration_route_dict r ON st.fk_administration_route_id = r.id
        JOIN chemicals c ON st.fk_dosed_chemical_id = c.id
        JOIN conc_time_values v ON v.fk_series_id = s.id
        WHERE r.administration_route_normalized = 'oral'
          AND c.preferred_name IS NOT NULL
          AND v.time_hr IS NOT NULL
          AND v.conc IS NOT NULL
        GROUP BY s.id, c.preferred_name
        HAVING COUNT(v.time_hr) >= ?
           AND MAX(v.time_hr) <= ?
        ORDER BY n_points DESC, max_time DESC, s.id ASC
        LIMIT ?
        """,
        (MIN_POINTS, MAX_TIME_HOURS, pool_size),
    ).fetchall()
    conn.close()

    specs: list[CandidateSeries] = []
    for series_id, preferred_name, n_points, max_time in rows:
        label = f"{preferred_name} oral"
        specs.append(
            CandidateSeries(
                series_id=int(series_id),
                label=label,
                n_points=int(n_points),
                max_time=float(max_time),
            )
        )
    return specs


def oracle_margin(fits: dict[str, object], oracle: str) -> float:
    oracle_bic = fits[oracle].bic
    other_best = min(fit.bic for name, fit in fits.items() if name != oracle)
    return float(other_best - oracle_bic)


def local_topt_score(initial_fits: dict[str, object], times: np.ndarray) -> float:
    ordered = sorted(initial_fits.values(), key=lambda fit: fit.bic)
    best_fit = ordered[0]
    rival_fit = ordered[1]
    pred_best = np.log(np.maximum(simulate_oral(best_fit.name, best_fit.params, times), EPS))
    pred_rival = np.log(np.maximum(simulate_oral(rival_fit.name, rival_fit.params, times), EPS))
    sigma2 = max(best_fit.sse / max(len(best_fit.pred_log_conc), 1), EPS)
    return float(np.sum((pred_best - pred_rival) ** 2) / sigma2)


def evaluate_series(spec: CandidateSeries) -> dict[str, object]:
    conn = get_connection()
    times, conc = load_series(conn, spec.series_id)
    conn.close()

    initial_idx, candidate_blocks = build_block_plan(times)
    fit_cache: dict[tuple[int, ...], dict[str, object]] = {}

    full_idx = np.arange(len(times), dtype=int)
    full_fits = fit_for_subset(times, conc, full_idx, fit_cache)
    oracle, full_bics = oracle_model(full_fits)

    initial_fits = fit_for_subset(times, conc, initial_idx, fit_cache)
    initial_state = identification_state(initial_fits, oracle)
    initial_margin = oracle_margin(initial_fits, oracle)

    h_current = make_h_block(initial_fits)
    u_tau, singular_values, tau = unresolved_subspace(h_current, tau_ratio=TAU_RATIO)

    candidate_results: list[dict[str, object]] = []
    for block_name, block_idx in candidate_blocks.items():
        h_e = predict_h_block(initial_fits, times[block_idx])
        cart_score, sigma_local = acquisition_score(h_e, u_tau)
        disagreement_score = float(np.linalg.norm(h_e, ord="fro") ** 2)
        topt_score = local_topt_score(initial_fits, times[block_idx])

        merged_idx = np.unique(np.concatenate([initial_idx, block_idx]))
        after_fits = fit_for_subset(times, conc, merged_idx, fit_cache)
        after_state = identification_state(after_fits, oracle)
        after_margin = oracle_margin(after_fits, oracle)

        candidate_results.append(
            {
                "block": block_name,
                "times": times[block_idx].tolist(),
                "block_size": int(len(block_idx)),
                "cartograph_score": float(cart_score),
                "sigma_local": float(sigma_local),
                "disagreement_score": disagreement_score,
                "topt_score": topt_score,
                "oracle_margin": after_margin,
                "identified": bool(after_state["identified"]),
                "best_model": after_state["best_model"],
                "gap": float(after_state["gap"]),
            }
        )

    cart_pick = max(candidate_results, key=lambda row: (row["cartograph_score"], row["sigma_local"]))
    disagreement_pick = max(candidate_results, key=lambda row: row["disagreement_score"])
    topt_pick = max(candidate_results, key=lambda row: row["topt_score"])
    oracle_best = max(candidate_results, key=lambda row: row["oracle_margin"])
    random_expected_margin = float(np.mean([row["oracle_margin"] for row in candidate_results]))

    return {
        "series_id": spec.series_id,
        "label": spec.label,
        "n_points": spec.n_points,
        "max_time": spec.max_time,
        "oracle": oracle,
        "full_bics": full_bics,
        "initial_idx": initial_idx.tolist(),
        "initial_times": times[initial_idx].tolist(),
        "initial_best_model": initial_state["best_model"],
        "initial_gap": float(initial_state["gap"]),
        "initial_oracle_margin": initial_margin,
        "unresolved_dim": int(u_tau.shape[1]),
        "tau": float(tau),
        "singular_values": singular_values.tolist(),
        "candidate_results": candidate_results,
        "cartograph_pick": cart_pick,
        "disagreement_pick": disagreement_pick,
        "topt_pick": topt_pick,
        "oracle_best_block": oracle_best["block"],
        "oracle_best_margin": float(oracle_best["oracle_margin"]),
        "random_expected_oracle_margin": random_expected_margin,
    }


def wins_ties_losses(rows: list[dict[str, object]], left_key: str, right_key: str) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for row in rows:
        left = float(row[left_key]["oracle_margin"])
        right = float(row[right_key]["oracle_margin"])
        if left > right + 1e-9:
            wins += 1
        elif right > left + 1e-9:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def hit_rate(rows: list[dict[str, object]], pick_key: str) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row[pick_key]["block"] == row["oracle_best_block"])
    return float(hits / len(rows))


def mean_margin(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    if key == "initial_oracle_margin":
        return float(np.mean([row[key] for row in rows]))
    if key == "random_expected_oracle_margin":
        return float(np.mean([row[key] for row in rows]))
    return float(np.mean([row[key]["oracle_margin"] for row in rows]))


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    active_rows = [row for row in rows if row["unresolved_dim"] > 0]
    return {
        "n_series": len(rows),
        "n_active": len(active_rows),
        "n_degenerate": len(rows) - len(active_rows),
        "mean_initial_margin": mean_margin(rows, "initial_oracle_margin"),
        "mean_cartograph_margin": mean_margin(rows, "cartograph_pick"),
        "mean_disagreement_margin": mean_margin(rows, "disagreement_pick"),
        "mean_topt_margin": mean_margin(rows, "topt_pick"),
        "mean_random_margin": mean_margin(rows, "random_expected_oracle_margin"),
        "cartograph_hit_rate": hit_rate(rows, "cartograph_pick"),
        "disagreement_hit_rate": hit_rate(rows, "disagreement_pick"),
        "topt_hit_rate": hit_rate(rows, "topt_pick"),
        "cart_vs_disagreement": wins_ties_losses(rows, "cartograph_pick", "disagreement_pick"),
        "cart_vs_topt": wins_ties_losses(rows, "cartograph_pick", "topt_pick"),
        "topt_vs_disagreement": wins_ties_losses(rows, "topt_pick", "disagreement_pick"),
        "active_mean_initial_margin": mean_margin(active_rows, "initial_oracle_margin"),
        "active_mean_cartograph_margin": mean_margin(active_rows, "cartograph_pick"),
        "active_mean_disagreement_margin": mean_margin(active_rows, "disagreement_pick"),
        "active_mean_topt_margin": mean_margin(active_rows, "topt_pick"),
        "active_mean_random_margin": mean_margin(active_rows, "random_expected_oracle_margin"),
        "active_cartograph_hit_rate": hit_rate(active_rows, "cartograph_pick"),
        "active_disagreement_hit_rate": hit_rate(active_rows, "disagreement_pick"),
        "active_topt_hit_rate": hit_rate(active_rows, "topt_pick"),
        "active_cart_vs_disagreement": wins_ties_losses(active_rows, "cartograph_pick", "disagreement_pick"),
        "active_cart_vs_topt": wins_ties_losses(active_rows, "cartograph_pick", "topt_pick"),
        "active_topt_vs_disagreement": wins_ties_losses(active_rows, "topt_pick", "disagreement_pick"),
    }


def plot_method_summary(output_dir: Path, summary: dict[str, object]) -> Path:
    methods = ["Initial", "CART", "Disagree", "T-opt", "Random"]
    full_margins = [
        summary["mean_initial_margin"],
        summary["mean_cartograph_margin"],
        summary["mean_disagreement_margin"],
        summary["mean_topt_margin"],
        summary["mean_random_margin"],
    ]
    active_margins = [
        summary["active_mean_initial_margin"],
        summary["active_mean_cartograph_margin"],
        summary["active_mean_disagreement_margin"],
        summary["active_mean_topt_margin"],
        summary["active_mean_random_margin"],
    ]
    full_hits = [
        0.0,
        summary["cartograph_hit_rate"],
        summary["disagreement_hit_rate"],
        summary["topt_hit_rate"],
        0.0,
    ]
    active_hits = [
        0.0,
        summary["active_cartograph_hit_rate"],
        summary["active_disagreement_hit_rate"],
        summary["active_topt_hit_rate"],
        0.0,
    ]

    x = np.arange(len(methods))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    axes[0].bar(x - width / 2, full_margins, width=width, color="#4c78a8", label="All series")
    axes[0].bar(x + width / 2, active_margins, width=width, color="#f58518", label="Active subset")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15)
    axes[0].set_ylabel("Mean oracle-model BIC margin")
    axes[0].set_title("One-step margin gain")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - width / 2, full_hits, width=width, color="#54a24b", label="All series")
    axes[1].bar(x + width / 2, active_hits, width=width, color="#e45756", label="Active subset")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15)
    axes[1].set_ylabel("Hidden-best block match rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Oracle-best block hit rate")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("EPA CvTdb One-Step Retrospective With Local T-opt Baseline")
    fig.tight_layout()
    path = output_dir / "figure_1_method_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    payload: dict[str, object],
) -> tuple[Path, Path, Path]:
    json_path = output_dir / "benchmark_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = payload["summary"]
    md_path = output_dir / "benchmark_summary.md"
    lines = [
        "# EPA Real-Data Retrospective With Local T-opt Baseline",
        "",
        f"Run timestamp (UTC): `{payload['timestamp_utc']}`",
        f"Dataset: `{payload['dataset_path']}`",
        f"Cohort rule: oral series, at least `{payload['min_points']}` points, max time <= `{payload['max_time_hours']}` h, one series per chemical",
        f"Successful series: `{summary['n_series']}`",
        f"Degenerate unresolved-space series: `{summary['n_degenerate']}`",
        f"Nondegenerate active subset: `{summary['n_active']}`",
        f"Identification margin (BIC gap): `{payload['identification_margin']:.2f}`",
        "",
        "## Cohort-Level Summary",
        "",
        "| Split | Initial Margin | CART Margin | Disagreement Margin | T-opt Margin | Random E[margin] | CART Hit | Disagreement Hit | T-opt Hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| All series | {summary['mean_initial_margin']:.3f} | {summary['mean_cartograph_margin']:.3f} | {summary['mean_disagreement_margin']:.3f} | {summary['mean_topt_margin']:.3f} | {summary['mean_random_margin']:.3f} | {summary['cartograph_hit_rate']:.2%} | {summary['disagreement_hit_rate']:.2%} | {summary['topt_hit_rate']:.2%} |",
        f"| Active subset | {summary['active_mean_initial_margin']:.3f} | {summary['active_mean_cartograph_margin']:.3f} | {summary['active_mean_disagreement_margin']:.3f} | {summary['active_mean_topt_margin']:.3f} | {summary['active_mean_random_margin']:.3f} | {summary['active_cartograph_hit_rate']:.2%} | {summary['active_disagreement_hit_rate']:.2%} | {summary['active_topt_hit_rate']:.2%} |",
        "",
        "## Pairwise One-Step Oracle-Margin Results",
        "",
        f"- All series, CARTOGRAPH vs Disagreement: `{summary['cart_vs_disagreement'][0]}W / {summary['cart_vs_disagreement'][1]}T / {summary['cart_vs_disagreement'][2]}L`",
        f"- All series, CARTOGRAPH vs T-opt: `{summary['cart_vs_topt'][0]}W / {summary['cart_vs_topt'][1]}T / {summary['cart_vs_topt'][2]}L`",
        f"- All series, T-opt vs Disagreement: `{summary['topt_vs_disagreement'][0]}W / {summary['topt_vs_disagreement'][1]}T / {summary['topt_vs_disagreement'][2]}L`",
        f"- Active subset, CARTOGRAPH vs Disagreement: `{summary['active_cart_vs_disagreement'][0]}W / {summary['active_cart_vs_disagreement'][1]}T / {summary['active_cart_vs_disagreement'][2]}L`",
        f"- Active subset, CARTOGRAPH vs T-opt: `{summary['active_cart_vs_topt'][0]}W / {summary['active_cart_vs_topt'][1]}T / {summary['active_cart_vs_topt'][2]}L`",
        f"- Active subset, T-opt vs Disagreement: `{summary['active_topt_vs_disagreement'][0]}W / {summary['active_topt_vs_disagreement'][1]}T / {summary['active_topt_vs_disagreement'][2]}L`",
        "",
        "## Working Vs Skipped",
        "",
        f"- Successful evaluations: `{summary['n_series']}`",
        f"- Skipped after fit/evaluation failure: `{len(payload['skipped_series'])}`",
        "",
    ]

    if payload["skipped_series"]:
        lines.extend(["### Skipped Series", ""])
        for row in payload["skipped_series"]:
            lines.append(f"- `{row['label']}` (series `{row['series_id']}`): `{row['error']}`")
        lines.append("")

    lines.extend([
        "## Per-Series Results",
        "",
        "| Series | Oracle | Initial Margin | Unresolved Dim | CART Pick | CART Margin | Disagreement Pick | Disagreement Margin | T-opt Pick | T-opt Margin | Hidden Best |",
        "|---|---|---:|---:|---|---:|---|---:|---|---:|---|",
    ])
    for row in payload["series_results"]:
        lines.append(
            f"| {row['label']} | {row['oracle']} | {row['initial_oracle_margin']:.3f} | {row['unresolved_dim']} | "
            f"{row['cartograph_pick']['block']} | {row['cartograph_pick']['oracle_margin']:.3f} | "
            f"{row['disagreement_pick']['block']} | {row['disagreement_pick']['oracle_margin']:.3f} | "
            f"{row['topt_pick']['block']} | {row['topt_pick']['oracle_margin']:.3f} | "
            f"{row['oracle_best_block']} |"
        )

    lines.extend(["", "## Artifact Paths", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    final_path = output_dir / "final_and_latest_results.md"
    final_lines = [
        "# Final And Latest Results",
        "",
        f"- Cohort: `{summary['n_series']}` real EPA oral series (plus `{len(payload['skipped_series'])}` skipped failures), one series per chemical, max time <= `{payload['max_time_hours']}` h.",
        f"- Degenerate unresolved-space cases: `{summary['n_degenerate']}` / `{summary['n_series']}`.",
        f"- All-series mean oracle margin: CART `{summary['mean_cartograph_margin']:.3f}`, Disagreement `{summary['mean_disagreement_margin']:.3f}`, T-opt `{summary['mean_topt_margin']:.3f}`, Random `{summary['mean_random_margin']:.3f}`.",
        f"- All-series pairwise: CART vs Disagreement `{summary['cart_vs_disagreement'][0]}W / {summary['cart_vs_disagreement'][1]}T / {summary['cart_vs_disagreement'][2]}L`, CART vs T-opt `{summary['cart_vs_topt'][0]}W / {summary['cart_vs_topt'][1]}T / {summary['cart_vs_topt'][2]}L`, T-opt vs Disagreement `{summary['topt_vs_disagreement'][0]}W / {summary['topt_vs_disagreement'][1]}T / {summary['topt_vs_disagreement'][2]}L`.",
        f"- Active-subset pairwise: CART vs Disagreement `{summary['active_cart_vs_disagreement'][0]}W / {summary['active_cart_vs_disagreement'][1]}T / {summary['active_cart_vs_disagreement'][2]}L`, CART vs T-opt `{summary['active_cart_vs_topt'][0]}W / {summary['active_cart_vs_topt'][1]}T / {summary['active_cart_vs_topt'][2]}L`, T-opt vs Disagreement `{summary['active_topt_vs_disagreement'][0]}W / {summary['active_topt_vs_disagreement'][1]}T / {summary['active_topt_vs_disagreement'][2]}L`.",
        f"- Hidden-best match rates: CART `{summary['cartograph_hit_rate']:.2%}`, Disagreement `{summary['disagreement_hit_rate']:.2%}`, T-opt `{summary['topt_hit_rate']:.2%}`.",
    ]
    with final_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(final_lines))

    return json_path, md_path, final_path


def main() -> None:
    output_dir = ensure_benchmark_output_dir()
    candidate_specs = fetch_candidate_series()

    series_results: list[dict[str, object]] = []
    skipped_series: list[dict[str, object]] = []
    successful_labels: set[str] = set()

    for spec in candidate_specs:
        if spec.label in successful_labels:
            continue
        print(f"Running EPA one-step baseline benchmark for {spec.label} (series {spec.series_id})...", flush=True)
        try:
            row = evaluate_series(spec)
        except Exception as exc:
            skipped_series.append(
                {
                    "series_id": spec.series_id,
                    "label": spec.label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  Skipping after failure: {type(exc).__name__}: {exc}", flush=True)
            continue

        series_results.append(row)
        successful_labels.add(spec.label)
        print(
            "  "
            f"CART {row['cartograph_pick']['block']} -> {row['cartograph_pick']['oracle_margin']:.3f}; "
            f"Disagree {row['disagreement_pick']['block']} -> {row['disagreement_pick']['oracle_margin']:.3f}; "
            f"T-opt {row['topt_pick']['block']} -> {row['topt_pick']['oracle_margin']:.3f}",
            flush=True,
        )
        if len(series_results) >= MAX_SUCCESSFUL_SERIES:
            break

    summary = build_summary(series_results)
    figure_path = plot_method_summary(output_dir, summary)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH.resolve()),
        "identification_margin": IDENTIFICATION_MARGIN,
        "min_points": MIN_POINTS,
        "max_time_hours": MAX_TIME_HOURS,
        "target_successful_series": MAX_SUCCESSFUL_SERIES,
        "candidate_pool": [spec.__dict__ for spec in candidate_specs],
        "series_results": series_results,
        "skipped_series": skipped_series,
        "summary": summary,
        "artifacts": {
            "figure_1_method_summary": str(figure_path.resolve()),
        },
    }

    json_path, md_path, final_path = write_outputs(output_dir, payload)
    payload["artifacts"].update(
        {
            "benchmark_results_json": str(json_path.resolve()),
            "benchmark_summary_markdown": str(md_path.resolve()),
            "final_and_latest_results": str(final_path.resolve()),
        }
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote benchmark summary to {md_path}", flush=True)
    print(f"Wrote latest results to {final_path}", flush=True)


if __name__ == "__main__":
    main()
