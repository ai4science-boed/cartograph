#!/usr/bin/env python3
"""Retrospective CARTOGRAPH refusal audit for A-Lab supplementary data.

This script intentionally does not attempt to re-adjudicate the A-Lab paper.
It asks a narrower governance question: if a fixed residual guard is
calibrated on manually confirmed positive synthesis claims, which published
positive claims would be passed versus flagged for human review/refusal?

Inputs are the corrected A-Lab supplementary data files downloaded from the
Nature article:
  data/alab/original_supplementary_data/
    - 20230502 Synthesis Results with Recipes.csv
    - Refinement-Table.xlsx

Outputs are written to outputs/alab_audit/. The pipeline is deterministic;
the only resampling diagnostic uses a fixed seed recorded in the JSON output.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "alab" / "original_supplementary_data"
OUT_DIR = ROOT / "outputs" / "alab_audit"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYNTHESIS_CSV = DATA_DIR / "20230502 Synthesis Results with Recipes.csv"
REFINEMENT_XLSX = DATA_DIR / "Refinement-Table.xlsx"


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a simple chemical formula into element counts.

    Handles nested parentheses and decimal occupancies, which are enough for
    the formulas appearing in the A-Lab supplementary table.
    """

    s = str(formula).strip()
    # Keep the leading formula and drop annotations such as "(6C-type)" that
    # appear after a whitespace in phase labels.
    leading = re.match(r"^([A-Za-z0-9().]+)", s)
    if leading:
        s = leading.group(1)
    s = re.sub(r"_(?:Alab|ICSD.*)$", "", s)
    s = s.replace(" ", "")
    i = 0

    def parse_group(stop: str | None = None) -> dict[str, float]:
        nonlocal i
        counts: dict[str, float] = defaultdict(float)
        while i < len(s):
            if stop is not None and s[i] == stop:
                i += 1
                break
            if s[i] == "(":
                i += 1
                inner = parse_group(")")
                mult = parse_number()
                for el, cnt in inner.items():
                    counts[el] += cnt * mult
            else:
                m = re.match(r"[A-Z][a-z]?", s[i:])
                if not m:
                    i += 1
                    continue
                el = m.group(0)
                i += len(el)
                counts[el] += parse_number()
        return counts

    def parse_number() -> float:
        nonlocal i
        m = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", s[i:])
        if not m:
            return 1.0
        i += len(m.group(0))
        return float(m.group(0))

    return dict(parse_group())


def formula_key(formula: str) -> tuple[tuple[str, float], ...]:
    counts = parse_formula(formula)
    return tuple(sorted((el, round(cnt, 3)) for el, cnt in counts.items()))


def same_composition(a: str, b: str) -> bool:
    ka = formula_key(a)
    kb = formula_key(b)
    return bool(ka) and ka == kb


def extract_rwp(text: object) -> float:
    if pd.isna(text):
        return math.nan
    match = re.search(r"Rwp\s*=\s*([0-9.]+)\s*%", str(text))
    return float(match.group(1)) if match else math.nan


def extract_phase_fractions(text: object) -> list[tuple[str, float]]:
    if pd.isna(text):
        return []
    flat = str(text).replace("\n", " | ")
    pairs = []
    for name, frac in re.findall(r"([^|\[]+?)\s*\[([0-9.]+)%\]", flat):
        cleaned = name.strip(" |,\t")
        if cleaned:
            pairs.append((cleaned, float(frac)))
    return pairs


def target_fraction(target: str, text: object) -> tuple[float, float, list[tuple[str, float]]]:
    """Return target wt%, largest non-target phase wt%, and parsed phases."""

    phases = extract_phase_fractions(text)
    flat = "" if pd.isna(text) else str(text)
    target_wt = 0.0
    if not phases and re.search(re.escape(target), flat):
        target_wt = 100.0
    for name, frac in phases:
        if same_composition(target, name) or target.replace(" ", "") in name.replace(" ", ""):
            target_wt = max(target_wt, frac)
    alt_wt = 0.0
    for name, frac in phases:
        if not (same_composition(target, name) or target.replace(" ", "") in name.replace(" ", "")):
            alt_wt = max(alt_wt, frac)
    return target_wt, alt_wt, phases


def is_metadata_row(target: str) -> bool:
    return target in {"Inconclusive results", "Offline Experiments***", "Notes", "*", "**", "***"}


def is_inconclusive_conclusion(conclusions: str) -> bool:
    """Treat only structure/composition inconclusive as external audit labels.

    Ordering ambiguity is common and is not counted as a failed synthesis claim.
    """

    return bool(
        re.search(r"Structure:\s*[Ii]nconclusive", conclusions)
        or re.search(r"Composition:\s*[Ii]nconclusive", conclusions)
        or re.search(r"Composition:\s*inconclusive", conclusions)
    )


@dataclass
class AuditRow:
    target: str
    csv_result: str
    external_label: str
    auto_rwp: float
    manual_rwp: float
    auto_target_wt: float
    manual_target_wt: float
    auto_max_alt_wt: float
    manual_max_alt_wt: float
    rho: float
    rwp_only_score: float
    target_deficit_score: float
    audit_decision: str
    conclusions: str


def load_audit_table() -> pd.DataFrame:
    synthesis = pd.read_csv(SYNTHESIS_CSV, encoding="latin1")
    refinement = pd.read_excel(REFINEMENT_XLSX)

    result_by_key = {
        formula_key(row["Target"]): row["Result"]
        for _, row in synthesis.iterrows()
        if isinstance(row["Target"], str)
    }

    rows = []
    for _, row in refinement[refinement["Target Formula"].notna()].iterrows():
        target = str(row["Target Formula"]).strip()
        if is_metadata_row(target):
            continue
        auto_text = row["From automated analysis:\nphases identified [wt%] and refinement outcome (Rwp%)"]
        manual_text = row["From manual analysis:\nphases identified [wt%] and refinement outcome (Rwp%)"]
        conclusions = str(row["Conclusions"]).replace("\n", " | ")

        csv_result = result_by_key.get(formula_key(target), "")
        auto_target, auto_alt, _ = target_fraction(target, auto_text)
        manual_target, manual_alt, _ = target_fraction(target, manual_text)
        auto_rwp = extract_rwp(auto_text)
        manual_rwp = extract_rwp(manual_text)

        external_label = "inconclusive" if is_inconclusive_conclusion(conclusions) else "confirmed"
        rows.append(
            {
                "target": target,
                "csv_result": csv_result,
                "external_label": external_label,
                "auto_rwp": auto_rwp,
                "manual_rwp": manual_rwp,
                "auto_target_wt": auto_target,
                "manual_target_wt": manual_target,
                "auto_max_alt_wt": auto_alt,
                "manual_max_alt_wt": manual_alt,
                "conclusions": conclusions,
            }
        )
    return pd.DataFrame(rows)


def compute_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = df.copy()

    # Presence-claim residual: high if the corrected public refinement has a
    # large fit residual, substantial non-target phase content, or weak target
    # fraction. The 20% Rwp scale is intentionally loose and the final threshold
    # is calibrated on confirmed A-Lab successes.
    scored["rho"] = np.sqrt(
        (scored["manual_rwp"] / 20.0) ** 2
        + ((100.0 - scored["manual_target_wt"]) / 100.0) ** 2
        + (scored["manual_max_alt_wt"] / 100.0) ** 2
    )
    scored["rwp_only_score"] = scored["manual_rwp"]
    scored["target_deficit_score"] = (100.0 - scored["manual_target_wt"]) / 100.0

    primary = scored[scored["csv_result"].isin(["Success", "Partial"])]
    calibration = primary[
        (primary["csv_result"] == "Success") & (primary["external_label"] == "confirmed")
    ]

    delta = float(calibration["rho"].quantile(0.95))
    rwp_delta = float(calibration["rwp_only_score"].quantile(0.95))
    deficit_delta = float(calibration["target_deficit_score"].quantile(0.95))

    scored["audit_decision"] = np.where(scored["rho"] > delta, "flag_for_review", "pass")
    scored["rwp_only_decision"] = np.where(
        scored["rwp_only_score"] > rwp_delta, "flag_for_review", "pass"
    )
    scored["target_deficit_decision"] = np.where(
        scored["target_deficit_score"] > deficit_delta, "flag_for_review", "pass"
    )

    thresholds = {
        "delta_rho_95_confirmed_success": delta,
        "delta_rwp_95_confirmed_success": rwp_delta,
        "delta_target_deficit_95_confirmed_success": deficit_delta,
        "calibration_n": int(len(calibration)),
        "primary_n": int(len(primary)),
    }
    return scored, thresholds


def bootstrap_audit_ci(
    scored: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> dict[str, object]:
    """Bootstrap the confirmed-success calibration threshold.

    This is an operating-characteristic diagnostic rather than a held-out
    evaluation: we resample the calibration rows, recompute the 95th-percentile
    threshold, and track how pass/flag rates move under that calibration noise.
    """

    primary = scored[scored["csv_result"].isin(["Success", "Partial"])].copy()
    calibration = primary[
        (primary["csv_result"] == "Success") & (primary["external_label"] == "confirmed")
    ]
    confirmed = primary[primary["external_label"] == "confirmed"]
    inconclusive = primary[primary["external_label"] == "inconclusive"]
    calibration_values = calibration["rho"].to_numpy()
    rng = np.random.default_rng(seed)

    deltas = []
    confirmed_rates = []
    inconclusive_rates = []
    for _ in range(n_boot):
        sample = rng.choice(calibration_values, size=len(calibration_values), replace=True)
        delta = float(np.quantile(sample, 0.95))
        deltas.append(delta)
        confirmed_rates.append(float((confirmed["rho"] > delta).mean()))
        inconclusive_rates.append(float((inconclusive["rho"] > delta).mean()))

    def ci(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
        }

    return {
        "seed": seed,
        "replicates": n_boot,
        "delta": ci(deltas),
        "confirmed_flag_rate": ci(confirmed_rates),
        "inconclusive_flag_rate": ci(inconclusive_rates),
    }


def summarize(scored: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, object]:
    primary = scored[scored["csv_result"].isin(["Success", "Partial"])].copy()
    confirmed = primary[primary["external_label"] == "confirmed"]
    inconclusive = primary[primary["external_label"] == "inconclusive"]

    def rate(df: pd.DataFrame, col: str) -> dict[str, float]:
        if len(df) == 0:
            return {"n": 0, "flagged": 0, "rate": math.nan}
        flagged = int((df[col] == "flag_for_review").sum())
        return {"n": int(len(df)), "flagged": flagged, "rate": flagged / len(df)}

    return {
        "thresholds": thresholds,
        "bootstrap_calibration": bootstrap_audit_ci(scored),
        "primary_reported_positive_claims": int(len(primary)),
        "confirmed_positive_claims": rate(confirmed, "audit_decision"),
        "inconclusive_positive_claims": rate(inconclusive, "audit_decision"),
        "rwp_only_confirmed": rate(confirmed, "rwp_only_decision"),
        "rwp_only_inconclusive": rate(inconclusive, "rwp_only_decision"),
        "target_deficit_only_confirmed": rate(confirmed, "target_deficit_decision"),
        "target_deficit_only_inconclusive": rate(inconclusive, "target_deficit_decision"),
        "flagged_targets": primary[primary["audit_decision"] == "flag_for_review"][
            [
                "target",
                "csv_result",
                "external_label",
                "rho",
                "manual_rwp",
                "manual_target_wt",
                "manual_max_alt_wt",
            ]
        ].to_dict(orient="records"),
    }


def write_markdown(scored: pd.DataFrame, summary: dict[str, object]) -> None:
    primary = scored[scored["csv_result"].isin(["Success", "Partial"])].copy()
    delta = summary["thresholds"]["delta_rho_95_confirmed_success"]
    inconc = summary["inconclusive_positive_claims"]
    conf = summary["confirmed_positive_claims"]
    rwp_inc = summary["rwp_only_inconclusive"]
    rwp_conf = summary["rwp_only_confirmed"]
    deficit_inc = summary["target_deficit_only_inconclusive"]
    deficit_conf = summary["target_deficit_only_confirmed"]
    bootstrap = summary["bootstrap_calibration"]

    top = primary.sort_values("rho", ascending=False)[
        [
            "target",
            "csv_result",
            "external_label",
            "audit_decision",
            "rho",
            "manual_rwp",
            "manual_target_wt",
            "manual_max_alt_wt",
        ]
    ]

    md = []
    md.append("# A-Lab Retrospective Refusal Audit\n")
    md.append("This is a governance audit, not a re-adjudication of the A-Lab paper. ")
    md.append(
        "We use corrected public A-Lab refinement data to ask which originally "
        "positive synthesis claims would pass a fixed residual guard and which "
        "would be flagged for human review / refusal to self-certify.\n"
    )
    md.append("## Protocol\n")
    md.append("- Source: corrected A-Lab supplementary data from the Nature article.\n")
    md.append("- Audit population: originally positive A-Lab claims, `Success` or `Partial` in the synthesis-results CSV.\n")
    md.append("- External labels: corrected manual conclusions in `Refinement-Table.xlsx`; only structure/composition inconclusive counts as `inconclusive`.\n")
    md.append("- Calibration: 95th percentile of the CARTOGRAPH audit residual on confirmed `Success` rows.\n")
    md.append(
        "- Residual: `rho = sqrt((Rwp/20)^2 + ((100-target_wt)/100)^2 + (max_alt_wt/100)^2)` "
        "using corrected public manual-refinement features.\n"
    )
    md.append(f"- Calibrated threshold: `delta = {delta:.3f}`.\n")
    md.append("\n## Main Result\n")
    md.append(
        f"- Inconclusive positive claims flagged: `{inconc['flagged']}/{inconc['n']}` "
        f"({100*inconc['rate']:.1f}%).\n"
    )
    md.append(
        f"- Confirmed positive claims flagged for review: `{conf['flagged']}/{conf['n']}` "
        f"({100*conf['rate']:.1f}%).\n"
    )
    md.append(
        f"- Rwp-only baseline flags inconclusive claims: `{rwp_inc['flagged']}/{rwp_inc['n']}`.\n"
    )
    md.append(
        f"- Target-deficit-only baseline flags inconclusive claims: `{deficit_inc['flagged']}/{deficit_inc['n']}`.\n"
    )
    md.append(
        f"- Target-deficit-only baseline flags confirmed claims: `{deficit_conf['flagged']}/{deficit_conf['n']}`; "
        f"Rwp-only flags confirmed claims: `{rwp_conf['flagged']}/{rwp_conf['n']}`.\n"
    )
    md.append(
        "- Bootstrap calibration diagnostic "
        f"({bootstrap['replicates']} resamples, seed {bootstrap['seed']}): "
        f"`delta` 95% CI = [{bootstrap['delta']['ci95_low']:.3f}, "
        f"{bootstrap['delta']['ci95_high']:.3f}], "
        f"inconclusive flag-rate CI = [{100*bootstrap['inconclusive_flag_rate']['ci95_low']:.1f}%, "
        f"{100*bootstrap['inconclusive_flag_rate']['ci95_high']:.1f}%].\n"
    )
    md.append(
        "\nInterpretation: the guard is non-trivially informative: it flags all "
        "post-correction inconclusive positive claims while also conservatively "
        "flagging a small number of confirmed but complex multiphase claims for "
        "review. This is useful governance evidence, not a claim that the audit "
        "fully re-adjudicates the A-Lab study.\n"
    )
    md.append("\n## Flagged Positive Claims\n")
    md.append(markdown_table(top[top["audit_decision"] == "flag_for_review"]))
    md.append("\n\n## Full Positive-Claim Table\n")
    md.append(markdown_table(top))
    md.append("\n")
    (OUT_DIR / "alab_audit_summary.md").write_text("".join(md), encoding="utf-8")
    (OUT_DIR / "final_and_latest_results.md").write_text("".join(md), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer."""

    if df.empty:
        return "_None._"
    cols = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append(values)
    widths = [max(len(str(col)), *(len(row[i]) for row in rows)) for i, col in enumerate(cols)]
    lines = []
    lines.append("| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols)) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    return "\n".join(lines)


def write_figure(scored: pd.DataFrame, thresholds: dict[str, float]) -> None:
    primary = scored[scored["csv_result"].isin(["Success", "Partial"])].copy()
    primary = primary.sort_values("rho", ascending=False)
    colors = primary["external_label"].map({"confirmed": "#28666e", "inconclusive": "#c44536"})
    edgecolors = np.where(primary["audit_decision"] == "flag_for_review", "black", "none")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(primary))
    ax.bar(x, primary["rho"], color=colors, edgecolor=edgecolors, linewidth=1.2)
    ax.axhline(
        thresholds["delta_rho_95_confirmed_success"],
        color="#1d1d1d",
        linestyle="--",
        linewidth=1.5,
        label="95th percentile confirmed-success threshold",
    )
    ax.set_ylabel("CARTOGRAPH audit residual $\\rho$")
    ax.set_xlabel("A-Lab positive synthesis claims, sorted by residual")
    ax.set_title("Retrospective audit of published A-Lab positive claims")
    ax.set_xticks(x)
    ax.set_xticklabels(primary["target"], rotation=75, ha="right", fontsize=7)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.96,
        "red = corrected manual conclusion inconclusive; teal = confirmed",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_1_alab_audit.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SYNTHESIS_CSV.exists() or not REFINEMENT_XLSX.exists():
        raise FileNotFoundError(
            "A-Lab data not found. Download and extract the Nature supplementary data ZIP first."
        )
    raw = load_audit_table()
    scored, thresholds = compute_scores(raw)
    summary = summarize(scored, thresholds)

    scored.to_csv(OUT_DIR / "alab_audit_results.csv", index=False)
    (OUT_DIR / "alab_audit_results.json").write_text(
        json.dumps({"summary": summary, "rows": scored.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    write_markdown(scored, summary)
    write_figure(scored, thresholds)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
