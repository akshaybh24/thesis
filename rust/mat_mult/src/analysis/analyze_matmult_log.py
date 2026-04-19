#!/usr/bin/env python3
r"""
Analyze a mat_mult log (e.g. matmult_21589422_1.out):

1. PDF visualization matching ``visualize_fitness_runs.py``: two pages (Fitness Difference,
   then Nr. Incorrect Cells); single-run logs use that run without min–max bands.
2. CSV summary: final average fitness (and cells), aggregate stats on h/c equation lengths
   parsed from each run's printed algorithm after ``Mutation counts``.
3. Curve fitting as in ``curve_fitting.py``: average every 1000th generation, exponential
   and rational fits to 10M — **separate pages** for Fitness Difference and Fitness Cells per
   fit-start slice (0, 100k, 250k).

Equation lengths:
  - h: character length of the RHS (after ``hNN:``).
  - c: character length of the RHS (after ``cNN:``), plus mean h-reference count per c equation
    (``h\\d+`` tokens), matching the spirit of log ``avg c-equation length``.

Usage:
  python3 analyze_matmult_log.py path/to/matmult_21589422_1.out
  python3 analyze_matmult_log.py path/to/log.out -o report.pdf --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from curve_fitting import (
    FIT_STARTS,
    FONT_AXIS_LABEL_SIZE,
    FONT_LEGEND_SIZE,
    FONT_RC as CF_FONT_RC,
    FONT_TICK_LABEL_SIZE,
    FONT_TITLE_SIZE,
    append_page,
    average_runs as cf_average_runs,
    fit_both,
    parse_log as parse_log_every_1000,
    suffix_from_start,
)

# Reuse parsing and plotting helpers from visualize_fitness_runs.py
from visualize_fitness_runs import (
    FONT_RC,
    GEN_LINE,
    append_separate_fitness_pages,
    build_average_series,
    parse_log,
    parse_num_triples_mutations,
    resolve_pdf_collision,
)

H_LINE = re.compile(r"^\s*h(\d+)\s*:\s*(.+?)\s*$")
C_LINE = re.compile(r"^\s*c(\d+)\s*:\s*(.+?)\s*$")
RUN_BRACKET = re.compile(r"\[run\s+(\d+)/(\d+)\]")


def c_h_ref_count(rhs: str) -> int:
    return len(re.findall(r"\bh\d+\b", rhs.strip()))


def parse_final_fitness_per_run(text: str) -> dict[int, tuple[float, float]]:
    """run_num -> (fitness_difference, fitness_cells) at generation 4999999 (last wins)."""
    out: dict[int, tuple[float, float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = GEN_LINE.search(line)
        if not m:
            continue
        gen = int(m.group(3))
        if gen != 4_999_999:
            continue
        run_num = int(m.group(1))
        fd = float(m.group(4).strip())
        fc = float(m.group(5).strip())
        out[run_num] = (fd, fc)
    return out


def parse_equation_stats_by_run(text: str) -> dict[int, dict[str, list]]:
    """
    After each ``[run x/y] Mutation counts`` block, collect h/c lines until the next
    ``[run`` or ``Run N of`` header.
    Returns run -> dict with keys h_char_lens, c_char_lens, c_href_counts (lists).
    """
    lines = text.splitlines()
    by_run: dict[int, dict[str, list]] = {}
    i = 0
    while i < len(lines):
        if "Mutation counts" in lines[i] and "[run" in lines[i]:
            m = RUN_BRACKET.search(lines[i])
            if not m:
                i += 1
                continue
            run_id = int(m.group(1))
            i += 1
            while i < len(lines):
                s = lines[i]
                if re.match(r"^\s*mutation type\s+\d+:", s):
                    i += 1
                    continue
                break
            h_char: list[int] = []
            c_char: list[int] = []
            c_href: list[int] = []
            while i < len(lines):
                s = lines[i]
                if re.match(r"^\[run\s+", s) or re.match(r"^Run\s+\d+\s+of\s+", s.strip()):
                    break
                hm = H_LINE.match(s)
                if hm:
                    rhs = hm.group(2).strip()
                    h_char.append(len(rhs))
                    i += 1
                    continue
                cm = C_LINE.match(s)
                if cm:
                    rhs = cm.group(2).strip()
                    c_char.append(len(rhs))
                    c_href.append(c_h_ref_count(rhs))
                    i += 1
                    continue
                i += 1
            by_run[run_id] = {
                "h_char_lens": h_char,
                "c_char_lens": c_char,
                "c_href_counts": c_href,
            }
            continue
        i += 1
    return by_run


def _safe_mean(vals: list[float]) -> float | None:
    return float(statistics.mean(vals)) if vals else None


def _safe_min(vals: list[float]) -> float | None:
    return float(min(vals)) if vals else None


def _safe_max(vals: list[float]) -> float | None:
    return float(max(vals)) if vals else None


def aggregate_equation_csv_rows(
    eq_by_run: dict[int, dict[str, list]],
) -> dict[str, float | str]:
    """Flatten per-run equation stats into summary fields for one CSV row."""
    if not eq_by_run:
        return {}

    run_avgs_h: list[float] = []
    run_max_h: list[float] = []
    run_min_h: list[float] = []
    run_avgs_c: list[float] = []
    run_max_c: list[float] = []
    run_min_c: list[float] = []
    run_avgs_href: list[float] = []

    all_h: list[int] = []
    all_c: list[int] = []
    all_href: list[int] = []

    for _rid, d in sorted(eq_by_run.items()):
        hc = d.get("h_char_lens") or []
        cc = d.get("c_char_lens") or []
        hr = d.get("c_href_counts") or []
        all_h.extend(hc)
        all_c.extend(cc)
        all_href.extend(hr)
        if hc:
            run_avgs_h.append(statistics.mean(hc))
            run_max_h.append(max(hc))
            run_min_h.append(min(hc))
        if cc:
            run_avgs_c.append(statistics.mean(cc))
            run_max_c.append(max(cc))
            run_min_c.append(min(cc))
        if hr:
            run_avgs_href.append(statistics.mean(hr))

    def fmt(x: float | None) -> str:
        return "" if x is None else f"{x:.6f}".rstrip("0").rstrip(".")

    return {
        "h_eq_char_len_mean_of_run_avgs": fmt(_safe_mean(run_avgs_h)),
        "h_eq_char_len_max_run_max": fmt(_safe_max(run_max_h)),
        "h_eq_char_len_min_run_min": fmt(_safe_min(run_min_h)),
        "h_eq_char_len_overall_min": fmt(float(min(all_h)) if all_h else None),
        "h_eq_char_len_overall_max": fmt(float(max(all_h)) if all_h else None),
        "c_eq_char_len_mean_of_run_avgs": fmt(_safe_mean(run_avgs_c)),
        "c_eq_char_len_max_run_max": fmt(_safe_max(run_max_c)),
        "c_eq_char_len_min_run_min": fmt(_safe_min(run_min_c)),
        "c_eq_char_len_overall_min": fmt(float(min(all_c)) if all_c else None),
        "c_eq_char_len_overall_max": fmt(float(max(all_c)) if all_c else None),
        "c_eq_href_mean_of_run_avgs": fmt(_safe_mean(run_avgs_href)),
        "c_eq_href_overall_mean": fmt(_safe_mean([float(x) for x in all_href]) if all_href else None),
    }


def xy_curve_averages(
    runs: dict[int, dict[int, tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean FD / FC across runs at each sampled generation (every 1000), matching
    ``curve_fitting.average_runs``. A single run uses its own series.
    """
    if not runs:
        raise ValueError("no run data")
    if len(runs) >= 2:
        return cf_average_runs(runs)
    sole = next(iter(runs.values()))
    x = np.array(sorted(sole.keys()), dtype=float)
    fd = np.array([sole[int(g)][0] for g in x], dtype=float)
    fc = np.array([sole[int(g)][1] for g in x], dtype=float)
    return x, fd, fc


def append_curve_fitting_pages(
    pdf: PdfPages,
    *,
    inp: Path,
    triples: int,
    mutations: int,
    source_label: str | None = None,
) -> None:
    """Average every-1000-gen series, exponential + rational fits (see ``curve_fitting.py``)."""
    runs_1k = parse_log_every_1000(inp)
    if not runs_1k:
        print(
            "No sampling points every 1000 generations — skipping curve-fit pages.",
            file=sys.stderr,
        )
        return
    try:
        x_full, fd_full, fc_full = xy_curve_averages(runs_1k)
    except ValueError as err:
        print(f"Curve fit skipped: {err}", file=sys.stderr)
        return

    curve_rc = {
        **CF_FONT_RC,
        "font.size": FONT_TICK_LABEL_SIZE,
        "xtick.labelsize": FONT_TICK_LABEL_SIZE,
        "ytick.labelsize": FONT_TICK_LABEL_SIZE,
        "axes.titlesize": FONT_TITLE_SIZE,
        "axes.labelsize": FONT_AXIS_LABEL_SIZE,
        "legend.fontsize": FONT_LEGEND_SIZE,
    }
    with plt.rc_context(curve_rc):
        for fit_start in FIT_STARTS:
            keep = x_full >= fit_start
            if not np.any(keep):
                print(
                    f"Curve fit skipped for start={fit_start}: no data",
                    file=sys.stderr,
                )
                continue
            x = x_full[keep]
            fd = fd_full[keep]
            fc = fc_full[keep]
            if len(x) < 4:
                print(
                    f"Curve fit skipped for start={fit_start}: need >= 4 points, got {len(x)}",
                    file=sys.stderr,
                )
                continue
            try:
                fd_exp, fd_rat = fit_both(x, fd)
                fc_exp, fc_rat = fit_both(x, fc)
            except Exception as err:
                print(
                    f"Curve fit failed for start={fit_start}: {err}",
                    file=sys.stderr,
                )
                continue
            suffix = suffix_from_start(fit_start)
            stem = f"{inp.stem} ({suffix})"
            fd_y_lim = (0.0, 10.0) if fit_start == 0 else None
            title_base = f"{triples} triples, {mutations} mutations ({suffix})"
            append_page(
                pdf=pdf,
                file_stem=stem,
                metric_label="Fitness Difference",
                x=x,
                y=fd,
                exp_fit=fd_exp,
                rat_fit=fd_rat,
                y_limits=fd_y_lim,
                title_override=f"{title_base} — Fitness Difference",
                source_file_name=source_label or inp.name,
            )
            append_page(
                pdf=pdf,
                file_stem=stem,
                metric_label="Fitness Cells",
                x=x,
                y=fc,
                exp_fit=fc_exp,
                rat_fit=fc_rat,
                y_limits=None,
                title_override=f"{title_base} — Fitness Cells",
                source_file_name=source_label or inp.name,
            )


def build_average_or_single(
    runs_data: dict[int, dict],
    run_nums: list[int],
):
    """Same shape as build_average_series output; single run uses narrow bands."""
    if len(run_nums) >= 2:
        out = build_average_series(runs_data, run_nums)
        if out is None:
            raise SystemExit("Could not align generations across runs for averaging.")
        return out
    if len(run_nums) == 1:
        r = runs_data[run_nums[0]]
        g, fd, fc = r["gens"], r["fd"], r["fc"]
        return (g, fd, fc, fd, fd, fc, fc)
    raise SystemExit("No run data.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="mat_mult log file (.out)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: next to input, *_analysis.pdf)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write summary CSV row (default: next to input, *_summary.csv)",
    )
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found or not a file: {inp}")

    runs_data, _, text = parse_log(inp)
    run_nums = sorted(runs_data.keys())
    if not run_nums:
        raise SystemExit("No generation fitness lines found (expected '[run x/y] Generation N, ...').")

    triples, mutations = parse_num_triples_mutations(text)
    total_runs = runs_data[run_nums[0]]["total"]
    final_per_run = parse_final_fitness_per_run(text)
    eq_stats = parse_equation_stats_by_run(text)

    fd_final_vals = [final_per_run[r][0] for r in sorted(final_per_run.keys()) if r in final_per_run]
    fc_final_vals = [final_per_run[r][1] for r in sorted(final_per_run.keys()) if r in final_per_run]
    final_avg_fd = statistics.mean(fd_final_vals) if fd_final_vals else ""
    final_avg_fc = statistics.mean(fc_final_vals) if fc_final_vals else ""

    agg_eq = aggregate_equation_csv_rows(eq_stats)

    pdf_path = args.output
    if pdf_path is None:
        pdf_path = inp.parent / f"{inp.stem}_analysis.pdf"
    else:
        pdf_path = pdf_path.resolve()
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"PDF output must end in .pdf, got: {pdf_path}")
    pdf_path = resolve_pdf_collision(pdf_path)

    csv_path = args.csv
    if csv_path is None:
        csv_path = inp.parent / f"{inp.stem}_summary.csv"
    else:
        csv_path = csv_path.resolve()

    g_av, fd_av, fc_av, fd_min, fd_max, fc_min, fc_max = build_average_or_single(runs_data, run_nums)

    row = {
        "source_file": inp.name,
        "num_triples": str(triples),
        "num_mutations": str(mutations),
        "num_runs": str(total_runs),
        "final_avg_fitness_difference": f"{final_avg_fd:.8f}".rstrip("0").rstrip(".") if final_avg_fd != "" else "",
        "final_avg_fitness_cells": f"{final_avg_fc:.8f}".rstrip("0").rstrip(".") if final_avg_fc != "" else "",
        **agg_eq,
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    with plt.rc_context(FONT_RC):
        with PdfPages(pdf_path) as pdf:
            append_separate_fitness_pages(
                pdf,
                gens=g_av,
                fd=fd_av,
                fc=fc_av,
                fd_min=fd_min,
                fd_max=fd_max,
                fc_min=fc_min,
                fc_max=fc_max,
                suptitle=(
                    f"{mutations} Mutations, {triples} Triples — "
                    f"{'Average' if len(run_nums) >= 2 else 'Single run'} "
                    f"({len(run_nums)} run(s) in file)"
                ),
                source_label=inp.name,
            )
            append_curve_fitting_pages(
                pdf,
                inp=inp,
                triples=triples,
                mutations=mutations,
                source_label=inp.name,
            )

    print(f"Wrote {pdf_path}", file=sys.stderr)
    print(f"Wrote {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
