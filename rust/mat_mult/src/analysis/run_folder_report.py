#!/usr/bin/env python3
"""
Analyze a mat_mult *output folder* that contains one log file per outer run (e.g.
``output/25matcappos2mut…/`` with files ``…1``, ``…2``, …).

Produces:
  1. A PDF with (a) the same dual-axis average fitness chart as
     ``visualize_fitness_runs.py`` (mean ± range across files), and (b) a curve-fit
     chart: final fitness difference vs run index; the fitted curve is drawn from
     run 1 through **2 × N** where **N** is the number of log files (endpoint =
     ``2 * num_runs``).
  2. A CSV summary with final average fitness (difference and cells), and
     average / max / min **character lengths** of h- and c-equation RHS strings
     (after ``hNN:`` / ``cNN:``), aggregated across runs / files.

Usage:
  python3 run_folder_report.py /path/to/output/prefix_folder
  python3 run_folder_report.py /path/to/output/prefix_folder -o /path/to/report_basename

With ``-o report``, writes ``report.pdf`` and ``report.csv`` next to each other.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from visualize_fitness_runs import (
    FONT_RC,
    append_dual_axis_fitness_page,
    build_average_series,
    collect_dir_files,
    parse_log,
    parse_num_triples_mutations,
)

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None  # type: ignore[misc, assignment]

RUN_HEADER = re.compile(r"\[run\s+(\d+)\s*/\s*(\d+)\s*\]", re.I)
H_LINE = re.compile(r"^\s*h(\d+)\s*:\s*(.+?)\s*$")
C_LINE = re.compile(r"^\s*c(\d+)\s*:\s*(.+?)\s*$")
START_ALGO = re.compile(r"START\s+ALGORITHM", re.I)


def parse_algo_char_lengths_by_run(text: str) -> list[dict]:
    """Per [run X/Y] block after START ALGORITHM: char lengths of h/c RHS strings."""
    lines = text.splitlines()
    runs_out: list[dict] = []
    run_num: int | None = None
    total: int | None = None
    before_start: list[str] = []
    h_exprs: list[str] = []
    c_exprs: list[str] = []
    seen_start = False

    def flush() -> None:
        nonlocal h_exprs, c_exprs, before_start, seen_start
        if run_num is None:
            return
        h_lens = [len(s) for s in h_exprs]
        c_lens = [len(s) for s in c_exprs]
        runs_out.append(
            {
                "run_num": run_num,
                "total": total or 0,
                "h_lens": h_lens,
                "c_lens": c_lens,
            }
        )
        h_exprs = []
        c_exprs = []
        before_start = []
        seen_start = False

    for line in lines:
        m = RUN_HEADER.search(line)
        if m:
            new_num = int(m.group(1))
            new_total = int(m.group(2))
            if run_num is not None and new_num != run_num:
                flush()
            run_num = new_num
            total = new_total
            if not seen_start:
                before_start.append(line)
            continue

        if run_num is None:
            continue

        if START_ALGO.search(line):
            seen_start = True
            continue

        if not seen_start:
            before_start.append(line)
            continue

        hm = H_LINE.match(line)
        if hm:
            h_exprs.append(hm.group(2).strip())
            continue
        cm = C_LINE.match(line)
        if cm:
            c_exprs.append(cm.group(2).strip())
            continue

    if run_num is not None:
        flush()

    return runs_out


def pooled_length_stats(runs: list[dict]) -> tuple[float, float, float, float, float, float]:
    """From per-run lists, return (h_avg, h_max, h_min, c_avg, c_max, c_min) pooled."""
    all_h: list[int] = []
    all_c: list[int] = []
    for r in runs:
        all_h.extend(r["h_lens"])
        all_c.extend(r["c_lens"])

    def agg(xs: list[int]) -> tuple[float, float, float]:
        if not xs:
            return (float("nan"), float("nan"), float("nan"))
        return (float(np.mean(xs)), float(np.max(xs)), float(np.min(xs)))

    ha, hmx, hmn = agg(all_h)
    ca, cmx, cmn = agg(all_c)
    return ha, hmx, hmn, ca, cmx, cmn


def merge_folder_runs(paths: list[Path]) -> tuple[dict[int, dict], list[Path], str]:
    """
    Treat each file as one independent series (run index = order among usable files).
    Returns merged runs_data for build_average_series, list of usable paths, sample text.
    """
    merged: dict[int, dict] = {}
    successful: list[Path] = []
    sample_text = ""
    idx = 1
    for p in paths:
        runs_data, _, text = parse_log(p)
        if not runs_data:
            print(f"Skip (no fitness lines): {p}", file=sys.stderr)
            continue
        if not sample_text:
            sample_text = text
        keys = sorted(runs_data.keys())
        if len(keys) > 1:
            av = build_average_series(runs_data, keys)
            if av is None:
                print(
                    f"Skip (cannot merge inner runs): {p}",
                    file=sys.stderr,
                )
                continue
            g_av, fd_av, fc_av, _, _, _, _ = av
            merged[idx] = {
                "total": len(paths),
                "gens": list(g_av),
                "fd": list(fd_av),
                "fc": list(fc_av),
            }
        else:
            r = runs_data[keys[0]]
            merged[idx] = {
                "total": r.get("total", len(paths)),
                "gens": list(r["gens"]),
                "fd": list(r["fd"]),
                "fc": list(r["fc"]),
            }
        successful.append(p)
        idx += 1
    return merged, successful, sample_text


def final_fitness_from_series(gens: list, fd: list, fc: list) -> tuple[float, float]:
    if not gens:
        return (float("nan"), float("nan"))
    last_i = len(gens) - 1
    return (float(fd[last_i]), float(fc[last_i]))


def fit_curve_run_index(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_end: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Fit y vs x (run indices). Returns (x_fine, y_fine, label).
    x_fine spans [1, x_end] (typically x_end = 2 * num_runs).
    """
    n = len(x)
    if n < 2 or not np.all(np.isfinite(y)):
        xf = np.linspace(1.0, x_end, max(2, int(x_end)))
        return xf, np.full_like(xf, np.nan), "insufficient data"

    xf = np.linspace(1.0, float(x_end), max(50, min(400, int(x_end * 20))))

    # Prefer quadratic if enough points and scipy available
    if curve_fit is not None and n >= 3:
        try:

            def quad(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
                return a * t * t + b * t + c

            popt, _ = curve_fit(quad, x, y, maxfev=10000)
            yf = quad(xf, *popt)
            return xf, yf, "quadratic fit"
        except Exception:
            pass

    coeffs = np.polyfit(x, y, deg=1)
    yf = np.polyval(coeffs, xf)
    return xf, yf, "linear fit"


def append_curve_fit_page(
    pdf: PdfPages,
    *,
    final_fds: list[float],
    num_runs: int,
    title: str,
) -> None:
    x = np.arange(1, num_runs + 1, dtype=float)
    y = np.array(final_fds, dtype=float)
    x_end = 2.0 * float(num_runs)
    xf, yf, fit_label = fit_curve_run_index(x, y, x_end=x_end)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(title, fontsize=16, ha="center", y=0.96)
    ax.scatter(x, y, color="tab:blue", s=36, zorder=3, label="Final fitness difference (per file)")
    mask = np.isfinite(yf)
    if np.any(mask):
        ax.plot(xf[mask], yf[mask], color="tab:orange", linewidth=1.5, label=f"{fit_label} (to x = 2×N = {x_end:g})")
    ax.axvline(num_runs, color="0.5", linestyle="--", linewidth=0.8, label="N (last data run)")
    ax.set_xlabel("Run index (file order)", fontsize=14)
    ax.set_ylabel("Final fitness difference", fontsize=14)
    ax.set_xlim(0.5, x_end + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.18, top=0.88)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "folder",
        type=Path,
        help="Folder containing one mat_mult log file per run (non-recursive)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output basename without extension (writes .pdf and .csv); default: folder_report in folder",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include files in subdirectories (same as visualize_fitness_runs)",
    )
    args = ap.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    paths = collect_dir_files(folder, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"No files under: {folder}")

    merged, successful_paths, sample_text = merge_folder_runs(paths)
    if len(merged) < 2:
        raise SystemExit(
            "Need at least two files with parseable fitness series to build averages. "
            f"Got {len(merged)} usable file(s)."
        )

    run_nums = sorted(merged.keys())
    triples, mutations = parse_num_triples_mutations(sample_text)
    avg = build_average_series(merged, run_nums)
    if avg is None:
        raise SystemExit("Could not align generations across files for averaging.")

    g_av, fd_av, fc_av, fd_min, fd_max, fc_min, fc_max = avg

    # Final fitness per merged run (last sample)
    final_fds: list[float] = []
    final_fcs: list[float] = []
    for n in run_nums:
        r = merged[n]
        fd, fc = final_fitness_from_series(r["gens"], r["fd"], r["fc"])
        final_fds.append(fd)
        final_fcs.append(fc)

    # Equation lengths: pool all runs inside each file, then summarize folder
    h_avgs: list[float] = []
    h_maxes: list[float] = []
    h_mins: list[float] = []
    c_avgs: list[float] = []
    c_maxes: list[float] = []
    c_mins: list[float] = []

    for p in successful_paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        by_run = parse_algo_char_lengths_by_run(text)
        if not by_run:
            continue
        ha, hmx, hmn, ca, cmx, cmn = pooled_length_stats(by_run)
        if np.isfinite(ha):
            h_avgs.append(ha)
        if np.isfinite(hmx):
            h_maxes.append(hmx)
        if np.isfinite(hmn):
            h_mins.append(hmn)
        if np.isfinite(ca):
            c_avgs.append(ca)
        if np.isfinite(cmx):
            c_maxes.append(cmx)
        if np.isfinite(cmn):
            c_mins.append(cmn)

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    out_base = args.output
    if out_base is None:
        out_base = folder / "folder_report"
    else:
        out_base = out_base.resolve()
        out_base.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = out_base.with_suffix(".pdf")
    csv_path = out_base.with_suffix(".csv")

    source_label = folder.name
    with plt.rc_context(FONT_RC):
        with PdfPages(pdf_path) as pdf:
            append_dual_axis_fitness_page(
                pdf,
                gens=g_av,
                fd=fd_av,
                fc=fc_av,
                fd_min=fd_min,
                fd_max=fd_max,
                fc_min=fc_min,
                fc_max=fc_max,
                suptitle=(
                    f"{mutations} Mutations, {triples} Triples — Average over files "
                    f"({len(run_nums)} runs)"
                ),
                source_label=source_label,
            )
            append_curve_fit_page(
                pdf,
                final_fds=final_fds,
                num_runs=len(run_nums),
                title=f"Final fitness difference vs run index (curve to 2×N = {2 * len(run_nums)})",
            )

    row = {
        "folder": str(folder),
        "num_run_files": len(run_nums),
        "triples": triples,
        "mutations": mutations,
        "final_avg_fitness_difference": f"{mean(final_fds):.6f}",
        "final_avg_fitness_cells": f"{mean(final_fcs):.6f}",
        "h_eq_char_len_avg": f"{mean(h_avgs):.6f}" if h_avgs else "",
        "h_eq_char_len_max": f"{max(h_maxes):.6f}" if h_maxes else "",
        "h_eq_char_len_min": f"{min(h_mins):.6f}" if h_mins else "",
        "c_eq_char_len_avg": f"{mean(c_avgs):.6f}" if c_avgs else "",
        "c_eq_char_len_max": f"{max(c_maxes):.6f}" if c_maxes else "",
        "c_eq_char_len_min": f"{min(c_mins):.6f}" if c_mins else "",
        "curve_fit_x_endpoint": str(2 * len(run_nums)),
        "curve_fit_note": "x runs from 1 to 2×num_run_files; y is quadratic or linear fit of final FD vs run index",
    }

    fieldnames = list(row.keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)

    print(f"Wrote {pdf_path.resolve()}")
    print(f"Wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
