#!/usr/bin/env python3
"""
Process all mat_mult log files and produce six output PDF files:
- fitness difference + fitness cells with fit start at 0
- fitness difference + fitness cells with fit start at 100k
- fitness difference + fitness cells with fit start at 250k

Each PDF has one page per input file.
For each page:
- average over runs at each sampled generation (every 1000)
- keep data from the selected fit-start generation and onward
- fit exponential and rational curves
- extrapolate fits to 10M and annotate the 10M endpoints
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MultipleLocator

GEN_RE = re.compile(
    r"\[run\s+(\d+)/(\d+)\]\s*(?:Generation|Gen)\s+(\d+)\s*,\s*"
    r"fitness difference:\s*([^,]+)\s*,\s*fitness_cells:\s*([^,]+)",
    re.IGNORECASE,
)
TRIPLES_RE = re.compile(r"num\s+triples:\s*(\d+)", re.IGNORECASE)
MUTATIONS_RE = re.compile(r"num\s+mutations:\s*(\d+)", re.IGNORECASE)

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "logs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "curve_fitting"
SAMPLE_STEP = 1000
FIT_STARTS = [0, 100_000, 250_000]
EXTRAPOLATE_TO = 10_000_000
EPS = 1e-12
FONT_RC = {
    "font.family": ["Bell MT", "DejaVu Serif", "Times New Roman", "serif"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
}

# Font sizes for plot readability (matplotlib doesn't automatically scale all elements).
FONT_TICK_LABEL_SIZE = 16
FONT_AXIS_LABEL_SIZE = 20
FONT_TITLE_SIZE = 22
FONT_LEGEND_SIZE = 16
FONT_ANNOTATE_SIZE = 16
FONT_SOURCE_LABEL_SIZE = 16


@dataclass
class FitResult:
    params: tuple[float, ...]
    r2: float
    y_10m: float


def exponential_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * x) + c


def rational_model(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (a * x + b) / (x + c) + d


def generation_fmt(x: float, _pos: int | None = None) -> str:
    if abs(x) < 1:
        return "0"
    if x >= 1_000_000:
        s = f"{x / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{s}M"
    return f"{x / 1000:.0f}k"


def parse_log(path: Path) -> dict[int, dict[int, tuple[float, float]]]:
    runs: dict[int, dict[int, tuple[float, float]]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        m = GEN_RE.search(raw.strip())
        if not m:
            continue
        run_num = int(m.group(1))
        gen = int(m.group(3))
        if gen % SAMPLE_STEP != 0:
            continue
        fd = float(m.group(4).strip())
        fc = float(m.group(5).strip())
        runs.setdefault(run_num, {})[gen] = (fd, fc)
    return runs


def parse_triples_mutations(path: Path) -> tuple[str, str]:
    triples = "?"
    mutations = "?"
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        m_t = TRIPLES_RE.search(raw)
        if m_t:
            triples = m_t.group(1)
        m_m = MUTATIONS_RE.search(raw)
        if m_m:
            mutations = m_m.group(1)
        if triples != "?" and mutations != "?":
            break
    return triples, mutations


def average_runs(runs: dict[int, dict[int, tuple[float, float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(runs) < 2:
        raise ValueError("need at least two runs")
    common: set[int] | None = None
    for series in runs.values():
        gs = set(series.keys())
        common = gs if common is None else (common & gs)
    if not common:
        raise ValueError("no common generations")
    x = np.array(sorted(common), dtype=float)
    run_list = list(runs.values())
    fd = np.array([np.mean([r[int(g)][0] for r in run_list]) for g in x], dtype=float)
    fc = np.array([np.mean([r[int(g)][1] for r in run_list]) for g in x], dtype=float)
    return x, fd, fc


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < EPS:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def fit_both(x: np.ndarray, y: np.ndarray) -> tuple[FitResult, FitResult]:
    y0, y_end = float(y[0]), float(y[-1])

    exp_popt, _ = curve_fit(
        exponential_model,
        x,
        y,
        p0=(max(y0 - y_end, EPS), 1e-6, y_end),
        bounds=([-math.inf, 0.0, -math.inf], [math.inf, math.inf, math.inf]),
        maxfev=200_000,
    )
    exp_y_hat = exponential_model(x, *exp_popt)
    exp_10m = float(exponential_model(np.array([EXTRAPOLATE_TO]), *exp_popt)[0])
    exp_fit = FitResult(tuple(float(v) for v in exp_popt), _r2(y, exp_y_hat), exp_10m)

    rat_popt, _ = curve_fit(
        rational_model,
        x,
        y,
        p0=(0.0, y0 * max(float(x[0]), 1.0), 100_000.0, y_end),
        bounds=([-math.inf, -math.inf, 0.0, -math.inf], [math.inf, math.inf, math.inf, math.inf]),
        maxfev=200_000,
    )
    rat_y_hat = rational_model(x, *rat_popt)
    rat_10m = float(rational_model(np.array([EXTRAPOLATE_TO]), *rat_popt)[0])
    rat_fit = FitResult(tuple(float(v) for v in rat_popt), _r2(y, rat_y_hat), rat_10m)

    return exp_fit, rat_fit


def append_page(
    *,
    pdf: PdfPages,
    file_stem: str,
    metric_label: str,
    x: np.ndarray,
    y: np.ndarray,
    exp_fit: FitResult,
    rat_fit: FitResult,
    y_limits: tuple[float, float] | None = None,
    title_override: str | None = None,
    source_file_name: str | None = None,
) -> None:
    x_ext = np.linspace(float(x.min()), float(EXTRAPOLATE_TO), 2000)
    exp_curve = exponential_model(x_ext, *exp_fit.params)
    rat_curve = rational_model(x_ext, *rat_fit.params)

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25)

    # Bold lines + z-order so averages and fits stay readable in PDF viewers.
    l_raw, = ax.plot(
        x,
        y,
        color="#1f77b4",
        lw=2.4,
        alpha=0.98,
        solid_capstyle="round",
        zorder=4,
        label=f"Average {metric_label}",
    )
    l_exp, = ax.plot(
        x_ext,
        exp_curve,
        color="#2ca02c",
        lw=2.8,
        alpha=0.98,
        solid_capstyle="round",
        zorder=2,
        label=f"Exponential Fitting (R² = {exp_fit.r2:.4f})",
    )
    l_rat, = ax.plot(
        x_ext,
        rat_curve,
        color="#8c1d40",
        lw=2.8,
        alpha=0.98,
        solid_capstyle="round",
        zorder=3,
        label=f"Rational Fitting (R² = {rat_fit.r2:.4f})",
    )
    ax.axvline(
        float(x.max()),
        color="gray",
        ls="--",
        lw=1.8,
        alpha=0.85,
        zorder=1,
        label="Data Threshold",
    )

    ax.scatter([EXTRAPOLATE_TO, EXTRAPOLATE_TO], [exp_fit.y_10m, rat_fit.y_10m], color=["#2ca02c", "#8c1d40"], zorder=5)
    ax.annotate(
        f"Y = {exp_fit.y_10m:.5f}",
        xy=(EXTRAPOLATE_TO, exp_fit.y_10m),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=FONT_ANNOTATE_SIZE,
        color="#2ca02c",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", alpha=0.8),
    )
    ax.annotate(
        f"Y = {rat_fit.y_10m:.5f}",
        xy=(EXTRAPOLATE_TO, rat_fit.y_10m),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=FONT_ANNOTATE_SIZE,
        color="#8c1d40",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8c1d40", alpha=0.8),
    )

    if title_override is not None:
        _tt = f"Curve Fitting - {title_override}\n(Extrapolated To 10M)"
    else:
        _tt = f"Curve Fitting - {file_stem} — {metric_label}\n(Extrapolated To 10M)"
    # pad: keep the two-line title out of the data/legend area; y>1.0 is in axes coords.
    ax.set_title(_tt, fontsize=FONT_TITLE_SIZE, pad=10, y=1.015)
    ax.set_xlabel("Generation", fontsize=FONT_AXIS_LABEL_SIZE)
    ax.set_ylabel(metric_label, fontsize=FONT_AXIS_LABEL_SIZE)
    ax.xaxis.set_major_locator(MultipleLocator(2_000_000))
    ax.xaxis.set_major_formatter(FuncFormatter(generation_fmt))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlim(float(x.min()), float(EXTRAPOLATE_TO))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(
        [l_raw, l_exp, l_rat],
        [l_raw.get_label(), l_exp.get_label(), l_rat.get_label()],
        loc="upper right",
        fontsize=FONT_LEGEND_SIZE,
        frameon=True,
    )
    if source_file_name:
        fig.text(
            0.995,
            0.01,
            source_file_name,
            ha="right",
            va="bottom",
            fontsize=FONT_SOURCE_LABEL_SIZE,
            color="black",
            alpha=0.8,
        )

    # Reserve a modest top strip for the title without a huge gap below it.
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), pad=0.45)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def suffix_from_start(start_gen: int) -> str:
    if start_gen == 0:
        return "from_0"
    if start_gen % 1000 == 0:
        return f"from_{start_gen // 1000}k"
    return f"from_{start_gen}"


def main() -> None:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log_paths = sorted([p for p in input_dir.glob("*.out") if p.is_file()], key=lambda p: p.name.lower())
    if not log_paths:
        raise SystemExit(f"No .out files found under: {input_dir}")

    created_files: list[Path] = []
    total_pages = 0
    total_skipped = 0

    plt.rcParams.update(FONT_RC)
    # Scale tick labels and general text.
    plt.rcParams.update(
        {
            "font.size": FONT_TICK_LABEL_SIZE,
            "xtick.labelsize": FONT_TICK_LABEL_SIZE,
            "ytick.labelsize": FONT_TICK_LABEL_SIZE,
            "axes.titlesize": FONT_TITLE_SIZE,
            "axes.labelsize": FONT_AXIS_LABEL_SIZE,
            "legend.fontsize": FONT_LEGEND_SIZE,
        }
    )

    for fit_start in FIT_STARTS:
        suffix = suffix_from_start(fit_start)
        fd_pdf = output_dir / f"fitness_difference_curve_fit_10m_{suffix}.pdf"
        fc_pdf = output_dir / f"fitness_cells_curve_fit_10m_{suffix}.pdf"
        pages = 0
        skipped = 0

        with PdfPages(fd_pdf) as fd_pages, PdfPages(fc_pdf) as fc_pages:
            for p in log_paths:
                runs = parse_log(p)
                if not runs:
                    skipped += 1
                    continue
                try:
                    x, fd, fc = average_runs(runs)
                except ValueError:
                    skipped += 1
                    continue

                keep = x >= fit_start
                if not np.any(keep):
                    skipped += 1
                    continue
                x, fd, fc = x[keep], fd[keep], fc[keep]

                fd_exp, fd_rat = fit_both(x, fd)
                fc_exp, fc_rat = fit_both(x, fc)

                triples, mutations = parse_triples_mutations(p)

                append_page(
                    pdf=fd_pages,
                    file_stem=f"{p.stem} ({suffix})",
                    metric_label="Fitness Difference",
                    x=x,
                    y=fd,
                    exp_fit=fd_exp,
                    rat_fit=fd_rat,
                    y_limits=(0.0, 10.0) if fit_start == 0 else None,
                    title_override=f"{triples} triples, {mutations} mutations - Fitness Difference",
                    source_file_name=p.name,
                )
                append_page(
                    pdf=fc_pages,
                    file_stem=f"{p.stem} ({suffix})",
                    metric_label="Fitness Cells",
                    x=x,
                    y=fc,
                    exp_fit=fc_exp,
                    rat_fit=fc_rat,
                    y_limits=(0.0, 10.0) if fit_start == 0 else None,
                    source_file_name=p.name,
                )
                pages += 1

        if pages == 0:
            raise SystemExit(f"No valid files produced plottable series for fit start {fit_start}.")
        created_files.extend([fd_pdf, fc_pdf])
        total_pages += pages
        total_skipped += skipped

    print("Generated 6 PDF files total:")
    for p in created_files:
        print(f"- {p.resolve()}")
    print(f"Total pages written across all PDFs: {total_pages * 2}")
    print(f"Total skipped file checks: {total_skipped}")


if __name__ == "__main__":
    main()
