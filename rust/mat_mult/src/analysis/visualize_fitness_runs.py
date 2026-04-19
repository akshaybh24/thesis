#!/usr/bin/env python3
"""
Parse mat_mult-style logs: for each [run X/Y], collect every line where generation is a multiple of 10
(Fitness Difference, Fitness Cells).

Single log file: writes one PDF next to it, named
  visualization_mut{num_mutations}_triples{num_triples}.pdf
(collision → …_2.pdf, …). Two pages per file: Fitness Difference, then Nr. Incorrect Cells.

Folder: reads every file in the directory (optional -r for subfolders), and writes one combined PDF
(default: visualization_all_averages.pdf in that folder) with two pages per file that yields a valid
average (2+ runs, shared generations). Skipped files are reported on stderr.

Curves use at most 5000 evenly spaced plot points when more samples exist in the log.

Optional -o sets the output PDF path for both modes (must end in .pdf).

Usage:
  python3 visualize_fitness_runs.py
  python3 visualize_fitness_runs.py path/to/log_file_or_folder
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

FONT_FAMILY = "Bell MT"
FONT_RC = {
    "font.family": [FONT_FAMILY, "serif"],
    "font.serif": [FONT_FAMILY, "DejaVu Serif", "Times New Roman"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
}
PLOT_TITLE_FONTSIZE = 16
PLOT_AXIS_LABEL_FONTSIZE = 14
PLOT_TICK_LABEL_FONTSIZE = 11

X_TICK_STEP = 500_000
Y_AXIS_LIM = (0, 30)
X_AXIS_RIGHT_PAD_FRACTION = 0.03
X_AXIS_RIGHT_PAD_MIN = 100_000

GEN_SAMPLE_STEP = 10
# At most this many (x, y) pairs drawn per run (evenly subsampled along generation).
MAX_PLOT_POINTS = 5000
# Cap visible markers so PDFs stay readable; markevery picks ~this many along the series.
_MAX_MARKERS_ALONG_LINE = 400
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "logs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "visualizations"
# Single-line entries like: [run 1/10] Generation 10, fitness difference: ..., fitness_cells: ...
GEN_LINE = re.compile(
    r"\[run\s+(\d+)/(\d+)\]\s*(?:Generation|Gen)\s+(\d+)\s*,\s*"
    r"fitness difference:\s*([^,]+)\s*,\s*fitness_cells:\s*([^,]+)",
    re.IGNORECASE,
)


def parse_float(s: str) -> float:
    return float(s.strip())


def format_generation_axis_compact(x: float, _pos: int | None) -> str:
    """X-axis ticks: 500k, 1M, 1.5M, …"""
    v = float(x)
    if abs(v) < 1.0:
        return "0"
    if v >= 1_000_000:
        m = v / 1_000_000
        if abs(m - round(m)) < 1e-7:
            return f"{int(round(m))}M"
        s = f"{m:.2f}".rstrip("0").rstrip(".")
        return f"{s}M"
    k = v / 1000.0
    if abs(k - round(k)) < 1e-7:
        return f"{int(round(k))}k"
    s = f"{k:.2f}".rstrip("0").rstrip(".")
    return f"{s}k"


def format_metric_value(v: float) -> str:
    """Compact label for endpoint values shown on lines."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def generation_ticks_for_range(gens: list[int], *, target_ticks: int = 7) -> list[int]:
    """Build readable generation tick positions that always include endpoints."""
    if not gens:
        return [0]
    g_min = int(min(gens))
    g_max = int(max(gens))
    if g_min == g_max:
        return [g_min]

    span = max(1, g_max - g_min)
    raw_step = span / max(1, target_ticks - 1)
    mag = 10 ** int(math.floor(math.log10(raw_step)))
    norm = raw_step / mag
    if norm <= 1.0:
        step_base = 1.0
    elif norm <= 2.0:
        step_base = 2.0
    elif norm <= 5.0:
        step_base = 5.0
    else:
        step_base = 10.0
    step = max(1, int(step_base * mag))

    start = (g_min // step) * step
    ticks: list[int] = []
    x = start
    while x <= g_max:
        if x >= g_min:
            ticks.append(x)
        x += step
    if ticks[0] != g_min:
        ticks.insert(0, g_min)
    if ticks[-1] != g_max:
        ticks.append(g_max)
    return ticks


def subsample_series(
    gens: list,
    fd: list,
    fc: list,
    max_points: int,
) -> tuple[list, list, list]:
    """Evenly pick up to max_points samples along the series (always keeps ends when n > max_points)."""
    n = len(gens)
    if max_points < 2 or n <= max_points:
        return list(gens), list(fd), list(fc)
    raw_idx = [
        int(round(k * (n - 1) / (max_points - 1)))
        for k in range(max_points)
    ]
    raw_idx = [max(0, min(n - 1, i)) for i in raw_idx]
    merged: list[int] = []
    for i in raw_idx:
        if not merged or i != merged[-1]:
            merged.append(i)
    if merged[0] != 0:
        merged.insert(0, 0)
    if merged[-1] != n - 1:
        merged.append(n - 1)
    return (
        [gens[i] for i in merged],
        [fd[i] for i in merged],
        [fc[i] for i in merged],
    )


def subsample_average_with_range(
    gens: list[int],
    fd_avg: list[float],
    fc_avg: list[float],
    fd_min: list[float],
    fd_max: list[float],
    fc_min: list[float],
    fc_max: list[float],
    max_points: int,
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Subsample average series and its range bands with the same indices."""
    n = len(gens)
    if max_points < 2 or n <= max_points:
        return (
            list(gens),
            list(fd_avg),
            list(fc_avg),
            list(fd_min),
            list(fd_max),
            list(fc_min),
            list(fc_max),
        )
    raw_idx = [
        int(round(k * (n - 1) / (max_points - 1)))
        for k in range(max_points)
    ]
    raw_idx = [max(0, min(n - 1, i)) for i in raw_idx]
    merged: list[int] = []
    for i in raw_idx:
        if not merged or i != merged[-1]:
            merged.append(i)
    if merged[0] != 0:
        merged.insert(0, 0)
    if merged[-1] != n - 1:
        merged.append(n - 1)
    return (
        [gens[i] for i in merged],
        [fd_avg[i] for i in merged],
        [fc_avg[i] for i in merged],
        [fd_min[i] for i in merged],
        [fd_max[i] for i in merged],
        [fc_min[i] for i in merged],
        [fc_max[i] for i in merged],
    )


def build_average_series(
    runs_data: dict[int, dict],
    run_nums: list[int],
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float], list[float]] | None:
    """
    Mean Fitness Difference and Fitness Cells at each generation that appears in every run.
    Returns None if there are fewer than two runs or no shared generations.
    """
    if len(run_nums) < 2:
        return None
    maps: list[dict[int, tuple[float, float]]] = []
    for n in run_nums:
        r = runs_data[n]
        g, fd, fc = r["gens"], r["fd"], r["fc"]
        if not g:
            return None
        maps.append({gi: (float(fdi), float(fci)) for gi, fdi, fci in zip(g, fd, fc)})
    common: set[int] = set(maps[0])
    for d in maps[1:]:
        common &= set(d)
    if not common:
        return None
    gens_sorted = sorted(common)
    n_r = len(maps)
    fd_avg = [sum(d[g][0] for d in maps) / n_r for g in gens_sorted]
    fc_avg = [sum(d[g][1] for d in maps) / n_r for g in gens_sorted]
    fd_min = [min(d[g][0] for d in maps) for g in gens_sorted]
    fd_max = [max(d[g][0] for d in maps) for g in gens_sorted]
    fc_min = [min(d[g][1] for d in maps) for g in gens_sorted]
    fc_max = [max(d[g][1] for d in maps) for g in gens_sorted]
    return gens_sorted, fd_avg, fc_avg, fd_min, fd_max, fc_min, fc_max


def append_separate_fitness_pages(
    pdf: PdfPages,
    *,
    gens: list,
    fd: list,
    fc: list,
    fd_min: list,
    fd_max: list,
    fc_min: list,
    fc_max: list,
    suptitle: str,
    source_label: str | None = None,
) -> None:
    """Two PDF pages: Fitness Difference only, then Nr. Incorrect Cells (each with mean + band)."""
    (
        gens_s,
        fd_s,
        fc_s,
        fd_min_s,
        fd_max_s,
        fc_min_s,
        fc_max_s,
    ) = subsample_average_with_range(
        gens, fd, fc, fd_min, fd_max, fc_min, fc_max, MAX_PLOT_POINTS
    )
    n_pts = len(gens_s)

    g_min = min(gens_s)
    g_max = max(gens_s)
    x_span = max(1, g_max - g_min)
    x_pad = max(X_AXIS_RIGHT_PAD_MIN, int(round(x_span * X_AXIS_RIGHT_PAD_FRACTION)))
    x_right = g_max + x_pad
    x_ticks = generation_ticks_for_range([g_min, g_max])

    def one_page(
        y_mean: list,
        y_min_band: list,
        y_max_band: list,
        ylabel: str,
        color: str,
        subtitle: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        # Two-line suptitle above axes; tweak y/top together so the gap isn’t excessive.
        fig.suptitle(
            f"{suptitle}\n{subtitle}",
            fontsize=PLOT_TITLE_FONTSIZE,
            ha="center",
            x=0.5,
            y=0.935,
        )
        ax.fill_between(gens_s, y_min_band, y_max_band, color=color, alpha=0.20, linewidth=0)
        ax.plot(gens_s, y_mean, **series_plot_kwargs(n_pts, color))
        ax.set_xlabel(
            "Generation",
            fontsize=PLOT_AXIS_LABEL_FONTSIZE,
            labelpad=12,
        )
        ax.set_ylabel(
            ylabel,
            color=color,
            fontsize=PLOT_AXIS_LABEL_FONTSIZE,
            labelpad=8,
        )
        ax.set_xlim(g_min, x_right)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(format_generation_axis_compact))
        ax.tick_params(
            axis="y",
            labelcolor=color,
            labelsize=PLOT_TICK_LABEL_FONTSIZE,
        )
        ax.tick_params(
            axis="x",
            bottom=True,
            labelbottom=True,
            rotation=30,
            labelsize=PLOT_TICK_LABEL_FONTSIZE,
        )
        for lb in ax.get_xticklabels():
            lb.set_horizontalalignment("right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*Y_AXIS_LIM)
        if n_pts > 0:
            x_last = gens_s[-1]
            y_last = y_mean[-1]
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_sp = max(x_hi - x_lo, 1.0)
            y_sp = max(y_hi - y_lo, 1.0)
            x_text = min(max(x_last - 0.03 * x_sp, x_lo + 0.01 * x_sp), x_hi - 0.01 * x_sp)
            y_text = min(max(y_last + 0.03 * y_sp, y_lo + 0.01 * y_sp), y_hi - 0.01 * y_sp)
            ax.text(
                x_text,
                y_text,
                format_metric_value(y_last),
                color=color,
                fontsize=PLOT_TICK_LABEL_FONTSIZE,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
                clip_on=True,
            )
        fig.subplots_adjust(
            left=0.10,
            right=0.92,
            bottom=0.22,
            top=0.79,
        )
        if source_label:
            fig.text(
                0.985,
                0.02,
                source_label,
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.35",
            )
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    one_page(
        fd_s,
        fd_min_s,
        fd_max_s,
        "Fitness Difference",
        "tab:blue",
        "Fitness Difference",
    )
    one_page(
        fc_s,
        fc_min_s,
        fc_max_s,
        "Nr. Incorrect Cells",
        "tab:orange",
        "Nr. Incorrect Cells",
    )


# Backward compatibility for scripts that still import the old name.
append_dual_axis_fitness_page = append_separate_fitness_pages


def series_plot_kwargs(n_pts: int, color: str) -> dict:
    """
    Line weight and markers scale with sample count: many points → thinner line + sparser
    markers so the polyline reads as discrete samples; few points → thicker line + every point marked.

    Lines are capped with a minimum width so long logs stay readable in PDFs (avoid sub-pixel lines).
    """
    n = max(int(n_pts), 1)
    lw = max(1.05, min(2.25, 60.0 / (n**0.5)))
    markevery = max(1, n // _MAX_MARKERS_ALONG_LINE)
    ms = max(0.55, min(2.0, 42.0 / (n**0.25)))
    return {
        "linewidth": lw,
        "marker": "o",
        "markersize": ms,
        "markevery": markevery,
        "color": color,
        "markerfacecolor": color,
        "markeredgecolor": color,
        "markeredgewidth": max(0.12, min(0.45, lw * 0.35)),
        "alpha": 0.96,
        "zorder": 3,
        # Rasterizing thick polylines can look faint in some PDF viewers; keep vector lines.
        "rasterized": False,
    }


def parse_num_triples_mutations(text: str) -> tuple[int, int]:
    """Last occurrence of num triples / num mutations in the file wins."""
    triples: int | None = None
    mutations: int | None = None
    for line in text.splitlines():
        m = re.search(r"num\s+triples:\s*(\d+)", line, re.IGNORECASE)
        if m:
            triples = int(m.group(1))
        m = re.search(r"num\s+mutations:\s*(\d+)", line, re.IGNORECASE)
        if m:
            mutations = int(m.group(1))
    if triples is None or mutations is None:
        raise SystemExit(
            "Could not find num triples / num mutations in the log "
            "(expected lines like 'num triples: N', 'num mutations: N')."
        )
    return triples, mutations


def resolve_pdf_collision(path: Path) -> Path:
    """If path exists, try path stem + _2, _3, … before .pdf."""
    if not path.exists():
        return path
    stem = path.stem
    parent = path.parent
    suffix = path.suffix
    n = 2
    while n < 1_000_000:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1
    raise SystemExit(f"Could not find a free filename near: {path}")


def next_free_pdf_path(path: Path) -> Path:
    """If path exists, use {stem}_2.pdf, {stem}_3.pdf, …"""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 2
    while n < 1_000_000:
        cand = parent / f"{stem}_{n}{suffix}"
        if not cand.exists():
            return cand
        n += 1
    raise SystemExit(f"Could not find a free PDF path near: {path}")


def collect_dir_files(directory: Path, *, recursive: bool) -> list[Path]:
    """Sorted list of files under directory (non-recursive or rglob)."""
    if recursive:
        files = [p for p in directory.rglob("*") if p.is_file()]
    else:
        files = [p for p in directory.iterdir() if p.is_file()]
    skip = {".ds_store"}
    files = [
        p
        for p in files
        if p.name.lower() not in skip and not p.name.startswith("._")
    ]
    return sorted((p.resolve() for p in files), key=lambda p: str(p).lower())


def parse_log(path: Path) -> tuple[dict[int, dict], dict[int, str], str]:
    """
    Returns:
      runs_data: run_num -> {"gens": [...], "fd": [...], "fc": [...], "total": int}
      final_lines: run_num -> full line text for generation 4999999 (last occurrence wins)
      text: full file text (for triples/mutations parsing)
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    runs_data: dict[int, dict] = {}
    final_lines: dict[int, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = GEN_LINE.search(line)
        if not m:
            continue
        run_num = int(m.group(1))
        total_runs = int(m.group(2))
        gen = int(m.group(3))
        fd_s = m.group(4)
        fc_s = m.group(5)

        if gen == 4_999_999:
            final_lines[run_num] = line

        if gen % GEN_SAMPLE_STEP != 0:
            continue

        if run_num not in runs_data:
            runs_data[run_num] = {
                "total": total_runs,
                "gens": [],
                "fd": [],
                "fc": [],
            }
        r = runs_data[run_num]
        r["total"] = total_runs
        r["gens"].append(gen)
        r["fd"].append(parse_float(fd_s))
        r["fc"].append(parse_float(fc_s))

    for r in runs_data.values():
        pairs = sorted(zip(r["gens"], r["fd"], r["fc"]), key=lambda t: t[0])
        r["gens"] = [p[0] for p in pairs]
        r["fd"] = [p[1] for p in pairs]
        r["fc"] = [p[2] for p in pairs]

    return runs_data, final_lines, text


def write_pdf(
    out_path: Path,
    runs_data: dict[int, dict],
    *,
    triples: int,
    mutations: int,
    source_label: str | None = None,
) -> None:
    run_nums = sorted(runs_data.keys())
    if not run_nums:
        raise SystemExit("No generation fitness lines found (expected 'Generation N, fitness difference: ...').")

    avg = build_average_series(runs_data, run_nums)
    if avg is None:
        raise SystemExit(
            "Cannot build an average chart: need at least two runs with at least one shared generation "
            "in the parsed samples (same GEN_SAMPLE_STEP grid in each run)."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Output must be a .pdf file, got: {out_path}")

    g_av, fd_av, fc_av, fd_min, fd_max, fc_min, fc_max = avg
    with plt.rc_context(FONT_RC):
        with PdfPages(out_path) as pdf:
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
                    f"{mutations} Mutations, {triples} Triples — Average "
                    f"({len(run_nums)} runs)"
                ),
                source_label=source_label,
            )


def write_folder_averages_pdf(out_path: Path, file_paths: list[Path]) -> int:
    """Append one average chart per eligible file into a single PDF. Returns page count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Output must be a .pdf file, got: {out_path}")

    pages = 0
    with plt.rc_context(FONT_RC):
        with PdfPages(out_path) as pdf:
            for p in file_paths:
                try:
                    runs_data, _, text = parse_log(p)
                except OSError as err:
                    print(f"Skip {p}: {err}", file=sys.stderr)
                    continue
                run_nums = sorted(runs_data.keys())
                if not run_nums:
                    print(f"Skip {p}: no generation fitness lines", file=sys.stderr)
                    continue
                try:
                    triples, mutations = parse_num_triples_mutations(text)
                except SystemExit as err:
                    print(f"Skip {p}: {err}", file=sys.stderr)
                    continue
                avg = build_average_series(runs_data, run_nums)
                if avg is None:
                    print(
                        f"Skip {p}: need 2+ runs with shared generations for average",
                        file=sys.stderr,
                    )
                    continue
                g_av, fd_av, fc_av, fd_min, fd_max, fc_min, fc_max = avg
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
                        f"{mutations} Mutations, {triples} Triples — Average "
                        f"({len(run_nums)} runs)"
                    ),
                    source_label=p.name,
                )
                pages += 2

    if pages == 0:
        raise SystemExit(
            "No average charts written (every file was skipped or produced no average)."
        )
    return pages




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=(
            "Path to a log file (.out, etc.) or a folder of logs "
            f"(default: {DEFAULT_INPUT_DIR})"
        ),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PDF path (default directory when omitted: "
            f"{DEFAULT_OUTPUT_DIR})"
        ),
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="If input is a folder, include files in subdirectories",
    )
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    if inp.is_dir():
        paths = collect_dir_files(inp, recursive=args.recursive)
        if not paths:
            raise SystemExit(f"No files under directory: {inp}")
        if args.output is not None:
            out_path = args.output.resolve()
        else:
            out_path = next_free_pdf_path(DEFAULT_OUTPUT_DIR / "visualization_all_averages.pdf")
        n_pages = write_folder_averages_pdf(out_path, paths)
        print(f"Wrote {out_path.resolve()} ({n_pages} average chart(s))")
        return

    if not inp.is_file():
        raise SystemExit(f"Not a file or directory: {inp}")

    runs_data, _final_lines, text = parse_log(inp)
    triples, mutations = parse_num_triples_mutations(text)
    if args.output is not None:
        out_path = args.output.resolve()
    else:
        base = DEFAULT_OUTPUT_DIR / f"visualization_mut{mutations}_triples{triples}.pdf"
        out_path = resolve_pdf_collision(base)
    write_pdf(
        out_path,
        runs_data,
        triples=triples,
        mutations=mutations,
        source_label=inp.name,
    )
    print(
        f"Wrote {out_path.resolve()} (2 pages: Fitness Difference + Nr. Incorrect Cells, "
        f"{len(runs_data)} run(s))"
    )


if __name__ == "__main__":
    main()
