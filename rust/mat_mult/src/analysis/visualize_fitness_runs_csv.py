#!/usr/bin/env python3
"""
Plot averaged fitness curves from compact CSV logs of the form::

    258000,0.14,3.5

where columns are: **generation**, **fitness level**, **fitness difference**.
Charts match ``visualize_fitness_runs.py``: **two separate pages** (Fitness Difference,
then Fitness Level).

Only lines matching three comma-separated numeric fields are used; other lines
(e.g. ``START ALGORITHM``, ``h1:``) are skipped. Like the original, samples are
kept when ``generation % 10 == 0``.

- **One file** with a single time series: **two** charts (bands collapse to the mean).
- **One folder (default):** files are grouped by basename triples/mutations (e.g. all
  ``10mat…1mut*`` together). Each group becomes **two** pages (**mean** curves). Groups
  with no overlapping generation grid are skipped.
- **``--per-file``:** **two** PDF pages per file (no grouping).
- **``--combine-files``:** ignore grouping; merge *all* files into a single average chart.

Triples/mutations for grouping and titles are parsed from the basename
(``{triples}mat…{mutations}mut…``), same convention as ``mat_mult`` output files.

Usage::

  python3 visualize_fitness_runs_csv.py path/to/file_or_folder \\
      -o path/to/out.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Reuse plotting, averaging, IO helpers from sibling module.
# (Script directory is on sys.path when invoked as python3 visualize_fitness_runs_csv.py.)
import visualize_fitness_runs as vfr


GEN_SAMPLE_STEP = getattr(vfr, "GEN_SAMPLE_STEP", 10)
STEM_TRIPLES = re.compile(r"^(\d+)mat", re.I)
STEM_MUTATIONS = re.compile(r"(\d+)mut", re.I)


def triples_and_mutations_from_stem(stem: str) -> tuple[int, int]:
    tm = STEM_TRIPLES.match(stem)
    mm = STEM_MUTATIONS.search(stem)
    if not tm or not mm:
        raise ValueError(
            f"Basename {stem!r} must look like '{{triples}}mat…{{mutations}}mut…'"
        )
    return int(tm.group(1)), int(mm.group(1))


def group_paths_by_triples_mutations(paths: list[Path]) -> dict[tuple[int, int], list[Path]]:
    """``(triples, mutations)`` → list of files sharing that config (from basename)."""
    groups: dict[tuple[int, int], list[Path]] = {}
    for p in paths:
        try:
            t, m = triples_and_mutations_from_stem(p.stem)
        except ValueError as err:
            print(f"Skip {p}: {err}", file=sys.stderr)
            continue
        groups.setdefault((t, m), []).append(p)
    return groups


def parse_csv_fitness_file(path: Path, *, run_key: int = 1) -> dict[int, dict]:
    """
    One CSV time series → one run entry: gens, fd (fitness difference), fl (fitness level).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    gens: list[int] = []
    fls: list[float] = []
    fds: list[float] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Support optional 4th/5th columns (holdout metrics); plotting uses first three only.
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gen = int(parts[0])
            fl = float(parts[1])
            fd = float(parts[2])
        except ValueError:
            continue
        if gen % GEN_SAMPLE_STEP != 0:
            continue
        gens.append(gen)
        fls.append(fl)
        fds.append(fd)

    pairs = sorted(zip(gens, fls, fds), key=lambda t: t[0])
    return {
        run_key: {
            "total": 1,
            "gens": [p[0] for p in pairs],
            "fd": [p[2] for p in pairs],
            "fl": [p[1] for p in pairs],
        }
    }


def runs_data_for_vfr(runs: dict[int, dict]) -> dict[int, dict]:
    """Map ``fl`` → ``fc`` keys expected by ``visualize_fitness_runs.build_average_series``."""
    out: dict[int, dict] = {}
    for k, r in runs.items():
        out[k] = {
            "total": r.get("total", 1),
            "gens": r["gens"],
            "fd": r["fd"],
            "fc": r["fl"],
        }
    return out


def average_or_single(
    runs_data: dict[int, dict],
    run_nums: list[int],
):
    """≥2 runs: mean/min/max; 1 run: band collapses to the series."""
    adapted = runs_data_for_vfr(runs_data)
    if len(run_nums) >= 2:
        return vfr.build_average_series(adapted, run_nums)
    if len(run_nums) == 1:
        n = run_nums[0]
        r = adapted[n]
        g, fd, fc = r["gens"], r["fd"], r["fc"]
        if not g:
            return None
        return (g, fd, fc, list(fd), list(fd), list(fc), list(fc))
    return None


def append_separate_csv_pages(
    pdf,
    *,
    gens: list,
    fd: list,
    fl: list,
    fd_min: list,
    fd_max: list,
    fl_min: list,
    fl_max: list,
    suptitle: str,
    source_label: str | None = None,
    shade_alpha: float = 0.28,
    range_caption: str | None = None,
) -> None:
    """Two PDF pages: Fitness Difference only, then Fitness Level only (y-lims from data)."""
    (
        gens_s,
        fd_s,
        fl_s,
        fd_min_s,
        fd_max_s,
        fl_min_s,
        fl_max_s,
    ) = vfr.subsample_average_with_range(
        gens, fd, fl, fd_min, fd_max, fl_min, fl_max, vfr.MAX_PLOT_POINTS
    )
    n_pts = len(gens_s)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    g_min = min(gens_s)
    g_max = max(gens_s)
    x_span = max(1, g_max - g_min)
    x_pad = max(vfr.X_AXIS_RIGHT_PAD_MIN, int(round(x_span * vfr.X_AXIS_RIGHT_PAD_FRACTION)))
    x_right = g_max + x_pad
    x_ticks = vfr.generation_ticks_for_range([g_min, g_max])

    def _ylim(lo_a: list[float], hi_a: list[float], mid: list[float]) -> tuple[float, float]:
        y_hi = max(max(hi_a) if hi_a else 0, max(mid) if mid else 0, 1e-6)
        y_lo = min(min(lo_a) if lo_a else 0, 0.0)
        pad = 0.05 * max(y_hi - y_lo, 1e-6)
        return y_lo, y_hi + pad

    def one_page(
        y_mean: list[float],
        y_min_b: list[float],
        y_max_b: list[float],
        ylabel: str,
        color: str,
        subtitle: str,
        *,
        show_range_footer: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle(
            f"{suptitle}\n{subtitle}",
            fontsize=vfr.PLOT_TITLE_FONTSIZE,
            ha="center",
            x=0.5,
            y=0.875,
        )
        ax.fill_between(
            gens_s, y_min_b, y_max_b, color=color, alpha=shade_alpha, linewidth=0, zorder=1
        )
        ax.plot(gens_s, y_mean, **vfr.series_plot_kwargs(n_pts, color), zorder=3)
        ax.set_xlabel("Generation", fontsize=vfr.PLOT_AXIS_LABEL_FONTSIZE, labelpad=12)
        ax.set_ylabel(
            ylabel,
            color=color,
            fontsize=vfr.PLOT_AXIS_LABEL_FONTSIZE,
            labelpad=8,
        )
        ax.set_xlim(g_min, x_right)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(vfr.format_generation_axis_compact))
        ax.tick_params(axis="y", labelcolor=color, labelsize=vfr.PLOT_TICK_LABEL_FONTSIZE)
        ax.tick_params(
            axis="x",
            bottom=True,
            labelbottom=True,
            rotation=30,
            labelsize=vfr.PLOT_TICK_LABEL_FONTSIZE,
        )
        for lb in ax.get_xticklabels():
            lb.set_horizontalalignment("right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*_ylim(y_min_b, y_max_b, y_mean))
        if n_pts > 0:
            x_last = gens_s[-1]
            y_last = y_mean[-1]
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_sp = max(x_hi - x_lo, 1.0)
            y_sp = max(y_hi - y_lo, 1.0)
            x_text = min(max(x_last - 0.03 * x_sp, x_lo + 0.01 * x_sp), x_hi - 0.01 * x_sp)
            y_ty = min(max(y_last + 0.03 * y_sp, y_lo + 0.01 * y_sp), y_hi - 0.01 * y_sp)
            ax.text(
                x_text,
                y_ty,
                vfr.format_metric_value(y_last),
                color=color,
                fontsize=vfr.PLOT_TICK_LABEL_FONTSIZE,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color),
                clip_on=True,
            )
        fig.subplots_adjust(left=0.10, right=0.92, bottom=0.22, top=0.82)
        if range_caption and show_range_footer:
            fig.text(
                0.02,
                0.02,
                range_caption,
                ha="left",
                va="bottom",
                fontsize=8,
                color="0.4",
                style="italic",
            )
        if source_label:
            fig.text(0.985, 0.02, source_label, ha="right", va="bottom", fontsize=8, color="0.35")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    one_page(
        fd_s,
        fd_min_s,
        fd_max_s,
        "Fitness Difference",
        "tab:blue",
        "Fitness Difference",
        show_range_footer=bool(range_caption),
    )
    one_page(
        fl_s,
        fl_min_s,
        fl_max_s,
        "Fitness Level",
        "tab:orange",
        "Fitness Level",
        show_range_footer=False,
    )


append_dual_axis_csv_page = append_separate_csv_pages


def write_pdf_csv(
    out_path: Path,
    runs_data: dict[int, dict],
    *,
    triples: int,
    mutations: int,
    source_label: str | None = None,
    range_caption: str | None = None,
) -> None:
    run_nums = sorted(runs_data.keys())
    if not run_nums:
        raise SystemExit(
            "No CSV rows like 'generation,fitness_level,fitness_difference' "
            f"(using generation % {GEN_SAMPLE_STEP} == 0)."
        )

    avg = average_or_single(runs_data, run_nums)
    if avg is None:
        raise SystemExit("No data to plot after averaging.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Output must be a .pdf file, got: {out_path}")

    g_av, fd_av, fl_av, fd_min, fd_max, fl_min, fl_max = avg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with plt.rc_context(vfr.FONT_RC):
        with PdfPages(out_path) as pdf:
            append_separate_csv_pages(
                pdf,
                gens=g_av,
                fd=fd_av,
                fl=fl_av,
                fd_min=fd_min,
                fd_max=fd_max,
                fl_min=fl_min,
                fl_max=fl_max,
                suptitle=(
                    f"{mutations} Mutations, {triples} Triples — Average "
                    f"({len(run_nums)} run(s), CSV format)"
                ),
                source_label=source_label,
                range_caption=range_caption,
            )


def write_folder_grouped_averages_pdf(out_path: Path, file_paths: list[Path]) -> int:
    """One page per (triples, mutations) group: mean lines + min–max shaded bands."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    groups = group_paths_by_triples_mutations(file_paths)
    if not groups:
        raise SystemExit("No files with parseable triples/mutations in basenames.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Output must be a .pdf file, got: {out_path}")

    caption = (
        "Shaded bands: min–max across runs at each generation (mean = solid line)."
    )
    pages = 0
    with plt.rc_context(vfr.FONT_RC):
        with PdfPages(out_path) as pdf:
            for triples, mutations in sorted(groups.keys()):
                paths_grp = sorted(groups[(triples, mutations)], key=lambda x: str(x).lower())
                merged: dict[int, dict] = {}
                for idx, p in enumerate(paths_grp, start=1):
                    part = parse_csv_fitness_file(p, run_key=idx)
                    merged[idx] = part[idx]
                valid_keys = sorted(i for i in merged if merged[i].get("gens"))
                if not valid_keys:
                    print(
                        f"Skip group ({triples}, {mutations}): no CSV rows",
                        file=sys.stderr,
                    )
                    continue
                slim = {i: merged[i] for i in valid_keys}
                avg = average_or_single(slim, valid_keys)
                if avg is None:
                    print(
                        f"Skip group ({triples}, {mutations}): could not build average",
                        file=sys.stderr,
                    )
                    continue
                g_av, fd_av, fl_av, fd_min, fd_max, fl_min, fl_max = avg
                n_runs = len(valid_keys)
                names = [p.name for p in paths_grp]
                src = f"{n_runs} file(s)" if n_runs > 3 else ", ".join(names)
                append_separate_csv_pages(
                    pdf,
                    gens=g_av,
                    fd=fd_av,
                    fl=fl_av,
                    fd_min=fd_min,
                    fd_max=fd_max,
                    fl_min=fl_min,
                    fl_max=fl_max,
                    suptitle=(
                        f"{mutations} Mutations, {triples} Triples — Average over "
                        f"{n_runs} run(s) (CSV)"
                    ),
                    source_label=src,
                    range_caption=caption if n_runs > 1 else None,
                )
                pages += 2

    if pages == 0:
        raise SystemExit("No pages written (all groups skipped).")
    return pages


def write_folder_pdf(out_path: Path, file_paths: list[Path]) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pages = 0
    with plt.rc_context(vfr.FONT_RC):
        with PdfPages(out_path) as pdf:
            for p in file_paths:
                try:
                    try:
                        triples, mutations = triples_and_mutations_from_stem(p.stem)
                    except ValueError as err:
                        print(f"Skip {p}: {err}", file=sys.stderr)
                        continue
                    runs = parse_csv_fitness_file(p, run_key=1)
                    if not runs.get(1, {}).get("gens"):
                        print(f"Skip {p}: no CSV fitness rows", file=sys.stderr)
                        continue
                except OSError as err:
                    print(f"Skip {p}: {err}", file=sys.stderr)
                    continue

                run_nums = sorted(runs.keys())
                avg = average_or_single(runs, run_nums)
                if avg is None:
                    print(f"Skip {p}: no plot data", file=sys.stderr)
                    continue
                g_av, fd_av, fl_av, fd_min, fd_max, fl_min, fl_max = avg
                append_separate_csv_pages(
                    pdf,
                    gens=g_av,
                    fd=fd_av,
                    fl=fl_av,
                    fd_min=fd_min,
                    fd_max=fd_max,
                    fl_min=fl_min,
                    fl_max=fl_max,
                    suptitle=(
                        f"{mutations} Mutations, {triples} Triples — Average "
                        f"(1 run, CSV format)"
                    ),
                    source_label=p.name,
                )
                pages += 2

    if pages == 0:
        raise SystemExit("No pages written (all files skipped).")
    return pages


def write_folder_combined_runs_pdf(out_path: Path, file_paths: list[Path]) -> None:
    """Treat each file as one run; one chart averaging over files."""
    merged: dict[int, dict] = {}
    triples_guess, mut_guess = 0, 0
    for idx, p in enumerate(sorted(file_paths, key=lambda x: str(x).lower()), start=1):
        try:
            t, m = triples_and_mutations_from_stem(p.stem)
            if triples_guess == 0:
                triples_guess, mut_guess = t, m
        except ValueError:
            pass
        part = parse_csv_fitness_file(p, run_key=idx)
        merged[idx] = part[idx]

    run_nums = sorted(merged.keys())
    valid = [i for i in run_nums if merged[i].get("gens")]
    if len(valid) < 1:
        raise SystemExit("No CSV data in any file.")

    slim = {i: merged[i] for i in valid}
    run_keys = sorted(slim.keys())
    avg = average_or_single(slim, run_keys)
    if avg is None:
        raise SystemExit("Could not build average.")

    write_pdf_csv(
        out_path,
        slim,
        triples=triples_guess,
        mutations=mut_guess,
        source_label=f"{len(valid)} files",
        range_caption=(
            "Shaded bands: min–max across runs at each generation (mean = solid line)."
            if len(valid) > 1
            else None
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        type=Path,
        help="CSV log file (gen,fitness_level,fitness_diff) or folder of such files",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="If input is a folder, include subdirectories",
    )
    ap.add_argument(
        "--per-file",
        action="store_true",
        help="If input is a folder: one PDF page per file (no grouping by triples/mutations).",
    )
    ap.add_argument(
        "--combine-files",
        action="store_true",
        help=(
            "If input is a folder: single chart merging every file into one average "
            "(ignore triples/mutations grouping)."
        ),
    )
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    default_out_dir = Path(__file__).resolve().parents[4] / "visualizations"

    import matplotlib

    matplotlib.use("Agg")

    if inp.is_dir():
        paths = vfr.collect_dir_files(inp, recursive=args.recursive)
        if not paths:
            raise SystemExit(f"No files under: {inp}")
        if args.output is not None:
            out_path = args.output.resolve()
        else:
            out_path = vfr.next_free_pdf_path(default_out_dir / "visualization_csv_averages.pdf")
        if args.combine_files:
            write_folder_combined_runs_pdf(out_path, paths)
            print(f"Wrote {out_path.resolve()} (2 pages: Fitness Difference + Fitness Level)")
        elif args.per_file:
            n = write_folder_pdf(out_path, paths)
            print(f"Wrote {out_path.resolve()} ({n} page(s))")
        else:
            n = write_folder_grouped_averages_pdf(out_path, paths)
            print(
                f"Wrote {out_path.resolve()} ({n} grouped average chart(s) "
                "by triples/mutations)"
            )
        return

    if not inp.is_file():
        raise SystemExit(f"Not a file or directory: {inp}")

    try:
        triples, mutations = triples_and_mutations_from_stem(inp.stem)
    except ValueError:
        triples, mutations = 0, 0

    runs = parse_csv_fitness_file(inp, run_key=1)
    if args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = vfr.resolve_pdf_collision(
            default_out_dir / f"visualization_csv_mut{mutations}_triples{triples}.pdf"
        )
    write_pdf_csv(
        out_path,
        runs,
        triples=triples,
        mutations=mutations,
        source_label=inp.name,
    )
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
