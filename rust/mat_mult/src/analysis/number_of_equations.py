#!/usr/bin/env python3
"""
Read mat_mult log files: for each [run X/Y] block, count h-equations and mean
c-equation length (number of hNN references in each c RHS), using only lines
after START ALGORITHM. Solved status is inferred from lines before START.

Typical use (folder with one log per job):
  python number_of_equations.py /path/to/logs -o NumberOfEquations.csv

Each input file is expected to contain multiple runs (e.g. 10).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

RUN_HEADER = re.compile(r"\[run\s+(\d+)\s*/\s*(\d+)\s*\]", re.I)
H_LINE = re.compile(r"^\s*h(\d+)\s*:\s*(.+?)\s*$")
C_LINE = re.compile(r"^\s*c(\d+)\s*:\s*(.+?)\s*$")
START_ALGO = re.compile(r"START\s+ALGORITHM", re.I)


def c_h_ref_count(rhs: str) -> int:
    return len(re.findall(r"\bh\d+\b", rhs.strip()))


# 5×5 output cells in row-major order (log labels c11 … c55).
C_EQ_COLUMN_KEYS = [f"{r}{c}" for r in range(1, 6) for c in range(1, 6)]


def c_lengths_by_cell(c_items: list[tuple[str, str]]) -> dict[str, int]:
    """Map c-term id string (e.g. ``11`` from ``c11:``) to h-reference count; last line wins on duplicates."""
    out: dict[str, int] = {}
    for cid, rhs in c_items:
        out[cid] = c_h_ref_count(rhs)
    return out


def avg_c_from_cells(c_by_id: dict[str, int]) -> float:
    return statistics.mean(c_by_id.values()) if c_by_id else float("nan")


def infer_solved(lines_before_start: list[str]) -> str:
    blob = "\n".join(lines_before_start).lower()
    if "not solved" in blob or "unsolved" in blob:
        return "no"
    if re.search(r"\bsolved\b", blob):
        return "yes"
    return ""


RUN_HEADER_HINT = re.compile(r"\[run\s+\d+\s*/", re.I)


def trailing_int_from_filename(path: Path) -> str:
    """Final digit run in the basename stem, e.g. ``2matnocappos1mut1`` → ``1``, ``...mut10`` → ``10``."""
    m = re.search(r"(\d+)$", path.stem)
    return m.group(1) if m else ""


def parse_file_with_start_blocks(text: str) -> list[dict]:
    """Logs with repeated 'START ALGORITHM' sections and no [run X/Y] lines (e.g. raw mat_mult file output)."""
    lines = text.splitlines()
    start_idxs = [i for i, line in enumerate(lines) if START_ALGO.search(line)]
    if not start_idxs:
        return []

    n = len(start_idxs)
    runs: list[dict] = []
    for idx, si in enumerate(start_idxs):
        lo = max(0, si - 50)
        before_start = lines[lo:si]
        end = start_idxs[idx + 1] if idx + 1 < n else len(lines)
        h_exprs: list[str] = []
        c_items: list[tuple[str, str]] = []
        for line in lines[si + 1 : end]:
            hm = H_LINE.match(line)
            if hm:
                h_exprs.append(hm.group(2).strip())
                continue
            cm = C_LINE.match(line)
            if cm:
                c_items.append((cm.group(1), cm.group(2).strip()))

        c_by_id = c_lengths_by_cell(c_items)
        avg_c = avg_c_from_cells(c_by_id)

        runs.append(
            {
                "run": f"{idx + 1}/{n}",
                "run_num": idx + 1,
                "total": n,
                "solved": infer_solved(before_start),
                "num_h_equations": len(h_exprs),
                "avg_c_equation_length": avg_c,
                "c_by_id": c_by_id,
            }
        )
    return runs


def parse_file_with_run_headers(text: str) -> list[dict]:
    lines = text.splitlines()
    runs: list[dict] = []
    run_num: int | None = None
    total: int | None = None
    before_start: list[str] = []
    h_exprs: list[str] = []
    c_items: list[tuple[str, str]] = []
    seen_start = False

    def flush() -> None:
        nonlocal h_exprs, c_items, before_start, seen_start
        if run_num is None:
            return
        n_h = len(h_exprs)
        c_by_id = c_lengths_by_cell(c_items)
        avg_c = avg_c_from_cells(c_by_id)
        runs.append(
            {
                "run": f"{run_num}/{total}",
                "run_num": run_num,
                "total": total or 0,
                "solved": infer_solved(before_start),
                "num_h_equations": n_h,
                "avg_c_equation_length": avg_c,
                "c_by_id": c_by_id,
            }
        )
        h_exprs = []
        c_items = []
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
            c_items.append((cm.group(1), cm.group(2).strip()))
            continue

    if run_num is not None:
        flush()

    return runs


def parse_file(text: str) -> list[dict]:
    if RUN_HEADER_HINT.search(text):
        return parse_file_with_run_headers(text)
    return parse_file_with_start_blocks(text)


def collect_files(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        if recursive:
            files = [p for p in path.rglob("*") if p.is_file()]
        else:
            files = [p for p in path.iterdir() if p.is_file()]
        skip = {".ds_store"}
        files = [
            p
            for p in files
            if p.name.lower() not in skip and not p.name.startswith("._")
        ]
        return sorted(files, key=lambda p: str(p).lower())
    raise SystemExit(f"Input not found: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        type=Path,
        help="Log file or directory of log files",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("NumberOfEquations.csv"),
        help="Output CSV (default: ./NumberOfEquations.csv)",
    )
    args = ap.parse_args()

    paths = collect_files(args.input, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"No files under: {args.input}")

    rows: list[dict[str, str | int | float]] = []

    c_headers = [f"c{k}" for k in C_EQ_COLUMN_KEYS]
    fieldnames = [
        "file_id",
        "run",
        "solved",
        "num_h_equations",
        "avg_c_equation_length",
        *c_headers,
    ]

    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        fid = trailing_int_from_filename(p)
        for r in parse_file(text):
            ac = r["avg_c_equation_length"]
            avg_str = "" if isinstance(ac, float) and ac != ac else f"{ac:.6f}"
            c_by_id: dict[str, int] = r["c_by_id"]
            row: dict[str, str | int] = {
                "file_id": fid,
                "run": r["run"],
                "solved": r["solved"],
                "num_h_equations": r["num_h_equations"],
                "avg_c_equation_length": avg_str,
            }
            for k in C_EQ_COLUMN_KEYS:
                row[f"c{k}"] = str(c_by_id[k]) if k in c_by_id else ""
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {args.out.resolve()} ({len(rows)} rows from {len(paths)} file(s))")


if __name__ == "__main__":
    main()
