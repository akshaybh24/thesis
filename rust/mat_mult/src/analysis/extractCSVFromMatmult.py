#!/usr/bin/env python3
"""
Parse mat_mult (or similar) logs grouped by [run X/Y].

For each run:
  - Read generation 4999999 line for fitness difference, fitness_cells, num h, temp.
  - Read num triples / num mutations from mat-size style lines (last occurrence per run).
  - Count h-equations (hNN:) and c-equations (cNN:).
  - Per h-equation: aNN / bNN token counts (min/avg/max and totals).
  - Per c-equation: average h-reference count.

Writes CSV (default: results_run.csv). Pass a file, directory, or stdin (-r for recursive dirs).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

RUN_HEADER = re.compile(r"\[run\s+(\d+)/(\d+)\]")
H_LINE = re.compile(r"^\s*h(\d+):\s*(.+?)\s*$")
C_LINE = re.compile(r"^\s*c(\d+):\s*(.+?)\s*$")

# Base column order (user-requested). Optional "file" prepended when multiple inputs.
CSV_FIELDS = [
    "run",
    "number of triples",
    "number of mutations",
    "fitness difference",
    "fitness_cells",
    "num h",
    "temp",
    "num_h_equations",
    "num_c_equations",
    "a_terms_per_h_equation (min/avg/max)",
    "b_terms_per_h_equation (min/avg/max)",
    "total_a_terms_all_h",
    "total_b_terms_all_h",
    "c_equation_avg_h_ref_count",
]


def is_final_gen_line(line: str) -> bool:
    if "4999999" not in line:
        return False
    return "Generation 4999999" in line or "Gen 4999999" in line


def parse_gen_line_metrics(line: str | None) -> dict[str, str]:
    """Extract scalar fields from the Generation 4999999 log line."""
    out = {
        "fitness difference": "",
        "fitness_cells": "",
        "num h": "",
        "temp": "",
    }
    if not line:
        return out
    m = re.search(r"fitness difference:\s*([^,]+)", line, re.I)
    if m:
        out["fitness difference"] = m.group(1).strip()
    m = re.search(r"fitness_cells:\s*([^,]+)", line, re.I)
    if m:
        out["fitness_cells"] = m.group(1).strip()
    m = re.search(r"num\s*h:\s*([^,]+)", line, re.I)
    if m:
        out["num h"] = m.group(1).strip()
    m = re.search(r"temp:\s*(\S+)", line, re.I)
    if m:
        out["temp"] = m.group(1).strip()
    return out


def count_a_b(expr: str) -> tuple[int, int]:
    a_n = len(re.findall(r"\ba\d+\b", expr))
    b_n = len(re.findall(r"\bb\d+\b", expr))
    return a_n, b_n


def c_h_ref_count(rhs: str) -> int:
    return len(re.findall(r"\bh\d+\b", rhs.strip()))


def update_triples_mutations(r: dict, line: str) -> None:
    """Set num triples / num mutations from a log line (last match in run wins)."""
    m = re.search(r"num\s+triples:\s*(\d+)", line, re.I)
    if m:
        r["num_triples"] = m.group(1)
    m = re.search(r"num\s+mutations:\s*(\d+)", line, re.I)
    if m:
        r["num_mutations"] = m.group(1)


def parse_runs(text: str) -> list[dict]:
    lines = text.splitlines()
    runs: list[dict] = []
    current: dict | None = None

    def start_run(run_num: int, total: int) -> dict:
        return {
            "run_num": run_num,
            "total": total,
            "gen_line": None,
            "num_triples": None,
            "num_mutations": None,
            "h_exprs": [],
            "c_rhs": [],
        }

    for line in lines:
        m = RUN_HEADER.search(line)
        if m:
            run_num = int(m.group(1))
            total = int(m.group(2))
            if current is not None and current["run_num"] != run_num:
                runs.append(current)
                current = None
            if current is None:
                current = start_run(run_num, total)
            if is_final_gen_line(line):
                current["gen_line"] = line.strip()
            update_triples_mutations(current, line)
            continue

        if current is None:
            continue

        if is_final_gen_line(line):
            current["gen_line"] = line.strip()

        update_triples_mutations(current, line)

        hm = H_LINE.match(line)
        if hm:
            current["h_exprs"].append(hm.group(2).strip())
            continue

        cm = C_LINE.match(line)
        if cm:
            current["c_rhs"].append(cm.group(2).strip())
            continue

    if current is not None:
        runs.append(current)

    return runs


def run_to_csv_row(r: dict, *, source_file: str = "") -> dict[str, str]:
    metrics = parse_gen_line_metrics(r.get("gen_line"))
    h_exprs = r["h_exprs"]
    c_rhs = r["c_rhs"]
    n_h = len(h_exprs)
    n_c = len(c_rhs)

    if h_exprs:
        per_h = [count_a_b(e) for e in h_exprs]
        all_a = [t[0] for t in per_h]
        all_b = [t[1] for t in per_h]
        a_triplet = (
            f"{min(all_a)}/{statistics.mean(all_a):.4f}/{max(all_a)}"
        )
        b_triplet = (
            f"{min(all_b)}/{statistics.mean(all_b):.4f}/{max(all_b)}"
        )
        total_a = str(sum(all_a))
        total_b = str(sum(all_b))
    else:
        a_triplet = ""
        b_triplet = ""
        total_a = ""
        total_b = ""

    if c_rhs:
        refs = [c_h_ref_count(s) for s in c_rhs]
        c_avg = f"{statistics.mean(refs):.4f}"
    else:
        c_avg = ""

    nt = r.get("num_triples")
    nm = r.get("num_mutations")
    row: dict[str, str] = {
        "run": f"{r['run_num']}/{r['total']}",
        "number of triples": "" if nt is None else str(nt),
        "number of mutations": "" if nm is None else str(nm),
        "fitness difference": metrics["fitness difference"],
        "fitness_cells": metrics["fitness_cells"],
        "num h": metrics["num h"],
        "temp": metrics["temp"],
        "num_h_equations": str(n_h),
        "num_c_equations": str(n_c),
        "a_terms_per_h_equation (min/avg/max)": a_triplet,
        "b_terms_per_h_equation (min/avg/max)": b_triplet,
        "total_a_terms_all_h": total_a,
        "total_b_terms_all_h": total_b,
        "c_equation_avg_h_ref_count": c_avg,
    }
    if source_file:
        row["file"] = source_file
    return row


def collect_input_files(path: Path, *, recursive: bool) -> list[Path]:
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
        nargs="?",
        type=Path,
        default=None,
        help="Log file or directory of log files (default: stdin)",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="If input is a directory, also read files in subdirectories",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("results_run.csv"),
        help="Output CSV (default: ./results_run.csv)",
    )
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    include_file = False

    if args.input is None:
        runs = parse_runs(sys.stdin.read())
        for r in runs:
            rows.append(run_to_csv_row(r))
    else:
        if not args.input.exists():
            raise SystemExit(f"Input not found: {args.input}")
        paths = collect_input_files(args.input, recursive=args.recursive)
        if not paths:
            raise SystemExit(f"No files to read under: {args.input}")
        include_file = args.input.is_dir() or len(paths) > 1
        for p in paths:
            text = p.read_text(encoding="utf-8", errors="replace")
            runs = parse_runs(text)
            label = str(p) if include_file else ""
            for r in runs:
                rows.append(run_to_csv_row(r, source_file=label))

    fieldnames = (["file"] + CSV_FIELDS) if include_file else list(CSV_FIELDS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for row in rows:
            if not include_file:
                row = {k: row[k] for k in CSV_FIELDS}
            w.writerow(row)

    print(f"Wrote {args.out.resolve()} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
