#!/usr/bin/env python3
"""
Find lines ``4999999,<fitness_difference>,<different_cells>`` in each file, then
read the following ``START ALGORITHM`` section and compute:

  - Columns 0–1: Triples, Mutations (from basename ``{triples}mat…{mutations}mut{run}``)
  - Columns 2–4 of the previous CSV layout are dropped (``file_id``, ``block``, and
    ``fitness_difference`` for one input file; ``source_file``, ``file_id``, and ``block``
    when reading multiple files — then ``fitness_difference`` shifts to column 2).
  - ``different_cells`` (from the ``4999999,`` CSV line), then h/c stats
  - num_h_equations (count of ``hNN:`` lines)
  - avg_c_equation_length (mean count of ``hNN`` references in each ``cNN:`` RHS)

Writes one CSV row per matching block per file. Blank lines between the CSV line
and ``START ALGORITHM`` are allowed.

Example::

  python3 extract_4999999_stats.py /path/to/logs -o summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

H_LINE = re.compile(r"^\s*h(\d+)\s*:\s*(.+?)\s*$")
C_LINE = re.compile(r"^\s*c(\d+)\s*:\s*(.+?)\s*$")
START_ALGO = re.compile(r"START\s+ALGORITHM", re.I)
FINAL_GEN_LOG = re.compile(
    r"Generation\s+4999999,\s*fitness difference:\s*([^,]+),\s*fitness_cells:\s*([^,]+)",
    re.I,
)
H_EQ_LOG = re.compile(
    r"h-equations:\s*(\d+),\s*avg c-equation length:\s*([0-9.]+)",
    re.I,
)
NUM_TRIPLES_LOG = re.compile(r"num triples:\s*(\d+)", re.I)
NUM_MUTATIONS_LOG = re.compile(r"num mutations:\s*(\d+)", re.I)
RUN_TAG_LOG = re.compile(r"\[run\s+(\d+)\s*/\s*\d+\]", re.I)
RUN_HEADER_LOG = re.compile(r"\bRun\s+(\d+)\s+of\s+\d+\b", re.I)

FINAL_GEN_PREFIX = "4999999,"

# Matches mat_mult output basenames, e.g. ``10matnocappos1mut5`` or
# ``5MNoChanges10matnocappos1mut1`` → triples ``10``, mutations ``1``.
STEM_TRIPLES = re.compile(r"(\d+)mat", re.I)
STEM_MUTATIONS = re.compile(r"(\d+)mut", re.I)

ARCH_DIFF_BY_PREFIX = {
    "matmult_21589422": "temperature 0.1, mutation 8 left out, 1/5 chance mutation 1 and 7 (1/10th for others)",
    "matmult_21589068": "temperature 0.1",
    "matmult_21587355": "mutation 8 left out",
}


def c_h_ref_count(rhs: str) -> int:
    return len(re.findall(r"\bh\d+\b", rhs.strip()))


def trailing_int_from_filename(path: Path) -> str:
    m = re.search(r"(\d+)$", path.stem)
    return m.group(1) if m else ""


def triples_and_mutations_from_stem(stem: str) -> tuple[str, str]:
    """``{triples}mat...{mutations}mut{run}`` as produced by mat_mult (``main.rs``)."""
    tm = STEM_TRIPLES.search(stem)
    triples = tm.group(1) if tm else ""
    mm = STEM_MUTATIONS.search(stem)
    mutations = mm.group(1) if mm else ""
    return triples, mutations


def triples_and_mutations_from_text(text: str) -> tuple[str, str]:
    triples_match = NUM_TRIPLES_LOG.search(text)
    triples = triples_match.group(1) if triples_match else ""
    mutations_match = NUM_MUTATIONS_LOG.search(text)
    mutations = mutations_match.group(1) if mutations_match else ""
    return triples, mutations


def architectural_difference_from_name(path: Path) -> str:
    stem = path.stem.lower()
    for prefix, description in ARCH_DIFF_BY_PREFIX.items():
        if stem.startswith(prefix):
            return description
    return ""


def parse_text_blocks(text: str) -> list[dict]:
    lines = text.splitlines()
    out: list[dict] = []
    i = 0
    block_idx = 0
    current_run = ""
    while i < len(lines):
        raw_line = lines[i]
        raw = raw_line.strip()
        run_tag_match = RUN_TAG_LOG.search(raw_line)
        if run_tag_match:
            current_run = run_tag_match.group(1).strip()
        else:
            run_header_match = RUN_HEADER_LOG.search(raw_line)
            if run_header_match:
                current_run = run_header_match.group(1).strip()
        gen_match = FINAL_GEN_LOG.search(raw_line)

        if not raw.startswith(FINAL_GEN_PREFIX) and not gen_match:
            i += 1
            continue

        if gen_match:
            fitness = gen_match.group(1).strip()
            cells = gen_match.group(2).strip()
            h_n = 0
            avg_s = ""

            # Newer logs print h/c stats on a nearby "h-equations" line.
            for j in range(i - 1, max(-1, i - 8), -1):
                hm = H_EQ_LOG.search(lines[j])
                if hm:
                    h_n = int(hm.group(1))
                    avg_s = hm.group(2).strip()
                    break

            block_idx += 1
            out.append(
                {
                    "block": str(block_idx),
                    "run": current_run,
                    "fitness_difference": fitness,
                    "different_cells": cells,
                    "num_h_equations": h_n,
                    "avg_c_equation_length": avg_s,
                }
            )
            i += 1
            continue

        rest = raw[len(FINAL_GEN_PREFIX) :]
        parts = rest.split(",", 1)
        fitness = parts[0].strip() if parts else ""
        cells = parts[1].strip() if len(parts) > 1 else ""

        j = i + 1
        while j < len(lines) and not START_ALGO.search(lines[j]):
            j += 1
        if j >= len(lines):
            i += 1
            continue

        h_n = 0
        c_rhss: list[str] = []
        k = j + 1
        while k < len(lines):
            s = lines[k].strip()
            if s.startswith(FINAL_GEN_PREFIX):
                break
            hm = H_LINE.match(lines[k])
            if hm:
                h_n += 1
                k += 1
                continue
            cm = C_LINE.match(lines[k])
            if cm:
                c_rhss.append(cm.group(2).strip())
                k += 1
                continue
            k += 1

        if c_rhss:
            lens = [c_h_ref_count(s) for s in c_rhss]
            avg_c = statistics.mean(lens)
            avg_s = f"{avg_c:.6f}"
        else:
            avg_s = ""

        block_idx += 1
        out.append(
            {
                "block": str(block_idx),
                "run": current_run,
                "fitness_difference": fitness,
                "different_cells": cells,
                "num_h_equations": h_n,
                "avg_c_equation_length": avg_s,
            }
        )
        i = k if k < len(lines) and lines[k].strip().startswith(FINAL_GEN_PREFIX) else i + 1
    return out


def mean_numeric(values: list[str]) -> str:
    nums: list[float] = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return ""
    return f"{statistics.mean(nums):.6f}"


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
        default=Path("generation_4999999_summary.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    paths = collect_files(args.input, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"No files under: {args.input}")

    multi = len(paths) > 1
    rows: list[dict[str, str]] = []
    grouped_values: dict[tuple[str, str], dict[str, list[str]]] = {}
    grouped_arch: dict[tuple[str, str], set[str]] = {}

    # Columns 2–4 (0-based) of the old layout are omitted: file_id, block, fitness_difference
    # (or for multi-input: source_file, file_id, block — fitness_difference then appears at index 2).
    leading = ["Triples", "Mutations", "Architectural Difference"]
    tail = [
        "different_cells",
        "num_h_equations",
        "avg_c_equation_length",
    ]
    fieldnames = leading + (["fitness_difference"] + tail if multi else tail)

    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        blocks = parse_text_blocks(text)
        if not blocks:
            continue
        triples, mutations = triples_and_mutations_from_stem(p.stem)
        architectural_difference = architectural_difference_from_name(p)
        if not triples or not mutations:
            # Newer output filenames may not encode triples/mutations.
            t2, m2 = triples_and_mutations_from_text(text)
            triples = triples or t2
            mutations = mutations or m2
        avg_different_cells = mean_numeric([b["different_cells"] for b in blocks])
        avg_num_h_equations = mean_numeric([str(b["num_h_equations"]) for b in blocks])
        avg_c_equation_length = mean_numeric([b["avg_c_equation_length"] for b in blocks])
        avg_fitness_difference = mean_numeric([b["fitness_difference"] for b in blocks])

        key = (triples, mutations)
        if key not in grouped_values:
            grouped_values[key] = {
                "different_cells": [],
                "num_h_equations": [],
                "avg_c_equation_length": [],
                "fitness_difference": [],
            }
            grouped_arch[key] = set()

        grouped_values[key]["different_cells"].append(avg_different_cells)
        grouped_values[key]["num_h_equations"].append(avg_num_h_equations)
        grouped_values[key]["avg_c_equation_length"].append(avg_c_equation_length)
        grouped_values[key]["fitness_difference"].append(avg_fitness_difference)
        if architectural_difference:
            grouped_arch[key].add(architectural_difference)

    for (triples, mutations), vals in sorted(grouped_values.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        arch_set = grouped_arch.get((triples, mutations), set())
        arch = " | ".join(sorted(arch_set)) if arch_set else ""
        row = {
            "Triples": triples,
            "Mutations": mutations,
            "Architectural Difference": arch,
            "different_cells": mean_numeric(vals["different_cells"]),
            "num_h_equations": mean_numeric(vals["num_h_equations"]),
            "avg_c_equation_length": mean_numeric(vals["avg_c_equation_length"]),
        }
        if multi:
            row["fitness_difference"] = mean_numeric(vals["fitness_difference"])
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
