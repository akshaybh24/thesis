#!/usr/bin/env python3
"""
Read a mat_mult log (.out). After the last occurrence of \"solved!\", extract
h/c equations; optionally verify them with random 5x5 matrices (100 trials).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

Matrix5 = list[list[int]]

STOP_MARKER = "=== TRUE C MATRIX ==="
H_LINE = re.compile(r"^\s*h\d+\s*:.+")
C_LINE = re.compile(r"^\s*c\d+\s*:.+")

# a45 / b32: one-based row and column (matches rust/mat_mult eval_single_h)
A_TERM = re.compile(r"a(\d)(\d)")
B_TERM = re.compile(r"b(\d)(\d)")
H_ID = re.compile(r"^h\d+$")


def extract_after_last_solved(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    solved_indices = [i for i, line in enumerate(lines) if "solved!" in line.lower()]
    if not solved_indices:
        raise ValueError('No line containing "solved!" found.')

    start = solved_indices[-1] + 1
    h_out: list[str] = []
    c_out: list[str] = []

    for line in lines[start:]:
        if line.strip().upper().startswith("===") and "TRUE C" in line.upper():
            break
        s = line.rstrip()
        if not s.strip():
            continue
        if H_LINE.match(s):
            h_out.append(s)
        elif C_LINE.match(s):
            c_out.append(s)

    return h_out, c_out


def generate_random_matrices_5x5(
    rng: random.Random | None = None,
    low: int = -50,
    high: int = 50,
    *,
    silent: bool = False,
) -> tuple[Matrix5, Matrix5, Matrix5]:
    """
    Build random 5x5 integer matrices A and B, return (A, B, C) with C = A @ B.
    Unless ``silent`` is True, prints A, B, and C.
    """
    if rng is None:
        rng = random.Random()

    def rand_mat() -> Matrix5:
        return [[rng.randint(low, high) for _ in range(5)] for _ in range(5)]

    a = rand_mat()
    b = rand_mat()
    c: Matrix5 = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            c[i][j] = sum(a[i][k] * b[k][j] for k in range(5))

    if not silent:
        print("A =")
        for row in a:
            print(" ", row)
        print("B =")
        for row in b:
            print(" ", row)
        print("C = A @ B =")
        for row in c:
            print(" ", row)

    return a, b, c


def _subst_ab(
    expr: str,
    a: Matrix5,
    b: Matrix5,
    *,
    use_a: bool,
) -> str:
    """Replace aXY / bXY with integer literals (Rust-style 1-based indices)."""

    def sub_a(m: re.Match[str]) -> str:
        r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
        return str(a[r][c])

    def sub_b(m: re.Match[str]) -> str:
        r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
        return str(b[r][c])

    if use_a:
        if B_TERM.search(expr):
            raise ValueError(f"b term in A-sum: {expr!r}")
        return A_TERM.sub(sub_a, expr)
    if A_TERM.search(expr):
        raise ValueError(f"a term in B-sum: {expr!r}")
    return B_TERM.sub(sub_b, expr)


def _eval_sum_expr(expr: str, a: Matrix5, b: Matrix5, *, use_a: bool) -> int:
    s = _subst_ab(expr, a, b, use_a=use_a)
    try:
        v = eval(s, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"bad sum after substitution {expr!r} -> {s!r}") from e
    if not isinstance(v, int):
        raise ValueError(f"expected int, got {type(v).__name__} for {expr!r}")
    return v


def split_h_product(rhs: str) -> tuple[str, str]:
    """Split `(Aexpr) * (Bexpr)` using balanced parentheses."""
    rhs = rhs.strip()
    if not rhs.startswith("("):
        raise ValueError(f"h rhs must start with '(': {rhs!r}")

    depth = 0
    i = 0
    while i < len(rhs):
        c = rhs[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inner_a = rhs[1:i]
                rest = rhs[i + 1 :].strip()
                if not rest.startswith("*"):
                    raise ValueError(f"expected '*' after first group: {rhs!r}")
                rest = rest[1:].strip()
                if not rest.startswith("("):
                    raise ValueError(f"expected '(' for B-sum: {rhs!r}")
                depth2 = 0
                for j, c2 in enumerate(rest):
                    if c2 == "(":
                        depth2 += 1
                    elif c2 == ")":
                        depth2 -= 1
                        if depth2 == 0:
                            inner_b = rest[1:j]
                            tail = rest[j + 1 :].strip()
                            if tail:
                                raise ValueError(f"trailing junk in h rhs: {tail!r}")
                            return inner_a, inner_b
                raise ValueError("unbalanced B-sum parens")
        i += 1
    raise ValueError("unbalanced A-sum parens")


def parse_h_line(line: str) -> tuple[str, str]:
    """Return (h_id, rhs) where rhs is `(a...) * (b...)`."""
    s = line.strip()
    m = re.match(r"^(h\d+)\s*:\s*(.+)$", s)
    if not m:
        raise ValueError(f"bad h line: {line!r}")
    return m.group(1), m.group(2).strip()


def eval_h_line(rhs: str, a: Matrix5, b: Matrix5) -> int:
    inner_a, inner_b = split_h_product(rhs)
    sa = _eval_sum_expr(inner_a, a, b, use_a=True)
    sb = _eval_sum_expr(inner_b, a, b, use_a=False)
    return sa * sb


def parse_c_line(line: str) -> tuple[tuple[int, int], str]:
    """Return ((row0, col0), rhs) for c_ij with i,j in 1..5."""
    s = line.strip()
    m = re.match(r"^c([1-5])([1-5])\s*:\s*(.+)$", s)
    if not m:
        raise ValueError(f"bad c line (expected c11..c55): {line!r}")
    i = int(m.group(1)) - 1
    j = int(m.group(2)) - 1
    return (i, j), m.group(3).strip()


def eval_c_rhs(rhs: str, h_vals: dict[str, int]) -> int:
    """Evaluate `h30 + h26 - h8 ...` using h-values."""
    expr = re.sub(r"\s+", "", rhs)
    if not expr:
        return 0
    if expr[0] not in "+-":
        expr = "+" + expr
    total = 0
    for sign, hid in re.findall(r"([+-])(h\d+)", expr):
        if hid not in h_vals:
            raise KeyError(f"missing {hid} in h_vals")
        v = h_vals[hid]
        if sign == "-":
            total -= v
        else:
            total += v
    return total


def verify_equations_once(
    a: Matrix5,
    b: Matrix5,
    c_true: Matrix5,
    h_lines: list[str],
    c_lines: list[str],
) -> bool:
    """
    Evaluate all h lines, then all c lines; return True iff C from c matches c_true.
    """
    h_vals: dict[str, int] = {}
    for line in h_lines:
        hid, rhs = parse_h_line(line)
        h_vals[hid] = eval_h_line(rhs, a, b)

    c_computed = [[0] * 5 for _ in range(5)]
    for line in c_lines:
        (i, j), rhs = parse_c_line(line)
        c_computed[i][j] = eval_c_rhs(rhs, h_vals)

    return c_computed == c_true


def run_verification_trials(
    h_lines: list[str],
    c_lines: list[str],
    trials: int = 100,
    seed: int | None = None,
    *,
    print_each_trial: bool = False,
) -> tuple[int, int]:
    """
    Call ``generate_random_matrices_5x5`` ``trials`` times, then
    ``verify_equations_once`` each time. Returns ``(passed_count, trials)``.
    """
    rng = random.Random(seed)
    passed = 0
    for t in range(trials):
        if print_each_trial:
            print(f"--- trial {t + 1}/{trials} ---")
        a, b, c_true = generate_random_matrices_5x5(
            rng=rng, silent=not print_each_trial
        )
        if verify_equations_once(a, b, c_true, h_lines, c_lines):
            passed += 1

    return passed, trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to log file (if omitted, you will be prompted)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="After extracting equations, run random 5x5 verification trials",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of random (A,B) checks when using --verify (default: 100)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --verify (optional, for reproducibility)",
    )
    ap.add_argument(
        "--show-matrices",
        action="store_true",
        help="With --verify, print A,B,C for each trial (very verbose)",
    )
    args = ap.parse_args()

    path = args.input
    if path is None:
        raw = input("Path to input file: ").strip().strip('"').strip("'")
        if not raw:
            sys.exit("No path given.")
        path = Path(raw)

    if not path.is_file():
        sys.exit(f"Not a file: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        h_lines, c_lines = extract_after_last_solved(text)
    except ValueError as e:
        sys.exit(str(e))

    if not h_lines and not c_lines:
        sys.exit(
            f'After last "solved!" in {path}, no hN: / cN: lines found '
            f'(expected before "{STOP_MARKER.lower()}").'
        )

    print(f"# Source: {path.resolve()}")
    print(f"# h-equations: {len(h_lines)}, c-equations: {len(c_lines)}")
    print()

    if args.verify:
        passed, n = run_verification_trials(
            h_lines,
            c_lines,
            trials=args.trials,
            seed=args.seed,
            print_each_trial=args.show_matrices,
        )
        print(f"Verification: {passed}/{n} trials matched C = A @ B")
        if passed < n:
            sys.exit(1)
        return

    for line in h_lines:
        print(line)
    for line in c_lines:
        print(line)


if __name__ == "__main__":
    main()
