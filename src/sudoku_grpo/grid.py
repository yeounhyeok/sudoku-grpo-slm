"""Sudoku grid utilities: parsing, text format, validation, scoring primitives.

Canonical internal form is an 81-char string, row-major, digits 1-9, blank = '.'.
"""
from __future__ import annotations

import json
import re

DIGITS = set("123456789")
BLANK = "."


def from_cells(cells) -> str:
    """Flatten list[int] / list[list[int]] (0 = blank) to canonical 81-char form."""
    flat = []
    for c in cells:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        else:
            flat.append(c)
    s = "".join(BLANK if int(v) == 0 else str(int(v)) for v in flat)
    if len(s) != 81:
        raise ValueError(f"expected 81 cells, got {len(s)}")
    return s


def normalize(s: str) -> str:
    """Accept '0' or '.' as blank; return canonical 81-char string."""
    s = s.strip().replace("0", BLANK)
    if len(s) != 81 or any(ch not in DIGITS and ch != BLANK for ch in s):
        raise ValueError("not a valid 81-char sudoku string")
    return s


def parse_grid_literal(text: str) -> str:
    """Parse '[[0,7,...],[...]]' (as stored in GridCorpus) to canonical form."""
    return from_cells(json.loads(text))


def format_grid(s81: str) -> str:
    """9 lines, digits separated by spaces, blanks as '.'."""
    return "\n".join(" ".join(s81[r * 9:(r + 1) * 9]) for r in range(9))


def _units():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
             for br in range(3) for bc in range(3)]
    return rows + cols + boxes


UNITS = _units()


def conflicts(s81: str) -> int:
    """Number of duplicate filled digits summed over all 27 units (0 = no rule violation)."""
    n = 0
    for u in UNITS:
        vals = [s81[i] for i in u if s81[i] in DIGITS]
        n += len(vals) - len(set(vals))
    return n


def is_complete_valid(s81: str) -> bool:
    return len(s81) == 81 and BLANK not in s81 and conflicts(s81) == 0 and set(s81) <= DIGITS


def givens_preserved(puzzle: str, answer: str) -> bool:
    return all(p == BLANK or p == a for p, a in zip(puzzle, answer))


def cell_accuracy(puzzle: str, pred: str, solution: str) -> float:
    """Fraction of originally blank cells predicted correctly."""
    blanks = [i for i, p in enumerate(puzzle) if p == BLANK]
    if not blanks:
        return 1.0
    return sum(pred[i] == solution[i] for i in blanks) / len(blanks)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_answer(text: str) -> str | None:
    """Extract the last <answer>...</answer> block and read 81 digits (spaces/newlines ignored).

    Returns canonical 81-char string, or None if the block is missing or not exactly 81 cells.
    """
    m = _ANSWER_RE.findall(text)
    if not m:
        return None
    body = re.sub(r"[\s|,\-+]", "", m[-1]).replace("0", BLANK)
    if len(body) != 81 or any(ch not in DIGITS and ch != BLANK for ch in body):
        return None
    return body
