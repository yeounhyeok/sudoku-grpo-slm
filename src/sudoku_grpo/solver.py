"""Solver-ordered reasoning traces (single-candidate techniques only).

Order follows the "solver-decomposed reasoning order" idea of Shah et al. (NeurIPS 2024):
at every state take the easiest applicable move, fill ONE cell, recompute.
  1. naked single  - scan cells r1c1..r9c9, first cell with exactly one candidate
  2. hidden single - scan units box1..9, row1..9, col1..9 and digits 1..9; first digit
                     that fits in exactly one empty cell of the unit
Puzzles that stall (need pairs, chains or search) return solved=False.

Step text formats (one step per line):
  L0  r4c5=3
  L1  r4c5=3 naked              | r7c2=6 hidden b7
  L2  r4c5=3 naked r:1258 c:467 b:9
          digits already in the cell's row / NEW digits in its column / NEW digits in its box
      r7c2=6 hidden b7 [r8c1:c r9c3:r]
          every other empty cell of the unit and why 6 cannot go there
          (:r/:c/:b = 6 already in that cell's row/column/box)
"""
from __future__ import annotations

from dataclasses import dataclass, field

ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOXES = [[(3 * (b // 3) + i) * 9 + 3 * (b % 3) + j for i in range(3) for j in range(3)] for b in range(9)]
UNIT_CELLS = {"r": ROWS, "c": COLS, "b": BOXES}
HIDDEN_ORDER = [("b", i) for i in range(9)] + [("r", i) for i in range(9)] + [("c", i) for i in range(9)]
LEVELS = (0, 1, 2)


def box_of(x: int) -> int:
    return (x // 27) * 3 + (x % 9) // 3


def cell_name(x: int) -> str:
    return f"r{x // 9 + 1}c{x % 9 + 1}"


def to_cells(s81: str) -> list[int]:
    return [0 if ch in ".0" else int(ch) for ch in s81]


def seen(g: list[int], x: int) -> tuple[set[int], set[int], set[int]]:
    """Digits present in the row / column / box of cell x."""
    R = {g[y] for y in ROWS[x // 9]} - {0}
    C = {g[y] for y in COLS[x % 9]} - {0}
    B = {g[y] for y in BOXES[box_of(x)]} - {0}
    return R, C, B


def candidates(g: list[int], x: int) -> set[int]:
    R, C, B = seen(g, x)
    return set(range(1, 10)) - R - C - B


def blocker(g: list[int], y: int, d: int) -> str | None:
    """Why digit d cannot go in empty cell y: 'r'/'c'/'b' (first unit, in that order) or None."""
    R, C, B = seen(g, y)
    return "r" if d in R else "c" if d in C else "b" if d in B else None


@dataclass
class Step:
    cell: int
    digit: int
    rule: str                      # "naked" | "hidden"
    unit: tuple[str, int] | None = None              # hidden: ("b", 6) = box 7
    seen: tuple[str, str, str] | None = None          # naked: (row, new col, new box) digit strings
    blocks: list[tuple[int, str]] = field(default_factory=list)  # hidden: (cell, 'r'|'c'|'b')

    def text(self, level: int) -> str:
        s = f"{cell_name(self.cell)}={self.digit}"
        if level >= 1:
            s += " naked" if self.rule == "naked" else f" hidden {self.unit[0]}{self.unit[1] + 1}"
        if level >= 2:
            if self.rule == "naked":
                s += f" r:{self.seen[0]} c:{self.seen[1]} b:{self.seen[2]}"
            else:
                s += " [" + " ".join(f"{cell_name(y)}:{w}" for y, w in self.blocks) + "]"
        return s


def _digits(S: set[int]) -> str:
    return "".join(map(str, sorted(S)))


def next_step(g: list[int]) -> Step | None:
    for x in range(81):
        if g[x]:
            continue
        R, C, B = seen(g, x)
        cand = set(range(1, 10)) - R - C - B
        if len(cand) == 1:
            return Step(x, cand.pop(), "naked", seen=(_digits(R), _digits(C - R), _digits(B - R - C)))
    for kind, i in HIDDEN_ORDER:
        u = UNIT_CELLS[kind][i]
        present = {g[y] for y in u}
        for d in range(1, 10):
            if d in present:
                continue
            free, blocks = [], []
            for y in u:
                if g[y]:
                    continue
                w = blocker(g, y, d)
                (blocks.append((y, w)) if w else free.append(y))
            if len(free) == 1:
                return Step(free[0], d, "hidden", unit=(kind, i), blocks=blocks)
    return None


def solve_steps(puzzle81: str) -> tuple[bool, list[Step], str]:
    """Run the single-candidate solver. Returns (solved, steps, final 81-char grid)."""
    g = to_cells(puzzle81)
    steps = []
    while (st := next_step(g)) is not None:
        g[st.cell] = st.digit
        steps.append(st)
    return all(g), steps, "".join(str(v) if v else "." for v in g)


def trace_text(steps: list[Step], level: int) -> str:
    return "\n".join(s.text(level) for s in steps)
