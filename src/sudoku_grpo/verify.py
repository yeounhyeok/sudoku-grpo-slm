"""Step-level verifier for solver traces (L0/L1/L2). Used for SFT data checks, analysis,
and later as the GRPO step reward (process signal checked by code, not by a judge model).

Each line is replayed on the puzzle state in order. Per step:
  parsed    line matches the step grammar
  legal     target cell is empty and the digit does not clash with its row/col/box
  correct   digit equals the solution digit (only if a solution is given)
  claim     rule tag holds in the current state (naked: exactly one candidate;
            hidden uK: digit fits nowhere else in that unit)          [L1, L2]
  evidence  the listed justification is exactly true in the current state  [L2, L3]
L3 lines (evidence before the digit) are parsed into the same fields; the rule is implied by the shape.
Legal steps are applied to the state; illegal ones are skipped (and counted).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .solver import UNIT_CELLS, blocker, box_of, candidates, seen, to_cells

STEP_RE = re.compile(
    r"^r(?P<r>[1-9])c(?P<c>[1-9])=(?P<d>[1-9])"
    r"(?:\s+(?P<rule>naked|hidden\s+(?P<uk>[rcb])(?P<ui>[1-9])))?"
    r"(?:\s+(?P<ev>.+?))?\s*$")
L3_NAKED = re.compile(r"^r(?P<r>[1-9])c(?P<c>[1-9])\s+(?P<ev>r:\d*\s+c:\d*\s+b:\d*)\s*=>\s*(?P<d>[1-9])\s*$")
L3_HIDDEN = re.compile(r"^(?P<uk>[rcb])(?P<ui>[1-9])\s+d(?P<d>[1-9])\s+(?P<ev>\[[^\]]*\])\s*=>\s*r(?P<r>[1-9])c(?P<c>[1-9])\s*$")
NAKED_EV = re.compile(r"^r:(?P<r>\d*)\s+c:(?P<c>\d*)\s+b:(?P<b>\d*)$")
HIDDEN_EV = re.compile(r"^\[(?P<body>[^\]]*)\]$")
BLOCK_RE = re.compile(r"^r([1-9])c([1-9]):([rcb])$")


def _digits(S) -> str:
    return "".join(map(str, sorted(S)))


def entailed(g, x: int, d: int) -> bool:
    """d at x is forced by a single: x has only candidate d, or d fits nowhere else in one of x's units."""
    if g[x] or d not in candidates(g, x):
        return False
    if candidates(g, x) == {d}:
        return True
    r, c, b = x // 9, x % 9, box_of(x)
    for unit in (UNIT_CELLS["r"][r], UNIT_CELLS["c"][c], UNIT_CELLS["b"][b]):
        if all(y == x or g[y] or blocker(g, y, d) for y in unit):
            return True
    return False


@dataclass
class StepCheck:
    line: str
    parsed: bool = False
    legal: bool = False
    correct: bool | None = None
    claim: bool | None = None      # None = no rule tag on this line
    evidence: bool | None = None   # None = no evidence on this line
    ev_score: float | None = None  # graded evidence in [0, 1] (share of evidence fields that are true)
    sound: bool = False            # legal AND logically forced: stated rule claim true (L1-L3) or entailed (L0)

    def ok(self, level: int) -> bool:
        if not (self.parsed and self.legal):
            return False
        if self.correct is False:
            return False
        if level >= 1 and self.claim is not True:
            return False
        if level >= 2 and self.evidence is not True:
            return False
        return True


@dataclass
class TraceCheck:
    steps: list[StepCheck] = field(default_factory=list)
    final: str = ""                # puzzle state after replaying legal steps

    def summary(self, level: int) -> dict:
        n = len(self.steps)
        f = lambda k: (sum(1 for s in self.steps if getattr(s, k) is True) / n) if n else 0.0
        first_bad = next((i for i, s in enumerate(self.steps) if not s.ok(level)), None)
        return {"n_steps": n, "parsed": f("parsed"), "legal": f("legal"), "correct": f("correct"),
                "claim": f("claim"), "evidence": f("evidence"),
                "step_ok": (sum(s.ok(level) for s in self.steps) / n) if n else 0.0,
                "all_ok": n > 0 and first_bad is None, "first_bad": first_bad,
                "filled": "." not in self.final}


def check_naked(g, x, d, ev) -> tuple[bool, bool | None, float | None]:
    claim = candidates(g, x) == {d}
    if ev is None:
        return claim, None, None
    m = NAKED_EV.match(ev)
    if not m:
        return claim, False, 0.0
    R, C, B = seen(g, x)
    want = (R, C - R, B - R - C)
    hits = [len(set(f)) == len(f) and {int(ch) for ch in f} == w for f, w in zip((m["r"], m["c"], m["b"]), want)]
    return claim, all(hits), sum(hits) / 3


def check_hidden(g, x, d, kind, i, ev) -> tuple[bool, bool | None, float | None]:
    unit = UNIT_CELLS[kind][i]
    others = [y for y in unit if y != x and not g[y]]
    claim = x in unit and all(blocker(g, y, d) for y in others)
    if ev is None:
        return claim, None, None
    m = HIDDEN_EV.match(ev)
    if not m:
        return claim, False, 0.0
    listed = {}
    for tok in m["body"].split():
        b = BLOCK_RE.match(tok)
        if not b:
            return claim, False, 0.0
        y = (int(b[1]) - 1) * 9 + int(b[2]) - 1
        if y in listed:
            return claim, False, 0.0
        listed[y] = b[3]
    good = 0
    for y, w in listed.items():  # the stated reason must be true (any true reason is accepted)
        unit_of_y = {"r": UNIT_CELLS["r"][y // 9], "c": UNIT_CELLS["c"][y % 9], "b": UNIT_CELLS["b"][box_of(y)]}[w]
        good += y in others and d in {g[z] for z in unit_of_y}
    denom = max(len(others), len(listed))
    score = good / denom if denom else 1.0
    return claim, score == 1.0 and set(listed) == set(others), score


def check_trace(puzzle81: str, trace: str, solution81: str | None = None, apply: str = "legal") -> TraceCheck:
    """apply="legal": every legal step is placed on the board (SFT data checks, analysis).
    apply="sound": only sound steps are placed, so an unjustified guess cannot corrupt the board that
    later steps are judged on (reward)."""
    g = to_cells(puzzle81)
    sol = to_cells(solution81) if solution81 else None
    out = TraceCheck()
    for line in (ln.strip() for ln in trace.splitlines()):
        if not line:
            continue
        sc = StepCheck(line)
        out.steps.append(sc)
        m = STEP_RE.match(line)
        if m:
            rule = "naked" if m["rule"] == "naked" else ("hidden" if m["rule"] else None)
            uk, ui = m["uk"], m["ui"]
        elif (m := L3_NAKED.match(line)):
            rule, uk, ui = "naked", None, None
        elif (m := L3_HIDDEN.match(line)):
            rule, uk, ui = "hidden", m["uk"], m["ui"]
        else:
            continue
        sc.parsed = True
        x, d = (int(m["r"]) - 1) * 9 + int(m["c"]) - 1, int(m["d"])
        if g[x] or d in set().union(*seen(g, x)):
            continue
        sc.legal = True
        if sol:
            sc.correct = sol[x] == d
        ev = m["ev"]
        if rule == "naked":
            sc.claim, sc.evidence, sc.ev_score = check_naked(g, x, d, ev)
        elif rule == "hidden":
            sc.claim, sc.evidence, sc.ev_score = check_hidden(g, x, d, uk, int(ui) - 1, ev)
        elif ev is not None:
            sc.evidence = False  # evidence without a rule tag is malformed
        sc.sound = bool(sc.claim) if sc.claim is not None else (ev is None and entailed(g, x, d))
        if apply == "legal" or sc.sound:
            g[x] = d
    out.final = "".join(str(v) if v else "." for v in g)
    return out
