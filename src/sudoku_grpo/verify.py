"""Step-level verifier for solver traces (L0/L1/L2). Used for SFT data checks, analysis,
and later as the GRPO step reward (process signal checked by code, not by a judge model).

Each line is replayed on the puzzle state in order. Per step:
  parsed    line matches the step grammar
  legal     target cell is empty and the digit does not clash with its row/col/box
  correct   digit equals the solution digit (only if a solution is given)
  claim     rule tag holds in the current state (naked: exactly one candidate;
            hidden uK: digit fits nowhere else in that unit)          [L1, L2]
  evidence  the listed justification is exactly true in the current state  [L2]
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
NAKED_EV = re.compile(r"^r:(?P<r>\d*)\s+c:(?P<c>\d*)\s+b:(?P<b>\d*)$")
HIDDEN_EV = re.compile(r"^\[(?P<body>[^\]]*)\]$")
BLOCK_RE = re.compile(r"^r([1-9])c([1-9]):([rcb])$")


def _digits(S) -> str:
    return "".join(map(str, sorted(S)))


@dataclass
class StepCheck:
    line: str
    parsed: bool = False
    legal: bool = False
    correct: bool | None = None
    claim: bool | None = None      # None = no rule tag on this line
    evidence: bool | None = None   # None = no evidence on this line

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


def check_naked(g, x, d, ev) -> tuple[bool, bool | None]:
    claim = candidates(g, x) == {d}
    if ev is None:
        return claim, None
    m = NAKED_EV.match(ev)
    if not m:
        return claim, False
    R, C, B = seen(g, x)
    return claim, (m["r"], m["c"], m["b"]) == (_digits(R), _digits(C - R), _digits(B - R - C))


def check_hidden(g, x, d, kind, i, ev) -> tuple[bool, bool | None]:
    unit = UNIT_CELLS[kind][i]
    others = [y for y in unit if y != x and not g[y]]
    claim = x in unit and all(blocker(g, y, d) for y in others)
    if ev is None:
        return claim, None
    m = HIDDEN_EV.match(ev)
    if not m:
        return claim, False
    listed = {}
    for tok in m["body"].split():
        b = BLOCK_RE.match(tok)
        if not b:
            return claim, False
        y = (int(b[1]) - 1) * 9 + int(b[2]) - 1
        if y in listed:
            return claim, False
        listed[y] = b[3]
    if set(listed) != set(others):
        return claim, False
    for y, w in listed.items():  # the stated reason must be true (any true reason is accepted)
        unit_of_y = {"r": UNIT_CELLS["r"][y // 9], "c": UNIT_CELLS["c"][y % 9], "b": UNIT_CELLS["b"][box_of(y)]}[w]
        if d not in {g[z] for z in unit_of_y}:
            return claim, False
    return claim, True


def check_trace(puzzle81: str, trace: str, solution81: str | None = None) -> TraceCheck:
    g = to_cells(puzzle81)
    sol = to_cells(solution81) if solution81 else None
    out = TraceCheck()
    for line in (ln.strip() for ln in trace.splitlines()):
        if not line:
            continue
        sc = StepCheck(line)
        out.steps.append(sc)
        m = STEP_RE.match(line)
        if not m:
            continue
        sc.parsed = True
        x, d = (int(m["r"]) - 1) * 9 + int(m["c"]) - 1, int(m["d"])
        if g[x] or d in set().union(*seen(g, x)):
            continue
        sc.legal = True
        if sol:
            sc.correct = sol[x] == d
        ev = m["ev"]
        if m["rule"] == "naked":
            sc.claim, sc.evidence = check_naked(g, x, d, ev)
        elif m["rule"]:
            sc.claim, sc.evidence = check_hidden(g, x, d, m["uk"], int(m["ui"]) - 1, ev)
        elif ev is not None:
            sc.evidence = False  # evidence without a rule tag is malformed
        g[x] = d
    out.final = "".join(str(v) if v else "." for v in g)
    return out
