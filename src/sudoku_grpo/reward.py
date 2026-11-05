"""Rule-based GRPO rewards for sudoku solver traces (no judge model).

Completion text = everything generated after the eval prefix "<think>\\n":
    {trace}\\n</think>\\n\\n<answer>\\n{9 rows}\\n</answer>

  format  1 if the thinking block is closed and the answer block parses to 81 cells
  step      fraction of the puzzle's empty cells filled by the LEADING run of correct placements
            (legal and equal to the solution digit), i.e. progress up to the first wrong digit. Steps
            after the first error earn nothing, so a locally plausible but globally wrong derivation is
            not rewarded (cf. AdaC-GRPO's critique of partial-credit rewards).
  evidence  share of those leading steps whose rule claim + evidence are exactly true (L1-L3 traces);
            0 if there is no correct leading step.
  (v1 required evidence inside `step`; the smoke showed step≈0 for 50-75% of groups -> no signal,
   so placement progress and evidence quality are now separate terms.)
  exact   1 if the answer grid equals the solution

Default weights mirror the G-RaR split (format 0.2 / answer 0.2 / process 0.6 = step 0.5 + evidence 0.1),
with the LLM judge replaced by the programmatic step verifier.
"""
from __future__ import annotations

from . import grid, verify

THINK_END = "</think>"


def split_completion(text: str) -> tuple[str, str, bool]:
    if THINK_END in text:
        trace, rest = text.split(THINK_END, 1)
        return trace, rest, True
    return text, "", False


def score(puzzle: str, solution: str, completion: str, level: int = 2) -> dict:
    trace, rest, closed = split_completion(completion)
    pred = grid.parse_answer(rest) if closed else None
    chk = verify.check_trace(puzzle, trace, solution)
    lead, ev_ok = 0, 0
    for st in chk.steps:
        if not st.ok(0):
            break
        lead += 1
        ev_ok += st.ok(level)
    n_empty = puzzle.count(".") or 1
    return {"format": float(pred is not None), "step": min(lead, n_empty) / n_empty,
            "evidence": (ev_ok / lead) if lead else 0.0,
            "exact": float(pred == solution), "n_steps": len(chk.steps), "lead": lead}


def make_reward_funcs(level: int = 2):
    """TRL reward functions: f(prompts, completions, puzzle, solution, **kw) -> list[float]."""
    cache: dict = {}

    def _s(c, p, s):
        key = (p, c)
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[key] = score(p, s, c, level)
        return cache[key]

    def format_reward(prompts, completions, puzzle, solution, **kw):
        return [_s(c, p, s)["format"] for c, p, s in zip(completions, puzzle, solution)]

    def step_reward(prompts, completions, puzzle, solution, **kw):
        return [_s(c, p, s)["step"] for c, p, s in zip(completions, puzzle, solution)]

    def evidence_reward(prompts, completions, puzzle, solution, **kw):
        return [_s(c, p, s)["evidence"] for c, p, s in zip(completions, puzzle, solution)]

    def exact_reward(prompts, completions, puzzle, solution, **kw):
        return [_s(c, p, s)["exact"] for c, p, s in zip(completions, puzzle, solution)]

    return [format_reward, step_reward, evidence_reward, exact_reward]
