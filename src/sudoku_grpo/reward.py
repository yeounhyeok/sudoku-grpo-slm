"""Rule-based GRPO rewards for sudoku solver traces (no judge model).

Completion text = everything generated after the eval prefix "<think>\\n":
    {trace}\\n</think>\\n\\n<answer>\\n{9 rows}\\n</answer>

  format  1 if the thinking block is closed and the answer block parses to 81 cells
  step    fraction of the puzzle's empty cells filled by the LEADING run of fully valid steps
          (legal, equal to the solution digit, rule claim and evidence true at `level`), i.e. progress
          up to the first mistake. Steps after the first error earn nothing, so a locally plausible
          but wrong derivation is not rewarded (cf. AdaC-GRPO's critique of partial-credit rewards).
  exact   1 if the answer grid equals the solution

Default weights mirror the G-RaR split (format 0.2 / answer 0.2 / process 0.6), with the LLM judge
replaced by the programmatic step verifier.
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
    lead = 0
    for st in chk.steps:
        if not st.ok(level):
            break
        lead += 1
    n_empty = puzzle.count(".") or 1
    return {"format": float(pred is not None), "step": min(lead, n_empty) / n_empty,
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

    def exact_reward(prompts, completions, puzzle, solution, **kw):
        return [_s(c, p, s)["exact"] for c, p, s in zip(completions, puzzle, solution)]

    return [format_reward, step_reward, exact_reward]
