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

  reason    (v3.1, SOLUTION-FREE) code judge of the reasoning itself (G-RaR's judge role). Steps are
            replayed on the board and only SOUND steps count: legal AND logically forced (the stated
            naked/hidden claim is true; for untagged L0 lines the digit must be entailed by a single).
              sound step score = 0.6 + 0.4 * graded evidence   (L1: 1.0, L0: 1.0); unsound = 0
            Unsound steps are not placed on the board, so a guess cannot corrupt later judgements.
            reason = sum(step scores) / #empty cells (cap 1); 0 if the thinking block never closes
            (rambling / loops are not rewarded; truncated rollouts are NOT masked out of the loss).
            The answer key is never consulted: a true single is a forced deduction, so a sound step can
            never be "locally right but globally wrong" (the failure AdaC-GRPO reports for partial-grid
            rewards); v3 gave 0.2 for merely legal guesses, which is exactly that failure -> removed.
  format    (v3.1) closed thinking + 81-cell answer + givens preserved + the answer agrees with every
            sound placement in the trace (the answer must follow from the reasoning).

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
    snd = verify.check_trace(puzzle, trace, None, apply="sound")   # judge: no answer key
    lead, ev_ok = 0, 0
    for st in chk.steps:
        if not st.ok(0):
            break
        lead += 1
        ev_ok += st.ok(level)
    n_empty = puzzle.count(".") or 1
    tot = 0.0
    for st in snd.steps:
        if not st.sound:
            continue
        tot += (0.6 + 0.4 * st.ev_score) if st.ev_score is not None else 1.0
    reason = min(tot / n_empty, 1.0) if closed else 0.0
    fmt = pred is not None and grid.givens_preserved(puzzle, pred) and all(
        pred[i] == ch for i, ch in enumerate(snd.final) if ch != "." and puzzle[i] == ".")
    return {"reason": reason, "format": float(fmt), "step": min(lead, n_empty) / n_empty,
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

    def reason_reward(prompts, completions, puzzle, solution, **kw):  # solution is NOT used inside
        return [_s(c, p, s)["reason"] for c, p, s in zip(completions, puzzle, solution)]

    def exact_reward(prompts, completions, puzzle, solution, **kw):
        return [_s(c, p, s)["exact"] for c, p, s in zip(completions, puzzle, solution)]

    return [format_reward, step_reward, evidence_reward, reason_reward, exact_reward]
