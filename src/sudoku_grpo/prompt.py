"""Prompt / completion format shared by SFT, GRPO rewards and evaluation."""
from __future__ import annotations

from .grid import format_grid

SYSTEM = (
    "You are a Sudoku solver. Think step by step inside <reasoning></reasoning>, "
    "then output the completed 9x9 grid inside <answer></answer> "
    "(9 lines, digits separated by spaces)."
)


def user_prompt(puzzle81: str) -> str:
    return "Solve this Sudoku. '.' marks an empty cell.\n\n" + format_grid(puzzle81)


SYSTEM_DIRECT = (
    "You are a Sudoku solver. Output ONLY the completed 9x9 grid inside <answer></answer> "
    "(9 lines, digits separated by spaces). Do not explain."
)


def messages_direct(puzzle81: str) -> list[dict]:
    """No-reasoning baseline: the model must emit the grid immediately."""
    return [{"role": "system", "content": SYSTEM_DIRECT},
            {"role": "user", "content": user_prompt(puzzle81)}]


SYSTEM_THINK = (
    "You are a Sudoku solver. Solve the puzzle, then give the final completed 9x9 grid "
    "inside <answer></answer> (9 lines, digits separated by spaces)."
)


def messages_think(puzzle81: str) -> list[dict]:
    """Native-thinking baseline: the model reasons in its own <think> block (chat template),
    and only the final grid format is prescribed."""
    return [{"role": "system", "content": SYSTEM_THINK},
            {"role": "user", "content": user_prompt(puzzle81)}]


def user_prompt_views(puzzle81: str) -> str:
    """Diagnostic variant (11-13 ablation, NOT the frozen protocol): the same grid plus its columns and
    3x3 boxes written out as lines, so column/box contents can be read without a vertical lookup."""
    s = puzzle81
    cols = "\n".join(f"C{c + 1}: " + " ".join(s[r * 9 + c] for r in range(9)) for c in range(9))
    boxes = "\n".join(
        f"B{b + 1}: " + " ".join(s[(b // 3 * 3 + i) * 9 + b % 3 * 3 + j] for i in range(3) for j in range(3))
        for b in range(9))
    return (user_prompt(puzzle81)
            + "\n\nThe same puzzle read by column (C1 = leftmost column, top to bottom):\n" + cols
            + "\n\nAnd by 3x3 box (B1 = top-left, B2 = top-middle, ..., B9 = bottom-right; "
              "each box read left to right, top to bottom):\n" + boxes)


def messages_think_views(puzzle81: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_THINK},
            {"role": "user", "content": user_prompt_views(puzzle81)}]


def messages(puzzle81: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(puzzle81)}]


def completion(reasoning: str, solution81: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{format_grid(solution81)}\n</answer>"
