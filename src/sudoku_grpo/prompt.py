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


def messages(puzzle81: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(puzzle81)}]


def completion(reasoning: str, solution81: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{format_grid(solution81)}\n</answer>"
