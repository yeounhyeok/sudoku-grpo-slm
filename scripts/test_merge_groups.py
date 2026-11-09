"""CPU unit test for dynamic-sampling merge_groups (different prompt/completion widths per round)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grpo import merge_groups

G, PAD = 2, 9


def mk(B, Pw, C, tag):
    ar = lambda w, v: torch.full((B, w), float(v))
    pm = torch.ones(B, Pw, dtype=torch.long); pm[:, 0] = 0   # one left pad
    cm = torch.ones(B, C, dtype=torch.long); cm[:, -1] = 0
    return {"prompt_ids": torch.full((B, Pw), tag, dtype=torch.long), "prompt_mask": pm,
            "completion_ids": torch.full((B, C), tag, dtype=torch.long), "completion_mask": cm,
            "advantages": torch.full((B,), float(tag)), "old_per_token_logps": ar(C, tag),
            "importance_sampling_ratio": ar(C, 1.0 + tag), "sampling_per_token_logps": ar(C, tag),
            "ref_per_token_logps": ar(C, tag), "num_items_in_batch": cm.sum()}

r0, r1 = mk(4, 5, 7, 1), mk(2, 8, 3, 2)
rounds = [(r0, 5, 7), (r1, 8, 3)]
o = merge_groups(rounds, [(0, 1), (1, 0), (0, 0)], G, PAD)
assert o["prompt_ids"].shape == (6, 8) and o["completion_ids"].shape == (6, 7), (o["prompt_ids"].shape, o["completion_ids"].shape)
assert o["advantages"].tolist() == [1, 1, 2, 2, 1, 1]
assert (o["prompt_ids"][0, :3] == PAD).all() and (o["prompt_ids"][0, 3:] == 1).all()      # left pad, rows from round 0
assert (o["prompt_ids"][2] == 2).all()
assert (o["completion_ids"][2, :3] == 2).all() and (o["completion_ids"][2, 3:] == PAD).all()  # right pad
assert (o["completion_mask"][2, 3:] == 0).all() and (o["prompt_mask"][0, :3] == 0).all()
assert (o["old_per_token_logps"][2, 3:] == 0).all() and (o["importance_sampling_ratio"][2, 3:] == 1.0).all()
assert o["num_items_in_batch"].item() == int(o["completion_mask"].sum())
for k, v in o.items():
    if k != "num_items_in_batch": assert v.shape[0] == 6, k
print("merge_groups OK", {k: tuple(v.shape) for k, v in o.items() if k != "num_items_in_batch"})
