#!/usr/bin/env python3
"""Upload a finished/ongoing HF Trainer run's log history to W&B after the fact
(for runs that were started without report_to=wandb, e.g. S1).

  python scripts/wandb_upload_history.py outputs/sft/<run> [--name <run>] [--group sft]

Reads <run>/run_meta.json if present (config + log_history), else the newest checkpoint's trainer_state.json.
Logged keys follow the Trainer's W&B names (train/loss, eval/loss, ...) with x-axis train/global_step.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", default=None)
    ap.add_argument("--group", default="sft")
    a = ap.parse_args()
    d = Path(a.run_dir)
    meta = {}
    if (d / "run_meta.json").exists():
        meta = json.loads((d / "run_meta.json").read_text())
        hist = meta.pop("log_history")
    else:
        ck = max(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        hist = json.loads((ck / "trainer_state.json").read_text())["log_history"]
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "sudoku-grpo-slm"),
                     entity=os.environ.get("WANDB_ENTITY", "2091126-hansung-university"),
                     name=a.name or d.name, group=a.group, config={**meta, "uploaded_after_training": True})
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")
    for h in hist:
        step = h.get("step")
        row = {"train/global_step": step}
        for k, v in h.items():
            if k == "step":
                continue
            key = k.replace("eval_", "eval/", 1) if k.startswith("eval_") else f"train/{k}"
            row[key] = v
        run.log(row)
    run.finish()
    print(f"uploaded {len(hist)} rows -> {run.url}")


if __name__ == "__main__":
    main()
