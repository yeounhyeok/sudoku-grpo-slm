#!/usr/bin/env python3
"""Merge a LoRA adapter into the base checkpoint *in its original layout* (for vLLM evaluation).

  python scripts/merge_lora.py ADAPTER_DIR OUT_DIR [--base models/qwen3.5-2b]

Works on the safetensors directly instead of save_pretrained(), so the merged directory keeps the
base model's exact config/architecture (Qwen3_5ForConditionalGeneration), tokenizer, chat template,
vision tower and MTP weights. For each adapted module X:
    X.weight += (alpha / r) * B @ A          (computed in fp32, stored back in the base dtype)
Adapter key  base_model.model.<X>.lora_A.weight  ->  base key  <X>.weight  (must exist, shapes checked).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("out")
    ap.add_argument("--base", default=None, help="default: base_model_name_or_path from adapter_config.json")
    args = ap.parse_args()
    ad, out = Path(args.adapter), Path(args.out)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    base = Path(args.base or cfg["base_model_name_or_path"])
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is supported"
    scale = cfg["lora_alpha"] / cfg["r"]

    A, B = {}, {}
    with safe_open(str(ad / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            assert k.startswith("base_model.model."), k
            mod = k[len("base_model.model."):]
            if mod.endswith(".lora_A.weight"):
                A[mod[:-len(".lora_A.weight")] + ".weight"] = f.get_tensor(k)
            elif mod.endswith(".lora_B.weight"):
                B[mod[:-len(".lora_B.weight")] + ".weight"] = f.get_tensor(k)
            else:
                raise SystemExit(f"unexpected adapter tensor {k}")
    assert A.keys() == B.keys()

    out.mkdir(parents=True, exist_ok=True)
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out / p.name)
    merged, zero = 0, 0
    for shard in sorted(base.glob("*.safetensors")):
        tensors = {}
        with safe_open(str(shard), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in A:
                    d = (B[k].float() @ A[k].float()) * scale
                    assert d.shape == t.shape, (k, d.shape, t.shape)
                    zero += int(d.abs().max() == 0)  # e.g. vision-tower adapters never receive gradients
                    t = (t.float() + d).to(t.dtype)
                    merged += 1
                tensors[k] = t.contiguous()
        save_file(tensors, str(out / shard.name), metadata=meta or {"format": "pt"})
    assert merged == len(A), f"merged {merged} of {len(A)} adapted modules"
    print(json.dumps({"merged_modules": merged, "zero_delta_modules": zero, "scale": scale, "out": str(out)}))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
