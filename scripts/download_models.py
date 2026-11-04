#!/usr/bin/env python3
"""Download the base models pinned in configs/models.toml (exact HF revisions).

  python scripts/download_models.py                    # qwen3.5-0.8b + qwen3.5-2b
  python scripts/download_models.py --all              # also the -Base checkpoints
  python scripts/download_models.py --only qwen3.5-2b
  python scripts/download_models.py --dry-run          # list files/sizes, download nothing

Target dir: $MODELS_DIR (default ./models, git-ignored). Needs `huggingface_hub`.
"""
from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="model keys to fetch")
    ap.add_argument("--all", action="store_true", help="include optional (-Base) models")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = tomllib.loads((ROOT / "configs/models.toml").read_text())
    out = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
    items = [m for m in cfg["model"]
             if (m["key"] in args.only if args.only else (args.all or not m.get("optional")))]
    api = HfApi()
    for m in items:
        info = api.model_info(m["repo"], revision=m["revision"], files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings)
        print(f"{m['key']:20s} {m['repo']}@{m['revision'][:8]}  {total / 1e9:.2f} GB  ({len(info.siblings)} files)")
        if args.dry_run:
            continue
        dest = out / m["key"]
        snapshot_download(m["repo"], revision=m["revision"], local_dir=dest)
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
