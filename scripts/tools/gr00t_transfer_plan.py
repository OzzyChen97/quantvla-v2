#!/usr/bin/env python3
"""Mask-only transfer plan (review round 5, item 5).

The held-out experiment must transfer ONLY the layer mask (which layers are
FP16 / W4) from a source plan; the DuQuant rotation packs, the A8 scales and
the checkpoint weights must all belong to the TARGET checkpoint. A source
plan's `packdirs` are checkpoint-specific — using them on another checkpoint
loads that checkpoint's weights with the WRONG rotation/pack metadata.

Usage:
    python scripts/tools/gr00t_transfer_plan.py \
        --mask-source <source_plan.json> \
        --packdir <target_packdir> \
        --mode mask-only|uniform --uniform-bits 6 \
        --out <transfer_plan.json>

--mode mask-only: layers/skip from the source plan, packdirs overridden.
--mode uniform:   every layer quantized at --uniform-bits (no skips), for the
                  same-budget uniform baseline on the target checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))


def build_transfer(mask_source: Path, packdir: str, mode: str, uniform_bits: int,
                   out: Path, mask_suite: str, target_suite: str) -> None:
    src = json.loads(mask_source.read_text())
    group = 64
    if mode == "mask-only":
        layers = {
            n: {"bits": e.get("bits"), "group": e.get("group", group), "skip": e.get("skip")}
            for n, e in (src.get("layers") or {}).items()
        }
    elif mode == "uniform":
        layers = {
            n: {"bits": uniform_bits, "group": group, "skip": False}
            for n in (src.get("layers") or {})
        }
    else:
        raise SystemExit(f"unknown mode {mode}")
    plan = {
        "meta": {
            "type": "transfer",
            "transfer_mode": "mask_only" if mode == "mask-only" else "uniform_baseline",
            "mask_source": str(mask_source),
            "mask_suite": mask_suite,
            "target_suite": target_suite,
            "pack_source": packdir,
            "note": ("layer FP16/W4 mask from the source plan; weights, DuQuant "
                     "rotation packs and A8 scales belong to the TARGET checkpoint"),
        },
        "packdirs": {str(group): packdir},
        "layers": layers,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    n_w4 = sum(1 for e in layers.values() if not e.get("skip"))
    print(f"[transfer] {mode} plan -> {out} "
          f"({n_w4} W4 / {len(layers) - n_w4} skip, packdir={packdir})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask-only transfer plan generator")
    p.add_argument("--mask-source", required=True)
    p.add_argument("--packdir", required=True, help="TARGET checkpoint pack dir.")
    p.add_argument("--mode", default="mask-only", choices=["mask-only", "uniform"])
    p.add_argument("--uniform-bits", type=int, default=6)
    p.add_argument("--mask-suite", default="spatial")
    p.add_argument("--target-suite", default="10")
    p.add_argument("--out", required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build_transfer(Path(a.mask_source), a.packdir, a.mode, a.uniform_bits,
                   Path(a.out), a.mask_suite, a.target_suite)
