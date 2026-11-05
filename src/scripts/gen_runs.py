#!/usr/bin/env python3
"""Expand a grid into a run manifest: one row per cell x seed, with status.

The manifest is the scheduler's state. `submit_packs.py` reads it to decide what to
launch; `monitor.py` reads it plus each run's `status.json` to report progress. It is a
plain JSON file so it can be inspected and hand-edited.

Usage:
    python src/scripts/gen_runs.py --grid phase1_repro
    python src/scripts/gen_runs.py --grid phase1_repro --dry-run
    python src/scripts/gen_runs.py --grid phase3_corner --append
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402

MANIFEST = os.path.join(C.EXPERIMENTS_DIR, "run_manifest.json")


def build(grid: str) -> list[dict]:
    spec = C.GRIDS[grid]
    rows = []
    for depth, size, agent, seed in spec.cells():
        cfg = C.run_config(grid, depth, size, agent, seed)
        rows.append({
            "run_id": cfg["run_id"],
            "grid": grid,
            "agent": cfg["agent"]["agent"],
            "depth": depth,
            "size": cfg["size"],
            "seed": seed,
            "subset_id": cfg["subset_id"],
            "step_cap": cfg["step_cap"],
            "critic_loss": cfg["agent"].get("critic_loss"),
            "repr_dim": cfg["arch"]["repr_dim"],
            "status": "pending",
            "attempts": 0,
            "job_id": None,
            "config_path": os.path.join(C.CONFIGS_DIR, f"{cfg['run_id']}.json"),
            "run_dir": os.path.join(C.RUNS_DIR, cfg["run_id"]),
        })
        os.makedirs(C.CONFIGS_DIR, exist_ok=True)
        with open(rows[-1]["config_path"], "w") as f:
            json.dump(cfg, f, indent=2)
    return rows


def load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as f:
        return json.load(f)


def save_manifest(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, MANIFEST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=list(C.GRIDS),
                    help="LEGACY. New experiments go through src/run.py.")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing manifest instead of replacing it")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = build(args.grid)
    spec = C.GRIDS[args.grid]
    print(f"grid {args.grid}: {len(rows)} runs")
    print(f"  depths {spec.depths}")
    print(f"  sizes  {spec.sizes or 'native'}")
    print(f"  agents {[a.agent for a in spec.agents]}  seeds {spec.seeds}")
    if spec.note:
        print(f"  note   {spec.note}")
    packs = -(-len(rows) // C.SLURM.runs_per_pack)
    print(f"  -> {packs} packs of {C.SLURM.runs_per_pack}, "
          f"{C.SLURM.max_concurrent_runs} runs concurrent")

    if args.dry_run:
        for r in rows[:6]:
            print(f"    {r['run_id']}")
        if len(rows) > 6:
            print(f"    ... and {len(rows) - 6} more")
        return 0

    existing = load_manifest() if args.append else []
    have = {r["run_id"] for r in existing}
    added = [r for r in rows if r["run_id"] not in have]
    save_manifest(existing + added)
    print(f"wrote {MANIFEST}: {len(existing) + len(added)} rows ({len(added)} new)")
    print(f"configs in {C.CONFIGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
