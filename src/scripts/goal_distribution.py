#!/usr/bin/env python3
"""Measure the temporal-distance distribution of sampled goals.

Geometric offsets are CLIPPED at the trajectory end
(`traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)`), so every overshoot
piles onto the final state instead of being resampled. With discount 0.99 the mean
offset is 100 steps, which is longer than what remains of a 201-step trajectory from a
typical start -- so a large fraction of the critic's positives are the same terminal
states. The probes have to bin that separately or it dominates the far-distance bins
that H-horizon is actually about.

The fraction is dataset-dependent (it falls as trajectories lengthen relative to
1/(1-gamma)), so re-run this per dataset rather than reusing a number.

Usage:
    python src/scripts/goal_distribution.py --dataset antmaze-medium-stitch-v0
    python src/scripts/goal_distribution.py --dataset cube-double-play-100m-v0 --shard 0
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_IMPLS = os.path.join(os.path.dirname(_SRC), "lib", "ogbench", "impls")
for p in (_SRC, _IMPLS):
    if p not in sys.path:
        sys.path.insert(0, p)

from offline import config as C  # noqa: E402

BINS = [0, 10, 25, 50, 100, 200, 500, 1000, 10**9]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="antmaze-medium-stitch-v0", choices=list(C.DATASETS))
    ap.add_argument("--shard", type=int)
    ap.add_argument("--agent", default="crl")
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    from utils.datasets import Dataset, GCDataset
    from offline.datasets import load_npz_list

    spec = C.DATASETS[args.dataset]
    train_rel, _ = spec.rel_paths(args.shard if spec.is_sharded else None)
    path = os.path.join(C.DATA_DIR, train_rel)
    print(f"dataset {os.path.basename(path)}")

    cfg = importlib.import_module(f"agents.{args.agent}").get_config()
    pub = C.PUBLISHED_FLAGS.get((args.dataset, args.agent)) or {}
    for k, v in pub.items():
        if k in cfg:
            cfg[k] = v
    ds = GCDataset(Dataset.create(**load_npz_list([path])), cfg)

    np.random.seed(args.seed)
    idxs = ds.dataset.get_random_idxs(args.samples)
    final = ds.terminal_locs[np.searchsorted(ds.terminal_locs, idxs)]

    out = {"dataset": args.dataset, "agent": args.agent, "discount": cfg["discount"],
           "samples": args.samples, "streams": {}}
    print(f"discount {cfg['discount']}, {args.samples:,} samples, "
          f"published flags {pub or '{}'}\n")

    for stream, p_traj, p_rand, geom in (
        ("value", cfg["value_p_trajgoal"], cfg["value_p_randomgoal"], cfg["value_geom_sample"]),
        ("actor", cfg["actor_p_trajgoal"], cfg["actor_p_randomgoal"], cfg["actor_geom_sample"]),
    ):
        g = ds.sample_goals(idxs, 0.0, p_traj, p_rand, geom)
        # Random goals come from other trajectories; distance is undefined for them.
        same_traj = ds.terminal_locs[np.searchsorted(ds.terminal_locs, g)] == final
        d = (g - idxs)[same_traj]
        frac_final = float(np.mean(g[same_traj] == final[same_traj])) if same_traj.any() else 0.0
        h = (np.histogram(d, bins=BINS)[0] / max(len(d), 1)).tolist()
        rec = {
            "p_trajgoal": p_traj, "p_randomgoal": p_rand, "geom_sample": geom,
            "frac_same_trajectory": round(float(np.mean(same_traj)), 4),
            "frac_goal_is_final_state": round(frac_final, 4),
            "mean_distance": round(float(d.mean()), 1) if len(d) else None,
            "median_distance": int(np.median(d)) if len(d) else None,
            "max_distance": int(d.max()) if len(d) else None,
            "bins": BINS, "bin_fractions": [round(x, 4) for x in h],
        }
        out["streams"][stream] = rec
        print(f"{stream} goals  (p_traj={p_traj}, p_rand={p_rand}, geom={geom})")
        print(f"   same-trajectory fraction      {rec['frac_same_trajectory']:.3f}")
        print(f"   P(goal == final state)        {rec['frac_goal_is_final_state']:.3f}"
              + ("   <-- bin this separately in probes"
                 if rec["frac_goal_is_final_state"] > 0.1 else ""))
        print(f"   distance mean/median/max      {rec['mean_distance']} / "
              f"{rec['median_distance']} / {rec['max_distance']}")
        print("   bins " + "  ".join(
            f"[{BINS[i]},{BINS[i+1] if BINS[i+1] < 10**9 else 'inf'})={h[i]:.3f}"
            for i in range(len(h)) if h[i] > 0))
        print()

    path_out = args.out or os.path.join(C.ANALYSIS_DIR, "goal_distribution",
                                        f"{args.dataset}_{args.agent}.json")
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    json.dump(out, open(path_out, "w"), indent=2)
    print(f"wrote {path_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
