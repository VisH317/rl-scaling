#!/usr/bin/env python3
"""Re-evaluate saved checkpoints at the published episode count.

Training evaluates with 15 episodes per task (75 per point) so the learning curves are
cheap. OGBench's published commands all use `--eval_episodes=50` (250 per point), so the
headline number has to be measured the same way or it is not comparable to their table.

This loads `params_best.pkl` and scores it at 50 episodes/task, and also reports the
`last3_eval_mean` from `eval.csv` (the convention the OGBench paper scores with: mean of
the final three evaluations). Both go in the research log, because they answer different
questions -- "how good did this run ever get" vs "where did it settle".

Usage:
    python src/scripts/reeval.py --grid phase1_anchor
    python src/scripts/reeval.py --grid phase1_repro --episodes 50
    python src/scripts/reeval.py --run-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_IMPLS = os.path.join(os.path.dirname(_SRC), "lib", "ogbench", "impls")
for p in (_SRC, _IMPLS):
    if p not in sys.path:
        sys.path.insert(0, p)

from offline import config as C  # noqa: E402


def last_n_eval_mean(run_dir: str, n: int = 3):
    p = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(p):
        return None, []
    vals = [float(r["eval/success"]) for r in csv.DictReader(open(p))
            if r.get("eval/success") not in (None, "")]
    if not vals:
        return None, []
    tail = vals[-n:]
    return sum(tail) / len(tail), tail


def reeval_one(run_dir: str, episodes: int, checkpoint: str = "best") -> dict | None:
    import numpy as np
    from utils.datasets import Dataset, GCDataset, HGCDataset
    from utils.evaluation import evaluate
    from utils.flax_utils import restore_agent
    import agents as og_agents

    from offline.datasets import build_datasets
    from offline.critic_losses import install_critic_loss
    from offline.patches import build_config, install_residual_arch

    cfg_p = os.path.join(run_dir, "config.json")
    ckpt = os.path.join(run_dir, "checkpoints", f"params_{checkpoint}.pkl")
    if not (os.path.exists(cfg_p) and os.path.exists(ckpt)):
        return None
    run_cfg = json.load(open(cfg_p))

    env, train_raw, val_raw, _ = build_datasets(run_cfg)
    agent_name = run_cfg["agent"]["agent"]
    agent_module = importlib.import_module(f"agents.{agent_name}")
    og_cfg = build_config(run_cfg, agent_module.get_config())
    dataset_class = {"GCDataset": GCDataset, "HGCDataset": HGCDataset}[og_cfg["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_raw), og_cfg)

    install_residual_arch(run_cfg["arch"])
    install_critic_loss(run_cfg["agent"])
    ex = train_dataset.sample(1)
    agent = og_agents.agents[agent_name].create(
        run_cfg["seed"], ex["observations"], ex["actions"], og_cfg
    )
    agent = restore_agent(agent, os.path.join(run_dir, "checkpoints"), checkpoint)

    per_task, successes = {}, []
    t0 = time.time()
    for task_id in range(1, 6):
        info, _, _ = evaluate(
            agent=agent, env=env, task_id=task_id, config=og_cfg,
            num_eval_episodes=episodes, num_video_episodes=0,
            video_frame_skip=3, eval_temperature=run_cfg["train"]["eval_temperature"],
            eval_gaussian=None,
        )
        per_task[f"task{task_id}"] = float(info["success"])
        successes.append(float(info["success"]))

    l3, tail = last_n_eval_mean(run_dir)
    return {
        "run_id": run_cfg["run_id"],
        "grid": run_cfg["grid"],
        "agent": agent_name,
        "depth": run_cfg["arch"]["depth"],
        "tower": run_cfg["arch"]["tower"],
        "seed": run_cfg["seed"],
        "checkpoint": checkpoint,
        "episodes_per_task": episodes,
        "reeval_success": round(float(np.mean(successes)), 4),
        "reeval_per_task": {k: round(v, 4) for k, v in per_task.items()},
        "train_best_success": json.load(open(os.path.join(run_dir, "summary.json")))
            .get("best_success") if os.path.exists(os.path.join(run_dir, "summary.json")) else None,
        "last3_eval_mean": round(l3, 4) if l3 is not None else None,
        "last3_evals": [round(v, 4) for v in tail],
        "reeval_secs": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid")  # no `choices`: see monitor.py
    ap.add_argument("--run-dir")
    ap.add_argument("--episodes", type=int, default=C.PUBLISHED_EVAL_EPISODES)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--out")
    args = ap.parse_args()

    dirs = ([args.run_dir] if args.run_dir
            else sorted(glob.glob(os.path.join(C.RUNS_DIR, f"{args.grid}__*"))))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "summary.json"))]
    if not dirs:
        print("no finished runs to re-evaluate")
        return 0
    print(f"re-evaluating {len(dirs)} runs at {args.episodes} episodes/task "
          f"({args.episodes * 5} per run)")

    out = []
    for d in dirs:
        try:
            r = reeval_one(d, args.episodes, args.checkpoint)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {os.path.basename(d)}: {type(e).__name__}: {e}")
            continue
        if r:
            out.append(r)
            print(f"  {r['agent']:<5} d{r['depth']:<3} s{r['seed']}  "
                  f"reeval={r['reeval_success']:.3f}  "
                  f"train_best={r['train_best_success']}  "
                  f"last3={r['last3_eval_mean']}")

    # Aggregate the way the published table does.
    print("\naggregate (mean +/- SD over seeds)")
    import numpy as np
    keys = sorted({(r["agent"], r["depth"]) for r in out})
    print(f"  {'agent':<6}{'depth':>6}{'n':>3}{'reeval@' + str(args.episodes):>14}{'last3':>14}")
    for a, dep in keys:
        sub = [r for r in out if r["agent"] == a and r["depth"] == dep]
        rv = [r["reeval_success"] for r in sub]
        l3 = [r["last3_eval_mean"] for r in sub if r["last3_eval_mean"] is not None]
        rs = f"{100*np.mean(rv):.1f}+/-{100*np.std(rv):.1f}"
        ls = f"{100*np.mean(l3):.1f}+/-{100*np.std(l3):.1f}" if l3 else "-"
        print(f"  {a:<6}{dep:>6}{len(sub):>3}{rs:>14}{ls:>14}")

    path = args.out or os.path.join(C.ANALYSIS_DIR, args.grid or "reeval",
                                    f"reeval_{args.episodes}ep.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
