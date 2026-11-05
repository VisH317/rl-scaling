#!/usr/bin/env python3
"""Training entrypoint for one run.

Wraps OGBench's reference agents rather than reimplementing them. What this adds over
`lib/ogbench/impls/main.py`:

  * the residual tower (via `patches.residual_arch`);
  * **plateau-based stopping with a size-dependent cap**, instead of a fixed 5M steps --
    the brief's key protocol decision, since a fixed budget across dataset sizes can
    manufacture a data plateau;
  * best-checkpoint saving (phases 2 and 5 need the best critic, not the last one);
  * parameter counts, epochs-over-data, samples-seen and peak-memory accounting;
  * a `status.json` heartbeat the monitor reads, and `metrics.jsonl` for plotting;
  * no wandb dependency and no rendering, so it runs headless and offline.

Usage:
    python -m offline.train --run-config path/to/config.json
    python -m offline.train --grid phase1_repro --depth 16 --agent crl --seed 0
    python -m offline.train --smoke            # tiny end-to-end validation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Any

# OGBench's impls/ is a flat package root (`from utils.x import y`), so it goes on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_IMPLS = os.path.join(os.path.dirname(_SRC), "lib", "ogbench", "impls")
for p in (_SRC, _IMPLS):
    if p not in sys.path:
        sys.path.insert(0, p)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_SRC, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


#: Stop this long before the Slurm wall clock, so there is time to write the final
#: checkpoint and summary.json.
DEADLINE_MARGIN_S = 20 * 60


def _deadline() -> float | None:
    """Epoch seconds at which to stop gracefully, or None for no limit.

    A hard Slurm timeout kills the process without a summary.json, so the run looks
    "started but vanished" and gets requeued from scratch -- losing hours of progress
    even though a perfectly good best-checkpoint was on disk. Stopping ourselves just
    before the limit turns that into an ordinary finished run with
    stop_reason="wall_clock".
    """
    for var in ("TRAIN_DEADLINE_EPOCH", "SLURM_JOB_END_TIME"):
        v = os.environ.get(var)
        if v:
            try:
                return float(v) - DEADLINE_MARGIN_S
            except ValueError:
                pass
    return None


class RunLogger:
    """CSV + JSONL + heartbeat. Flushes eagerly so a killed run still has its history."""

    def __init__(self, run_dir: str, run_cfg: dict):
        self.run_dir = run_dir
        self.run_cfg = run_cfg
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "probes"), exist_ok=True)
        self._csv: dict[str, Any] = {}
        self.jsonl = open(os.path.join(run_dir, "metrics.jsonl"), "a", buffering=1)
        self.t0 = time.time()

    def csv_row(self, name: str, row: dict) -> None:
        path = os.path.join(self.run_dir, f"{name}.csv")
        new = path not in self._csv
        if new:
            f = open(path, "a", newline="", buffering=1)
            self._csv[path] = (f, csv.DictWriter(f, fieldnames=list(row)))
            if os.path.getsize(path) == 0:
                self._csv[path][1].writeheader()
        f, w = self._csv[path]
        # Keep the header stable even if a later row gains keys.
        w.writerow({k: row.get(k) for k in w.fieldnames})

    def log(self, kind: str, step: int, metrics: dict) -> None:
        rec = {"kind": kind, "step": step, "wall_clock_s": round(time.time() - self.t0, 1)}
        rec.update(metrics)
        self.jsonl.write(json.dumps(rec) + "\n")
        self.csv_row(kind, rec)

    def status(self, **kw) -> None:
        """Atomic heartbeat so the monitor never reads a half-written file."""
        rec = {
            "run_id": self.run_cfg["run_id"],
            "grid": self.run_cfg["grid"],
            "agent": self.run_cfg["agent"]["agent"],
            "depth": self.run_cfg["arch"]["depth"],
            "size": self.run_cfg["size"],
            "seed": self.run_cfg["seed"],
            "host": platform.node(),
            "pid": os.getpid(),
            "updated": time.time(),
            "wall_clock_s": round(time.time() - self.t0, 1),
            "peak_rss_gb": round(_peak_rss_gb(), 2),
            **kw,
        }
        path = os.path.join(self.run_dir, "status.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, path)

    def close(self) -> None:
        for f, _ in self._csv.values():
            f.close()
        self.jsonl.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-config", help="path to a resolved run config json")
    ap.add_argument("--grid")
    ap.add_argument("--depth", type=int)
    ap.add_argument("--agent", choices=("crl", "gcbc"))
    ap.add_argument("--size", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: 400 steps, 2 eval episodes, 1 checkpoint")
    ap.add_argument("--max-steps", type=int, help="override the step cap")
    args = ap.parse_args()

    from offline import config as C
    from offline.critic_losses import install_critic_loss
    from offline.patches import agent_param_report, build_config, install_residual_arch

    # ---- resolve the run config -------------------------------------------------
    if args.run_config:
        with open(args.run_config) as f:
            run_cfg = json.load(f)
    else:
        if not (args.grid and args.depth and args.agent):
            ap.error("need --run-config, or --grid --depth --agent")
        spec = C.GRIDS[args.grid]
        agent_cfg = next(a for a in spec.agents if a.agent == args.agent)
        run_cfg = C.run_config(args.grid, args.depth, args.size, agent_cfg, args.seed)

    if args.smoke:
        run_cfg["step_cap"] = 400
        run_cfg["train"]["eval_interval"] = 200
        run_cfg["train"]["log_interval"] = 100
        run_cfg["train"]["eval_episodes"] = 2
        run_cfg["train"]["plateau_patience"] = 10**9
        run_cfg["run_id"] = "smoke__" + run_cfg["run_id"]
    if args.max_steps:
        run_cfg["step_cap"] = args.max_steps

    run_cfg["code_commit"] = _git_hash()
    run_dir = args.run_dir or os.path.join(C.RUNS_DIR, run_cfg["run_id"])
    os.makedirs(run_dir, exist_ok=True)

    log = RunLogger(run_dir, run_cfg)
    log.status(state="starting", step=0, best_success=None)

    # ---- heavy imports after the run dir exists, so a failure is recorded -------
    try:
        import jax
        import numpy as np
        import optax
        from utils.datasets import Dataset, GCDataset, HGCDataset
        from utils.evaluation import evaluate
        from utils.flax_utils import save_agent

        import agents as og_agents
        from offline.datasets import build_datasets

        run_cfg["devices"] = [str(d) for d in jax.devices()]
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(run_cfg, f, indent=2)

        # ---- data ---------------------------------------------------------------
        t_data = time.time()
        env, train_raw, val_raw, data_info = build_datasets(run_cfg)
        run_cfg["data_info"] = data_info
        log.status(state="data_loaded", step=0, best_success=None, data_info=data_info)

        agent_name = run_cfg["agent"]["agent"]
        # `agents.agents` maps name -> agent CLASS; `get_config()` lives on the module
        # (OGBench's main.py reaches it through config_flags on the file path).
        import importlib
        agent_module = importlib.import_module(f"agents.{agent_name}")
        agent_class = og_agents.agents[agent_name]
        base_cfg = agent_module.get_config()
        og_cfg = build_config(run_cfg, base_cfg)

        dataset_class = {"GCDataset": GCDataset, "HGCDataset": HGCDataset}[
            og_cfg["dataset_class"]
        ]
        train_dataset = dataset_class(Dataset.create(**train_raw), og_cfg)
        val_dataset = dataset_class(Dataset.create(**val_raw), og_cfg)
        data_secs = time.time() - t_data

        # ---- agent --------------------------------------------------------------
        example = train_dataset.sample(1)
        # Process-lifetime patch: Flax re-runs setup() on every apply, so the shim must
        # stay installed for as long as the agent is used, not just while it is built.
        run_cfg["residual_tower_installed"] = install_residual_arch(run_cfg["arch"])
        # Must happen BEFORE the agent is built, and for the whole process: Flax
        # re-enters the loss on every apply.
        run_cfg["critic_loss_patched"] = install_critic_loss(run_cfg["agent"])
        agent = agent_class.create(
            run_cfg["seed"], example["observations"], example["actions"], og_cfg
        )
        params = agent_param_report(agent)
        run_cfg["params"] = params
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(run_cfg, f, indent=2)
        print(f"[{run_cfg['run_id']}] params={params['total']:,} "
              f"devices={run_cfg['devices']} data_load={data_secs:.0f}s", flush=True)

        # ---- training loop ------------------------------------------------------
        tr = run_cfg["train"]
        ck = run_cfg["checkpoint"]
        cap = run_cfg["step_cap"]
        batch = og_cfg["batch_size"]
        n_transitions = data_info["n_transitions"]

        best_success = -1.0
        best_step = 0
        evals_since_best = 0
        best_smoothed = -1.0
        recent: list[float] = []
        stop_reason = "cap_reached"
        periodic: list[int] = []
        deadline = _deadline()
        if deadline:
            print(f"[{run_cfg['run_id']}] wall-clock deadline in "
                  f"{(deadline - time.time()) / 3600:.1f}h", flush=True)
        t_train = time.time()

        for step in range(1, cap + 1):
            b = train_dataset.sample(batch)
            agent, info = agent.update(b)

            if step % tr["log_interval"] == 0:
                vb = val_dataset.sample(batch)
                _, val_info = agent.total_loss(vb, grad_params=agent.network.params)
                samples = step * batch
                log.log("train", step, {
                    **{f"train/{k}": float(v) for k, v in info.items()},
                    **{f"val/{k}": float(v) for k, v in val_info.items()},
                    "samples_seen": samples,
                    "epochs": round(samples / n_transitions, 4),
                    "steps_per_s": round(step / (time.time() - t_train), 1),
                    "peak_rss_gb": round(_peak_rss_gb(), 2),
                })

            if step % ck["heartbeat_interval"] == 0:
                log.status(state="training", step=step, step_cap=cap,
                           best_success=best_success if best_success >= 0 else None,
                           best_step=best_step,
                           frac_done=round(step / cap, 4))

            if step % tr["eval_interval"] == 0 or step == cap:
                t_eval = time.time()
                per_task = {}
                successes = []
                for task_id in range(1, 6):
                    eval_info, _, _ = evaluate(
                        agent=agent, env=env, task_id=task_id,
                        config=og_cfg,
                        num_eval_episodes=tr["eval_episodes"],
                        num_video_episodes=0,
                        video_frame_skip=3,
                        eval_temperature=tr["eval_temperature"],
                        eval_gaussian=None,
                    )
                    per_task[f"eval/task{task_id}/success"] = float(eval_info["success"])
                    successes.append(float(eval_info["success"]))
                success = float(np.mean(successes))
                samples = step * batch

                log.log("eval", step, {
                    "eval/success": success, **per_task,
                    "samples_seen": samples,
                    "epochs": round(samples / n_transitions, 4),
                    "eval_secs": round(time.time() - t_eval, 1),
                })

                # Best checkpoint still tracks the raw per-eval success -- that is the
                # checkpoint phases 2 and 5 want.
                if success > best_success:
                    best_success, best_step = success, step
                    if ck["save_best"]:
                        save_agent(agent, os.path.join(run_dir, "checkpoints"), "best")

                # The PLATEAU test runs on a moving average instead, because a single
                # evaluation over 75 episodes is noisy enough to fake a plateau.
                recent.append(success)
                recent = recent[-tr["plateau_window"]:]
                smoothed = float(np.mean(recent))
                if smoothed > best_smoothed + tr["plateau_min_delta"]:
                    best_smoothed, evals_since_best = smoothed, 0
                else:
                    evals_since_best += 1

                print(f"[{run_cfg['run_id']}] step {step:,}/{cap:,} "
                      f"success={success:.3f} best={best_success:.3f}@{best_step:,} "
                      f"ma{tr['plateau_window']}={smoothed:.3f} stale={evals_since_best}",
                      flush=True)

                log.status(state="training", step=step, step_cap=cap,
                           best_success=best_success, best_step=best_step,
                           last_success=success, frac_done=round(step / cap, 4),
                           evals_since_best=evals_since_best)

                # A floor on gradient steps: OGBench's published runs train for 1M,
                # so stopping earlier is not comparable no matter how flat the curve is.
                if (evals_since_best >= tr["plateau_patience"]
                        and step >= tr.get("plateau_min_steps", 0)):
                    stop_reason = "plateau"
                    break

                # Checked at eval boundaries so the stop lands right after a saved
                # checkpoint and a logged evaluation.
                if deadline and time.time() > deadline:
                    stop_reason = "wall_clock"
                    print(f"[{run_cfg['run_id']}] stopping early: wall-clock deadline "
                          f"reached at step {step:,}/{cap:,}", flush=True)
                    break

            if step % ck["periodic_interval"] == 0:
                save_agent(agent, os.path.join(run_dir, "checkpoints"), step)
                periodic.append(step)
                for old in periodic[:-ck["keep_periodic"]]:
                    p = os.path.join(run_dir, "checkpoints", f"params_{old}.pkl")
                    if os.path.exists(p):
                        os.remove(p)
                periodic = periodic[-ck["keep_periodic"]:]

        if ck["save_final"]:
            save_agent(agent, os.path.join(run_dir, "checkpoints"), "final")

        summary = {
            "run_id": run_cfg["run_id"],
            "state": "done",
            "stop_reason": stop_reason,
            "best_success": best_success,
            "best_step": best_step,
            # The published scoring convention: mean of the last 3 evaluations.
            "last3_eval_mean": round(float(np.mean(recent)), 4) if recent else None,
            "last3_evals": [round(v, 4) for v in recent],
            "final_step": step,
            "step_cap": cap,
            "gradient_steps": step,
            "samples_seen": step * batch,
            "epochs_over_data": round(step * batch / n_transitions, 4),
            "params": params,
            "wall_clock_s": round(time.time() - log.t0, 1),
            "train_secs": round(time.time() - t_train, 1),
            "data_load_secs": round(data_secs, 1),
            "peak_rss_gb": round(_peak_rss_gb(), 2),
            "devices": run_cfg["devices"],
            "code_commit": run_cfg["code_commit"],
            "data_info": data_info,
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log.status(**{k: v for k, v in summary.items() if k != "run_id"}, step=step)
        print(f"[{run_cfg['run_id']}] DONE best={best_success:.3f}@{best_step:,} "
              f"({stop_reason}) {summary['wall_clock_s']:.0f}s", flush=True)
        return 0

    except Exception as e:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        log.status(state="failed", step=-1, error=f"{type(e).__name__}: {e}")
        with open(os.path.join(run_dir, "error.txt"), "w") as f:
            f.write(tb)
        print(tb, file=sys.stderr, flush=True)
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
