#!/usr/bin/env python3
"""Hydra entrypoint: compose a config, expand it into runs, optionally submit.

This sits ABOVE the existing interface rather than replacing it. It resolves the
composed config into exactly the same per-run JSON that `train.py --run-config`
already consumes, so jobs already in the Slurm queue keep working unchanged.

    # see what you would run, change nothing
    python src/scripts/launch.py experiment=phase1_anchor --cfg job
    python src/scripts/launch.py experiment=phase1_repro dry_run=true

    # override anything
    python src/scripts/launch.py arch.depth=64 optim.lr=1e-4 train.plateau_patience=8
    python src/scripts/launch.py agent=crl_infonce arch=residual_cold
    python src/scripts/launch.py experiment=phase3_corner dataset=cube_double_play_100m

    # generate and submit
    python src/scripts/launch.py experiment=phase1_anchor submit=chain
    python src/scripts/launch.py experiment=phase1_repro submit=chain after=2993272

    # sweep across groups (one manifest entry per combination)
    python src/scripts/launch.py -m agent=crl,crl_infonce arch.depth=4,32

Run `--help` for Hydra's own flags. `--cfg job --resolve` prints the fully resolved
config without doing anything, which is the fastest way to check an override landed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from offline import config as C  # noqa: E402
from offline.schema import register, test_schema_matches_config  # noqa: E402

register()

CONF_DIR = os.path.join(os.path.dirname(_SRC), "conf")


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_SRC, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def step_cap(cfg: DictConfig, size: int) -> int:
    if cfg.train.fixed_step_cap:
        return int(cfg.train.fixed_step_cap)
    caps = {int(k): int(v) for k, v in cfg.train.step_caps.items()}
    if size in caps:
        return caps[size]
    keys = sorted(caps)
    if size <= keys[0]:
        return caps[keys[0]]
    if size >= keys[-1]:
        return caps[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= size <= hi:
            f = (size - lo) / (hi - lo)
            return int(caps[lo] + f * (caps[hi] - caps[lo]))
    raise AssertionError("unreachable")


def subset_id(cfg: DictConfig, size: int | None) -> str:
    ds = cfg.dataset.name
    if size is None:
        return f"{ds}__native"
    m = "M" if size >= 1_000_000 else "K"
    n = size // (1_000_000 if m == "M" else 1_000)
    return f"{ds}__{n}{m}__seed{cfg.subset.seed}"


def optim_deviations(o: DictConfig) -> list[str]:
    """Anything departing from the verified reference setup (plain constant-LR Adam)."""
    out = []
    if o.schedule != "constant":
        out.append(f"schedule={o.schedule} (both baselines: constant)")
    if o.warmup_steps:
        out.append(f"warmup_steps={o.warmup_steps} (both baselines: 0)")
    if o.max_grad_norm:
        out.append(f"max_grad_norm={o.max_grad_norm} (both baselines: unclipped)")
    if o.weight_decay:
        out.append(f"weight_decay={o.weight_decay} (both baselines: 0)")
    if o.optimizer != "adam":
        out.append(f"optimizer={o.optimizer} (both baselines: adam)")
    lr_a, lr_c = o.actor_lr or o.lr, o.critic_lr or o.lr
    if lr_a != 3e-4 or lr_c != 3e-4:
        out.append(f"lr actor={lr_a} critic={lr_c} (both baselines: 3e-4)")
    return out


#: Stable-CRL's deltas (Zheng et al. 2023), verified from its source. Held in code
#: rather than only in YAML so they can be RE-APPLIED after the published per-dataset
#: flags: published `alpha` is CRL's BC coefficient for that environment, and would
#: otherwise silently overwrite stable-CRL's bc_coef=0.05 with 0.1.
STABLE_AGENT_DELTAS = {"batch_size": 2048, "alpha": 0.05}
STABLE_ARCH_DELTAS = {"repr_dim": 16, "init": "cold"}


def build_run_config(cfg: DictConfig, depth: int, size, agent_name: str,
                     seed: int) -> dict:
    """Resolve one cell into the run-config dict `train.py --run-config` consumes."""
    agent = OmegaConf.to_container(cfg.agent, resolve=True)
    agent["agent"] = agent_name
    arch = OmegaConf.to_container(cfg.arch, resolve=True)
    arch["depth"] = int(depth)

    # Precedence, lowest to highest:
    #   1. agent/arch group defaults
    #   2. published per-(dataset, agent) flags from hyperparameters.sh
    #   3. an explicit recipe's deltas
    # Step 3 must come last. Published `alpha` is CRL's BC coefficient for that
    # environment (0.1 on antmaze-medium-stitch) and would otherwise overwrite
    # stable-CRL's 0.05, silently turning a "stable recipe" run into something else.
    published, recipe_over_published = {}, {}
    if agent.get("use_published_flags", True):
        published = dict(C.PUBLISHED_FLAGS.get((cfg.dataset.name, agent_name)) or {})
        agent.update(published)

    if agent.get("recipe") == "stable":
        arch.update(STABLE_ARCH_DELTAS)
        for k, v in STABLE_AGENT_DELTAS.items():
            if k in published and published[k] != v:
                recipe_over_published[k] = {"published": published[k], "recipe": v}
            agent[k] = v

    native = cfg.dataset.n_train_trajs * (cfg.dataset.traj_len - 1)
    resolved_size = int(size) if size is not None else native

    train = OmegaConf.to_container(cfg.train, resolve=True)
    train.pop("step_caps", None)
    train.update(OmegaConf.to_container(cfg.eval, resolve=True))
    # train.py reads these names.
    train["eval_episodes"] = int(cfg.eval.episodes)
    train["eval_temperature"] = float(cfg.eval.temperature)

    out = {
        "grid": cfg.experiment.name,
        "dataset": cfg.dataset.name,
        "env_name": cfg.dataset.env_name,
        "subset_id": subset_id(cfg, size),
        "size": resolved_size,
        "seed": int(seed),
        "arch": arch,
        "agent": agent,
        "train": train,
        "optim": OmegaConf.to_container(cfg.optim, resolve=True),
        "checkpoint": OmegaConf.to_container(cfg.checkpoint, resolve=True),
        "step_cap": step_cap(cfg, resolved_size),
        "published_flags": published,
        "recipe_overrode_published": recipe_over_published,
        "optim_deviations": optim_deviations(cfg.optim),
        "code_commit": _git_hash(),
        "config_source": "hydra",
    }
    out["run_id"] = C._run_id(out)
    return out


def expand(cfg: DictConfig) -> list[dict]:
    e = cfg.experiment
    depths = list(e.depths) or [int(cfg.arch.depth)]
    sizes = list(e.sizes) or [cfg.dataset.size]
    agents = list(e.agents) or [cfg.agent.agent]
    seeds = list(e.seeds) or list(cfg.train.seeds)
    return [build_run_config(cfg, d, s, a, sd)
            for d in depths for s in sizes for a in agents for sd in seeds]


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> int:
    drift = test_schema_matches_config()
    if drift:
        print("WARNING: schema.py has drifted from config.py:")
        for d in drift:
            print("   ", d)
        print()

    runs = expand(cfg)
    e = cfg.experiment
    if list(e.depths) and int(cfg.arch.depth) not in [int(d) for d in e.depths]:
        print(f"NOTE: arch.depth={cfg.arch.depth} is ignored -- experiment "
              f"'{e.name}' sweeps depths {list(e.depths)}. To run a single depth, "
              f"use experiment=adhoc, or override experiment.depths=[{cfg.arch.depth}].\n")
    per_pack = int(cfg.slurm.gpus_per_job) * int(cfg.slurm.runs_per_gpu)
    n_packs = -(-len(runs) // per_pack)

    print("=" * 76)
    print(f"experiment: {e.name}      {len(runs)} runs, {n_packs} packs of {per_pack}")
    print("=" * 76)
    r0 = runs[0]
    a0, k0 = r0["agent"], r0["arch"]
    print(f"  dataset   {cfg.dataset.name}  (env {cfg.dataset.env_name})")
    # Resolved, not cfg.* -- a recipe can change arch and agent after composition.
    print(f"  arch      tower={k0['tower']} width={k0['width']} "
          f"repr_dim={k0['repr_dim']} init={k0['init']}")
    print(f"  agent cfg recipe={a0['recipe']} batch={a0['batch_size']} "
          f"alpha={a0['alpha']} loss={a0['critic_loss']}/{a0['logits']} "
          f"lse={a0['logsumexp_penalty']}")
    print(f"  goals     actor traj/random={a0['actor_p_trajgoal']}/"
          f"{a0['actor_p_randomgoal']} geom={a0['actor_geom_sample']} | "
          f"value traj/random={a0['value_p_trajgoal']}/{a0['value_p_randomgoal']} "
          f"geom={a0['value_geom_sample']}")
    print(f"  depths    {list(e.depths) or [int(cfg.arch.depth)]}")
    print(f"  agents    {list(e.agents) or [cfg.agent.agent]}   "
          f"seeds {list(e.seeds) or list(cfg.train.seeds)}")
    print(f"  optim     {cfg.optim.optimizer} lr={cfg.optim.lr} "
          f"schedule={cfg.optim.schedule} clip={cfg.optim.max_grad_norm}")
    print(f"  eval      {cfg.eval.episodes} eps/task x {cfg.eval.n_tasks} tasks "
          f"every {cfg.train.eval_interval:,} steps")
    print(f"  plateau   patience={cfg.train.plateau_patience} "
          f"window={cfg.train.plateau_window} min_steps={cfg.train.plateau_min_steps:,}")
    if e.note:
        print(f"  note      {' '.join(str(e.note).split())}")
    if e.targets:
        print(f"  TARGETS   {' '.join(str(e.targets).split())}")

    dev = runs[0]["optim_deviations"]
    if dev:
        print("\n  !! OPTIMIZER DEVIATES FROM BOTH REFERENCE BASELINES:")
        for d in dev:
            print(f"       {d}")
        print("     These runs are not comparable to the published numbers; keep them "
              "out of the scaling fit.")
    if r0.get("recipe_overrode_published"):
        print("\n  recipe took precedence over published flags:")
        for k, v in r0["recipe_overrode_published"].items():
            print(f"       {k}: published={v['published']} -> recipe={v['recipe']}")

    pub = runs[0]["published_flags"]
    print(f"\n  published flags applied: {pub if pub else '(none for this dataset/agent)'}")
    if not cfg.agent.get("use_published_flags", True):
        print("  !! use_published_flags=false -- NOT what OGBench ran.")

    print(f"\n  step caps: {sorted({r['step_cap'] for r in runs})}")
    print(f"  example run_id: {runs[0]['run_id']}")

    if cfg.dry_run:
        print("\ndry_run=true -- nothing written")
        for r in runs[:8]:
            print(f"    {r['run_id']}")
        if len(runs) > 8:
            print(f"    ... and {len(runs) - 8} more")
        return 0

    # Write run configs and append to the manifest `submit_packs.py` reads.
    os.makedirs(cfg.paths.configs_dir, exist_ok=True)
    rows = []
    for r in runs:
        p = os.path.join(cfg.paths.configs_dir, f"{r['run_id']}.json")
        with open(p, "w") as f:
            json.dump(r, f, indent=2)
        rows.append({
            "run_id": r["run_id"], "grid": r["grid"], "agent": r["agent"]["agent"],
            "depth": r["arch"]["depth"], "size": r["size"], "seed": r["seed"],
            "subset_id": r["subset_id"], "step_cap": r["step_cap"],
            "critic_loss": r["agent"].get("critic_loss"),
            "repr_dim": r["arch"]["repr_dim"],
            "status": "pending", "attempts": 0, "job_id": None,
            "config_path": p,
            "run_dir": os.path.join(cfg.paths.runs_dir, r["run_id"]),
        })

    manifest = cfg.paths.manifest
    existing = []
    if os.path.exists(manifest):
        with open(manifest) as f:
            existing = json.load(f)
    have = {x["run_id"] for x in existing}
    new = [x for x in rows if x["run_id"] not in have]
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest + ".tmp", "w") as f:
        json.dump(existing + new, f, indent=2)
    os.replace(manifest + ".tmp", manifest)
    print(f"\nwrote {len(rows)} configs to {cfg.paths.configs_dir}")
    print(f"manifest {manifest}: {len(existing) + len(new)} rows ({len(new)} new)")

    # Record the exact composed config alongside the runs.
    snap = os.path.join(cfg.paths.configs_dir, f"_composed_{e.name}.yaml")
    with open(snap, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"composed config snapshot: {snap}")

    if cfg.submit == "none":
        print(f"\nsubmit=none -- nothing queued. To launch:\n"
              f"  python src/scripts/submit_packs.py --grid {e.name} --chain")
        return 0

    cmd = [sys.executable, os.path.join(_HERE, "submit_packs.py"),
           "--grid", str(e.name)]
    if cfg.submit == "chain":
        cmd.append("--chain")
    if cfg.after:
        cmd += ["--after", str(cfg.after)]
    print(f"\nsubmitting: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
