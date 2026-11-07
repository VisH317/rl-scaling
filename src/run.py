#!/usr/bin/env python3
"""One entry point for every offline CRL scaling experiment.

    python src/run.py experiments/foo.yaml              # resolve + print the plan
    python src/run.py experiments/foo.yaml --submit     # write configs + queue packs
    python src/run.py experiments/foo.yaml --status     # progress digest
    python src/run.py experiments/foo.yaml --figures    # regenerate plots + CSVs
    python src/run.py experiments/foo.yaml --reeval     # queue the 50 eps/task rescore

The YAML is the whole experiment. Any key may be a list, which makes it a sweep axis;
runs are the full product of every list times `seeds`. `experiments/TEMPLATE.yaml`
documents every key. `src/offline/config.py` is now a *registry* -- datasets, paths,
published per-dataset flags -- and no longer a place you tune anything.

This replaces the Hydra layer (`conf/`, `schema.py`, `scripts/launch.py`), which is
removed; it lives in git history at 53177a7 if ever needed. Hydra's composition bought
override syntax we were not using and cost a second place for every default to live,
and a drift test to keep the two in sync. One file, read once, is less to get wrong.

Output of a resolve is exactly the per-run JSON `offline/train.py --run-config` already
consumes, so nothing downstream changes.
"""

# Interpreter guard and import paths, in one line. MUST come first and stay
# 3.6-parseable: Stellar's system python3 is 3.6.8 and importing this re-execs into
# the project venv. See src/rl_scaling/bootstrap.py for why.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_scaling.bootstrap  # noqa: E402,F401

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline import config as C  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("pyyaml is required:  pip install pyyaml")


# ---------------------------------------------------------------------------
# Profiles: named bundles of fields that are only meaningful together.
# ---------------------------------------------------------------------------
#: The Q1 fields move as a unit. Forward InfoNCE's row softmax is invariant to adding
#: c(s,a) to a row, so it pins the density ratio only up to a function of (s,a) -- and
#: the actor's max_a Q(s,a,g) compares across exactly that freedom. The logsumexp
#: penalty is what removes the invariance. Setting `critic_loss: infonce` with
#: `logsumexp_penalty: 0` would break actor extraction for reasons unrelated to depth,
#: so the profiles exist to make the coupled choice the easy one.
#: `critic_ensemble` IS now implemented (offline/critic_losses.py) -- it needs two
#: coordinated patches, ensemblize's axis_size and the actor's `q1, q2` unpack, so it
#: was a hard error until both existed. n=2 stays a no-op on OGBench's own code.
PROFILES: dict[str, dict] = {
    "ogbench": dict(
        critic_loss="binary_nce", logits="dot", logsumexp_penalty=0.0,
        critic_ensemble=2, batch_size=1024,
    ),
    # NOTE on `critic_ensemble`: BOTH codebases have two critic ENCODERS, phi(s,a)
    # and psi(g) -- that is the bilinear critic, not an ensemble. What differs is
    # that OGBench vmaps each encoder into a 2-member ENSEMBLE (4 towers, giving
    # q1/q2 for the actor's min), while scaling-crl uses 1 member (2 towers).
    # We keep OGBench's 2: `crl.py`'s ddpgbc actor does `q1, q2 = critic(...)`, so
    # an ensemble of 1 would crash. Recorded as a known deviation, not silently
    # accepted.
    "scaling_crl": dict(
        critic_loss="infonce", logits="neg_l2", logsumexp_penalty=0.1,
        critic_ensemble=1, batch_size=256, repr_dim=64,
    ),
    "stable_crl": dict(
        critic_loss="binary_nce", logits="dot", logsumexp_penalty=0.0,
        critic_ensemble=2, batch_size=2048, repr_dim=16, init="cold", alpha=0.05,
    ),
}

#: Keys that may be swept (appear as a list). Everything else must be a scalar.
SWEEPABLE = {"agent", "depth", "data_size", "profile", "repr_dim", "width",
             "critic_loss", "logits", "logsumexp_penalty", "batch_size", "lr",
             "alpha", "init", "tower", "block_size", "critic_ensemble", "seeds",
             "discount"}

DEFAULTS: dict = dict(
    name=None, dataset="antmaze-medium-stitch-v0", seeds=[0, 1, 2], data_size=None,
    agent=["crl", "gcbc"], depth=[4, 8, 16, 32], profile="ogbench",
    critic_loss=None, logits=None, logsumexp_penalty=None, critic_ensemble=None,
    repr_dim=64,
    tower="residual", width=256, block_size=4, activation="swish", norm="layer_norm",
    init="lecun_uniform",
    actor_loss="ddpgbc", alpha=None, discount=None,
    actor_p_trajgoal=None, actor_p_randomgoal=None,
    use_published_flags=True,
    lr=3.0e-4, batch_size=None, schedule="constant", warmup_steps=0,
    max_grad_norm=0.0, weight_decay=0.0,
    step_cap=2_000_000, plateau_min_steps=1_000_000, plateau_patience=5,
    plateau_window=3, plateau_min_delta=0.01,
    eval_interval=100_000, eval_episodes=15,
    gpus_per_job=2, runs_per_gpu=3, time_limit="24:00:00",
    #: Two-stage extraction (see src/offline/extract.py). `extract_from` names the
    #: grid whose trained critics to freeze; each run pairs with the source run at the
    #: SAME agent/depth/seed. `actor_depth` pins the policy trunk while `depth` keeps
    #: selecting the critic. Either may be used alone: actor_depth on its own simply
    #: trains a mismatched pair from scratch.
    extract_from=None, extract_checkpoint="best", actor_depth=None,
)

#: Deviating from any of these confounds the depth axis.
OPTIM_BASELINE = dict(lr=3.0e-4, schedule="constant", warmup_steps=0,
                      max_grad_norm=0.0, weight_decay=0.0)

#: YAML keys that may override a profile or a published flag. Anything here is read
#: from the experiment file when set; `None` means "take the profile/published value".
OVERRIDABLE = ("critic_loss", "logits", "logsumexp_penalty", "critic_ensemble",
               "batch_size", "repr_dim", "alpha", "init", "discount",
               "actor_p_trajgoal", "actor_p_randomgoal")

#: Knobs the config can express but NOTHING downstream reads. Setting one of these
#: silently does nothing, which is worse than not offering it -- so `run.py` refuses
#: to resolve rather than hand back a config that lies about what will run.
#:
#: Audited 2026-11-05 by tracing each field to its consumer. The optimizer is built
#: inside OGBench's `crl.py` as a bare `optax.adam(config['lr'])`; `train.py` imports
#: optax but never constructs one. So of the eight optimizer fields, exactly one -- lr
#: -- reaches the optimizer.
UNIMPLEMENTED: dict[str, tuple[object, str]] = {
    "schedule": ("constant",
                 "crl.py builds a bare optax.adam(lr); no schedule is applied"),
    "warmup_steps": (0, "same: no schedule wrapper exists"),
    "max_grad_norm": (0.0, "same: no optax.clip_by_global_norm in the chain"),
    "weight_decay": (0.0, "same: adam, not adamw"),
    "actor_loss": ("ddpgbc",
                   "awr needs a separate value network; build_config does not create "
                   "one, so it would fail at agent construction"),
}


def load(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        sys.exit(f"unknown key(s) in {path}: {sorted(unknown)}\n"
                 f"see experiments/TEMPLATE.yaml for the full list")
    cfg = {**DEFAULTS, **raw}
    cfg["name"] = cfg["name"] or os.path.splitext(os.path.basename(path))[0]
    for k, v in cfg.items():
        if isinstance(v, list) and k not in SWEEPABLE:
            sys.exit(f"'{k}' is not a sweepable axis; give it a single value")
    if cfg["dataset"] not in C.DATASETS:
        sys.exit(f"unknown dataset {cfg['dataset']!r}; known: {sorted(C.DATASETS)}")
    if cfg["extract_from"] is not None:
        agents = cfg["agent"] if isinstance(cfg["agent"], list) else [cfg["agent"]]
        if set(agents) - {"crl"}:
            sys.exit(f"extract_from is CRL-only: gcbc has no critic to freeze "
                     f"(got agent={agents})")
        if cfg["actor_depth"] is None:
            sys.exit("extract_from without actor_depth re-extracts at the critic's own "
                     "depth, which reproduces the joint run and tests nothing; set "
                     "actor_depth explicitly (use the critic depth if that is intended)")

    bad = []
    for k, (supported, why) in UNIMPLEMENTED.items():
        vals = cfg[k] if isinstance(cfg[k], list) else [cfg[k]]
        for v in vals:
            if v is not None and v != supported:
                bad.append(f"  {k}: {v!r} is not implemented (only {supported!r} is)\n"
                           f"      {why}")
    if bad:
        sys.exit(f"{path} asks for settings nothing downstream reads:\n"
                 + "\n".join(bad)
                 + "\n\n  A knob that silently does nothing is worse than no knob, so "
                   "this is a hard error.\n  Implement it, or drop the key.")
    return cfg


def _axes(cfg: dict) -> tuple[list[str], list[list]]:
    names, values = [], []
    for k in sorted(SWEEPABLE - {"seeds"}):
        v = cfg.get(k)
        if isinstance(v, list):
            names.append(k)
            values.append(v)
    return names, values


def resolve_runs(cfg: dict) -> list[dict]:
    """Expand the sweep into fully-resolved per-run config dicts."""
    names, values = _axes(cfg)
    seeds = cfg["seeds"] if isinstance(cfg["seeds"], list) else [cfg["seeds"]]
    runs = []
    for combo in itertools.product(*values) if values else [()]:
        point = {**cfg, **dict(zip(names, combo))}
        for seed in seeds:
            runs.append(_one(point, seed, names))
    return runs


#: Axes that `plot_results.py` groups by. Anything else varying inside ONE grid is
#: averaged together silently, which is why `grid_suffix` exists.
VISIBLE_AXES = {"agent", "depth", "data_size", "seeds"}


def _grid_name(p: dict, swept: list[str]) -> str:
    """Grid name for one sweep point.

    Figures group by (agent, depth), so two profiles at the same depth inside one grid
    would be silently averaged into a single point. Splitting the grid per profile
    keeps every downstream plot correct without teaching them a new axis.
    """
    if "profile" in swept:
        return f"{p['name']}__{p['profile']}"
    return p["name"]


def _one(p: dict, seed: int, swept: list[str] = ()) -> dict:
    ds = C.DATASETS[p["dataset"]]
    size = p["data_size"] if p["data_size"] is not None else ds.transitions_per_shard
    agent_name = p["agent"]

    # Precedence, lowest first: profile -> published flags -> explicit YAML keys.
    # Explicit wins, because the only reason to write a key by hand is to override.
    prof = dict(PROFILES[p["profile"]])
    pub = (C.PUBLISHED_FLAGS.get((p["dataset"], agent_name), {})
           if p["use_published_flags"] else {})
    explicit = {k: p[k] for k in OVERRIDABLE if p.get(k) is not None}
    merged = {**prof, **pub, **explicit}

    # Build the defaults first, then overlay `merged` wholesale. Listing the keys to
    # copy by hand is how `discount: 0.995` -- OGBench's published value for
    # humanoidmaze -- ended up recorded in `published_flags` but never applied: the
    # hand-written list did not mention it. Anything a profile or a published command
    # can carry now lands, and an unknown key is an error rather than a silent drop.
    agent = dict(
        agent=agent_name, recipe=p["profile"],
        critic_loss="binary_nce", logits="dot", logsumexp_penalty=0.0,
        critic_ensemble=2, batch_size=1024,
        lr=float(p["lr"]), discount=0.99, alpha=0.1,
        actor_loss=p["actor_loss"], const_std=True,
        value_p_trajgoal=1.0, value_p_randomgoal=0.0, value_geom_sample=True,
        actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=False,
    )
    arch_only = {"repr_dim", "init"}
    for k, v in merged.items():
        if k in arch_only:
            continue
        if k not in agent:
            sys.exit(f"profile/published flag {k!r} has nowhere to go in the agent "
                     f"config; add it to the defaults in run.py._one")
        agent[k] = type(agent[k])(v) if not isinstance(v, str) else v
    unapplied = {k: v for k, v in pub.items() if agent.get(k) != v
                 and k not in explicit}
    if unapplied:
        sys.exit(f"published flags {unapplied} did not reach the agent config -- "
                 f"this is the bug class that produced the actor_p_trajgoal error")
    arch = dict(
        depth=int(p["depth"]), width=int(p["width"]),
        repr_dim=int(merged.get("repr_dim", p["repr_dim"])),
        block_size=int(p["block_size"]), norm=p["norm"], activation=p["activation"],
        init=merged.get("init", p["init"]), tower=p["tower"],
        ogbench_hidden_dims=[512, 512, 512],
    )
    # Absent unless set, so no existing run id moves -- see ArchConfig.actor_depth.
    if p["actor_depth"] is not None:
        arch["actor_depth"] = int(p["actor_depth"])
    cfg = dict(
        grid=_grid_name(p, list(swept)), dataset=p["dataset"], env_name=ds.env_name,
        subset_id=C._subset_id(p["dataset"], p["data_size"]), size=size, seed=int(seed),
        arch=arch, agent=agent,
        train=dict(
            eval_interval=int(p["eval_interval"]), log_interval=10_000,
            save_interval=500_000, eval_episodes=int(p["eval_episodes"]),
            eval_temperature=0.0, video_episodes=0,
            plateau_patience=int(p["plateau_patience"]),
            plateau_min_delta=float(p["plateau_min_delta"]),
            plateau_window=int(p["plateau_window"]),
            plateau_min_steps=int(p["plateau_min_steps"]),
            seeds=list(p["seeds"]) if isinstance(p["seeds"], list) else [p["seeds"]],
        ),
        step_cap=int(p["step_cap"]),
        published_flags=pub,
        optim=dict(lr=float(p["lr"]), actor_lr=None, critic_lr=None, optimizer="adam",
                   b1=0.9, b2=0.999, eps=1e-8, weight_decay=float(p["weight_decay"]),
                   schedule=p["schedule"], warmup_steps=int(p["warmup_steps"]),
                   end_lr_fraction=0.1, max_grad_norm=float(p["max_grad_norm"])),
        checkpoint=dict(save_best=True, save_final=True, periodic_interval=500_000,
                        keep_periodic=2, compress=False, heartbeat_interval=5_000),
        # Only fields that actually reach the optimizer; `load()` has already
        # rejected the ones that do not.
        optim_deviations=[k for k, v in OPTIM_BASELINE.items()
                          if p[k] != v and k not in UNIMPLEMENTED],
    )
    if p["extract_from"] is not None:
        # The SOURCE RUN IS NOT NAMED HERE, deliberately. Storing a concrete run id
        # would mean this config could only be written after the source grid had
        # finished, and would break if the source were ever re-run. We store the grid
        # plus the checkpoint, and `train.py` resolves it by (agent, depth, seed) --
        # the triple already in this config -- erroring unless exactly one run matches.
        cfg["extract"] = dict(source_grid=str(p["extract_from"]),
                              checkpoint=str(p["extract_checkpoint"]),
                              actor_depth=int(p["actor_depth"])
                              if p["actor_depth"] is not None else None)
    # The id is a hash of the config, so provenance must be attached AFTER it is
    # computed -- otherwise renaming the YAML would rename every run and orphan
    # finished work. Verified: with this ordering, phase1_repro.yaml and
    # phase1_anchor.yaml reproduce the run ids of the runs already on disk exactly.
    cfg["run_id"] = C._run_id(cfg)
    cfg["experiment_file"] = p.get("_source")
    return cfg


# ---------------------------------------------------------------------------
# Warnings: things that are legal but will quietly ruin an experiment.
# ---------------------------------------------------------------------------

def check(cfg: dict, runs: list[dict]) -> list[str]:
    warn = []
    for k, base in OPTIM_BASELINE.items():
        if k not in UNIMPLEMENTED and cfg[k] != base:
            warn.append(f"optim.{k}={cfg[k]} (baseline {base}) -- warmup, clipping and "
                        f"decay help DEEP nets specifically, so this confounds depth")
    if not cfg["use_published_flags"]:
        warn.append("use_published_flags=false -- CRL will run crl.py's bare defaults "
                    "(actor_p_trajgoal=1.0), not OGBench's published command")
    for r in runs:
        a = r["agent"]
        if a["critic_loss"] == "infonce" and a["logsumexp_penalty"] == 0.0:
            warn.append("infonce with logsumexp_penalty=0 -- the row softmax is "
                        "invariant to adding c(s,a) to a row, which is exactly what "
                        "max_a Q(s,a,g) compares across. Actor extraction will likely "
                        "fail for reasons unrelated to depth. Use profile: scaling_crl")
            break
    for r in runs:
        if r["agent"]["logits"] == "neg_l2":
            warn.append("logits=neg_l2 -- every logit is <= 0, so the logged "
                        "`binary_accuracy` is meaningless. Read categorical_accuracy")
            break
    swept = set(_axes(cfg)[0])
    hidden = sorted(swept - VISIBLE_AXES - {"profile"})
    if hidden:
        grids = {r["grid"] for r in runs}
        if len(grids) == 1:
            warn.append(f"sweeping {hidden} inside one grid -- figures group by "
                        f"(agent, depth), so these variants will be AVERAGED "
                        f"together. Split them into separate experiment files, or "
                        f"sweep `profile` instead (which splits the grid for you)")
    if any(r["arch"]["tower"] == "ogbench_mlp" for r in runs) and \
            isinstance(cfg["depth"], list) and len(cfg["depth"]) > 1:
        warn.append("tower=ogbench_mlp ignores `depth` -- the sweep will produce "
                    "duplicate runs that differ only in their id")
    return sorted(set(warn))




def plan(cfg: dict, runs: list[dict], source: str) -> None:
    names, values = _axes(cfg)
    print(f"\n{cfg['name']}   ({source})")
    print(f"  dataset   {cfg['dataset']}  size {runs[0]['size']:,}")
    print(f"  axes      " + ("  ".join(f"{n}={v}" for n, v in zip(names, values))
                             or "(none -- single point)"))
    print(f"  seeds     {cfg['seeds']}")
    print(f"  runs      {len(runs)}")

    # Show the SET of resolved values, not run[0]'s. Printing the first run's
    # settings while a sweep varies them is how a summary ends up disagreeing
    # with what actually runs.
    def spread(get) -> str:
        vals = sorted({str(get(r)) for r in runs})
        return vals[0] if len(vals) == 1 else "{" + ",".join(vals) + "}"

    print(f"  critic    {spread(lambda r: r['agent']['critic_loss'])} / "
          f"{spread(lambda r: r['agent']['logits'])} / "
          f"lse={spread(lambda r: r['agent']['logsumexp_penalty'])} / "
          f"ensemble={spread(lambda r: r['agent']['critic_ensemble'])} / "
          f"repr_dim={spread(lambda r: r['arch']['repr_dim'])}")
    print(f"  actor     {spread(lambda r: r['agent']['actor_loss'])}  "
          f"alpha={spread(lambda r: r['agent']['alpha'])}  "
          f"discount={spread(lambda r: r['agent']['discount'])}  "
          f"p_trajgoal={spread(lambda r: r['agent']['actor_p_trajgoal'])}/"
          f"p_randomgoal={spread(lambda r: r['agent']['actor_p_randomgoal'])}")
    print(f"  arch      {spread(lambda r: r['arch']['tower'])}  "
          f"width={spread(lambda r: r['arch']['width'])}  "
          f"block={spread(lambda r: r['arch']['block_size'])}  "
          f"init={spread(lambda r: r['arch']['init'])}")
    print(f"  optim     adam lr={cfg['lr']} {cfg['schedule']} "
          f"batch={spread(lambda r: r['agent']['batch_size'])}"
          + ("  [BASELINE]" if not runs[0]["optim_deviations"] else
             f"  [DEVIATES: {runs[0]['optim_deviations']}]"))
    print(f"  train     step_cap={cfg['step_cap']:,}  "
          f"plateau>={cfg['plateau_min_steps']:,}  "
          f"eval every {cfg['eval_interval']:,} x {cfg['eval_episodes']} eps/task")

    # One cost model, in scripts/estimate_runtime.py. Having a second rough one here
    # only produced two numbers that disagreed.
    slots = max(1, cfg["gpus_per_job"] * cfg["runs_per_gpu"])
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "scripts"))
        import estimate_runtime as E
        ds = C.DATASETS[cfg["dataset"]]
        for r in runs:
            r["_ob_dim"], r["_act_dim"] = ds.ob_dim, ds.act_dim
        est = E.estimate(runs, E.max_episode_steps(cfg["dataset"]))
        print(f"  cost      {est['total_h']:.0f} GPU-h "
              f"({est['train_h']:.0f} train + {est['eval_h']:.0f} eval); "
              f"~{est['total_h'] / slots:.0f} h wall across {slots} slots "
              f"({-(-len(runs) // slots)} packs)"
              + ("   [extrapolated]" if est["extrapolated"] else ""))
        print(f"            full breakdown: python src/scripts/estimate_runtime.py "
              f"{cfg.get('_source', '<yaml>')} --verbose")
    except SystemExit as e:
        print(f"  cost      unavailable: {e}")

    for w in check(cfg, runs):
        print(f"\n  WARNING  {w}")

    grids = sorted({r["grid"] for r in runs})
    if grids != [cfg["name"]]:
        print(f"  grids     {len(grids)} (split so figures stay separable): "
              + ", ".join(grids))
    print(f"\n  {'run_id':<62} {'agent':<5} {'d':>3} {'seed':>4}")
    for r in runs[:12]:
        print(f"  {r['run_id']:<62} {r['agent']['agent']:<5} "
              f"{r['arch']['depth']:>3} {r['seed']:>4}")
    if len(runs) > 12:
        print(f"  ... and {len(runs) - 12} more")


def write(runs: list[dict], cfg: dict, source: str) -> list[dict]:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "scripts"))
    import gen_runs
    os.makedirs(C.CONFIGS_DIR, exist_ok=True)
    rows = []
    for r in runs:
        path = os.path.join(C.CONFIGS_DIR, f"{r['run_id']}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        rows.append(dict(
            run_id=r["run_id"], grid=r["grid"], agent=r["agent"]["agent"],
            depth=r["arch"]["depth"], size=r["size"], seed=r["seed"],
            subset_id=r["subset_id"], step_cap=r["step_cap"],
            critic_loss=r["agent"]["critic_loss"], repr_dim=r["arch"]["repr_dim"],
            status="pending", attempts=0, job_id=None, config_path=path,
            run_dir=os.path.join(C.RUNS_DIR, r["run_id"]),
        ))
    # Resolved copy of the experiment file, beside the run configs.
    with open(os.path.join(C.CONFIGS_DIR, f"_{cfg['name']}.resolved.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "_source": source, "_n_runs": len(runs)}, f,
                       sort_keys=True)

    existing = {row["run_id"]: row for row in gen_runs.load_manifest()}
    for row in rows:
        existing.setdefault(row["run_id"], row)
    gen_runs.save_manifest(list(existing.values()))
    print(f"\nwrote {len(rows)} configs to {C.CONFIGS_DIR}")
    return rows


def _sub(script: str, *args: str) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.call([sys.executable, os.path.join(here, "scripts", script),
                            *args])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", help="path to an experiments/*.yaml")
    ap.add_argument("--submit", action="store_true", help="write configs and queue")
    ap.add_argument("--write", action="store_true", help="write configs, queue nothing")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--reeval", action="store_true", help="queue the 50 eps/task rescore")
    ap.add_argument("--force", action="store_true", help="submit despite warnings")
    args = ap.parse_args()

    cfg = load(args.experiment)
    cfg["_source"] = args.experiment
    runs = resolve_runs(cfg)

    if args.status:
        return _sub("monitor.py", "--grid", cfg["name"])
    if args.figures:
        return _sub("plot_results.py", "--grid", cfg["name"])
    if args.reeval:
        from rl_scaling import slurm as S
        snap = S.snapshot()
        env = os.environ.copy()
        env["GRIDS"] = cfg["name"]
        return subprocess.call(
            ["sbatch", f"--export={S.exports(snap, GRIDS=cfg['name'])}",
             f"--output={S.log_option('eval')}",
             S.sbatch_script(snap, "reeval.sbatch")], env=env)

    plan(cfg, runs, args.experiment)
    if not (args.submit or args.write):
        print("\n(nothing written -- add --write to generate configs, "
              "--submit to also queue)")
        return 0

    warns = check(cfg, runs)
    if warns and args.submit and not args.force:
        print("\nrefusing to submit with warnings above; re-run with --force "
              "if they are intended")
        return 2

    write(runs, cfg, args.experiment)
    if args.submit:
        return _sub("submit_packs.py", "--grid", cfg["name"], "--chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
