#!/usr/bin/env python3
"""Estimate wall-clock and GPU-hours for an experiment YAML, before submitting it.

    python src/scripts/estimate_runtime.py experiments/foo.yaml
    python src/scripts/estimate_runtime.py experiments/foo.yaml --verbose
    python src/scripts/estimate_runtime.py --calibrate       # refit from finished runs

Two costs, measured separately because they scale with different things:

  TRAIN  gradient steps / throughput. Throughput falls with depth and with the
         critic's width, and CRL costs ~2.6x GCBC at equal depth (two extra towers
         plus the actor's critic query).

  EVAL   (step_cap / eval_interval) evaluations, each `5 tasks x eval_episodes`
         rollouts of up to `max_episode_steps`. This is the term people forget. On
         humanoidmaze the episode limit is 2000 instead of antmaze's 1000, so every
         evaluation costs twice as much there for the same settings -- and because a
         failing policy never terminates early, eval is MOST expensive exactly when
         the run is going badly.

Calibrated from this project's own finished runs (`--calibrate` refits and prints the
constants to paste back). Anything not covered by a measurement is extrapolated and
labelled as such, so an estimate never quietly pretends to be a measurement.
"""

# --- interpreter guard ------------------------------------------------------
# Stellar's system python3 is 3.6.8; this project needs 3.9+. Running this file
# directly picks up /usr/bin/python3 via the shebang and used to die with
# "SyntaxError: future feature annotations is not defined" -- an error that names
# neither the real cause nor the fix. Re-exec into the project venv instead, so
# `./src/scripts/<name>.py` and `$VENV/bin/python src/scripts/<name>.py` both work.
#
# This block must stay 3.6-parseable and above every other import. Annotations
# below are fine: they are evaluated when each `def` runs, which is after this.
import os
import sys

_PY = os.environ.get("RL_SCALING_PYTHON") or os.path.join(
    os.environ.get("RL_SCALING_SCRATCH", "/scratch/gpfs/vt7993"),
    "rl-scaling", "venv", "bin", "python")
if sys.version_info < (3, 9):
    if not os.path.exists(_PY):
        sys.stderr.write(
            "this needs Python 3.9+ (found %d.%d) and the project venv is not at\n"
            "  %s\nSet RL_SCALING_PYTHON to your interpreter, or run it as\n"
            "  <venv>/bin/python %s\n"
            % (sys.version_info[0], sys.version_info[1], _PY, __file__))
        raise SystemExit(1)
    os.execv(_PY, [_PY, os.path.abspath(__file__)] + sys.argv[1:])
# ----------------------------------------------------------------------------

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402

# --- calibration ------------------------------------------------------------
#: Measured steps/s, antmaze-medium-stitch (ob 29, act 8), width 256, repr_dim 64,
#: batch 1024, one A100. Refit with --calibrate.
THROUGHPUT = {
    ("crl", 4): 151.6, ("crl", 8): 85.2, ("crl", 16): 48.5,
    ("gcbc", 4): 400.2, ("gcbc", 8): 263.2, ("gcbc", 16): 186.1, ("gcbc", 32): 106.9,
}
#: Measured eval wall-clock per evaluation (5 tasks x 15 episodes, 1000-step limit),
#: antmaze-medium-stitch. Scales with episodes and with the episode limit.
EVAL_SECS_PER_EVAL = 160.3   # refit --calibrate, n=272 evaluations
EVAL_REF = dict(episodes=15, tasks=5, max_steps=1000)

#: Deeper-than-measured extrapolation: throughput roughly halves per depth doubling
#: once the tower dominates. Measured 151.6 -> 85.2 -> 48.5 is x0.56, x0.57.
DEPTH_FALLOFF = 0.57


def throughput(agent: str, depth: int, ob_dim: int, act_dim: int,
               batch: int, width: int, repr_dim: int,
               ensemble: int = 2) -> tuple[float, bool]:
    """steps/s, and whether the value came from a measurement at that depth."""
    exact = THROUGHPUT.get((agent, depth))
    measured = exact is not None
    if exact is None:
        known = sorted(d for a, d in THROUGHPUT if a == agent)
        base_d = min(known, key=lambda d: abs(d - depth))
        exact = THROUGHPUT[(agent, base_d)]
        steps = 0
        d = base_d
        while d < depth:
            d *= 2; steps += 1
        while d > depth:
            d //= 2; steps -= 1
        exact *= DEPTH_FALLOFF ** steps

    # Input width enters only the first layer, so it is a small correction, not a
    # proportional one. Reference: ob 29 + act 8.
    ref_in = 29 + 8
    this_in = ob_dim + act_dim
    exact *= (ref_in / this_in) ** 0.15
    exact *= (1024 / batch) ** 0.25          # larger batches are sublinearly slower
    exact *= (256 / width) ** 0.6

    # Ensemble size. CRL's critic is `ensemble` copies each of phi and psi; the actor
    # adds one more tower plus a critic query. At ensemble 2 the critic is ~4 of ~5
    # towers, at 1 it is ~2 of ~3, so dropping to 1 should buy roughly 5/3 on the
    # critic-bound part. EXTRAPOLATED -- we have no ensemble-1 timing yet, so the
    # caller is told.
    ensemble_extrapolated = False
    if agent == "crl" and ensemble != 2:
        towers_2 = 2 * 2 + 1
        towers_n = 2 * ensemble + 1
        exact *= towers_2 / towers_n
        ensemble_extrapolated = True
    return exact, measured and not ensemble_extrapolated


def eval_secs(episodes: int, tasks: int, max_steps: int, agent_scale: float) -> float:
    r = EVAL_REF
    return (EVAL_SECS_PER_EVAL
            * (episodes / r["episodes"]) * (tasks / r["tasks"])
            * (max_steps / r["max_steps"]) * agent_scale)


def max_episode_steps(dataset_name: str) -> int:
    """Read the episode limit from the gym registry. Never silently guesses.

    OGBench's registered env id is NOT the dataset name: `make_env_and_datasets`
    drops the dataset-type token, so `humanoidmaze-medium-navigate-v0` is served by
    the env `humanoidmaze-medium-v0` (`ogbench/utils.py` L180). Looking the dataset
    name up directly fails, and an earlier version of this function then fell back to
    1000 -- halving the estimate for every humanoid run, since that env's limit is
    2000. Defaulting on a lookup failure is how an estimate becomes a wrong number
    with no warning, so this raises instead.
    """
    import gymnasium
    import ogbench  # noqa: F401  (registers the ids)

    splits = dataset_name.split("-")
    env_id = "-".join(splits[:-2] + splits[-1:])   # drop the dataset-type token
    for candidate in (env_id, dataset_name):
        try:
            n = gymnasium.spec(candidate).max_episode_steps
            if n:
                return int(n)
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit(
        f"could not resolve an episode limit for {dataset_name!r} "
        f"(tried {env_id!r}). Refusing to guess -- the eval term depends on it.")


def estimate(runs: list[dict], env_steps: int, verbose: bool = False) -> dict:
    rows, total_train, total_eval = [], 0.0, 0.0
    any_extrapolated = False
    for r in runs:
        a, arch = r["agent"], r["arch"]
        tp, measured = throughput(a["agent"], arch["depth"],
                                  r.get("_ob_dim", 29), r.get("_act_dim", 8),
                                  a["batch_size"], arch["width"], arch["repr_dim"],
                                  int(a.get("critic_ensemble", 2)))
        any_extrapolated |= not measured
        train_s = r["step_cap"] / tp
        n_evals = r["step_cap"] // r["train"]["eval_interval"]
        # A policy that fails never terminates early, so eval cost is an upper bound
        # that is actually ATTAINED by the runs you most want to watch.
        ev_s = n_evals * eval_secs(r["train"]["eval_episodes"], 5, env_steps,
                                   1.0 if a["agent"] == "crl" else 0.85)
        total_train += train_s
        total_eval += ev_s
        rows.append(dict(run_id=r["run_id"], agent=a["agent"], depth=arch["depth"],
                         steps_per_s=tp, measured=measured, n_evals=n_evals,
                         train_h=train_s / 3600, eval_h=ev_s / 3600,
                         total_h=(train_s + ev_s) / 3600))
    return dict(rows=rows, train_h=total_train / 3600, eval_h=total_eval / 3600,
                total_h=(total_train + total_eval) / 3600,
                extrapolated=any_extrapolated)


def calibrate() -> None:
    """Refit the constants from every finished run on scratch."""
    import statistics as st
    tp: dict[tuple[str, int], list[float]] = {}
    ev: list[float] = []
    for d in sorted(glob.glob(os.path.join(C.RUNS_DIR, "*"))):
        sp, cp = os.path.join(d, "summary.json"), os.path.join(d, "config.json")
        if not (os.path.exists(sp) and os.path.exists(cp)):
            continue
        s, c = json.load(open(sp)), json.load(open(cp))
        if c["arch"]["tower"] != "residual":
            continue
        tp.setdefault((c["agent"]["agent"], c["arch"]["depth"]), []).append(
            s["gradient_steps"] / max(s["train_secs"], 1e-9))
        ep = os.path.join(d, "eval.csv")
        if os.path.exists(ep):
            v = [float(r["eval_secs"]) for r in csv.DictReader(open(ep))
                 if r.get("eval_secs")]
            if v and c["train"]["eval_episodes"] == 15:
                ev += v
    print("THROUGHPUT = {")
    for k in sorted(tp):
        print(f'    ("{k[0]}", {k[1]}): {st.mean(tp[k]):.1f},'
              f'   # n={len(tp[k])}')
    print("}")
    if ev:
        print(f"EVAL_SECS_PER_EVAL = {st.mean(ev):.1f}"
              f"   # n={len(ev)}, median {st.median(ev):.1f}")


def report(path: str, verbose: bool = False) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import run as R

    cfg = R.load(path)
    cfg["_source"] = path
    runs = R.resolve_runs(cfg)
    ds = C.DATASETS[cfg["dataset"]]
    for r in runs:
        r["_ob_dim"], r["_act_dim"] = ds.ob_dim, ds.act_dim
    env_steps = max_episode_steps(cfg["dataset"])

    e = estimate(runs, env_steps, verbose)
    slots = max(1, cfg["gpus_per_job"] * cfg["runs_per_gpu"])
    packs = -(-len(runs) // slots)

    print(f"\n{cfg['name']}   ({path})")
    print(f"  {len(runs)} runs on {cfg['dataset']}  "
          f"(episode limit {env_steps}, ob {ds.ob_dim}, act {ds.act_dim})")
    print(f"\n  train      {e['train_h']:7.1f} GPU-h")
    print(f"  eval       {e['eval_h']:7.1f} GPU-h   "
          f"({e['eval_h'] / max(e['total_h'], 1e-9) * 100:.0f}% of the total)")
    print(f"  TOTAL      {e['total_h']:7.1f} GPU-h")

    wall = e["total_h"] / slots
    print(f"\n  {slots} concurrent run slots ({cfg['gpus_per_job']} GPUs x "
          f"{cfg['runs_per_gpu']} runs), {packs} pack(s)")
    print(f"  wall-clock ~{wall:.1f} h  ({wall / 24:.1f} days) if nothing queues")
    lim = cfg["time_limit"]
    lim_h = int(lim.split(":")[0])
    worst = max(r["total_h"] for r in e["rows"])
    print(f"  longest single run ~{worst:.1f} h vs the {lim_h} h job limit"
          + ("  -- OK" if worst < lim_h * 0.9 else
             "  -- WILL HIT THE WALL, the graceful deadline stop will truncate it"))

    if e["extrapolated"]:
        print("\n  NOTE: some depths have no measurement and are extrapolated at "
              f"x{DEPTH_FALLOFF} per doubling. Treat as +/- 40%, not a measurement.")

    if verbose:
        print(f"\n  {'agent':<6}{'depth':>6}{'steps/s':>10}{'src':>6}"
              f"{'evals':>7}{'train_h':>9}{'eval_h':>8}{'total_h':>9}")
        seen = set()
        for r in e["rows"]:
            k = (r["agent"], r["depth"])
            if k in seen:
                continue
            seen.add(k)
            print(f"  {r['agent']:<6}{r['depth']:>6}{r['steps_per_s']:>10.1f}"
                  f"{'meas' if r['measured'] else 'extr':>6}{r['n_evals']:>7}"
                  f"{r['train_h']:>9.1f}{r['eval_h']:>8.1f}{r['total_h']:>9.1f}")
        print("  (one row per distinct cell; multiply by the seed count)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?")
    ap.add_argument("--verbose", action="store_true", help="per-cell breakdown")
    ap.add_argument("--calibrate", action="store_true",
                    help="refit the constants from finished runs and print them")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
        return 0
    if not args.experiment:
        ap.error("give an experiments/*.yaml, or --calibrate")
    return report(args.experiment, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
