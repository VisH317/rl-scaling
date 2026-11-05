#!/usr/bin/env python3
"""One entry point for every interpretability eval of a trained checkpoint.

    python src/eval.py                       # resolve + print the plan
    python src/eval.py --write               # write the task files
    python src/eval.py --submit              # write + queue one Slurm job
    python src/eval.py --figures             # rebuild figures from finished tasks
    python src/eval.py --list                # every eval that exists, and its status
    python src/eval.py --status              # which tasks are done, failed, pending
    python src/eval.py --run <task.json>     # execute ONE task (what the job calls)

`src/offline/evals/evals.yaml` is the suite: which grids, which depths, how the one
checkpoint per depth is chosen, and which evals to run on it. Each eval is a directory
under `src/offline/evals/` with its own `eval.yaml`, `README.md` and `impl.py`; this
file never knows what any of them measure.

The split from `src/run.py` is deliberate. `run.py` creates runs; this consumes them and
writes only into the analysis directory. Nothing here can modify a run.
"""

# --- interpreter guard ------------------------------------------------------
# Stellar's system python3 is 3.6.8; this project needs 3.9+. Must stay 3.6-parseable
# and above every other import. See the same block in src/run.py.
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
import glob
import json
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scripts"))

from offline import config as C          # noqa: E402
from offline.evals import common         # noqa: E402

try:
    import yaml
except ImportError:                        # pragma: no cover
    sys.exit("pyyaml is required:  pip install pyyaml")

DEFAULT_SUITE = common.SUITE_FILE

SUITE_KEYS = {"name", "grids", "agents", "depths", "seed_select", "checkpoint",
              "require_finished", "evals", "env_family", "time_limit"}
EVAL_ENTRY_KEYS = {"name", "enabled", "params"}


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def load_suite(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - SUITE_KEYS
    if unknown:
        sys.exit(f"unknown key(s) in {path}: {sorted(unknown)}\n"
                 f"known: {sorted(SUITE_KEYS)}")
    cfg = dict(name=os.path.splitext(os.path.basename(path))[0], grids=[],
               agents=["crl"], depths=[], seed_select="reeval", checkpoint="best",
               require_finished=True, evals=[], time_limit=None)
    cfg.update(raw)
    cfg["_source"] = path
    if not cfg["grids"] or not cfg["depths"]:
        sys.exit(f"{path}: `grids` and `depths` are both required")
    if cfg["seed_select"] not in common.SELECT_RULES:
        sys.exit(f"{path}: seed_select must be one of {common.SELECT_RULES}")
    for e in cfg["evals"]:
        if not isinstance(e, dict) or "name" not in e:
            sys.exit(f"{path}: each `evals` entry needs a `name`")
        bad = set(e) - EVAL_ENTRY_KEYS
        if bad:
            sys.exit(f"{path}: unknown key(s) {sorted(bad)} in eval {e['name']!r}")
    return cfg


def suite_dir(cfg: dict) -> str:
    return os.path.join(common.EVALS_OUT, cfg["name"])


def _env_family(dataset: str) -> str:
    """Which family of environment a dataset belongs to, for `applies_to`-style gating."""
    head = dataset.split("-")[0]
    return {"antmaze": "locomaze", "humanoidmaze": "locomaze", "pointmaze": "locomaze",
            "cube": "manipulation", "scene": "manipulation",
            "puzzle": "manipulation"}.get(head, head)


def resolve(cfg: dict) -> tuple:
    """(tasks, notes). A note is a point that was deliberately dropped, with the why."""
    specs = common.discover()
    unknown = [e["name"] for e in cfg["evals"] if e["name"] not in specs]
    if unknown:
        sys.exit(f"unknown eval(s) {unknown}; found: {sorted(specs)}")

    picks = common.select_runs(cfg["grids"], cfg["agents"], cfg["depths"],
                               rule=cfg["seed_select"],
                               require_finished=cfg["require_finished"])
    tasks, notes = [], []
    for e in cfg["evals"]:
        if e.get("enabled", True) is False:
            notes.append(f"{e['name']}: disabled in {os.path.basename(cfg['_source'])}")
            continue
        spec = specs[e["name"]]
        if spec.status != "ready":
            notes.append(f"{spec.name}: status={spec.status} -- {spec.readme}")
            continue
        params = dict(spec.params)
        params.update(e.get("params") or {})
        unknown_p = set(params) - set(spec.params)
        if unknown_p:
            sys.exit(f"{cfg['_source']}: eval {spec.name!r} has no param(s) "
                     f"{sorted(unknown_p)}; see {spec.dir}/eval.yaml")
        for pick in picks:
            if pick["run_id"] is None:
                notes.append(f"{spec.name}: {pick['grid']} {pick['agent']} "
                             f"d{pick['depth']} -- {pick['reason']}")
                continue
            if pick["agent"] not in spec.applies_to:
                notes.append(f"{spec.name}: {pick['agent']} not in applies_to "
                             f"{list(spec.applies_to)} -- dropped")
                continue
            run_cfg = common.load_run_config(pick["run_dir"])
            fam = _env_family(run_cfg["dataset"])
            if "*" not in spec.env_family and fam not in spec.env_family:
                notes.append(f"{spec.name}: {run_cfg['dataset']} is {fam!r}, eval "
                             f"handles {list(spec.env_family)} -- dropped")
                continue
            out_dir = os.path.join(suite_dir(cfg), spec.name, pick["run_id"])
            tasks.append(dict(
                task_id=f"{spec.name}__{pick['run_id']}", eval=spec.name,
                suite=cfg["name"], run_id=pick["run_id"], run_dir=pick["run_dir"],
                grid=pick["grid"], agent=pick["agent"], depth=pick["depth"],
                seed=run_cfg["seed"], checkpoint=cfg["checkpoint"],
                dataset=run_cfg["dataset"], params=params,
                selection={k: pick[k] for k in
                           ("score", "rule_used", "n_candidates", "n_scored", "spread",
                            "runner_up", "all_seeds")},
                out_dir=out_dir, resources=spec.resources,
            ))
    return tasks, notes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def plan(cfg: dict, tasks: list, notes: list) -> None:
    specs = common.discover()
    print(f"\n{cfg['name']}   ({cfg['_source']})")
    print(f"  grids     {cfg['grids']}   agents {cfg['agents']}   "
          f"depths {cfg['depths']}")
    print(f"  select    {cfg['seed_select']}  checkpoint={cfg['checkpoint']}  "
          f"require_finished={cfg['require_finished']}")
    print(f"  output    {suite_dir(cfg)}")
    print(f"  tasks     {len(tasks)}")

    if tasks:
        print(f"\n  {'eval':<16}{'d':>4}{'seed':>5}{'select':>9}  {'rule':<16}"
              f"{'spread':>8}  run_id")
        for t in tasks:
            s = t["selection"]
            print(f"  {t['eval']:<16}{t['depth']:>4}{t['seed']:>5}"
                  f"{s['score']:>9.3f}  {s['rule_used']:<16}"
                  f"{s['spread']:>8.3f}  {t['run_id']}")
        print(f"\n  seeds not picked (the spread is as real as the depth effect):")
        seen = set()
        for t in tasks:
            k = (t["grid"], t["agent"], t["depth"])
            if k in seen:
                continue
            seen.add(k)
            print(f"    d{t['depth']:<3} {t['selection']['all_seeds']}")

    if notes:
        print("\n  not run:")
        for n in sorted(set(notes)):
            print(f"    {n}")

    if tasks:
        cap = max(_hms(t["resources"].get("time_limit", "01:00:00")) for t in tasks)
        print(f"\n  cost      {len(tasks)} tasks, run sequentially in ONE job "
              f"(the architecture patch is process-lifetime, so one checkpoint per "
              f"process).\n            Job wall limit "
              f"{_fmt_hms(cap * len(tasks))} = {len(tasks)} x the longest eval's cap. "
              f"That is a CAP, not an estimate;\n            each task records its real "
              f"`secs` in meta.json, which is what to calibrate from.")
    print(f"\n  evals on disk:")
    for name, s in sorted(specs.items()):
        mark = "x" if any(t["eval"] == name for t in tasks) else " "
        print(f"    [{mark}] {name:<16} {s.status:<8} {s.title}")


def _hms(s: str) -> int:
    parts = [int(x) for x in str(s).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _fmt_hms(sec: int) -> str:
    sec = min(sec, 24 * 3600)
    return "%02d:%02d:00" % (sec // 3600, (sec % 3600) // 60)


# ---------------------------------------------------------------------------
# Write / submit / run
# ---------------------------------------------------------------------------

def write(cfg: dict, tasks: list) -> list:
    tdir = os.path.join(suite_dir(cfg), "tasks")
    os.makedirs(tdir, exist_ok=True)
    paths = []
    for t in tasks:
        p = os.path.join(tdir, f"{t['task_id']}.json")
        with open(p, "w") as f:
            json.dump(t, f, indent=2)
        paths.append(p)
    with open(os.path.join(suite_dir(cfg), "suite.resolved.yaml"), "w") as f:
        yaml.safe_dump({**{k: v for k, v in cfg.items()},
                        "_n_tasks": len(tasks),
                        "_resolved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                       f, sort_keys=True)
    print(f"\nwrote {len(paths)} task files to {tdir}")
    return paths


def submit(cfg: dict, tasks: list, paths: list) -> int:
    repo = os.path.dirname(_HERE)
    script = os.path.join(repo, "src", "slurm", "eval_pack.sbatch")
    cap = _fmt_hms(sum(_hms(t["resources"].get("time_limit", "01:00:00"))
                       for t in tasks))
    env = os.environ.copy()
    env["TASKS"] = " ".join(paths)
    cmd = ["sbatch", f"--time={cap}", f"--job-name=crl-eval-{cfg['name']}",
           "--export=ALL,TASKS=" + " ".join(paths), script]
    print("  " + " ".join(cmd[:4]) + " ... " + script)
    return subprocess.call(cmd, env=env)


def run_one(task_path: str) -> int:
    task = json.load(open(task_path))
    spec = common.discover()[task["eval"]]
    os.makedirs(task["out_dir"], exist_ok=True)
    marker = os.path.join(task["out_dir"], "status.json")
    t0 = time.time()
    print(f"== {task['task_id']}")
    try:
        spec.impl().run(task)
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        json.dump(dict(state="failed", error=f"{type(e).__name__}: {e}",
                       secs=round(time.time() - t0, 1), task=task),
                  open(marker, "w"), indent=2)
        return 1
    json.dump(dict(state="ok", secs=round(time.time() - t0, 1), task=task),
              open(marker, "w"), indent=2)
    return 0


def status(cfg: dict, tasks: list) -> int:
    rc = 0
    print(f"\n{cfg['name']}  ({suite_dir(cfg)})")
    for t in tasks:
        p = os.path.join(t["out_dir"], "status.json")
        if not os.path.exists(p):
            print(f"  pending  {t['task_id']}")
            rc = max(rc, 1)
            continue
        s = json.load(open(p))
        print(f"  {s['state']:<8} {t['task_id']}  ({s['secs']}s)"
              + (f"  {s.get('error', '')}" if s["state"] != "ok" else ""))
        if s["state"] != "ok":
            rc = 2
    return rc


def figures(cfg: dict) -> int:
    specs = common.discover()
    sd = suite_dir(cfg)
    made = []
    for e in cfg["evals"]:
        spec = specs.get(e["name"])
        if spec is None or not spec.figures:
            continue
        mod = spec.figure_module()
        made += mod.make_figures(sd, os.path.join(sd, "figures", spec.name))
    for m in made:
        print(f"  {m}")
    if not made:
        print("  nothing to draw yet")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", nargs="?", default=DEFAULT_SUITE)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="fill offline/evals/results_<suite>.md from the template")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", metavar="TASK_JSON")
    args = ap.parse_args()

    if args.run:
        return run_one(args.run)
    if args.list:
        for name, s in sorted(common.discover().items()):
            print(f"{name:<18} {s.status:<9} applies_to={list(s.applies_to)}  "
                  f"{s.title}\n   {s.readme}")
        return 0

    cfg = load_suite(args.suite)
    tasks, notes = resolve(cfg)
    if args.figures:
        return figures(cfg)
    if args.report:
        from offline.evals import report
        figures(cfg)
        report.build(cfg, suite_dir(cfg))
        return 0
    if args.status:
        return status(cfg, tasks)

    plan(cfg, tasks, notes)
    if not (args.write or args.submit):
        print("\n(nothing written -- add --write to generate task files, "
              "--submit to also queue)")
        return 0
    if not tasks:
        print("\nnothing to run")
        return 1
    paths = write(cfg, tasks)
    if args.submit:
        return submit(cfg, tasks, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
