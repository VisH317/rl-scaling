#!/usr/bin/env python3
"""Live view of everything you have on the cluster, across every experiment.

    python src/scripts/dashboard.py              # one snapshot
    python src/scripts/dashboard.py --watch 30   # refresh every 30s
    python src/scripts/dashboard.py --grid humanoid_ensemble1   # just one
    python src/scripts/dashboard.py --all        # include finished runs

`monitor.py` answers "how is ONE grid doing". This answers "what do I have running
right now", which is the question at 2am.

Reads `squeue` plus each run's `status.json` heartbeat. Nothing here writes anything,
so it is safe to leave running in a second terminal.

On "live updates on episodes": the trainer does not emit per-episode progress -- OGBench's
`evaluate()` loops internally with tqdm disabled, so there is nothing to tail. What this
does instead is INFER the evaluation phase, and label it as inferred. The heartbeat is
written only on step boundaries, and `eval_interval` is a multiple of
`heartbeat_interval`, so a run sitting exactly on an eval boundary with a heartbeat
older than one normal heartbeat gap is mid-evaluation. The finest real granularity is
one evaluation, which `eval.csv` records with its per-task breakdown.
"""

# Interpreter guard and import paths, in one line. MUST come first and stay
# 3.6-parseable: Stellar's system python3 is 3.6.8 and importing this re-execs into
# the project venv. See src/rl_scaling/bootstrap.py for why.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_scaling.bootstrap  # noqa: E402,F401

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402

#: A heartbeat older than this on a job Slurm still calls RUNNING means stalled --
#: unless the run is mid-evaluation, which this script detects separately.
STALL_S = 1800

BAR = "█"      # full block
BAR_BG = "░"   # light shade


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       timeout=30).decode()
    except Exception:  # noqa: BLE001
        return ""


def squeue_rows() -> list[dict]:
    out = sh(["squeue", "-u", os.environ.get("USER", ""), "-h",
              "-o", "%i|%T|%M|%L|%N|%j|%r"])
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 7:
            rows.append(dict(zip(("job", "state", "elapsed", "left", "node",
                                  "name", "reason"), parts)))
    return rows


def hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def bar(frac: float, width: int = 18) -> str:
    frac = min(max(frac, 0.0), 1.0)
    n = int(round(frac * width))
    return BAR * n + BAR_BG * (width - n)


def last_eval(run_dir: str) -> tuple[int, float, list[float]] | None:
    p = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(p):
        return None
    try:
        rows = [r for r in csv.DictReader(open(p)) if r.get("eval/success")]
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    r = rows[-1]
    per = []
    for i in range(1, 6):
        v = r.get(f"eval/task{i}/success")
        if v not in (None, ""):
            per.append(float(v))
    return int(r["step"]), float(r["eval/success"]), per


def collect(grid: str | None, include_done: bool) -> list[dict]:
    recs = []
    pattern = f"{grid}__*" if grid else "*"
    for d in sorted(glob.glob(os.path.join(C.RUNS_DIR, pattern))):
        sp = os.path.join(d, "status.json")
        if not os.path.exists(sp):
            continue
        try:
            st = json.load(open(sp))
        except Exception:  # noqa: BLE001
            continue
        summary = os.path.join(d, "summary.json")
        st["_done"] = os.path.exists(summary)
        if st["_done"] and not include_done:
            # Keep finished runs out of the live view but count them.
            st["_hidden"] = True
        st["_dir"] = d
        st["_eval"] = last_eval(d)
        recs.append(st)
    return recs


def classify(r: dict, live_nodes: set[str], now: float) -> tuple[str, str]:
    """state, detail. 'evaluating' is INFERRED -- see the module docstring."""
    if r.get("_done"):
        return "done", ""
    age = now - r.get("updated", 0)
    on_cluster = r.get("host") in live_nodes
    step, cap = r.get("step") or 0, r.get("step_cap") or 1
    rate = (step / r["wall_clock_s"]) if r.get("wall_clock_s") else 0.0
    normal_gap = (5000 / rate) if rate > 0 else 60.0

    if not on_cluster and age > STALL_S:
        return "orphaned", f"no job on {r.get('host')}, heartbeat {hms(age)} old"
    if age > STALL_S:
        return "stalled", f"heartbeat {hms(age)} old"
    if age > normal_gap * 2.5 and step > 0:
        return "evaluating", f"inferred: no step for {hms(age)}"
    return "training", ""


def render(args) -> int:
    now = time.time()
    jq = squeue_rows()
    live_nodes = {r["node"] for r in jq if r["node"]}
    recs = collect(args.grid, args.all)

    width = 100
    print("=" * width)
    print(f" {os.environ.get('USER', '?')} on Stellar     "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
          + (f"     refreshing every {args.watch}s" if args.watch else ""))
    print("=" * width)

    # ---- Slurm ------------------------------------------------------------
    if not jq:
        print("\n SLURM: nothing queued or running.")
    else:
        run = [r for r in jq if r["state"] == "RUNNING"]
        pend = [r for r in jq if r["state"] != "RUNNING"]
        print(f"\n SLURM  {len(run)} running, {len(pend)} pending"
              f"   (GPU QOS allows 2 running jobs, 8 GPUs, 2 nodes)")
        print(f"   {'job':>9}  {'name':<14}{'state':<9}{'elapsed':>9}{'left':>10}"
              f"  {'node':<15}reason")
        for r in run + pend:
            reason = "" if r["reason"] in ("None", "none") else r["reason"]
            print(f"   {r['job']:>9}  {r['name']:<14}{r['state']:<9}"
                  f"{r['elapsed']:>9}{r['left']:>10}  {r['node'] or '-':<15}{reason}")

    # ---- runs -------------------------------------------------------------
    by_grid: dict[str, list] = {}
    for r in recs:
        by_grid.setdefault(r.get("grid", "?"), []).append(r)

    if not by_grid:
        print("\n no runs found on scratch yet.")
        return 0

    for grid in sorted(by_grid):
        rs = by_grid[grid]
        shown = [r for r in rs if not r.get("_hidden")]
        n_done = sum(1 for r in rs if r.get("_done"))
        print(f"\n {grid}   {n_done}/{len(rs)} finished"
              + ("" if shown else "   (all finished -- --all to list them)"))
        if not shown:
            continue
        print(f"   {'agent':<5}{'d':>3}{'s':>2}  {'progress':<20}{'step':>18}"
              f"{'best':>7}{'last eval':>11}{'rate':>9}{'eta':>8}  state")
        for r in sorted(shown, key=lambda x: (x.get("agent", ""), x.get("depth", 0),
                                              x.get("seed", 0))):
            state, detail = classify(r, live_nodes, now)
            step, cap = r.get("step") or 0, r.get("step_cap") or 1
            frac = step / cap
            rate = (step / r["wall_clock_s"]) if r.get("wall_clock_s") else 0.0
            eta = hms((cap - step) / rate) if rate > 0 else "-"
            ev = r.get("_eval")
            ev_s = f"{ev[1]:.3f}" if ev else "-"
            best = r.get("best_success")
            print(f"   {r.get('agent', '?'):<5}{r.get('depth', 0):>3}"
                  f"{r.get('seed', 0):>2}  {bar(frac)} {frac * 100:3.0f}%"
                  f"{step:>11,}/{cap / 1e6:.0f}M"
                  f"{(f'{best:.3f}' if best is not None else '-'):>7}"
                  f"{ev_s:>11}{rate:>8.0f}/s{eta:>8}  {state}"
                  + (f"  ({detail})" if detail else ""))
            if ev and ev[2] and args.tasks:
                print(f"        tasks @ {ev[0]:,}: "
                      + "  ".join(f"t{i + 1} {v:.2f}" for i, v in enumerate(ev[2])))

    attention = [r for r in recs if not r.get("_done")
                 and classify(r, live_nodes, now)[0] in ("stalled", "orphaned")]
    if attention:
        print(f"\n NEEDS ATTENTION ({len(attention)})")
        for r in attention:
            print(f"   {r['run_id']}  {classify(r, live_nodes, now)[1]}")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", help="restrict to one grid")
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="refresh forever every N seconds")
    ap.add_argument("--all", action="store_true", help="list finished runs too")
    ap.add_argument("--tasks", action="store_true",
                    help="show the per-task breakdown of each run's last evaluation")
    args = ap.parse_args()

    if not args.watch:
        return render(args)
    try:
        while True:
            print("\033[2J\033[H", end="")   # clear + home
            render(args)
            print(f"\n (ctrl-c to stop)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
