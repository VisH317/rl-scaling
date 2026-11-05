#!/usr/bin/env python3
"""Status digest for an experiment grid. Designed to be read at a glance.

Reads each run's `status.json` heartbeat (written every `heartbeat_interval` gradient
steps) plus the Slurm queue, and reports: how far along everything is, the current
success-vs-depth table, anything that failed, and -- importantly -- anything that is
STALLED, meaning the job is nominally alive but its heartbeat has gone cold. A stalled
run is the failure mode that would otherwise waste a whole night.

Exit code is 0 normally, 2 if there is something a human should look at. That makes it
usable as a cron/`/loop` check: only the nonzero exits need attention.

Usage:
    python src/scripts/monitor.py --grid phase1_repro
    python src/scripts/monitor.py --grid phase1_repro --watch 300
    python src/scripts/monitor.py --grid phase1_repro --plots      # refresh figures too
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402

#: A heartbeat older than this, on a job Slurm still lists as running, means stalled.
STALL_SECONDS = 1800


def squeue_rows() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-o", "%i|%T|%M|%L|%N|%j"],
            stderr=subprocess.DEVNULL, timeout=60,
        ).decode()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 6:
            rows.append(dict(zip(("job", "state", "elapsed", "left", "node", "name"),
                                 parts)))
    return rows


def read_json(p: str):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def collect(grid: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(C.RUNS_DIR, f"{grid}__*"))):
        cfg = read_json(os.path.join(d, "config.json"))
        st = read_json(os.path.join(d, "status.json"))
        summary = read_json(os.path.join(d, "summary.json"))
        if cfg is None and st is None:
            continue
        rec = {
            "run_id": os.path.basename(d),
            "dir": d,
            "agent": (cfg or st or {}).get("agent")
                     if isinstance((cfg or {}).get("agent"), str)
                     else ((cfg or {}).get("agent") or {}).get("agent")
                          or (st or {}).get("agent"),
            "depth": ((cfg or {}).get("arch") or {}).get("depth") or (st or {}).get("depth"),
            "seed": (cfg or st or {}).get("seed"),
            "size": (cfg or st or {}).get("size"),
            "status": st, "summary": summary,
        }
        out.append(rec)
    return out


def classify(rec: dict, live_jobs: set[str], now: float) -> tuple[str, str]:
    """Returns (state, detail)."""
    if rec["summary"]:
        s = rec["summary"]
        return "done", (f"best={s.get('best_success'):.3f}@{s.get('best_step'):,} "
                        f"{s.get('stop_reason')} "
                        f"{s.get('wall_clock_s', 0) / 3600:.1f}h")
    st = rec["status"]
    if not st:
        return "pending", ""
    if st.get("state") == "failed":
        return "failed", str(st.get("error"))[:70]
    age = now - st.get("updated", 0)
    frac = st.get("frac_done")
    best = st.get("best_success")
    detail = (f"step {st.get('step', 0):,}"
              + (f"/{st.get('step_cap'):,}" if st.get("step_cap") else "")
              + (f" ({100 * frac:.0f}%)" if frac else "")
              + (f" best={best:.3f}" if isinstance(best, (int, float)) and best >= 0 else "")
              + f" hb {age / 60:.0f}m ago")
    if age > STALL_SECONDS:
        return "stalled", detail
    return "running", detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="phase1_repro", choices=list(C.GRIDS))
    ap.add_argument("--watch", type=int, metavar="SECONDS")
    ap.add_argument("--plots", action="store_true", help="regenerate figures each pass")
    ap.add_argument("--verbose", action="store_true", help="one line per run")
    args = ap.parse_args()

    while True:
        now = time.time()
        jq = squeue_rows()
        live = {r["job"] for r in jq}
        recs = collect(args.grid)
        spec = C.GRIDS[args.grid]
        expected = spec.n_runs()

        buckets: dict[str, list] = {}
        for r in recs:
            state, detail = classify(r, live, now)
            r["_state"], r["_detail"] = state, detail
            buckets.setdefault(state, []).append(r)

        n_known = len(recs)
        print("=" * 78)
        print(f"{args.grid}   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 78)
        order = ("done", "running", "stalled", "failed", "pending")
        line = "  ".join(f"{k}={len(buckets.get(k, []))}" for k in order)
        print(f"runs: {n_known} seen / {expected} expected     {line}")

        if jq:
            print("\nslurm queue")
            for r in jq:
                print(f"  {r['job']:>10} {r['state']:<10} elapsed {r['elapsed']:>8} "
                      f"left {r['left']:>10} {r['node']:<16} {r['name']}")
        else:
            print("\nslurm queue: empty")

        # The results table: this is what the night is for.
        done = buckets.get("done", [])
        if done:
            print("\nbest success by agent x depth")
            agents = sorted({r["agent"] for r in done if r["agent"]})
            depths = sorted({r["depth"] for r in done if r["depth"]})
            print("  depth  " + "".join(f"{a:>18}" for a in agents))
            for d in depths:
                cells = []
                for a in agents:
                    vals = [r["summary"]["best_success"] for r in done
                            if r["agent"] == a and r["depth"] == d
                            and r["summary"].get("best_success") is not None]
                    cells.append(f"{sum(vals)/len(vals):.3f} (n={len(vals)})"
                                 if vals else "-")
                print(f"  {d:>5}  " + "".join(f"{c:>18}" for c in cells))

        for state in ("stalled", "failed"):
            if buckets.get(state):
                print(f"\n{state.upper()} -- needs attention")
                for r in buckets[state]:
                    print(f"  {r['run_id']}")
                    print(f"      {r['_detail']}")
                    err = os.path.join(r["dir"], "error.txt")
                    if state == "failed" and os.path.exists(err):
                        with open(err) as f:
                            tail = [ln for ln in f.read().strip().splitlines() if ln][-2:]
                        for ln in tail:
                            print(f"      | {ln[:110]}")

        if args.verbose or buckets.get("running"):
            print("\nin flight")
            for r in sorted(buckets.get("running", []),
                            key=lambda x: (x["agent"] or "", x["depth"] or 0)):
                print(f"  {r['agent']:<5} d{r['depth']:<3} s{r['seed']}  {r['_detail']}")

        if args.plots and done:
            print("\nrefreshing figures")
            subprocess.run(
                [sys.executable,
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "plot_results.py"),
                 "--grid", args.grid],
                check=False, timeout=600,
            )

        needs_attention = bool(buckets.get("stalled") or buckets.get("failed"))
        all_done = done and len(done) == expected
        if all_done:
            print(f"\nGRID COMPLETE: {len(done)}/{expected} runs finished")

        if not args.watch:
            return 2 if needs_attention else 0
        if all_done:
            return 2 if needs_attention else 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
