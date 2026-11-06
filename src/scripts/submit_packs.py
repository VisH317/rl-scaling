#!/usr/bin/env python3
"""Keep the Slurm queue topped up with packs of runs.

The Stellar GPU QOS allows 2 running jobs / 8 GPUs / 2 nodes per user, so a job array
is the wrong unit -- `--array=...%2` hits the same 2-job ceiling. The unit is a PACK:
one job, several GPUs, several runs per GPU (see src/slurm/train_pack.sbatch).

This script reconciles the run manifest against reality (each run's status.json and
summary.json, plus squeue) and submits packs for pending runs up to the job cap. It is
idempotent, so it is safe to run from cron, from `/loop`, or by hand.

Usage:
    python src/scripts/submit_packs.py --grid phase1_repro --dry-run
    python src/scripts/submit_packs.py --grid phase1_repro
    python src/scripts/submit_packs.py --grid phase1_repro --after <jobid>   # gate on smoke
    python src/scripts/submit_packs.py --grid phase1_repro --watch 600      # self-topping
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402
from gen_runs import MANIFEST, load_manifest, save_manifest  # noqa: E402

PACK_SBATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "slurm", "train_pack.sbatch")
TERMINAL = ("done", "failed_permanent")
#: A run only burns an attempt if it actually STARTED (wrote status.json). A job
#: cancelled or preempted while still PENDING costs nothing.
MAX_ATTEMPTS = 3


def squeue_jobs() -> dict[str, str]:
    """{job_id: state} for this user's jobs. Empty dict if squeue is unavailable."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %T"],
            stderr=subprocess.DEVNULL, timeout=60,
        ).decode()
    except Exception:  # noqa: BLE001
        return {}
    jobs = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        jid, _, state = line.strip().partition(" ")
        jobs[jid.split("_")[0]] = state.strip()
    return jobs


def reconcile(rows: list[dict], jobs: dict[str, str]) -> list[dict]:
    """Update each row's status from what is actually on disk and in the queue."""
    for r in rows:
        if r["status"] in TERMINAL:
            continue
        run_dir = r["run_dir"]
        summary_p = os.path.join(run_dir, "summary.json")
        status_p = os.path.join(run_dir, "status.json")

        if os.path.exists(summary_p):
            r["status"] = "done"
            try:
                with open(summary_p) as f:
                    s = json.load(f)
                r["best_success"] = s.get("best_success")
                r["best_step"] = s.get("best_step")
                r["stop_reason"] = s.get("stop_reason")
                r["wall_clock_s"] = s.get("wall_clock_s")
                r["params"] = (s.get("params") or {}).get("total")
            except Exception:  # noqa: BLE001
                pass
            continue

        st = None
        if os.path.exists(status_p):
            try:
                with open(status_p) as f:
                    st = json.load(f)
            except Exception:  # noqa: BLE001
                st = None

        job_alive = r.get("job_id") and str(r["job_id"]) in jobs

        if st and st.get("state") == "failed":
            r["status"] = ("failed_permanent" if r.get("attempts", 0) >= MAX_ATTEMPTS
                           else "pending")
            r["error"] = st.get("error")
            r["job_id"] = None
        elif job_alive:
            r["status"] = "running"
            r["step"] = st.get("step") if st else None
            r["frac_done"] = st.get("frac_done") if st else None
            r["best_success"] = st.get("best_success") if st else None
        elif r["status"] == "running":
            # The job left the queue without writing a summary.
            if st is None:
                # It never started -- cancelled or preempted while PENDING. Requeue and
                # refund the attempt, otherwise a couple of manual scancels would retire
                # the whole grid.
                r["status"] = "pending"
                r["attempts"] = max(0, r.get("attempts", 1) - 1)
                r["job_id"] = None
                r["error"] = None
            else:
                # It ran and died: OOM-killed, node failure, or wall clock.
                r["status"] = ("failed_permanent" if r.get("attempts", 0) >= MAX_ATTEMPTS
                               else "pending")
                r["error"] = (f"started but vanished at step {st.get('step')} "
                              f"without summary.json")
                r["job_id"] = None
    return rows


def submit_pack(run_ids: list[str], *, after: str | None, dry: bool) -> str | None:
    """`after` is a full sbatch dependency expression, e.g. "afterok:123"."""
    from rl_scaling import slurm as S

    # Freeze the code and submit the frozen copy. RL_SCALING_REPO is required, not
    # cosmetic: Slurm copies the batch script into its spool directory, so the script
    # cannot locate `_prelude.sh` from its own path.
    snap = S.snapshot()
    env = dict(os.environ)
    env["RUN_IDS"] = " ".join(run_ids)
    env["RUNS_PER_GPU"] = str(C.SLURM.runs_per_gpu)
    env["MEM_FRACTION"] = str(C.SLURM.xla_mem_fraction)
    env["RL_SCALING_REPO"] = snap

    cmd = [
        "sbatch",
        f"--account={C.SLURM.account}",
        f"--partition={C.SLURM.partition}",
        f"--gres=gpu:{C.SLURM.gpus_per_job}",
        f"--cpus-per-task={C.SLURM.cpus_per_run * len(run_ids)}",
        f"--mem-per-cpu={C.SLURM.mem_per_cpu}",
        f"--time={C.SLURM.time_limit}",
        "--nodes=1",
    ]
    if after:
        cmd.append(f"--dependency={after}")
    cmd.append(S.sbatch_script(snap, os.path.basename(PACK_SBATCH)))

    if dry:
        print("  would submit:", " ".join(cmd))
        print(f"  RUN_IDS={env['RUN_IDS']}")
        return None

    out = subprocess.check_output(cmd, env=env, timeout=120).decode().strip()
    job_id = out.split()[-1]
    print(f"  submitted job {job_id} with {len(run_ids)} runs")
    return job_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", help="restrict to one grid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--after", help="job id this batch must wait for (e.g. the smoke test)")
    ap.add_argument("--max-packs", type=int, help="submit at most this many packs now")
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="loop forever, topping up every SECONDS")
    ap.add_argument("--reset-failed", action="store_true",
                    help="return failed_permanent runs to pending and zero their attempts")
    ap.add_argument("--chain", action="store_true",
                    help="submit EVERY pending pack now, chaining those beyond the job "
                         "cap with afterany dependencies so they self-sequence. Use this "
                         "for unattended overnight runs -- it needs no live watcher, "
                         "since PENDING jobs do not count against the running-job cap.")
    args = ap.parse_args()

    first = True
    while True:
        rows = load_manifest()
        if not rows:
            print(f"no manifest at {MANIFEST}; run gen_runs.py first")
            return 1
        if args.grid:
            rows_all, rows = rows, [r for r in rows if r["grid"] == args.grid]
        else:
            rows_all = rows

        if args.reset_failed:
            n = 0
            for r in rows:
                if r["status"] == "failed_permanent":
                    r.update(status="pending", attempts=0, job_id=None, error=None)
                    n += 1
            print(f"reset {n} failed runs to pending")

        jobs = squeue_jobs()
        reconcile(rows_all, jobs)

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        n_our_jobs = len({str(r["job_id"]) for r in rows_all
                          if r.get("job_id") and str(r["job_id"]) in jobs})
        print(f"[{time.strftime('%H:%M:%S')}] "
              + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
              + f"   slurm jobs in queue: {n_our_jobs}/{C.SLURM.max_running_jobs}")

        pending = [r for r in rows if r["status"] == "pending"]
        per = C.SLURM.runs_per_pack
        free_slots = max(0, C.SLURM.max_running_jobs - n_our_jobs)
        n_packs_pending = -(-len(pending) // per)

        if args.chain:
            # Submit everything; anything past the cap waits on an earlier pack.
            n_submit = n_packs_pending
        else:
            n_submit = free_slots
        if args.max_packs is not None:
            n_submit = min(n_submit, args.max_packs)

        if not pending:
            print("  nothing pending")
        elif n_submit == 0:
            print("  job cap reached; nothing submitted")
        else:
            # Group a pack by dataset size so co-resident runs have similar RAM needs.
            pending.sort(key=lambda r: (r["size"], r["depth"]))
            # Seed the chain with packs ALREADY in the queue, so a second invocation
            # chains onto them instead of submitting an ungated pack that races ahead.
            queued_jobs = sorted({str(r["job_id"]) for r in rows_all
                                  if r.get("job_id") and str(r["job_id"]) in jobs})
            submitted: list[str] = list(queued_jobs)
            n_preexisting = len(submitted)
            for k in range(n_submit):
                chunk = pending[k * per:(k + 1) * per]
                if not chunk:
                    break
                ids = [r["run_id"] for r in chunk]

                # Dependency: the external gate for the first wave, then afterany on the
                # pack `max_running_jobs` slots earlier, so at most that many ever run.
                deps = []
                if args.after:
                    # The external gate (e.g. the smoke test) applies to EVERY pack, not
                    # just the first wave -- otherwise a later pack runs even if the
                    # smoke test failed.
                    deps.append(f"afterok:{args.after}")
                if args.chain:
                    idx = n_preexisting + k - C.SLURM.max_running_jobs
                    if 0 <= idx < len(submitted) and submitted[idx]:
                        deps.append(f"afterany:{submitted[idx]}")
                dep = ",".join(deps) or None

                print(f"  pack {k + 1}/{n_submit}: {len(ids)} runs, sizes "
                      f"{sorted({r['size'] for r in chunk})}"
                      + (f", waits on {dep}" if dep else ""))
                job_id = submit_pack(ids, after=dep, dry=args.dry_run)
                submitted.append(job_id or "")
                if job_id:
                    for r in chunk:
                        r["status"] = "running"
                        r["job_id"] = job_id
                        r["attempts"] = r.get("attempts", 0) + 1

        if not args.dry_run:
            save_manifest(rows_all)

        failed = [r for r in rows if r["status"] == "failed_permanent"]
        if failed:
            print(f"  {len(failed)} permanently failed:")
            for r in failed[:5]:
                print(f"    {r['run_id']}: {r.get('error')}")

        first = False
        if not args.watch:
            return 0
        if all(r["status"] in TERMINAL for r in rows):
            print("all runs terminal; exiting watch loop")
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
