#!/usr/bin/env python3
"""Every check that does not need a GPU, a dataset or a checkpoint.

    src/check.py            verify            (exit 0 / 1)
    src/check.py --write    refresh goldens   (review the diff before committing)
    src/check.py -v         show each command's output

This is the regression harness for the structure refactor. The one that matters is
**run id stability**: `config._run_id` hashes the resolved config dict, so any change
that perturbs that dict -- a reordered key, a new field, a default moved -- renames
every run and orphans the run directories on scratch. `tests/run_ids.json` pins the
ids every committed experiment currently produces; a refactor is "faithful" exactly
when this file still matches.

`tests/plan_golden/` does the same for the human-readable plan output, which is the
proof for the lifecycle extraction (docs/repo_structure.md Stage 3): if `run.py` and
`eval.py` print byte-identical plans before and after, the behaviour moved intact.

Designed to run on the login node in seconds. Anything needing a GPU belongs in
`src/slurm/smoke.sbatch`.
"""

# Interpreter guard and import paths, in one line. MUST come first and stay
# 3.6-parseable: Stellar's system python3 is 3.6.8 and importing this re-execs into
# the project venv. See src/rl_scaling/bootstrap.py for why.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_scaling.bootstrap  # noqa: E402,F401

import argparse
import glob
import json
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TESTS = os.path.join(REPO, "tests")
RUN_IDS = os.path.join(TESTS, "run_ids.json")
PLAN_DIR = os.path.join(TESTS, "plan_golden")

#: Env that makes every check deterministic and keeps JAX off any GPU that happens
#: to be visible -- the point of this harness is that it runs anywhere.
ENV = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL="osmesa", TQDM_DISABLE="1",
           PYTHONHASHSEED="0", XLA_PYTHON_CLIENT_PREALLOCATE="false")

_RESULTS = []


def _run(cmd, cwd=REPO):
    p = subprocess.run(cmd, cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def check(name, fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:                                        # noqa: BLE001
        import traceback
        ok, detail = False, traceback.format_exc()
    _RESULTS.append((name, ok, detail, time.time() - t0))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {time.time() - t0:5.1f}s")
    if not ok and detail:
        for line in str(detail).rstrip().splitlines()[-25:]:
            print(f"        {line}")
    return ok


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def check_impls():
    """The vendored OGBench implementations must be importable.

    `lib/**` is gitignored, so a fresh clone or a `git worktree` has no `lib/` at all
    -- and several checks below then pass only because the code path that needs it is
    never reached. `offline/critic_losses.py`'s self-test is the example: its maths
    runs fine, while `install_critic_loss`, which monkeypatches `agents.crl`, is
    silently not exercised. A harness that reports PASS in an environment where half
    the project cannot import is the exact failure class this repo refuses elsewhere,
    so this is a hard FAIL with the fix in the message.
    """
    def fn():
        import rl_scaling.bootstrap as b
        if not os.path.isdir(b.IMPLS):
            return False, (
                f"no OGBench implementations at {b.IMPLS}\n"
                f"`lib/**` is gitignored, so this tree has none. Either clone OGBench "
                f"there, or symlink an existing checkout:\n"
                f"  ln -s /home/vt7993/rl-scaling/lib {b.REPO}/lib")
        missing = [m for m in ("agents", "utils") if not os.path.isdir(
            os.path.join(b.IMPLS, m))]
        if missing:
            return False, f"{b.IMPLS} is missing {missing}"
        rc, out = _run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, '.'); "
                        "import rl_scaling.bootstrap; "
                        "import agents, utils.networks, ogbench; print('ok')"],
                       cwd=os.path.join(REPO, "src"))
        return rc == 0, out
    return fn


# ---------------------------------------------------------------------------
# Self-checks that already exist in the modules
# ---------------------------------------------------------------------------

SELFTESTS = [
    ("offline.config resolves", ["-m", "offline.config"]),
    ("residual tower self-test", ["-m", "offline.networks"]),
    ("critic losses self-test", ["-m", "offline.critic_losses"]),
    ("critic_heatmap metrics self-test",
     ["-m", "offline.evals.critic_heatmap.impl"]),
]


def _selftest(args):
    def fn():
        rc, out = _run([sys.executable] + args, cwd=os.path.join(REPO, "src"))
        return rc == 0, out
    return fn


# ---------------------------------------------------------------------------
# Config resolution + run id stability
# ---------------------------------------------------------------------------

def experiment_files():
    return sorted(f for f in glob.glob(os.path.join(REPO, "experiments", "*.yaml"))
                  if os.path.basename(f) != "TEMPLATE.yaml")


def _resolved_ids() -> dict:
    sys.path.insert(0, os.path.join(REPO, "src"))
    import run as R
    out = {}
    for f in experiment_files():
        cfg = R.load(f)
        cfg["_source"] = f
        out[os.path.basename(f)[:-5]] = [r["run_id"] for r in R.resolve_runs(cfg)]
    return out


def check_run_ids(write: bool):
    def fn():
        got = _resolved_ids()
        if write:
            os.makedirs(TESTS, exist_ok=True)
            json.dump(got, open(RUN_IDS, "w"), indent=2, sort_keys=True)
            return True, f"wrote {sum(len(v) for v in got.values())} ids"
        if not os.path.exists(RUN_IDS):
            return False, f"{RUN_IDS} missing; run with --write"
        want = json.load(open(RUN_IDS))
        msgs = []
        for name in sorted(set(want) | set(got)):
            a, b = want.get(name), got.get(name)
            if a is None:
                msgs.append(f"NEW experiment {name} ({len(b)} runs) -- rerun --write")
            elif b is None:
                msgs.append(f"REMOVED experiment {name}")
            elif a != b:
                lost = [x for x in a if x not in b]
                msgs.append(f"{name}: run ids CHANGED ({len(lost)} orphaned), e.g. "
                            f"{lost[0] if lost else '?'}")
        return not msgs, "\n".join(msgs)
    return fn


def check_run_ids_on_disk():
    """Every id in the golden file that already has a run directory must still exist.

    The golden file alone cannot catch this: if both the code and the golden are
    regenerated together, a renamed run looks consistent while the work on scratch is
    orphaned. This compares against what is actually on the filesystem.
    """
    def fn():
        sys.path.insert(0, os.path.join(REPO, "src"))
        from offline import config as C
        if not os.path.isdir(C.RUNS_DIR):
            return True, "no runs directory on this machine -- skipped"
        on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(C.RUNS_DIR, "*"))}
        got = {i for ids in _resolved_ids().values() for i in ids}
        known_grids = {n for n in _resolved_ids()}
        orphans = sorted(r for r in on_disk - got
                         if r.split("__")[0] in known_grids)
        if orphans:
            return False, ("these run directories are no longer produced by any "
                           "experiment file:\n  " + "\n  ".join(orphans[:10]))
        return True, f"{len(on_disk & got)} of {len(on_disk)} run dirs still addressed"
    return fn


# ---------------------------------------------------------------------------
# Golden plan output
# ---------------------------------------------------------------------------

#: Lines whose content legitimately varies between machines or over time. They are
#: blanked rather than dropped so a changed line count still shows up as a diff.
_VOLATILE = [
    # The repo root first, and ANYWHERE in the line, not just at its start. A golden
    # that embeds the absolute path can only be verified from the tree that wrote it,
    # which defeats the point when the same goldens have to hold in a worktree and on
    # a code snapshot on scratch.
    (re.compile(re.escape(os.path.realpath(REPO))), "<repo>"),
    (re.compile(re.escape(os.environ.get("RL_SCALING_SCRATCH",
                                         "/scratch/gpfs/vt7993"))), "<scratch>"),
    (re.compile(r"^(\s*)(/|~).*$"), r"\1<path>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"), "<time>"),
    (re.compile(r"^(  cost      ).*$"), r"\1<cost>"),
    (re.compile(r"^(            full breakdown: ).*$"), r"\1<cmd>"),
    (re.compile(r"^(\s*select\s+\w+\s+checkpoint=).*$"), r"\1<sel>"),
]


def _normalise(text: str) -> str:
    out = []
    for line in text.splitlines():
        for pat, rep in _VOLATILE:
            line = pat.sub(rep, line)
        out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


def check_plans(write: bool):
    def fn():
        os.makedirs(PLAN_DIR, exist_ok=True)
        targets = [(os.path.basename(f)[:-5], ["src/run.py", f])
                   for f in experiment_files()]
        targets.append(("_evalsuite", ["src/eval.py"]))
        msgs = []
        for name, cmd in targets:
            rc, out = _run([sys.executable] + cmd)
            if rc != 0:
                msgs.append(f"{name}: exit {rc}\n{out[-600:]}")
                continue
            norm = _normalise(out)
            p = os.path.join(PLAN_DIR, f"{name}.txt")
            if write:
                open(p, "w").write(norm)
            elif not os.path.exists(p):
                msgs.append(f"{name}: no golden at {p}; run with --write")
            elif open(p).read() != norm:
                import difflib
                d = list(difflib.unified_diff(open(p).read().splitlines(),
                                              norm.splitlines(), "golden", "now",
                                              lineterm="", n=1))
                msgs.append(f"{name}: plan output changed\n    "
                            + "\n    ".join(d[:20]))
        return not msgs, "\n".join(msgs)
    return fn


# ---------------------------------------------------------------------------
# Eval registry
# ---------------------------------------------------------------------------

def check_evals():
    def fn():
        sys.path.insert(0, os.path.join(REPO, "src"))
        from offline.evals import common
        specs = common.discover()          # raises on a malformed eval.yaml
        if not specs:
            return False, "no evals discovered"
        bad = []
        for name, spec in specs.items():
            if not os.path.exists(spec.readme):
                bad.append(f"{name}: no README")
            if spec.status == "ready":
                try:
                    spec.impl()
                except Exception as e:                            # noqa: BLE001
                    bad.append(f"{name}: impl import failed: {type(e).__name__}: {e}")
                if spec.figures:
                    try:
                        spec.figure_module()
                    except Exception as e:                        # noqa: BLE001
                        bad.append(f"{name}: figures import failed: {e}")
        return not bad, "\n".join(bad) or f"{len(specs)} evals: " + ", ".join(
            f"{n}({s.status})" for n, s in sorted(specs.items()))
    return fn


def check_sbatch_prelude():
    """Every sbatch script must find `_prelude.sh` the way SLURM runs it.

    Slurm copies the batch script into /var/spool/slurmd/job<id>/ before executing
    it, so ${BASH_SOURCE[0]} does NOT point into the repo and the script cannot
    locate the prelude from its own path. The submitter therefore exports
    RL_SCALING_REPO. This failed in production while `sbatch --test-only` accepted
    the submission, because --test-only never runs the script -- so the check has to
    actually execute the resolution block, with the script moved out of the repo.

    Only the prologue is run (everything up to and including the `source` line), so
    this needs no GPU and starts no work.
    """
    def fn():
        import shutil
        import subprocess
        import tempfile
        scripts = sorted(glob.glob(os.path.join(REPO, "src", "slurm", "*.sbatch")))
        if not scripts:
            return False, "no sbatch scripts found"
        bad, good = [], []
        for sc in scripts:
            lines = open(sc).read().splitlines(keepends=True)
            cut = [i for i, ln in enumerate(lines) if "_PRELUDE" in ln
                   and ln.strip().startswith("source")]
            if not cut:
                # smoke.sbatch predates the prelude and hardcodes its paths.
                if "_prelude" not in "".join(lines):
                    continue
                bad.append(f"{os.path.basename(sc)}: sources a prelude, "
                           f"but not via the resolved $_RL_PRELUDE")
                continue
            prologue = "".join(lines[:cut[0] + 1]) + '\necho "REPO=$REPO"\n'
            for label, spool, env_repo in (
                    ("slurm", True, REPO),      # copied away + told where the repo is
                    ("hand-run", False, None)):  # run in place, nothing exported
                d = tempfile.mkdtemp()
                try:
                    path = (os.path.join(d, "slurm_script") if spool
                            else os.path.join(os.path.dirname(sc), ".check_prologue.sh"))
                    open(path, "w").write(prologue)
                    env = {k: v for k, v in os.environ.items()
                           if k != "RL_SCALING_REPO"}
                    if env_repo:
                        env["RL_SCALING_REPO"] = env_repo
                    r = subprocess.run(["bash", path], capture_output=True, text=True,
                                       env=env, timeout=120)
                    out = [ln for ln in r.stdout.splitlines()
                           if ln.startswith("REPO=")]
                    if r.returncode != 0 or not out:
                        bad.append(f"{os.path.basename(sc)} [{label}]: exit "
                                   f"{r.returncode}: "
                                   f"{(r.stderr.strip().splitlines() or [''])[-1]}")
                    elif os.path.realpath(out[-1][5:]) != os.path.realpath(REPO):
                        bad.append(f"{os.path.basename(sc)} [{label}]: REPO resolved "
                                   f"to {out[-1][5:]}, expected {REPO}")
                    else:
                        good.append(f"{os.path.basename(sc)}[{label}]")
                finally:
                    if not spool:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                    shutil.rmtree(d, ignore_errors=True)
        return not bad, "\n".join(bad) or "resolved: " + ", ".join(good)
    return fn


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="refresh the goldens instead of checking them")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print(f"\ncheck  ({'WRITE goldens' if args.write else 'verify'})  "
          f"python {sys.version.split()[0]}  {REPO}\n")
    t0 = time.time()
    if not check("OGBench implementations importable", check_impls()):
        print("\nstopping: the checks below would pass vacuously without them")
        return 1
    for name, cmd in SELFTESTS:
        check(name, _selftest(cmd))
    check("every experiment resolves + run ids stable", check_run_ids(args.write))
    check("no run directory orphaned", check_run_ids_on_disk())
    check("eval registry valid", check_evals())
    check("sbatch scripts find the prelude as slurm runs them",
          check_sbatch_prelude())
    check("plan output unchanged", check_plans(args.write))

    bad = [n for n, ok, _, _ in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed in "
          f"{time.time() - t0:.1f}s")
    if args.verbose:
        for n, ok, d, _ in _RESULTS:
            print(f"\n--- {n}\n{d}")
    if bad:
        print("FAILED: " + ", ".join(bad))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
