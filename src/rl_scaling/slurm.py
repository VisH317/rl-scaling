"""Submit-time code snapshots.

THE PROBLEM. `train_pack.sbatch` and `eval_pack.sbatch` run out of `$REPO`, which is
the live working tree. A job that sits in the queue for fourteen hours executes the
repo as it is when it *starts*, not as it was when it was submitted — so editing a
file can change the meaning of work already queued, and a finished run records its
hyperparameters (`config.json`) while nothing records the code that read them.

THE FIX. `--submit` copies `src/` into `<CODE_DIR>/<tag>/` and submits the sbatch
*from that copy*. `slurm/_prelude.sh` derives `REPO` from its own location, so the
snapshot needs no edit to be self-consistent: a job launched from the snapshot sees
the snapshot as the repo.

`tag` is a content hash, so resubmitting unchanged code reuses one snapshot rather
than accumulating copies. `manifest.json` beside it records the git commit, whether
the tree was dirty, and a hash of `lib/` — which is **symlinked, not copied**, since
it is 54 MB of vendored OGBench that is gitignored and effectively never edited. The
hash is what makes a change to it detectable rather than invisible.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time

from rl_scaling import paths

#: Copied into the snapshot. Everything a job imports, nothing it does not.
#: `tests` and `experiments` are a few KB and make the snapshot a complete,
#: self-checking unit: `<snap>/src/check.py` runs there.
_INCLUDE = ("src", "tests", "experiments")
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "figures"}
_SKIP_SUFFIX = (".pyc", ".pyo", ".png", ".npz")


def _walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for f in sorted(filenames):
            if f.endswith(_SKIP_SUFFIX):
                continue
            yield os.path.join(dirpath, f)


def _tree_hash(root: str) -> str:
    h = hashlib.sha256()
    for p in _walk(root):
        h.update(os.path.relpath(p, root).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _git(*args, repo=None) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo or paths.REPO, *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                             # noqa: BLE001
        return ""


def snapshot(repo: str = None, verbose: bool = True) -> str:
    """Freeze `src/` and return the snapshot's root (a usable `$REPO`)."""
    repo = repo or paths.REPO
    tag = _tree_hash(os.path.join(repo, "src"))
    dest = os.path.join(paths.CODE_DIR, tag)
    if os.path.exists(os.path.join(dest, "manifest.json")):
        if verbose:
            print(f"  code snapshot {tag} (reused)")
        return dest

    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    for part in _INCLUDE:
        shutil.copytree(os.path.join(repo, part), os.path.join(tmp, part),
                        ignore=shutil.ignore_patterns(*_SKIP_DIRS, "*.pyc"))
    # lib/ is 54 MB of vendored OGBench and is gitignored; link it and hash it rather
    # than copying it per job.
    lib = os.path.join(repo, "lib")
    if os.path.exists(lib):
        os.symlink(os.path.realpath(lib), os.path.join(tmp, "lib"))
    json.dump(dict(
        tag=tag, created=time.strftime("%Y-%m-%dT%H:%M:%S"), source_repo=repo,
        git_commit=_git("rev-parse", "HEAD", repo=repo),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        git_dirty=bool(_git("status", "--porcelain", repo=repo)),
        lib_realpath=os.path.realpath(lib) if os.path.exists(lib) else None,
        lib_hash=_tree_hash(lib) if os.path.exists(lib) else None,
    ), open(os.path.join(tmp, "manifest.json"), "w"), indent=2)
    os.replace(tmp, dest)
    if verbose:
        print(f"  code snapshot {tag} -> {dest}")
    return dest


def sbatch_script(snapshot_root: str, name: str) -> str:
    return os.path.join(snapshot_root, "src", "slurm", name)


def log_option(job_kind: str) -> str:
    """`--output` for the submitter to pass.

    sbatch parses `#SBATCH --output=` before any shell runs, so it cannot come from
    `_prelude.sh`; the directive inside each script is only a hand-run fallback.
    """
    d = paths.SLURM_LOGS_DIR if job_kind == "train" else paths.LOGS_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%x-%j.out")
