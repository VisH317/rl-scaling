"""Every path in the project, derived from two environment variables.

    RL_SCALING_SCRATCH   where data, venv, runs and analysis live  (big, not backed up)
    RL_SCALING_REPO      the checkout                              (small, in git)

Nothing else in the repo may hold an absolute path. Before this module there were 23
hardcoded `/home/vt7993` or `/scratch/gpfs/vt7993` strings outside the config, 19 of
them inside `src/slurm/`, so moving scratch or changing the account was a four-file
edit with no test behind it.

`slurm/_prelude.sh` is the shell half of this file and must agree with it; the one
value that cannot come from here is a `#SBATCH --output` directive, which sbatch
parses before any shell runs -- so the submitter passes `--output` explicitly and the
directive in each script is only a fallback for a hand-run `sbatch`.
"""

from __future__ import annotations

import os

SCRATCH_ROOT = os.environ.get("RL_SCALING_SCRATCH", "/scratch/gpfs/vt7993")

#: The tree this code is RUNNING from. Under a submit-time snapshot that is the
#: snapshot, which is the point: a job's code is frozen.
REPO = os.environ.get("RL_SCALING_REPO") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

#: The tree human-facing output belongs in -- generated results documents and the
#: figures beside them. Normally the same as REPO; under a snapshot it must NOT be,
#: or `eval.py --report` would write `results_<suite>.md` into scratch where nobody
#: would ever see it. The submitter exports RL_SCALING_OUTPUT_REPO for exactly this.
OUTPUT_REPO = os.environ.get("RL_SCALING_OUTPUT_REPO") or REPO

#: Everything the project writes lives under here.
PROJECT = os.path.join(SCRATCH_ROOT, "rl-scaling")
VENV = os.path.join(PROJECT, "venv")
PYTHON = os.path.join(VENV, "bin", "python")

EXPERIMENTS_DIR = os.path.join(PROJECT, "experiments")
RUNS_DIR = os.path.join(EXPERIMENTS_DIR, "runs")
CONFIGS_DIR = os.path.join(EXPERIMENTS_DIR, "configs")
ANALYSIS_DIR = os.path.join(EXPERIMENTS_DIR, "analysis")
MANIFEST_DIR = os.path.join(EXPERIMENTS_DIR, "data")
LOGS_DIR = os.path.join(EXPERIMENTS_DIR, "logs")
SLURM_LOGS_DIR = os.path.join(EXPERIMENTS_DIR, "slurm")
#: Submit-time snapshots of `src/`, one per job -- see `rl_scaling.slurm.snapshot`.
CODE_DIR = os.path.join(EXPERIMENTS_DIR, "code")

#: OGBench data, shared across projects, so it sits beside rl-scaling and not inside.
DATA_DIR = os.path.join(SCRATCH_ROOT, "ogbench", "data")
SUBSET_DIR = os.path.join(SCRATCH_ROOT, "ogbench", "subsets")
EXPERTS_DIR = os.path.join(SCRATCH_ROOT, "ogbench", "experts")

#: In-repo, read from wherever the code runs.
EXPERIMENTS_YAML = os.path.join(REPO, "experiments")
SRC = os.path.join(REPO, "src")
SLURM = os.path.join(SRC, "slurm")
TESTS = os.path.join(REPO, "tests")
#: In-repo, WRITTEN -- so these follow OUTPUT_REPO, not REPO.
DOCS = os.path.join(OUTPUT_REPO, "docs")
EVALS_DOCS = os.path.join(OUTPUT_REPO, "src", "offline", "evals")

DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"


def shell_exports() -> str:
    """The same values as `KEY=value` lines, for `slurm/_prelude.sh` to eval.

    Generated rather than transcribed, so the shell and Python halves cannot drift.
    """
    keys = ("SCRATCH_ROOT", "REPO", "OUTPUT_REPO", "PROJECT", "VENV", "PYTHON", "EXPERIMENTS_DIR",
            "RUNS_DIR", "CONFIGS_DIR", "ANALYSIS_DIR", "LOGS_DIR", "SLURM_LOGS_DIR",
            "CODE_DIR", "DATA_DIR", "EXPERTS_DIR")
    g = globals()
    return "\n".join(f'RL_{k}="{g[k]}"' for k in keys)


if __name__ == "__main__":
    print(shell_exports())
