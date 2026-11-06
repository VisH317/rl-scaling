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
#: One folder per experiment grid or eval suite, holding that result's figures AND
#: the document that reads them. Co-located on purpose: a figure and the sentence
#: interpreting it go stale together, and a reader should never have to join two
#: directories by hand to see what a number refers to.
RESULTS = os.path.join(OUTPUT_REPO, "results")


#: Output-folder names for grids and eval suites.
#:
#: An ALIAS, not a rename. A run id is a hash over its config, and the grid name is
#: part of that config, so renaming an experiment would change all of its run ids and
#: orphan every finished run directory on scratch. The grid names therefore stay
#: frozen forever; only the folder a human reads is renamed.
#:
#: The scheme is `<environment>__<what varies>`, so folders sort by environment and a
#: name says what the experiment DOES rather than when it was run. "phase1_repro" had
#: to be looked up; "antmaze_stitch__depth" does not.
RESULTS_ALIAS = {
    # grid name (frozen)      -> folder a human reads
    "phase1_anchor":          "antmaze_stitch__baseline_arch",
    "phase1_repro":           "antmaze_stitch__depth",
    "antmaze_ensemble1":      "antmaze_stitch__depth_ensemble1",
    "antmaze_navigate1":      "antmaze_navigate__depth",
    "humanoid_ensemble1":     "humanoidmaze_navigate__depth_ensemble1",
    "interp01":               "interp__critic_depth",
}


def results_name(name: str) -> str:
    """The folder name for a grid or suite. Unaliased names pass through."""
    return RESULTS_ALIAS.get(name, name)


def results_dir(name: str) -> str:
    """`results/<aliased name>/`, created on demand."""
    d = os.path.join(RESULTS, results_name(name))
    os.makedirs(d, exist_ok=True)
    return d

DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"


def shell_exports() -> str:
    """The same values as `KEY=value` lines, for `slurm/_prelude.sh` to eval.

    Generated rather than transcribed, so the shell and Python halves cannot drift.
    """
    keys = ("SCRATCH_ROOT", "REPO", "OUTPUT_REPO", "PROJECT", "VENV", "PYTHON", "EXPERIMENTS_DIR",
            "RUNS_DIR", "CONFIGS_DIR", "ANALYSIS_DIR", "LOGS_DIR", "SLURM_LOGS_DIR",
            "CODE_DIR", "DATA_DIR", "EXPERTS_DIR", "RESULTS")
    g = globals()
    return "\n".join(f'RL_{k}="{g[k]}"' for k in keys)


if __name__ == "__main__":
    print(shell_exports())
