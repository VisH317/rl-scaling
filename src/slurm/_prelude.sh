# Sourced by every sbatch script. Not executable on its own.
#
# One hardcoded value lives here -- the scratch root, and only as a fallback for
# $RL_SCALING_SCRATCH. Everything else is read from `rl_scaling/paths.py`, so the
# shell and Python halves of the path layout cannot drift: there is one definition
# and this evaluates it.
#
#   RL_REPO RL_PROJECT RL_VENV RL_PYTHON RL_EXPERIMENTS_DIR RL_RUNS_DIR
#   RL_CONFIGS_DIR RL_ANALYSIS_DIR RL_LOGS_DIR RL_SLURM_LOGS_DIR RL_CODE_DIR
#   RL_DATA_DIR RL_EXPERTS_DIR
#
# `#SBATCH --output=` is NOT set from here: sbatch parses directives before any shell
# runs. The submitter passes `--output` explicitly; the directive inside each script
# is only a fallback for a hand-run `sbatch`.

: "${RL_SCALING_SCRATCH:=/scratch/gpfs/vt7993}"
export RL_SCALING_SCRATCH

# REPO: the tree this PRELUDE is in. Inside this file BASH_SOURCE is reliable,
# because a `source` resolves the real path -- unlike the sbatch script that sourced
# it, which Slurm copies into its spool directory before running. The submitter
# therefore exports RL_SCALING_REPO, and that wins when present.
REPO="${REPO:-${RL_SCALING_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
export RL_SCALING_REPO="$REPO"
PY="${PY:-$RL_SCALING_SCRATCH/rl-scaling/venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "prelude: no interpreter at $PY (set PY or RL_SCALING_SCRATCH)" >&2
  exit 1
fi
eval "$("$PY" "$REPO/src/rl_scaling/paths.py")" || {
  echo "prelude: could not read paths from $REPO/src/rl_scaling/paths.py" >&2
  exit 1
}

export REPO PY
export EXP="$RL_EXPERIMENTS_DIR"          # legacy name, still used in places

# Runtime environment every job needs.
export MUJOCO_GL="${MUJOCO_GL:-egl}"      # headless rendering
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$RL_LOGS_DIR" "$RL_SLURM_LOGS_DIR"

rl_banner () {
  echo "node:    $(hostname)"
  echo "repo:    $REPO"
  echo "python:  $PY"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
}
