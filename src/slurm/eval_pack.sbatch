#!/bin/bash
# Run a suite of interpretability evals, ONE TASK PER PROCESS.
#
# The per-process rule is not a style choice. `offline/patches.py` swaps OGBench's `MLP`
# for the residual shim for the lifetime of the process, because Flax re-runs `setup()`
# on every `apply` -- so a second checkpoint of a different depth loaded into the same
# process would be built with the first run's tower and die on a parameter-structure
# mismatch. Each task therefore gets its own `python src/eval.py --run`.
#
# Read-only with respect to the runs: loads checkpoints, writes only under
# <ANALYSIS_DIR>/evals/. Safe to run while training packs are in flight.
#
# Submitted by `src/eval.py --submit`, which passes TASKS and --time.
#
#SBATCH --job-name=crl-eval
#SBATCH --account=mae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/vt7993/rl-scaling/experiments/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/vt7993/rl-scaling/experiments/logs/%x-%j.out

set -uo pipefail

# Slurm COPIES this script into its spool directory before running it, so
# ${BASH_SOURCE[0]} points at /var/spool/..., NOT into the repo. The submitter
# exports RL_SCALING_REPO (the code snapshot's root) for that reason; the
# BASH_SOURCE fallback is for running this file directly with bash.
_RL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
_RL_PRELUDE="${RL_SCALING_REPO:-${_RL_HERE%/src/slurm}}/src/slurm/_prelude.sh"
[ -r "$_RL_PRELUDE" ] || _RL_PRELUDE="$_RL_HERE/_prelude.sh"
if [ ! -r "$_RL_PRELUDE" ]; then
  echo "no prelude found: set RL_SCALING_REPO to the repo (or snapshot) root" >&2
  exit 1
fi
source "$_RL_PRELUDE"
# A FILE of task paths, one per line -- not a space-separated --export value. sbatch
# splits --export on commas and a value containing spaces round-trips unreliably, which
# would silently truncate the task list rather than fail.
: "${TASK_FILE:?TASK_FILE must be set (a file of task json paths, one per line)}"
[ -r "$TASK_FILE" ] || { echo "cannot read TASK_FILE=$TASK_FILE"; exit 1; }
mapfile -t TASKS < "$TASK_FILE"
[ "${#TASKS[@]}" -gt 0 ] || { echo "TASK_FILE is empty: $TASK_FILE"; exit 1; }

cd "$REPO" || exit 1
rl_banner
echo "tasks (${#TASKS[@]} from $TASK_FILE):"
for t in "${TASKS[@]}"; do echo "  $t"; done

rc=0
n=0
for t in "${TASKS[@]}"; do
  n=$((n + 1))
  echo
  echo "================ [$n] $(basename "$t") ================"
  $PY "$REPO/src/eval.py" --run "$t" || rc=1
done

echo
echo "== figures + report =="
# --report draws the figures, copies them into the repo and fills the AUTO blocks of
# offline/evals/results_<suite>.md. Prose written between those blocks is preserved.
$PY "$REPO/src/eval.py" --report || rc=1

echo
echo "exit $rc  ($n tasks)"
exit $rc
