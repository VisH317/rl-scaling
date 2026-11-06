#!/bin/bash
# Post-hoc re-evaluation of saved `params_best.pkl` at the PUBLISHED episode count.
#
# Why this exists: training evaluates with 15 episodes/task (75 per point) to keep the
# learning curves cheap. Every OGBench published command uses --eval_episodes=50 (250
# per point). A number measured at 15 episodes is not comparable to their table, so the
# headline figure has to be re-measured here.
#
# Read-only with respect to the runs: it loads checkpoints and writes a JSON into the
# analysis directory. Safe to re-run; safe to run while other packs are training.
#
# Submit:  sbatch src/slurm/reeval.sbatch
#SBATCH --job-name=crl-reeval
#SBATCH --account=mae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=06:00:00
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
cd "$REPO" || exit 1

EPISODES=${EPISODES:-50}
GRIDS=${GRIDS:-"phase1_anchor phase1_repro"}

rl_banner
echo "episodes/task: $EPISODES    grids: $GRIDS"

rc=0
for g in $GRIDS; do
  echo
  echo "================ $g ================"
  # Only runs with a summary.json are picked up, so in-flight runs are skipped
  # rather than evaluated half-trained.
  $PY src/scripts/reeval.py --grid "$g" --episodes "$EPISODES" || rc=1
done

echo
echo "exit $rc"
exit $rc
