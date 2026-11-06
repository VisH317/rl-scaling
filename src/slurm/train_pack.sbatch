#!/bin/bash
# Run a PACK of runs: several per GPU, several GPUs per job.
#
# Why packs: the Stellar GPU QOS allows only 2 running jobs per user, so one-run-per-job
# caps throughput at 2 regardless of how small the networks are. These networks are tiny
# (~4.3M params per tower at depth 64, width 256), so VRAM is not the constraint -- host
# RAM is, since each process holds its own dataset copy. Packing gets us to
# SLURM.max_concurrent_runs (24) instead of 2.
#
# Submitted by src/scripts/submit_packs.py, which passes RUN_IDS.
#
#SBATCH --job-name=crl-pack
#SBATCH --account=mae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/gpfs/vt7993/rl-scaling/experiments/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/vt7993/rl-scaling/experiments/slurm/%x-%j.out

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
: "${RUN_IDS:?RUN_IDS must be set (space-separated run ids)}"
RUNS_PER_GPU=${RUNS_PER_GPU:-3}
MEM_FRACTION=${MEM_FRACTION:-0.28}

export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION

# Hand the runs a graceful-stop deadline. Without it a Slurm timeout kills each process
# without a summary.json, so the run looks like it vanished and is requeued from scratch.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  end=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | tr ' ' '\n' | grep '^EndTime=' | cut -d= -f2)
  if [ -n "$end" ]; then
    export TRAIN_DEADLINE_EPOCH=$(date -d "$end" +%s 2>/dev/null || echo "")
    echo "job ends at $end (deadline epoch ${TRAIN_DEADLINE_EPOCH:-unset})"
  fi
fi

cd "$REPO/src" || exit 1

N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "job $SLURM_JOB_ID on $(hostname): $N_GPUS GPUs, ${RUNS_PER_GPU} runs/GPU"
echo "runs: $RUN_IDS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

i=0
declare -a PIDS=()
for run_id in $RUN_IDS; do
  gpu=$(( (i / RUNS_PER_GPU) % N_GPUS ))
  cfg="$EXP/configs/${run_id}.json"
  log="$EXP/runs/${run_id}/driver.log"
  mkdir -p "$(dirname "$log")"
  echo "  -> gpu $gpu : $run_id"
  CUDA_VISIBLE_DEVICES=$gpu \
    $PY -m offline.train --run-config "$cfg" >> "$log" 2>&1 &
  PIDS+=($!)
  i=$((i + 1))
  # Stagger starts so the processes do not all load datasets at once.
  sleep 20
done

echo "launched ${#PIDS[@]} runs; waiting"
rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done

echo "pack $SLURM_JOB_ID finished (rc=$rc)"
exit $rc
