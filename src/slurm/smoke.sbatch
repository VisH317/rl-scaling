#!/bin/bash
# GPU validation. Everything the real grid depends on, in ~10 minutes on one GPU.
# Nothing else should be submitted until this exits 0.
#SBATCH --job-name=crl-smoke
#SBATCH --account=mae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -uo pipefail

REPO=/home/vt7993/rl-scaling
VENV=/scratch/gpfs/vt7993/rl-scaling/venv
PY=$VENV/bin/python
SMOKE=/scratch/gpfs/vt7993/rl-scaling/experiments/smoke

export MUJOCO_GL=egl          # headless rendering; state-based eval does not need it
export TQDM_DISABLE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd "$REPO/src" || exit 1
mkdir -p "$SMOKE"

fail=0
step () { echo; echo "===== $* ====="; }
check () { if [ "$1" -eq 0 ]; then echo "PASS"; else echo "FAIL (exit $1)"; fail=1; fi; }

step "node"
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

step "1. jax sees the GPU"
$PY - <<'EOF'
import jax, jax.numpy as jnp
d = jax.devices()
print("devices:", d)
assert any(x.platform == "gpu" for x in d), "no GPU visible to jax"
x = jnp.ones((2048, 2048))
print("matmul ok, trace =", float((x @ x).trace()))
EOF
check $?

step "2. residual tower self-tests"
$PY -m offline.networks
check $?

step "3. config self-check"
$PY -m offline.config >/dev/null && echo "config resolves"
check $?

step "3b. critic losses: infonce / neg_l2 / logsumexp penalty"
$PY -m offline.critic_losses
check $?

step "3c. every experiments/*.yaml resolves, and the committed ones are unchanged"
for f in "$REPO"/experiments/*.yaml; do
  case "$(basename "$f")" in TEMPLATE.yaml) continue ;; esac
  $PY "$REPO/src/run.py" "$f" >/dev/null 2>&1 \
    || { echo "  FAILED to resolve $(basename "$f")"; false; }
done
check $?
# Regression: the YAMLs must reproduce the run ids already on disk, or a re-run
# would orphan finished work instead of recognising it.
$PY - <<'EOF'
import sys, os, glob
sys.path.insert(0, "/home/vt7993/rl-scaling/src")
import run as R
ok = True
for name in ("phase1_repro", "phase1_anchor"):
    f = f"/home/vt7993/rl-scaling/experiments/{name}.yaml"
    cfg = R.load(f); cfg["_source"] = f
    want = {r["run_id"] for r in R.resolve_runs(cfg)}
    have = {os.path.basename(p)[:-5] for p in
            glob.glob(f"/scratch/gpfs/vt7993/rl-scaling/experiments/configs/{name}__*.json")}
    if have and want != have:
        print(f"  {name}: run ids CHANGED"); ok = False
    else:
        print(f"  {name}: {len(want)} run ids stable")
raise SystemExit(0 if ok else 1)
EOF
check $?

step "4. mujoco env + EGL rendering"
$PY - <<'EOF'
import numpy as np, ogbench
for name in ("antmaze-medium-stitch-v0", "cube-double-play-v0"):
    env = ogbench.make_env_and_datasets(name, env_only=True)
    ob, info = env.reset(options=dict(task_id=1, render_goal=True))
    env.step(env.action_space.sample())
    print(f"{name}: ob {np.asarray(ob).shape}, tasks {len(env.unwrapped.task_infos)}")
    try:
        print("   render:", np.asarray(env.render()).shape)
    except Exception as e:
        print(f"   render UNAVAILABLE ({type(e).__name__}) -- fine, video_episodes=0")
EOF
check $?

step "5. CRL depth 4, end to end"
$PY -m offline.train --smoke --grid phase1_repro --depth 4 --agent crl --seed 0 \
    --run-dir "$SMOKE/crl_d4"
check $?

step "6. GCBC depth 32, end to end"
$PY -m offline.train --smoke --grid phase1_repro --depth 32 --agent gcbc --seed 0 \
    --run-dir "$SMOKE/gcbc_d32"
check $?

step "7. deepest cell times out at depth 64 (throughput for the grid estimate)"
$PY -m offline.train --grid phase3_corner --depth 64 --agent crl --size 1000000 --seed 0 \
    --max-steps 2000 --run-dir "$SMOKE/crl_d64_timing"
check $?

step "8. three runs share one GPU (the packing assumption)"
for i in 0 1 2; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 \
    $PY -m offline.train --smoke --grid phase1_repro --depth 16 --agent crl --seed $i \
        --run-dir "$SMOKE/share_$i" > "$SMOKE/share_$i.log" 2>&1 &
done
wait
grep -h "DONE\|Traceback" "$SMOKE"/share_*.log | tail -6
if [ "$(grep -lc DONE "$SMOKE"/share_*.log 2>/dev/null | wc -l)" -eq 3 ]; then
  echo "PASS (3/3 co-resident runs finished)"
else
  echo "FAIL (co-residency); reduce SLURM.runs_per_gpu in config.py"; fail=1
fi

step "9. checkpoints and logs written"
$PY - <<'EOF'
import json, os
root = "/scratch/gpfs/vt7993/rl-scaling/experiments/smoke"
ok = True
for run in ("crl_d4", "gcbc_d32"):
    d = os.path.join(root, run)
    for f in ("config.json", "summary.json", "status.json", "eval.csv",
              "train.csv", "metrics.jsonl", "checkpoints/params_best.pkl",
              "checkpoints/params_final.pkl"):
        p = os.path.join(d, f)
        print(f"  {'ok  ' if os.path.exists(p) else 'MISS'} {run}/{f}")
        ok &= os.path.exists(p)
    s = json.load(open(os.path.join(d, "summary.json")))
    print(f"  {run}: params={s['params']['total']:,} "
          f"steps/s={s['gradient_steps']/max(s['train_secs'],1e-9):.1f} "
          f"peak_rss={s['peak_rss_gb']}GB")
raise SystemExit(0 if ok else 1)
EOF
check $?

step "RESULT"
if [ "$fail" -eq 0 ]; then
  echo "ALL CHECKS PASSED -- safe to submit the grid"
else
  echo "SMOKE TEST FAILED -- do not submit the grid"
fi
exit $fail
