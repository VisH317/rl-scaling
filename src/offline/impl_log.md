# Implementation Log

Running record of build work. Companion to [`offline_exp.md`](offline_exp.md) (the plan) and
[`research_log.md`](research_log.md) (verified facts, decisions, experiment results).

This file is for **implementation**: what was built, what it was tested against, what is
stubbed, and what broke. Append newest at the bottom. Keep entries short and factual.

Status key: `DONE` verified working · `BUILT` written but not yet executed · `WIP` · `BLOCKED` · `TODO`

---

## Session 1 — 2026-11-05

### Layout established

```
src/offline/
  offline_exp.md        the plan (stable, authored by Vish)
  research_log.md       verified facts, decisions, experiment log
  impl_log.md           this file
  config.py             ALL experiment configuration
  networks.py           1000-layer residual tower (Flax)
  patches.py            inject the tower into OGBench's agents
  train.py              training entrypoint
src/scripts/            download, subsets, manifest, run generation, monitor, plots
src/slurm/              sbatch templates + pack submitter
lib/ogbench/            vendored OGBench clone (gitignored)
```

### I1 — `src/offline/config.py` · DONE

Single source of truth. Contents: path roots (scratch, never `~`), a dataset registry with
*measured* structure, the download plan, nested-subset plans implementing D5, `ArchConfig` for the
residual tower, `AgentConfig` with the three codebases' values recorded side by side, `TrainConfig`
with size-dependent step caps, the four experiment grids, and Slurm packing limits.

Self-check: `python -m offline.config` prints a resolved summary and validates that every subset
size is a whole number of trajectories and every depth is divisible by the residual block size.

### I2 — `src/scripts/download_datasets.py` · DONE

Deliberately does **not** use `ogbench.make_env_and_datasets`' auto-download: its default is
`~/.ogbench/data` (93 GB home quota) and it would fire inside a Slurm job where outbound internet
is not guaranteed. Streams with a HEAD size pre-check, atomic `.tmp` → rename, resumes by skipping
complete files, re-fetches size mismatches, and warns when a file's real size drifts >15% from the
registry so `config.py` stays honest.

Verified: `--dry-run` correctly expands 28 files (5 datasets, cube-double shards 000–009 plus vals).

### I3 — vendored OGBench · DONE

`git clone --depth 1` into `lib/ogbench` (gitignored via `lib/**`). Provides `impls/agents/crl.py`
and `impls/agents/gcbc.py` (decision D2) plus `impls/utils/` (datasets, evaluation, flax_utils,
log_utils). We patch rather than fork, so OGBench's published defaults stay visible.

### I4 — `src/offline/networks.py` · DONE

The residual tower, transcribed from `scaling-crl/train.py`. Three differences from
OGBench's `MLP` that all matter at depth: no residuals, GELU vs Swish, and LayerNorm
*after* the activation instead of before it. We match scaling-crl: `Dense → LN → Swish`,
blocks of 4, `x + identity` after the block's last activation, `lecun_uniform`
(`variance_scaling(1/3, 'fan_in', 'uniform')`) on every Dense, bare linear head.

`PlainTower` is the non-residual control with identical norm/activation placement, so
residual connections are the *only* difference when comparing.

Self-tests (`python -m offline.networks`) — all pass:

| depth | params (measured) | analytic | out std |
|---|---|---|---|
| 4 | 293,184 | 293,184 | 0.496 |
| 8 | 558,400 | 558,400 | — |
| 16 | 1,088,832 | 1,088,832 | — |
| 32 | 2,149,696 | 2,149,696 | — |
| 64 | 4,271,424 | 4,271,424 | 2.441 |

Measured counts match the closed form exactly at every depth; deep forward passes stay
finite; cold init gives absmax 3.1e-11; the headless trunk returns width-dim features.

**Bug found and fixed:** a dataclass field named `init` silently shadows
`nn.Module.init()`, so `tower.init(...)` called the string. Renamed to `init_scheme`
with a docstring warning. Also moved the divisibility check out of `setup()` into
`__call__`, since `setup` alongside `@nn.compact` is fragile in Flax.

**Observation worth keeping:** output std grows with depth (0.50 at depth 4 → 2.44 at
depth 64) because residual blocks accumulate without a final norm. The reference does
the same, so we match it rather than "fixing" it — but if deep runs show loss scale
problems, this is the first place to look.

### I5 — `src/offline/patches.py` · DONE

Swaps OGBench's `MLP` for a residual shim. OGBench calls `MLP` in exactly two shapes, and
`activate_final` discriminates them cleanly: `False` → a φ/ψ tower whose head width is
`hidden_dims[-1]` (= `latent_dim`, the representation dim); `True` → the actor trunk with
no head, since `GCActor` adds `mean_net` itself. `ensemblize`'s `nn.vmap` is agnostic to
what it wraps, so the 2-critic ensemble keeps working.

**Bug found and fixed (important).** The patch was first written as a context manager
around `create()`. That fails: Flax re-runs `setup()` on every `apply`, so a tree built
with the shim and applied with the original `MLP` raises
`ScopeParamNotFoundError: /modules_critic/vmap(phi)/Dense_0/kernel`. The patch must be
**process-lifetime** — `install_residual_arch()`. One run per process, one arch per run,
so that is the correct scope. The context-manager form is kept for tests only, with a
docstring saying not to use it in training.

Also patches the name in each importing module, since `from utils.networks import MLP`
binds a separate reference per module.

### I6 — `src/offline/datasets.py` · DONE

Resolves a subset id to concrete files for all three cases (native / whole-shards /
materialized prefix). Loads with `compact_dataset`-style `valids` (halves observation
memory; `Dataset.create` reconstructs next-observations by index shift) and
`add_info=False` (training never needs `qpos`/`qvel`, which are ~45% of the bytes; the
probe and coverage scripts read them from the npz directly). Multi-shard loads
preallocate and fill rather than `np.concatenate`, so peak RAM is 1× not 2×. Actions are
clipped to ±(1−1e-5) as Horizon Reduction does.

`dataset_info()` records `n_transitions` (valid transitions, **not** stored rows) in every
run config, so `D` in the scaling fit is never ambiguous. Measured on
antmaze-medium-stitch: 1,005,000 rows → 5,000 trajs → **1,000,000 transitions**, traj_len
201, ob 29, act 8, 157 MB in RAM. Matches the hand measurement in `research_log.md` §1.2.

### I7 — `src/offline/train.py` · DONE

One run per process. Adds, over OGBench's `main.py`: the residual tower; plateau-based
stopping with a size-dependent cap; best-checkpoint saving; parameter / epochs /
samples-seen / peak-RSS accounting; a `status.json` heartbeat and `metrics.jsonl`; and no
wandb or rendering, so it runs headless and offline.

**Bug found and fixed:** `agents.agents[name]` is the agent *class*; `get_config()` lives
on the *module* (OGBench reaches it via `config_flags` on the file path). Now imports both.

Verified end-to-end on CPU and then on an A100:

| check | result |
|---|---|
| CRL depth 4 (CPU) | 1,446,408 params, 400 steps + 5-task eval + both checkpoints, 90 s |
| GCBC depth 32 (CPU) | 2,139,400 params, `n_transitions`=1,000,000 confirmed, 84 s |
| CRL depth 4 (A100) | 1,446,408 params, `devices=['cuda:0']`, 58 s |
| CRL depth 16 (A100, real run) | 5,424,648 params, training |

### I8 — `src/scripts/gen_runs.py` · DONE

Expands a grid into `experiments/run_manifest.json`, one row per cell × seed, and writes
a resolved `config.json` per run. The manifest is the scheduler's state and is plain JSON
so it can be inspected and hand-edited. `phase1_repro` → 24 runs.

### I9 — `src/slurm/smoke.sbatch` · DONE (passing)

Nine gated checks on one GPU in ~10 min: jax sees the GPU, tower self-tests, config
resolves, MuJoCo + EGL, CRL end-to-end, GCBC end-to-end, depth-64 throughput timing,
**three runs co-resident on one GPU** (the packing assumption), and that all expected
log/checkpoint files exist. Exits nonzero if anything fails, so the grid can be gated on
it with `afterok`.

Result on `stellar-m01g2`: checks 1–6 PASS. EGL rendering works on the GPU nodes
(answering research-log Q5) — the `EGLError` in the traceback is from the destructor
*after* a successful render, so `video_episodes` could be raised later if videos are wanted.

### I10 — `src/slurm/train_pack.sbatch` + `src/scripts/submit_packs.py` · DONE

Packs, not job arrays: `--array=...%2` hits the same 2-job QOS ceiling, so the unit is a
pack (one job, N GPUs, M runs per GPU). Runs are staggered 20 s apart so they don't all
load datasets simultaneously, and grouped by dataset size so co-resident runs have
similar RAM needs.

`submit_packs.py` reconciles the manifest against `status.json` / `summary.json` / `squeue`
and submits up to the cap. `--chain` submits *every* pending pack at once, with later
packs held by `afterany` on an earlier one — PENDING jobs don't count against the
running-job cap, so the whole grid self-sequences with **no live watcher required**.
That is the mode to use for unattended overnight runs.

**Two bugs found and fixed, both from real failures:**

1. *Attempt accounting.* A manual `scancel` of a still-PENDING job was counted as a
   failed attempt, and two cancels retired all 24 runs as `failed_permanent`. Now a run
   only burns an attempt if it actually **started** (wrote `status.json`); a job killed
   while pending is refunded. `MAX_ATTEMPTS` 2 → 3, plus a `--reset-failed` flag.
2. *Chain seeding.* `--chain` seeded its dependency list only from packs submitted in the
   same invocation, so a second invocation submitted an **ungated** pack that raced ahead
   of the smoke gate. Now it seeds from packs already in the queue, and the external
   `--after` gate is applied to every pack rather than only the first wave.

### I11 — `src/scripts/monitor.py` · DONE

Status digest: counts by state, the Slurm queue, the live success-by-agent×depth table,
failures with the tail of `error.txt`, and runs in flight. Its real job is detecting
**stalled** runs — alive in Slurm but heartbeat cold for >30 min — which is the failure
mode that would otherwise waste a whole night. Exit code 2 when something needs a human,
so it works as a cron/`/loop` check where only nonzero exits matter.

### I12 — `src/scripts/plot_results.py` · DONE

Five figures, colors from the dataviz reference palette used unchanged (categorical slots
1/2 for CRL/GCBC; the documented blue and orange ramps, ordinal range, for depth).
`palette.md` certifies the first three slots all-pairs in both modes, so the two-series
charts are covered — `validate_palette.js` needs node, which Stellar does not have, and
is not required for the unmodified reference set. Every figure writes a `.csv` beside the
`.png`: these are static images with no hover layer, so the table *is* the accessibility
channel.

No dual axes anywhere; sequential encodings are one hue light→dark; the rebuttal's
reference values are drawn as grey annotations, never as a series, so paper numbers are
never confused with our runs.

### I13 — cluster facts that changed the plan · DONE

Two real blockers hit during submission, both now encoded in `config.py` with comments:

1. **Maintenance window.** A `--time=7-00:00:00` request sits PENDING with
   `ReqNodeNotAvail, Reserved for maintenance` — Stellar reserves the second Tuesday
   (next: 2026-11-10 06:00–14:00) and won't start a job that can't finish first.
   `SlurmConfig.time_limit` is now **24:00:00**, which schedules immediately.
2. **The 8-GPU node is down.** `stellar-k09g1` is in state `inval`, leaving six nodes
   with 2 GPUs each, so a single-node 4-GPU request is unschedulable.
   `SlurmConfig.gpus_per_job` is now **2** → 6 runs/pack, 12 concurrent (not 24).
   Re-check `sinfo -p gpu` before raising it.

### I14 — review feedback, six items · 2026-11-05

**1. Published CRL flags — a real bug, now fixed.** Our CRL used `crl.py`'s bare defaults
(`actor_p_trajgoal=1.0, actor_p_randomgoal=0.0`). OGBench's published command for this
environment (`hyperparameters.sh` L166) is `0.5/0.5` with `alpha=0.1`. Verified verbatim.
Material on a `stitch` dataset: stitching trajectory segments is the entire point, and
trajectory-only actor goals never pose that problem. Added `PUBLISHED_FLAGS[(dataset,
agent)]` in `config.py`, applied in `run_config()`, recorded per run as
`published_flags`. GCBC confirmed to use bare defaults (L158), so its rows are unchanged.
Cancelled the 12 in-flight runs and regenerated all configs — ~25 min of valid GCBC
progress lost, against 12 CRL runs × ~12 h that would have used the wrong goal
distribution.

**2. `phase1_anchor` grid.** Stock OGBench architecture (3×512 GELU MLP, `latent_dim`
512) via `tower="ogbench_mlp"`, which `patches.py` already supported as an opt-out;
`build_config` now passes the stock dims through untouched when it is set. CRL + GCBC,
3 seeds, 1M steps, published flags. 6 runs. Targets recorded in the grid's `note`:
CRL 53±6, GCBC 45±11, scored as the mean of the last 3 evaluations.

**3. Plateau rule.** `plateau_min_steps=1_000_000` floor (OGBench's published runs train
for 1M, so stopping earlier is not comparable) and `plateau_window=3` moving average
(single-eval success over 75 episodes is noisy enough to fake a plateau). Best-checkpoint
saving still tracks the raw per-eval success — that is the checkpoint phases 2 and 5
want. `summary.json` now also records `last3_eval_mean` and `last3_evals`.

**4. `src/scripts/reeval.py`.** Restores `params_best.pkl` and re-scores at 50
episodes/task (250 per run, the published count), alongside `last3_eval_mean` from
`eval.csv`. Prints a mean±SD aggregate in the published `XX.X±Y.Y` form and writes JSON
to `analysis/<grid>/reeval_50ep.json`.

**5. Resolved head width — verified from a real checkpoint, not the config.**
φ is `Dense(37→256)` → `4×(256→256)` → `Dense(256→64)`, with a leading ensemble dim of 2.
So the head is 64 (`repr_dim`) and the **last hidden layer is still 256** (`arch.width`).
`value_hidden_dims=(256,)` did not change it: OGBench builds `MLP((*hidden_dims,
latent_dim))` and the shim reads only `hidden_dims[-1]`, so that tuple carries the head
width and nothing else. Documented in `research_log.md` §1.10. The checkpoint also
confirms the 2-critic ensemble and that Adam state is stored (resumable).

**6. `src/scripts/goal_distribution.py`.** Measures the pile-up rather than estimating it.
On antmaze-medium-stitch: **43.6%** of value goals are the trajectory's final state
(mean distance 56.5), versus **2.5%** for actor goals — the effect is specific to the
critic, because value goals use geometric sampling and actor goals uniform. Mechanism is
`np.minimum(idxs + offsets, final_state_idxs)`: geometric offsets with mean 100 overshoot
a 201-step trajectory and are clipped rather than resampled. Also measured on the grid
task: cube-double-play is **10.1%** (1001-step trajectories), 4× lower. Logged in §1.8
with the probe consequence: bin "goal == final state" separately, or it silently
dominates the far-distance bins that H-horizon is about.

### I15 — Hydra configuration layer · 2026-11-05

Added `hydra-core==1.3.8`. Hydra sits **above** the existing interface, not in place of
it: `launch.py` composes a config and resolves it into exactly the per-run JSON that
`train.py --run-config` already consumes. Jobs already in the Slurm queue read configs
written before this change and are unaffected — verified by composing a preset and
feeding the result through the unmodified `train.py` end-to-end (400 steps, 5-task eval,
both checkpoints, `last3_eval_mean` populated).

Layout: `conf/` with 10 groups and 38 YAML files (`paths`, `dataset`, `arch`, `agent`,
`optim`, `train`, `eval`, `checkpoint`, `slurm`, `subset`, `experiment`), plus
`conf/README.md`. Schema in `src/offline/schema.py` as Hydra structured configs, so
typos and wrong types fail at composition rather than at train time.

**Drift guard.** `config.py` stays the authoritative annotated record (each value cites
its source); `schema.py` restates the defaults in OmegaConf-friendly types.
`test_schema_matches_config()` asserts they agree on every shared field, and runs on
every `launch.py` invocation plus as smoke check 3b. Currently clean.

**Two real bugs found while testing the layer:**

1. **Precedence inversion.** Published per-`(dataset, agent)` flags were applied *after*
   the recipe, so `agent=crl_stable` came out with `alpha=0.1` (the published BC
   coefficient for antmaze-medium-stitch) instead of stable-CRL's `0.05` — a run that
   was neither recipe. Order is now: group defaults → published flags → recipe deltas,
   with the stable deltas held in code (`STABLE_AGENT_DELTAS`) so they can be re-applied
   last. The launcher prints the conflict when it fires.
2. **Misleading summary.** The printed summary read `cfg.arch`, but a recipe mutates the
   per-run arch after composition, so `crl_stable` displayed `repr_dim=64 init=lecun_uniform`
   while the runs actually got `16 / cold`. The summary now prints the resolved run
   config. A config display that disagrees with what runs is worse than no display.

Also: Hydra rejected the composition at first ("paths appears more than once in the
final defaults list") because both `RootConf` and `config.yaml` declared a defaults
list. The YAML owns it now.

Guard rails in the launcher: a loud warning when the optimizer departs from plain
constant-LR Adam (warmup and clipping help deep nets *specifically*, so they confound
the depth axis); a warning when `use_published_flags=false`; and a note when
`arch.depth=N` is shadowed by an experiment's `depths` sweep. Each generation also
writes `_composed_<experiment>.yaml`, the fully resolved config, beside the run configs.

All 8 presets compose: anchor 6 runs, repro 24, q1 pilot 2, corner 54, full 150,
smoke 2, adhoc 1, actor_thread 30 (declared but not yet runnable — frozen-critic
loading is unimplemented).

