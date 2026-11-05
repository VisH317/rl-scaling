# Hydra configuration

Every experiment variable lives here. Nothing is hard-coded in a script, and nothing is
submitted unless you ask for it.

```bash
V=/scratch/gpfs/vt7993/rl-scaling/venv/bin/python
cd /home/vt7993/rl-scaling
```

## The four things you will actually do

```bash
# 1. See what a preset would run, change nothing
$V src/scripts/launch.py experiment=phase1_anchor dry_run=true

# 2. See the fully resolved config (every value, after composition)
$V src/scripts/launch.py experiment=phase1_repro --cfg job --resolve

# 3. Generate configs + manifest, still queue nothing
$V src/scripts/launch.py experiment=phase1_repro

# 4. Generate and submit
$V src/scripts/launch.py experiment=phase1_repro submit=chain
$V src/scripts/launch.py experiment=phase1_repro submit=chain after=<smoke_job_id>
```

`submit=chain` queues every pack at once with `afterany` dependencies. The QOS caps
*running* jobs at 2 but leaves PENDING unlimited, so the grid self-sequences overnight
with no watcher.

## Groups

| Group | Options | Default |
|---|---|---|
| `experiment` | `phase1_anchor` `phase1_repro` `q1_loss_pilot` `phase3_corner` `phase3_full` `smoke` `adhoc` `actor_thread` | `phase1_repro` |
| `dataset` | `antmaze_medium_stitch` `antmaze_medium_navigate` `antmaze_large_navigate` `cube_double_play_100m` `cube_double_play` | `antmaze_medium_stitch` |
| `arch` | `residual` `residual_cold` `plain_mlp` `ogbench_stock` | `residual` |
| `agent` | `crl` `crl_stable` `crl_infonce` `crl_defaults_only` `gcbc` | `crl` |
| `optim` | `adam_constant` `warmup_clip` `adamw_decay` | `adam_constant` |
| `train` | `default` `anchor_1m` `long` `smoke` | `default` |
| `eval` | `default` `published` | `default` |
| `checkpoint` | `default` `frequent` | `default` |
| `slurm` | `stellar` `stellar_bignode` `debug` | `stellar` |
| `subset` | `default` | `default` |

Swap a group with `group=name`; override any leaf with a dotted path:

```bash
$V src/scripts/launch.py agent=crl_infonce arch=residual_cold optim=warmup_clip
$V src/scripts/launch.py optim.lr=1e-4 train.plateau_patience=8 eval.episodes=50
$V src/scripts/launch.py experiment.depths=[8,16] experiment.seeds=[0,1,2,3,4]
$V src/scripts/launch.py -m agent=crl,crl_infonce arch.depth=4,32   # Hydra multirun
```

## Precedence — read this before trusting an override

Applied lowest to highest:

1. the `agent` / `arch` group defaults
2. **published per-`(dataset, agent)` flags** from `hyperparameters.sh`
   (`use_published_flags: true`)
3. **an explicit recipe's deltas** (`recipe: stable`)

Step 3 must come last, and the launcher prints it when it fires. Published `alpha` is
CRL's BC coefficient *for that environment* (0.1 on antmaze-medium-stitch) and would
otherwise silently overwrite stable-CRL's `bc_coef=0.05`, turning a "stable recipe" run
into something that is neither recipe.

Two further gotchas the launcher warns about:

- `arch.depth=N` is **ignored** when the chosen `experiment` sweeps a `depths` list.
  Use `experiment=adhoc`, or set `experiment.depths=[N]`.
- Any optimizer setting that departs from plain constant-LR Adam prints a loud warning.
  Warmup and gradient clipping help deep networks *specifically*, so turning them on
  confounds the depth axis — keep those runs out of the scaling fit.

## What is deliberately conservative

- **Optimizer**: plain Adam, constant 3e-4, no schedule/warmup/clipping/weight decay.
  Verified identical in OGBench and scaling-crl. The 1000-layer stability came from the
  architecture, not optimizer tricks.
- **Eval**: 15 episodes/task during training (cheap curves); the headline number is
  `params_best.pkl` re-scored at 50 episodes/task via `src/scripts/reeval.py`, which is
  what every published OGBench command uses.
- **Plateau**: never fires before 1M gradient steps, and tests a 3-eval moving average,
  because one 75-episode evaluation is noisy enough to fake a plateau.

## Relationship to `src/offline/config.py`

`config.py` remains the authoritative, annotated record of what the reference codebases
do — each value cites its source. `src/offline/schema.py` restates those defaults in
OmegaConf-friendly types for Hydra. They are kept in sync by
`test_schema_matches_config()`, which runs on every `launch.py` invocation and in the
GPU smoke test. **Change a default in `config.py` first**; the drift check will tell you
what else to update.

Hydra composes the config and resolves it into exactly the per-run JSON that
`train.py --run-config` already consumes, so this layer sits *above* the existing
pipeline rather than replacing it — jobs already in the Slurm queue are unaffected.

Each generation writes `_composed_<experiment>.yaml` next to the run configs: the fully
resolved config, so a batch can always be traced back to what produced it.
