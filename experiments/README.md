# Experiments

One experiment = one YAML file here. Nothing else needs editing.

```bash
V=/scratch/gpfs/vt7993/rl-scaling/venv/bin/python

$V src/run.py experiments/scaling_crl_replication.yaml            # show the plan
$V src/run.py experiments/scaling_crl_replication.yaml --submit   # generate + queue
$V src/run.py experiments/scaling_crl_replication.yaml --status   # progress
$V src/run.py experiments/scaling_crl_replication.yaml --figures  # plots + CSVs
$V src/run.py experiments/scaling_crl_replication.yaml --reeval   # 50 eps/task rescore
```

`--submit` refuses if `run.py` raised a warning; add `--force` when it is intended.
Without `--write` or `--submit` nothing is created.

## The files

| file | what |
|---|---|
| `TEMPLATE.yaml` | every key, commented. Copy this. |
| `phase1_anchor.yaml` | control: OGBench's own architecture, targets CRL 53±6 / GCBC 45±11 |
| `phase1_repro.yaml` | the depth sweep we ran: CRL 34.1 → 49.3 → 71.4 at depths 4/8/16 |
| `scaling_crl_replication.yaml` | the 1000-layer paper's critic, offline. 12 runs, ~131 GPU-h |
| `q1_loss_pilot.yaml` | 4-run cheap version of the above. Run this first. |

## Sweeps

Any key may be a list; runs are the full product of every list, times `seeds`.

```yaml
depth: [4, 8, 16, 32]
profile: [ogbench, scaling_crl]
seeds: [0, 1, 2]
# -> 4 x 2 x 3 = 24 runs
```

`run.py` prints the run count and estimated GPU-hours before anything is queued.

## Profiles

Three fields — `critic_loss`, `logits`, `logsumexp_penalty` — are only meaningful
together, so they are bundled:

| profile | critic_loss | logits | logsumexp | ensemble | batch | repr_dim |
|---|---|---|---|---|---|---|
| `ogbench` | `binary_nce` | `dot` | 0.0 | 2 | 1024 | (yours) |
| `scaling_crl` | `infonce` | `neg_l2` | **0.1** | 1 | 256 | 64 |
| `stable_crl` | `binary_nce` | `dot` | 0.0 | 2 | 2048 | 16 (+ cold init, α=0.05) |

Forward InfoNCE's row softmax is invariant to adding any `c(s,a)` to a row, so it pins
the density ratio only up to a function of `(s,a)` — and the actor's `max_a Q(s,a,g)`
compares across exactly that freedom. The logsumexp penalty is what removes it.
**`infonce` with `logsumexp_penalty: 0` would break actor extraction for reasons
unrelated to depth**, so `run.py` warns and refuses to submit. Set the fields
individually only to override one, deliberately.

Precedence, lowest first: **profile → OGBench published flags → explicit YAML keys.**
Writing a key by hand always wins, because that is the only reason to write it.

## Guarantees

- `phase1_repro.yaml` and `phase1_anchor.yaml` **reproduce the run ids already on disk
  exactly**, verified in the smoke test. Re-running them recognises finished work
  instead of retraining it. Provenance (`experiment_file`) is attached after the id is
  hashed, so renaming a YAML never renames a run.
- A resolved copy of the experiment lands at
  `$CONFIGS_DIR/_<name>.resolved.yaml` beside the per-run JSONs.
- `run.py` warns, before queueing, about: optimizer settings that confound the depth
  axis; `use_published_flags: false`; `infonce` without the penalty; `neg_l2` making
  the logged `binary_accuracy` meaningless; and `tower: ogbench_mlp` silently ignoring
  a depth sweep.

## Where the rest lives

`src/offline/config.py` is now a **registry** — datasets, paths, OGBench's published
flags, run-id construction — not a place to tune anything. `src/offline/train.py` is
unchanged and still consumes the same per-run JSON. The Hydra layer it replaced
(`conf/`, `schema.py`, `scripts/launch.py`) is in git history at `53177a7`.

## Knobs that are not implemented

`run.py` **hard-errors** rather than accept a setting nothing downstream reads. A knob
that silently does nothing is worse than no knob — it produces a config file that lies
about what ran. Audited 2026-11-05 by tracing each field to its consumer:

| key | only value that works | why |
|---|---|---|
| `schedule` | `constant` | `crl.py` builds a bare `optax.adam(lr)`; `train.py` never constructs one |
| `warmup_steps` | `0` | same — no schedule wrapper exists |
| `max_grad_norm` | `0.0` | same — no `clip_by_global_norm` in the chain |
| `weight_decay` | `0.0` | same — `adam`, not `adamw` |
| `critic_ensemble` | `2` | `crl.py` hardcodes `ensemble=True` and its ddpgbc actor unpacks `q1, q2 = critic(...)`; 1 would crash |
| `actor_loss` | `ddpgbc` | `awr` needs a separate value network `build_config` does not create |

So of the eight optimizer fields the config can express, exactly one — `lr` — reaches
the optimizer.

**Known deviation from the paper:** both use two critic *encoders*, `phi(s,a)` and
`psi(g)` — that is the bilinear critic. OGBench additionally vmaps each into a 2-member
*ensemble* (4 towers, giving `q1`/`q2` for the actor's `min`); the paper uses 1 member
(2 towers). We keep OGBench's 2. Recorded, not silently accepted.

## Estimating runtime before you submit

```bash
$V src/scripts/estimate_runtime.py experiments/humanoid_ensemble1.yaml --verbose
$V src/scripts/estimate_runtime.py --calibrate    # refit from finished runs
```

Two costs, modelled separately because they scale with different things:

- **train** — `step_cap / throughput`. Throughput falls with depth and width, and CRL
  costs ~2.6× GCBC at equal depth.
- **eval** — `(step_cap / eval_interval)` evaluations, each `5 tasks × eval_episodes`
  rollouts of up to `max_episode_steps`. The term people forget. On humanoidmaze the
  episode limit is **2000** rather than antmaze's 1000, so eval is **31%** of that
  experiment's total rather than 13%. A failing policy never terminates early, so eval
  costs most exactly when a run is going badly.

Constants are calibrated from this project's finished runs (`--calibrate` refits and
prints them to paste back). Cells with no measurement at that depth are extrapolated
and **labelled** — an estimate never quietly presents itself as a measurement. The
episode limit is read from the gym registry and the script **refuses to guess** if it
cannot resolve one, because OGBench's env id is not the dataset name
(`humanoidmaze-medium-navigate-v0` → env `humanoidmaze-medium-v0`) and a silent
fallback to 1000 halves every humanoid estimate.

`run.py` prints the same model's headline, so there is only one cost number.

## Watching things run

```bash
$V src/scripts/dashboard.py                 # everything you have, one snapshot
$V src/scripts/dashboard.py --watch 30      # live, refreshing
$V src/scripts/dashboard.py --tasks         # + per-task breakdown of the last eval
$V src/scripts/dashboard.py --grid NAME     # just one experiment
```

`dashboard.py` answers "what do I have running right now" across every experiment;
`monitor.py --grid NAME` answers "how is this one grid doing" in more detail. Both are
read-only and safe to leave open in a second terminal.

**On per-episode progress:** there isn't any to show. OGBench's `evaluate()` loops over
episodes internally with tqdm disabled, so nothing is emitted until an evaluation
finishes. The finest real granularity is one evaluation, which `eval.csv` records with
its per-task breakdown — that is what `--tasks` prints. The dashboard does *infer* the
evaluating phase and labels it as inferred: the heartbeat is only written on step
boundaries and `eval_interval` is a multiple of `heartbeat_interval`, so a run parked
exactly on an eval boundary with a heartbeat older than one normal gap is mid-eval.
