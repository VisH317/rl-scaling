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
the optimizer. **Known deviation from the paper:** it uses a single critic encoder; we
use OGBench's two. Recorded, not silently accepted.
