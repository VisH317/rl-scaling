# `offline/evals` — interpretability evals on trained checkpoints

An **eval** here is not a policy score. `scripts/reeval.py` already answers *how good is
this agent* by rolling it out. These answer *what did the network learn*, and they are
read-only with respect to the runs: they load `params_*.pkl` and write only under
`<ANALYSIS_DIR>/evals/`. Nothing in this directory can modify a run.

```bash
V=/scratch/gpfs/vt7993/rl-scaling/venv/bin/python

src/eval.py                     # resolve the suite and print the plan
src/eval.py --list              # every eval that exists, and its status
src/eval.py --write             # write the task files
src/eval.py --submit            # write + queue one Slurm job
src/eval.py --status            # which tasks are done, failed, still pending
src/eval.py --figures           # rebuild figures from whatever has finished
src/eval.py path/to/other.yaml  # a different suite
```

`src/eval.py` re-execs into the venv, so the `$V` prefix is optional (it is still what
the sbatch scripts use). The suite file is `evals.yaml` in this directory.

## The two layers

| layer | file | what it decides |
|---|---|---|
| **suite** | `evals.yaml` | which grids, agents and depths; how the one checkpoint per depth is chosen; which evals run |
| **eval** | `<name>/eval.yaml` | what that eval measures and with what knobs; which agents and environments it applies to |

A suite entry may override any of an eval's own params, key by key; an override of a key
the eval does not define is a hard error, not a silent no-op.

```
evals.yaml                      the suite
critic_heatmap/
    eval.yaml                   config + status + applies_to + params
    README.md                   what the test is and how to read it   (required)
    impl.py                     run(task) -> dict                     (required)
    figures.py                  make_figures(suite_dir, out_dir)      (optional)
state_probes/                   same four files; status: planned
common.py                       checkpoint loading, maze geometry, run selection
```

`common.discover()` finds an eval by the presence of `eval.yaml`. A directory without
one is not an eval. A malformed one **raises** rather than being skipped: an eval that
silently fails to register is an eval you think you ran.

## Adding an eval

1. `mkdir src/offline/evals/<name>` and write the four files. Copy
   `critic_heatmap/eval.yaml` as the template — every key is documented there.
2. `name:` must equal the directory name. The directory is the id.
3. Start at `status: planned`. It then appears in `--list` and in the plan, priced and
   gated, and `--submit` refuses it by name. That is the point: a design nothing
   executes should still be visible to the tool that would execute it.
4. `impl.run(task)` gets the resolved task dict — `run_dir`, `checkpoint`, `params`,
   `out_dir`, and the `selection` record — and writes into `task["out_dir"]`.
5. Write a metric self-check that runs without a GPU or a checkpoint (see
   `critic_heatmap/impl.py`'s `_selftest`, which pins the metrics against synthetic
   fields whose answers are known). The smoke test runs it.

## How the checkpoint is chosen

`seed_select` picks one run per (grid, agent, depth):

| rule | source | note |
|---|---|---|
| `reeval` | `analysis/<grid>/reeval_50ep.json` | the 50-episodes/task number the report quotes. Falls back to `best_success` per run, **recording** that it did |
| `best_success` | each run's `summary.json` | the best training-time eval, 15 episodes/task |
| `last3` | the last three rows of `eval.csv` | where the run settled |
| `all` | — | no selection; every seed becomes its own task |

Every task carries the **whole seed set**, the runner-up and the spread, and the plan
prints them. At depth 4 the spread is 0.24 (0.208 / 0.376 / 0.444) — wider than some
depth gaps — so a one-seed-per-depth figure is partly a statement about that seed, and
the number that says so travels with the figure.

`require_finished: false` additionally admits runs that are still training, scored from
the tail of `eval.csv` and labelled `in_flight@<step>` wherever they appear. Use it to
look early; never to quote a number.

## Why one task per process

`offline/patches.py` swaps OGBench's `MLP` for the residual shim for the **lifetime of
the process**, because Flax re-runs `setup()` on every `apply`. Loading a second
checkpoint of a different depth into the same process would build it with the first
run's tower and die on a parameter-structure mismatch. So `eval_pack.sbatch` loops and
calls `src/eval.py --run <task.json>` once per task. Tasks are short (minutes), so the
repeated JAX start-up is not worth engineering around.

## The evals

| eval | status | question |
|---|---|---|
| [`critic_heatmap`](critic_heatmap/README.md) | ready | Does `Q(s, π(s,g), g)` rank maze positions by *geodesic* distance to the goal, or only by straight-line distance — and does that change with depth? |
| [`state_probes`](state_probes/README.md) | planned | Do `φ(s,a)` and `ψ(g)` *linearly* encode position, maze cell and geodesic distance — above a raw-observation and an untrained-network baseline? |

The two can disagree, which is the interesting case: a representation can carry geodesic
distance linearly while the dot product fails to use it, or the field can rank correctly
while neither representation is linearly decodable.
