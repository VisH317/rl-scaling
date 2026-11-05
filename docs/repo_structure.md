# Repo structure: what to reuse instead of rewrite

**Status:** plan, nothing done yet · **Written** 2026-11-05, while `2993276` /
`2993926` / `2994202` are in the queue

The question this answers: *as more experiments and evals are added, what gets written
once and reused, and what gets rewritten each time?* Everything below is measured
against the repo as it stands, with file and line references, not estimated.

---

## 0. First, an operational hazard that changes the order of everything

**A queued job runs the repo as it is when the job STARTS, not as it was when the job
was submitted.** `train_pack.sbatch` does `cd "$REPO/src"` and `eval_pack.sbatch` runs
`$PY "$REPO/src/eval.py"`, with `REPO=/home/vt7993/rl-scaling` — the live working tree.

Right now `2994202` is pending behind ~14 h of training and will execute whatever
`src/eval.py` and `src/offline/evals/` contain when it starts. So:

> **Freeze `src/eval.py`, `src/offline/evals/**` and `src/offline/{train,networks,
> patches,critic_losses,datasets}.py` until `2994202` finishes.** Stages 1–2 below are
> chosen to avoid those paths entirely.

The permanent fix is in Stage 1: **snapshot the code into scratch at submit time** and
point the sbatch at the snapshot. That also makes a finished run reproducible — today a
run's `config.json` records its hyperparameters but nothing records the code that read
them, so a run from last week cannot be re-executed exactly.

---

## 1. What is already right — do not "improve" these

Worth stating, because a refactor that loses them is a regression:

| thing | why it works |
|---|---|
| One YAML per experiment, `src/run.py` as the only entry point | killed the Hydra layer and the second place every default lived |
| An eval is a **directory** with `eval.yaml` + `README.md` + `impl.py` | self-contained; `common.discover()` finds it by the presence of `eval.yaml` and **raises** on a malformed one rather than skipping |
| `status: planned` as a first-class state | a design is registered, priced and listed by the tool that would run it |
| `UNIMPLEMENTED` in `run.py` (a knob nothing reads is a hard error) | this is the single best idea in the repo and should be the pattern everywhere |
| Metric self-checks against synthetic inputs with known answers | `critic_heatmap/impl.py::_selftest`, `critic_losses.py`, `networks.py` |
| Run ids are a hash of the resolved config | provenance is automatic; see §2, it is also the main refactor constraint |

---

## 2. The one hard constraint

`C._run_id(cfg)` hashes the **resolved config dict**. Any change that alters that dict —
a reordered key, a new field, a default moved — renames every run and **orphans the 36
run directories on scratch**, which represent several hundred GPU-hours.

`smoke.sbatch` already guards this for two grids (step 3c, "run ids CHANGED"). That test
is the backbone of this whole plan and must come first and be extended to every
experiment file. **No stage below may be merged with that check failing.**

---

## 3. Measured duplication

| # | what is duplicated | evidence | cost as the repo grows |
|---|---|---|---|
| D1 | **Two parallel compilers.** `run.py` (509 ln) and `eval.py` (367 ln) implement the same lifecycle: `load → resolve → plan → write → submit → status → figures`, with two YAML loaders, two unknown-key checks, two sbatch submitters, two time-formatters | `run.py:156,196,352,426,463` vs `eval.py:70,107,166,228,246,318` | every lifecycle fix gets made twice or, worse, once |
| D2 | **Interpreter guard, 18 lines, copied 5×** | `run.py`, `eval.py`, `scripts/{monitor,dashboard,estimate_runtime}.py` | every new CLI either copies it or breaks on the login node's Python 3.6.8 |
| D3 | **`sys.path.insert` bootstrap, 20 occurrences across 15 files** | `grep -rn sys.path.insert src/` | import order becomes load-bearing and undocumented |
| D4 | **sbatch boilerplate ×4.** Each of `train_pack`/`reeval`/`eval_pack`/`smoke` re-declares account, partition, `REPO`, `PY`, `MUJOCO_GL`, `TQDM_DISABLE`, `XLA_PYTHON_CLIENT_PREALLOCATE` and its own log path | **19 hardcoded absolute paths** live in `src/slurm/`, outside `config.py` | moving scratch, or changing the account, is a 4-file edit with no test |
| D5 | **Two agent loaders.** `scripts/reeval.py:52` and `evals/common.py::load_agent` rebuild the agent and restore the checkpoint | already flagged in `common.py`'s docstring | a fix to one silently does not reach the other |
| D6 | **Two progress viewers.** `monitor.py` (249 ln, one grid) and `dashboard.py` (266 ln, everything) both parse `squeue` and `status.json` | their own docstrings state the split | the `squeue` parser exists twice |
| D7 | **`config.py` is 939 lines of two different things**: a live registry (`DATASETS`, `PUBLISHED_FLAGS`, `SUBSET_PLANS`, paths) and superseded config classes (`GRIDS`, `GridSpec`, `ArchConfig`, `AgentConfig`, `TrainConfig`, `OptimConfig`, `STOCK_ARCH`, `run_config`) that `run.py::DEFAULTS` now **mirrors** | `config.py:366,372,678,706,819`; its own docstring says "what is superseded" | a reader cannot tell which half is authoritative without reading the docstring |
| D8 | **No config inheritance.** Every experiment restates `tower`, `width`, `block_size`, `step_cap`, `eval_interval`… `humanoid_ensemble1.yaml` restates 9 lines identical to `phase1_repro.yaml` to change 2 things | `experiments/*.yaml` | **this is the one the user asked about**: the Nth experiment copies N−1's boilerplate and silently inherits its mistakes |
| D9 | **Hand-maintained schema docs drift, demonstrably.** `TEMPLATE.yaml:32` says `critic_ensemble: 2 # 2 only -- crl.py's actor unpacks q1, q2` and `experiments/README.md:94` says "1 would crash" — while `humanoid_ensemble1.yaml` with `critic_ensemble: 1` **is in the queue right now** | those two lines | docs that lie are worse than no docs |
| D10 | **Dead scaffolding.** `src/rl_scaling/` contains only `print("Hello from rl-scaling!")`, is not installed, and `pyproject.toml` declares `requires-python = ">=3.13"` (venv is **3.11.15**), `dependencies = []` (real deps are in `src/offline/requirements.txt`), and an entry point `rl_scaling:main` nothing calls | `pyproject.toml`, `src/rl_scaling/__init__.py` | a newcomer's first `uv sync` fails or installs nothing |
| D11 | **3,336 lines of markdown, a 2-line root README, no index** | `find . -name '*.md'` | the work is undiscoverable from the top |
| D12 | **No CPU test entry point.** 6 modules have `_selftest`s; the only thing that runs them is `smoke.sbatch`, which **needs a GPU** | `grep -l _selftest src/**/*.py` | nothing catches drift on the login node in the 10 s it would take |

---

## 4. Target shape

Two module contracts, and everything else is library code behind them.

### 4.1 An **experiment** is pure configuration

No code. One YAML, which may inherit:

```yaml
# experiments/phase1_repro.yaml
base: [antmaze_stitch, residual_depth_sweep]   # fragments in experiments/_base/
agent: [crl, gcbc]
profile: ogbench
```

```yaml
# experiments/_base/antmaze_stitch.yaml     the dataset and its schedule
dataset: antmaze-medium-stitch-v0
seeds: [0, 1, 2]
step_cap: 2_000_000
plateau_min_steps: 1_000_000
eval_interval: 100_000
eval_episodes: 15

# experiments/_base/residual_depth_sweep.yaml   the architecture axis
tower: residual
width: 256
block_size: 4
repr_dim: 64
depth: [4, 8, 16, 32]
```

`humanoid_ensemble1` then becomes the four lines that are actually its experiment:

```yaml
base: [humanoidmaze_medium, residual_depth_sweep]
agent: [crl]
seeds: [0, 1]
critic_ensemble: 1          # the one deliberate change
```

**Merge rule, stated once and tested:** shallow merge; later `base` beats earlier; the
file's own keys beat every base. **Lists are replaced, never merged** — a list is a
sweep axis, and silently concatenating two axes would multiply the grid without
anyone asking for it. The resolved config is already written to
`configs/_<name>.resolved.yaml`, so provenance survives.

### 4.2 An **eval** is a self-contained directory

Already true, and it is the right shape. Keep `eval.yaml + README.md + impl.py
(+ figures.py)` colocated. The asymmetry with `experiments/` is deliberate and worth
writing down: *an experiment is pure config, an eval is code plus config*, so forcing
them into the same layout would be a worse fit, not a better one.

Add only: `evals/suites/*.yaml` with the same `base:` mechanism, so a second suite
(e.g. one that runs on humanoidmaze) does not copy `evals.yaml`.

### 4.3 The library behind both

Reuse the dead `src/rl_scaling/` name rather than inventing another:

```
src/rl_scaling/
    bootstrap.py    the interpreter guard + sys.path, imported first by every CLI   [D2,D3]
    paths.py        every path, derived from two env vars. Nothing else holds one.  [D4]
    yamlconf.py     load + `base:` inheritance + unknown-key error + UNIMPLEMENTED  [D8,D9]
    pipeline.py     load -> resolve -> plan -> write -> submit -> status -> report  [D1]
    slurm.py        submission, squeue parsing, the shared sbatch prelude           [D4,D6]
    checkpoints.py  THE agent loader; reeval and the evals both call it             [D5]
    viz.py          palette, style_axes, save(png + csv)                            [--]
src/offline/        the research domain: networks, patches, train, datasets, critic_losses
src/offline/evals/  eval units (unchanged shape)
src/run.py          thin CLI over pipeline, subject = runs
src/eval.py         thin CLI over pipeline, subject = evals
```

`pipeline.py` holds the lifecycle; `run.py` and `eval.py` supply a *subject*: how to
resolve a config into units of work, how to print them, which sbatch to submit. That is
the only real difference between them today.

---

## 5. Stages

Ordered so that each is independently useful, independently revertible, and so the
risky ones come after the queue drains. Sizes are rough.

### Stage 0 — guardrails (do first, blocks everything) · ~80 ln · no risk

- Extend the `smoke.sbatch` run-id regression from 2 grids to **every** `experiments/*.yaml`, and snapshot the expected ids into a committed `tests/run_ids.json`.
- Add `tests/check.py`: a **CPU-only, login-node-safe** entry point that runs every `_selftest`, resolves every experiment and suite, and diffs `TEMPLATE.yaml` against the generated schema. Target: **under 15 s, no GPU, no dataset**.
- `make check` / `src/check.py` so it is one word to run.

*Verifies:* itself — it must pass before and after every later stage.

### Stage 1 — delete copy-paste with no design content · ~200 ln net **deletion** · low risk

Touches no file the queued jobs execute.

1. `rl_scaling/bootstrap.py` — the guard and path setup, once. The 5 copies become `import rl_scaling.bootstrap`. [D2, D3]
2. `rl_scaling/paths.py` — move the path block out of `config.py`; `slurm/_prelude.sh` sourced by all four sbatch files, exporting `REPO`, `PY`, `EXP`, the GL and XLA vars and the log directory. **19 hardcoded paths → 2 env vars.** [D4]
3. Fix `pyproject.toml` (`requires-python = ">=3.11"`, real dependencies from `requirements.txt`) and delete `src/rl_scaling/__init__.py`'s Hello World. [D10]
4. **Submit-time code snapshot**: `--submit` copies `src/` to `<EXP>/code/<job_id>/` and the sbatch runs from there. Fixes §0 permanently and makes a finished run re-executable.

*Verifies:* Stage 0 passes; `sbatch --test-only` on each script; one real 1-task eval job against the snapshot.

### Stage 2 — config inheritance and a generated template · ~250 ln · low risk, high payoff

This is the stage that answers the original question.

1. `yamlconf.py` with `base:` resolution and the merge rule from §4.1.
2. Rewrite the five existing experiment files as `base:` + deltas. **The run-id test must stay green — that is the whole proof the rewrite was faithful.**
3. **Generate `TEMPLATE.yaml` from `DEFAULTS` + `UNIMPLEMENTED` + docstrings** (`src/run.py --template`), and make Stage 0 fail if the committed file differs. This structurally kills D9: the `critic_ensemble` comment could not have gone stale, because nobody would have written it by hand.
4. Same mechanism for `evals/suites/`.

*Verifies:* run ids unchanged for all five experiments; `--template` output == committed file.

### Stage 3 — one lifecycle · ~400 ln moved, ~250 deleted · medium risk · **after `2994202`**

Extract `pipeline.py`; `run.py` and `eval.py` become subject definitions. Fold
`--report` (currently eval-only) into the pipeline so experiments get the same
AUTO-block results document that the evals now have — `results_phase1.md` is
hand-maintained today and has already been wrong once (the `last3_mean` pin bug).

*Verifies:* byte-identical `plan` output on all five experiments and the suite, captured
as golden files in Stage 0.

### Stage 4 — consolidate domain duplicates · ~300 ln deleted · medium risk

1. One agent loader; `reeval.py` calls it. Verify by re-scoring **one** finished run and matching its recorded `reeval_success` exactly. [D5]
2. `monitor.py` becomes `dashboard.py --grid` — one `squeue` parser. [D6]
3. Split `config.py`: `registry.py` (datasets, published flags, subset plans) keeps the verified values; the superseded config classes move to `docs/source_comparison.md`, where the three-codebase comparison tables are genuinely valuable as **prose** and actively confusing as **code**. [D7]

### Stage 5 — docs index · ~100 ln · no risk

Root `README.md`: what this is, the three commands, and a table pointing at the 3,336
lines that already exist (`offline_exp.md` = the brief, `algorithm.md` = what runs,
`research_log.md` = verified facts, `results_phase1.md` = findings, `evals/README.md`
= the eval architecture). [D11]

---

## 6. Deliberately not proposed

- **A plugin registry / entry-point system for evals.** Directory discovery already works and is one function. Entry points would add packaging ceremony for zero reach.
- **A config schema library (pydantic/attrs).** `run.py`'s hand-written unknown-key + `UNIMPLEMENTED` checks produce better error messages than a validator would, because they say *why* a knob is refused. Keep them.
- **Merging `experiments/` and `evals/` into one config system.** They answer different questions (what to run vs what to measure), have different lifecycles, and one of them carries code. Symmetry here would be cosmetic.
- **Moving the eval code out of `src/offline/evals/`.** Colocating an eval's code, config and README is what makes it a module.
- **A generic "Task" abstraction over training runs and eval tasks.** They differ in cost by three orders of magnitude and in failure mode entirely. The shared part is the *lifecycle*, which Stage 3 extracts; the subjects should stay separate.

---

## 7. Decisions needed

1. **Stage ordering.** Stage 2 is the one that was actually asked for. Stage 1 is a prerequisite only for the code-snapshot fix, not for inheritance — so **2-before-1** is available if the inheritance payoff is wanted sooner.
2. **`experiments/_base/` naming.** `_base/` keeps fragments visually separate from runnable experiments and sorts first. Alternatives: `experiments/base/`, or a top-level `base.yaml` with named sections.
3. **Does `results_phase1.md` become an AUTO-block document** (Stage 3)? It would stop the numbers drifting, at the cost of a one-time rewrite of a 651-line file that currently reads well.
4. **Is the code snapshot worth the scratch space?** `src/` is ~8k lines, so a few hundred KB per job — negligible, but it does mean a run's code and the repo's code can diverge, which is the point and also a thing to remember when debugging.

---

## 8. Scorecard

If all five stages land:

| | now | after |
|---|---|---|
| lines in the two compilers | 876 | ~450 + one shared pipeline |
| copies of the interpreter guard | 5 | 1 |
| hardcoded absolute paths outside the path module | 23 | 0 |
| agent loaders | 2 | 1 |
| `squeue` parsers | 2 | 1 |
| lines in a new experiment's YAML | 10–15 | 3–5 |
| hand-maintained schema docs | 2 (both stale) | 0 (generated, drift-tested) |
| CPU-only test entry point | none | `make check`, <15 s |
