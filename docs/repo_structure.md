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

> **Nothing in `/home/vt7993/rl-scaling` may change until `2994202` finishes.** Not
> `src/eval.py`, not `src/offline/evals/**`, not `src/offline/{train,networks,patches,
> critic_losses,datasets}.py`. §0.1 is how the refactor proceeds anyway.

The permanent fix is in Stage 1: **snapshot the code into scratch at submit time** and
point the sbatch at the snapshot. That also makes a finished run reproducible — today a
run's `config.json` records its hyperparameters but nothing records the code that read
them, so a run from last week cannot be re-executed exactly.

### 0.1 Why this work is in a git *worktree*, not a branch

A branch is not isolation here. `git checkout` rewrites the files **in place**, so
switching `/home/vt7993/rl-scaling` to a refactor branch would hand the queued job the
refactored code at 08:40 tomorrow — precisely the failure the freeze exists to prevent.

So the work happens in a second checkout at a different path:

```
/home/vt7993/rl-scaling            main                 <- what the queue executes
/home/vt7993/rl-scaling-refactor   refactor/structure   <- where the work happens
```

One repository, two working trees, one `.git`. The main tree is never touched, so the
freeze is **lifted for the refactor**: every stage below, including the ones that
rewrite `src/eval.py`, is safe to do right now.

The constraint moves to the merge. **Do not merge `refactor/structure` into `main`
until `2994202` has finished** — merging is the moment the live tree changes. After
Stage 1 lands the constraint weakens to "no job submitted before the snapshot existed
is still queued".

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
| D13 | **The vendored OGBench implementations are not versioned.** `.gitignore` line 1 is `lib/**`, so 54 MB of `impls/{agents,utils}` — which `patches.py` monkeypatches and the project cannot run without — is outside git. A fresh clone or a `git worktree` has no `lib/` at all | found by creating the worktree for this refactor | the repo is not reproducible from git alone, and checks pass *vacuously* where it is missing (see Stage 0's fix) |
| D12 | **No CPU test entry point.** 6 modules have `_selftest`s; the only thing that runs them is `smoke.sbatch`, which **needs a GPU** | `grep -l _selftest src/**/*.py` | nothing catches drift on the login node in the 10 s it would take |

---

## 4. Target shape

Two module contracts, and everything else is library code behind them.

### 4.1 An **experiment** is one flat, complete file — generated from a template

*(Revised 2026-11-05 after review. The first draft proposed runtime `base:`
inheritance; that is rejected, and §6 records why.)*

An experiment file stays exactly what it is today: **every key present, in one file you
can read top to bottom**, with no indirection to chase. Reuse happens at the moment the
file is **created**, not every time it is read.

```bash
src/run.py --new humanoid_ensemble2 --template depth_sweep \
           --set dataset=humanoidmaze-medium-navigate-v0 --set critic_ensemble=1
# -> writes experiments/humanoid_ensemble2.yaml: flat, complete, every key commented
```

```
experiments/
  _templates/
    depth_sweep.yaml        the common shape: a depth axis over one dataset
    loss_comparison.yaml    extends: depth_sweep, sweeps `profile` instead
    single_point.yaml       one cell, for a pilot
  phase1_repro.yaml         flat. Reads top to bottom. No `base:` key.
  humanoid_ensemble1.yaml   flat.
```

**Templates may extend templates** — `extends:` inside `_templates/` only — and that
chain is resolved **at generation time**. What lands in `experiments/` is always fully
expanded. Three consequences worth being explicit about:

- *You see every knob for an experiment in its own file.* That is the requirement, and
  it is what runtime inheritance would have taken away.
- *The file on disk is exactly what runs*, so run-id stability is obvious rather than
  argued — there is no resolution step between the file and the hash.
- *Editing a template cannot silently change an old experiment.* For a research repo
  that is a feature: a finished run's config means what it meant when it ran.

The cost is real and I am not going to hide it: **a new default has to be applied to
each existing experiment**, not inherited. `--regenerate <name>` re-emits a file from
its template while preserving the keys that differ from the template's values (recorded
in a `# generated-from:` header), so this is one command per file rather than a hand
edit — but it is still N files, and the run-id test is what proves each one is faithful.

The comments in a generated file are **not hand-written**. They come from the schema
(§4.4), which is why the `critic_ensemble` comment in D9 could not go stale again.

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
    schema.py       the ONE definition of every config key: default, doc,
                    choices, sweepable, unimplemented. Everything else is
                    derived from it -- templates, docs, validation.        [D8,D9]
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

### 4.4 Observability: knowing what exists without reading the code

A standing requirement, not a one-off: **it should be possible to see what the repo can
do, and what belongs in a config file, without opening a `.py`.** Today that is false —
the answer to "what keys does an experiment take?" lives in `run.py:DEFAULTS` and is
*transcribed* by hand into `TEMPLATE.yaml` and `experiments/README.md`, which is exactly
how D9 happened.

The fix is one source and generated views of it.

**The source.** `rl_scaling/schema.py` holds one record per config key:

```python
Key("critic_ensemble", default=2, group="agent", sweepable=True,
    doc="Critic ensemble members. OGBench uses 2 (q1/q2 for the actor's min); "
        "scaling-crl uses 1. Implemented in offline/critic_losses.py.",
    choices=(1, 2))
```

`DEFAULTS`, `SWEEPABLE` and `UNIMPLEMENTED` are **derived** from it, so `run.py` keeps
behaving identically while nothing is stated twice.

**The generated views**, all under `docs/reference/`, all rebuilt by `src/check.py
--write` and **drift-tested** (the check fails if a committed file differs from what
regenerating would produce):

| file | generated from | answers |
|---|---|---|
| `config_keys.md` | `schema.py` | every experiment key: default, choices, what it does, whether it is a sweep axis, whether anything reads it |
| `evals.md` | each `eval.yaml` + its README's first paragraph | every eval: status, what it measures, which agents it applies to, its params and defaults, its outputs |
| `experiments.md` | `experiments/*.yaml` resolved | every experiment: run count, grid, cost estimate, current status |
| `cli.md` | the argparse parsers | every command and flag across `run.py`, `eval.py` and `scripts/` |
| `outputs.md` | `impl.py` writers | what every artefact file contains, field by field |

**The skill.** `.claude/skills/rl-scaling/SKILL.md` is the entry point for a Claude Code
session in this repo and for a human reading over its shoulder. It states:

- where the generated reference lives and that it is generated (never hand-edit);
- the two module contracts from §4.1 and §4.2, as checklists;
- **what to put in a config file** — which keys are required, which are sweep axes,
  which are deliberately refused and why;
- the non-negotiables: run ids are config hashes; a knob nothing reads is a hard error;
  an eval must ship a self-check that runs without a GPU;
- the operational rules from §0 — the live tree is what the queue executes, so check
  the queue before editing it;
- what to regenerate and re-run after a change (`src/check.py`).

A skill is the right shape for this because it loads *before* the work, which is where
instructions change the outcome. A README is read after someone has already guessed.

---

## 5. Stages

Ordered so that each is independently useful and independently revertible. Because the
work is in a worktree (§0.1) none of them has to wait for the queue — the ordering is
about dependency and proof, not about the scheduler. Sizes are rough.

### Stage 0 — guardrails · DONE (`6fce227`) · ~290 ln

`src/check.py` — 9 checks, CPU-only, ~13 s. `tests/run_ids.json` pins the **54 run
ids** the five committed experiments produce; `tests/plan_golden/` pins the plan text
for each experiment and the eval suite, machine-specific lines normalised. A third
check compares resolved ids against the run directories actually on scratch, because
the golden alone cannot catch a rename where code and golden are regenerated together.
`make check` / `make check-write`.

**Verified the guards bite rather than merely pass.** Changing one default `run.py`
inherits (`plateau_patience` 5→6) fails run-id stability on all five experiments, fails
the plan goldens, and makes the orphan check name the humanoid run directories now
training. Changing a default two experiments override (`repr_dim`) correctly fails only
the three that rely on it.

A precondition check was added after the worktree exposed D13: if `lib/ogbench/impls`
is absent the harness **fails** instead of reporting a vacuous pass — `critic_losses`'
self-test passes without it, because the maths runs while `install_critic_loss`, which
monkeypatches `agents.crl`, is never reached.

### Stage 1 — delete copy-paste with no design content · DONE · ~200 ln net **deletion**

1. `rl_scaling/bootstrap.py` — the guard and path setup, once. The 5 copies become `import rl_scaling.bootstrap`. [D2, D3]
2. `rl_scaling/paths.py` — move the path block out of `config.py`; `slurm/_prelude.sh` sourced by all four sbatch files, exporting `REPO`, `PY`, `EXP`, the GL and XLA vars and the log directory. **19 hardcoded paths → 2 env vars.** [D4]
3. Fix `pyproject.toml` (`requires-python = ">=3.11"`, real dependencies from `requirements.txt`) and delete `src/rl_scaling/__init__.py`'s Hello World. [D10]
4. **Submit-time code snapshot**: `--submit` copies `src/` to `<EXP>/code/<job_id>/` and the sbatch runs from there. Fixes §0 permanently and makes a finished run re-executable.

**Done.** `rl_scaling/{bootstrap,paths,slurm}.py`; the 6 guard copies are 3 lines each
(verified: every entry point still works under the system `python3` *and* the venv
python). `slurm/_prelude.sh` `eval`s `paths.py`'s own output, so the shell and Python
halves cannot drift; hardcoded paths in `src/slurm/` went **19 → 7**, and the 7 are the
`#SBATCH --output` directives sbatch parses before any shell runs (the submitter now
passes `--output`; the directives are hand-run fallbacks) plus one deliberate scratch
fallback. `pyproject.toml` now declares the real dependencies and `>=3.9` instead of
`>=3.13` against a 3.11 venv, and says in a comment that it does *not* cover `lib/`.

The snapshot threw up a design point worth recording: `paths.REPO` (where the code
runs) had to be split from `paths.OUTPUT_REPO` (where human-facing output lands).
Without that, `eval.py --report` would have written `results_<suite>.md` into the
snapshot on scratch, where nobody would ever see it.

*Verified:* Stage 0 green throughout; the snapshot is self-sufficient
(`<snap>/src/check.py` runs there and `_prelude.sh` resolves `REPO` to the snapshot);
`sbatch --test-only` accepts the snapshot-based submission.

### Stage 2 — the schema as the single source · ~350 ln · low risk, high payoff

This is the stage that makes §4.4 true, and it subsumes the template work.

1. `rl_scaling/schema.py`: one `Key` per config key. `DEFAULTS` / `SWEEPABLE` /
   `UNIMPLEMENTED` become derived views. **Values must be byte-identical — the run-id
   test is the proof.**
2. `src/run.py --new <name> --template <t> [--set k=v]` writes a flat, complete,
   commented experiment file. `experiments/_templates/` with `extends:` resolved **at
   generation time**, so nothing in `experiments/` has a `base:` key.
3. `--regenerate <name>` re-emits from the template, preserving keys that differ.
4. `TEMPLATE.yaml` and `docs/reference/config_keys.md` are generated; Stage 0's check
   fails on drift. **D9 becomes structurally impossible.**

*Verifies:* run ids unchanged for all five experiments; regenerating each committed
experiment is a no-op diff; `--template` output == committed file.

### Stage 3 — one lifecycle · ~400 ln moved, ~250 deleted · medium risk

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

### Stage 5 — generated reference + docs index · ~300 ln · no risk

The five generated files in §4.4, plus a root `README.md` that is an index: what this
is, the three commands, and a table pointing at the 3,336 lines of prose that already
exist (`offline_exp.md` = the brief, `algorithm.md` = what runs, `research_log.md` =
verified facts, `results_phase1.md` = findings, `evals/README.md` = the eval
architecture, `docs/reference/` = generated). [D11]

### Stage 6 — the Claude skill · ~150 ln · no risk

`.claude/skills/rl-scaling/SKILL.md` per §4.4. Written last on purpose: it documents
the contracts the earlier stages establish, and a skill that describes a structure
that does not exist yet is the same drift problem in a new place.

---

## 6. Deliberately not proposed

- **Runtime config inheritance (`base:` in an experiment file).** This was in the first
  draft and is rejected. Seeing every knob for an experiment in one file, read top to
  bottom, is worth more than not repeating `tower: residual` five times — the knobs are
  the experiment, and an inherited one is a knob you did not know you had. Reuse moves
  to generation time (§4.1), which keeps the DRY benefit where it costs nothing and
  additionally makes a finished run's config immune to later template edits.
- **A plugin registry / entry-point system for evals.** Directory discovery already works and is one function. Entry points would add packaging ceremony for zero reach.
- **A config schema library (pydantic/attrs).** `run.py`'s hand-written unknown-key + `UNIMPLEMENTED` checks produce better error messages than a validator would, because they say *why* a knob is refused. Keep them.
- **Merging `experiments/` and `evals/` into one config system.** They answer different questions (what to run vs what to measure), have different lifecycles, and one of them carries code. Symmetry here would be cosmetic.
- **Moving the eval code out of `src/offline/evals/`.** Colocating an eval's code, config and README is what makes it a module.
- **A generic "Task" abstraction over training runs and eval tasks.** They differ in cost by three orders of magnitude and in failure mode entirely. The shared part is the *lifecycle*, which Stage 3 extracts; the subjects should stay separate.

---

## 7. Decisions needed

1. **Merge timing.** The branch cannot land while `2994202` is queued (§0.1). Merge
   right after it finishes, or hold until Stage 1's snapshot makes merge timing
   irrelevant?
2. **`--regenerate` semantics.** When a template changes, should regenerating an
   experiment preserve *every* key that differs from the template, or only keys listed
   in its `# generated-from:` header? The first is safer, the second is tidier.
3. **Does `results_phase1.md` become an AUTO-block document** (Stage 3)? It would stop
   the numbers drifting, at the cost of a one-time rewrite of a 651-line file that
   currently reads well.
4. **How much does `docs/reference/` duplicate the skill?** Current split: the
   reference is exhaustive and generated, the skill is short, opinionated and
   hand-written, and points at the reference. If the skill starts restating key
   defaults it has become a third copy.

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
| writing a new experiment | copy the nearest file, edit, hope | `--new --template --set`, flat output |
| hand-maintained schema docs | 2 (both stale) | 0 (generated, drift-tested) |
| CPU-only test entry point | none | `src/check.py`, <15 s |
| "what can this repo do?" answered without reading code | no | `docs/reference/` (5 generated files) + the skill |
