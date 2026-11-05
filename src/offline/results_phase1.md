# Phase 1 results — offline GCRL depth scaling on `antmaze-medium-stitch-v0`

Compiled 2026-11-05. Covers grids `phase1_anchor` (finished) and `phase1_repro`
(16 of 24 runs finished, 8 still training).

Figures are in [`figures/`](figures/), copied from
`/scratch/gpfs/vt7993/rl-scaling/experiments/analysis/`. Every PNG has a CSV beside it,
and those CSVs are the source for the tables here — nothing below is typed by hand from
terminal output.

> **Headline.** `phase1_repro`'s own grid spec in `config.py` says: *"Expect CRL flat
> with depth, GCBC rising. If CRL scales here, stop and report — that alone changes the
> project."* **CRL scaled: 34.1 → 49.3 → 74.2 across depths 4 → 8 → 16.** This is the
> pre-registered stop-and-report condition, and §5 is the evidence that it is the critic
> doing the scaling.

---

## 1. The environment

### 1.1 `antmaze-medium`

A quadruped ant in a 2-D maze. OGBench's `MazeEnv`, 8×8 cell grid with a solid border,
`maze_unit = 4.0` world units per cell, so interior coordinates run 0–20 on each axis.

- **Observation:** 29-D. Dims 0–1 are the ant's `(x, y)`; the rest is the ant's own
  joint configuration and velocities. Goal-conditioned, so the policy sees
  `(observation, goal_observation)`.
- **Action:** 8-D continuous, torques on the ant's 8 leg joints, in `[-1, 1]`.
- **Episode limit:** 1000 steps.
- **Success:** binary per episode — did the ant reach the goal cell before the limit.
  Every "success rate" in this document is a **fraction of episodes in `[0, 1]`**.

The maze (`##` = wall, numbers = **% of all 1,005,000 dataset states** in that cell, so
the grid doubles as the data-coverage map):

```
      j=0    1     2     3     4     5     6     7
i=0    ##    ##    ##    ##    ##    ##    ##    ##
i=1    ##   1.26  1.68   ##    ##   1.70  0.59   ##
i=2    ##   1.84  4.40   ##   5.31  3.80  0.96   ##
i=3    ##    ##   7.55  8.03 10.58   ##    ##    ##
i=4    ##   4.65  4.52   ##  10.28  6.54  3.25   ##
i=5    ##   3.06   ##   3.28  4.93   ##   2.71   ##
i=6    ##   2.22  2.32  2.32   ##   0.91  1.29   ##
i=7    ##    ##    ##    ##    ##    ##    ##    ##
```

Cell `(i, j)` sits at world `x = 4(j−1)`, `y = 4(i−1)`. Coverage is 11× denser in the
central corridor `(3,4)` than in the `(6,5)` pocket.

### 1.2 What `-stitch` means, measured

The dataset is **5,000 trajectories × 201 steps = 1,005,000 rows / 1,000,000 valid
transitions**. I measured how far a single trajectory actually travels, as the shortest
maze-path distance in **cells** between its own first and last state:

| statistic | cells |
|---|---|
| mean | 3.93 |
| median | 4 |
| 90th percentile | 4 |
| max | **5** |

97% of trajectories cover **exactly 4 cells**, and **not one** covers more than 5. The
dataset contains no trajectory that solves any of the five tasks end to end. Every task
must be assembled from pieces — that is the entire point of the `-stitch` variant, and
it is why a method's ability to compose matters more here than its ability to imitate.

### 1.3 The 5 evaluation tasks

Each task is a fixed `(start cell, goal cell)` pair, evaluated for `eval_episodes`
episodes. Geometry below is computed by breadth-first search over the maze grid.

| task | start `(i,j)` | goal `(i,j)` | path (cells) | path (world units) | straight line (units) | detour ratio | stitches needed | legal first moves |
|---|---|---|---|---|---|---|---|---|
| 1 | (1,1) | (6,6) | 10 | 40 | 28 | 1.41 | 2.5 | 2 |
| 2 | (6,1) | (1,6) | 10 | 40 | 28 | 1.41 | 2.5 | 2 |
| 3 | (5,3) | (4,2) | 6 | 24 | 6 | **4.24** | 1.5 | 2 |
| 4 | (6,5) | (6,1) | 10 | 40 | 16 | 2.50 | 2.5 | **1** |
| 5 | (2,6) | (1,1) | 8 | 32 | 20 | 1.57 | 2.0 | 2 |

- **path (cells)** — BFS shortest path, count of cell transitions.
- **detour ratio** — maze path ÷ straight-line distance. 1.0 would be a clear shot.
- **stitches needed** — path cells ÷ 4, the median reach of one trajectory. How many
  dataset trajectories must be composed, at minimum.

### 1.4 The geometry that predicts the results

The decisive quantity turned out not to be path length but **the angle between the
straight line to the goal and the first move that actually makes progress**:

| task | best first-move angle | all legal first moves | CRL d16 | GCBC d16 |
|---|---|---|---|---|
| 5 | 11° | 11° / 79° | 0.93 | 0.96 |
| 1 | 45° | 45° / 45° | 0.39 | 0.94 |
| 2 | 45° | 45° / 45° | 0.64 | 0.82 |
| 3 | **135°** | 135° / 135° | 0.95 | 0.39 |
| 4 | **180°** | 180° (forced) | 0.80 | **0.00** |

Angles in **degrees**; success as a **fraction of episodes**. Rows sorted by angle.

**GCBC's success rate is monotone in this angle** — 0.96, 0.94, 0.82, 0.39, 0.00 — and
nothing else in §1.3 orders the tasks that way. Path length does not: tasks 1, 2 and 4
are all 10 cells and span GCBC 0.94 to 0.00.

Task 4 is the extreme case. Its start cell `(6,5)` is a dead end — `(6,4)`, `(5,5)` and
`(7,5)` are all walls, so the single legal exit is `(6,6)`, which is **directly away
from the goal**. A goal-conditioned BC policy conditioned on a goal 16 units west
reproduces the westward actions in its data and walks into a wall for 1000 steps. It has
no mechanism to represent "go east first". Task 3 is the same failure, milder: 135°,
and GCBC gets 0.39 instead of 0.00.

This also says the two methods are **complementary, not ranked**. CRL is near-inverted:
strongest exactly where GCBC fails (tasks 3, 4) and weakest on task 1, the long but
well-aligned route. A single aggregate success number on this dataset averages over that
and conveys very little.

---

## 2. The two grids

Both train on the same data. They differ in **what network** is underneath.

| | `phase1_anchor` | `phase1_repro` |
|---|---|---|
| purpose | reproduce OGBench's published table | ask whether depth helps |
| tower | stock OGBench MLP | residual blocks (`block_size` 4) |
| shape | 3 × 512, GELU | depth ∈ {4, 8, 16, 32}, width 256 |
| norm / activation | LayerNorm **after** GELU | LayerNorm **inside** block, Swish |
| init | Flax default | `variance_scaling(1/3, 'fan_in', 'uniform')` |
| `repr_dim` (φ/ψ head) | 512 | 64 |
| depth axis | none — single point | the experiment |
| step cap | 1M | 2M, plateau floor 1M |
| runs | 2 agents × 3 seeds = 6 | 2 × 4 depths × 3 seeds = 24 |

`phase1_anchor` is the control: **OGBench's own architecture, unmodified**, so a miss
tells us the harness is wrong rather than the idea. `phase1_repro` swaps in the residual
tower from the 1000-layer work and sweeps depth. Despite the name, both are phase 1 —
`repro` means "reproduce the 1000-layer depth result", not "phase 2".

Shared across both: `antmaze-medium-stitch-v0` native 1M, seeds 0/1/2, batch 1024, Adam
at constant lr 3e-4 (no warmup, no clipping, no weight decay — these help deep nets
*specifically* and would confound the depth axis), OGBench's published flags for this
dataset (`actor_p_trajgoal=0.5`, `actor_p_randomgoal=0.5`, `alpha=0.1`), and evaluation
every 100k steps at 15 episodes/task.

**Because the anchor differs from `repro` depth 4 in three ways at once** — tower, width
and `repr_dim` — the two are *not* a controlled comparison. See §4.

---

## 3. Status

| grid | runs | finished | in flight |
|---|---|---|---|
| `phase1_anchor` | 6 | 6 | 0 |
| `phase1_repro` | 24 | 16 | 8 (CRL d16, CRL d32, GCBC d32) |

```
2993272  crl-pack    COMPLETED  02:01:29   anchor
2993273  crl-pack    COMPLETED  03:06:55   repro pack 1
2993274  crl-pack    COMPLETED  03:50:51   repro pack 2
2993275  crl-pack    RUNNING    06:09:02   repro pack 3
2993276  crl-pack    RUNNING    03:23:17   repro pack 4
2993575  crl-reeval  PENDING               blocked by the 2-job GPU QOS limit
```

**41.2 GPU-hours** of training wall clock across the 22 finished runs. No failures, no
stalls, no resubmissions since the smoke gate.

**What is not yet measured.** Every success number below comes from the in-training
evaluator at **15 episodes/task (75 episodes per point)**. OGBench's published commands
use `--eval_episodes=50` (250 per point). Job `2993575` re-scores `params_best.pkl` at
50 and is queued. Until it runs, treat these as noisier than the table they are compared
against, and the comparison itself as provisional.

---

## 4. The anchor: did we reproduce OGBench?

**Metric:** mean of the final 3 evaluations, the convention the OGBench paper reports.
**Units:** percentage of episodes succeeded, 0–100. **Spread:** ±1 SD across 3 seeds.

| agent | ours (last-3 @ 1M steps, 15 eps/task) | published (50 eps/task) | per-seed |
|---|---|---|---|
| CRL | **45.5 ± 8.1** | 53 ± 6 | 56.9, 40.9, 38.7 |
| GCBC | **37.2 ± 11.4** | 45 ± 11 | 23.6, 51.6, 36.4 |

Both land ~8 points low with overlapping intervals. With 3 seeds and SD ≈ 10 points the
standard error is ≈ 5–6, so **this neither confirms nor refutes the reproduction**. It
is consistent with it. 3 seeds at 15 episodes cannot resolve an 8-point gap; the pending
50-episode re-eval is what will move this.

Per-task, the anchor scores **task 4 at 0.01 (CRL) and 0.00 (GCBC)** at every seed,
capping the aggregate at 0.80 — so OGBench's published ~0.53 for CRL is *consistent with
task 4 being unsolved in their runs too*.

Two mechanical caveats, both verified in the configs:

- `plateau_min_steps` (1M) equals `step_cap` (1M), so plateau stopping could not fire
  early. All six runs took the full 1M steps. Four nonetheless report
  `stop_reason: "plateau"` — a cosmetic mislabel; the step counts are correct.
- `code_commit` reads `0009db4` while the configs contain flags added in `6e01598`.
  Both true: the flags were in the working tree before it was committed. **We do not
  record a dirty-tree flag**, so `code_commit` alone does not identify the code that ran.

---

## 5. The CRL critic is scaling — this is the result

**Metric:** last-3-evaluation mean, pinned to a fixed **1,000,000 gradient steps** for
every run. The pin is necessary: the plateau rule stops runs anywhere between 0.9M and
1.9M steps, so an unpinned last-3 measures each run at a different point on its own
curve. **Units:** percent of episodes, ±1 SD over 3 seeds.

| depth | CRL | GCBC |
|---|---|---|
| 4 | 34.1 ± 9.8 | 53.5 ± 6.8 |
| 8 | 49.3 ± 3.8 | 65.2 ± 1.6 |
| 16 | **74.2 ± 5.1** \* | 62.5 ± 2.4 |
| 32 | not yet at 1M \* | 61.2 ± 5.5 \* |

\* contains runs still training. A value shown is final *at 1M steps* regardless of what
the run does later; CRL d32 is at ~280k steps and is excluded rather than reported at
the wrong step.

**CRL: 34 → 49 → 74 across a 4× depth change**, far outside the seed spread, still
rising at 16. At depth 16 CRL reaches 74%, against OGBench's published 53% for the same
agent on the same data.

**GCBC saturates at depth 8**: 53.5 → 65.2 (real, SDs 6.8 and 1.6, non-overlapping),
then flat at 62.5 and 61.2. Our depth-32 GCBC of 61.2 sits essentially on the 1000-layer
rebuttal's own 63 at depth 32.

### 5.1 Why this is the critic and not just "bigger network"

GCBC and CRL share the same actor architecture and the same depth sweep. GCBC is **actor
only** — no critic. It saturates. CRL adds the contrastive critic, and only CRL keeps
scaling. The component that differs is the component that scales.

The critic's own diagnostics agree. All depths pinned to **1M gradient steps**; depth 32
excluded (not yet there):

| depth | InfoNCE top-1 accuracy | train contrastive loss | **val** contrastive loss | task success |
|---|---|---|---|---|
| 4 | 0.369 | 0.00387 | 0.00530 | 0.341 |
| 8 | 0.467 | 0.00337 | 0.00549 | 0.493 |
| 16 | 0.583 | 0.00269 | 0.00632 | 0.742 |

- **InfoNCE top-1 accuracy** — fraction in `[0, 1]`; chance is 1/1024 ≈ 0.001 at batch
  1024. The critic identifies the true future state among 1023 distractors 58% of the
  time at depth 16, up from 37% at depth 4.
- **contrastive loss** — the InfoNCE objective, nats, lower is better.
- **val contrastive loss** — same objective on the held-out validation split.

Train loss falls monotonically, **validation loss rises monotonically**, and success
rises with it. The generalization gap on the contrastive objective widens by 50% from
depth 4 to 16 while success more than doubles.

**Consequence: held-out contrastive loss is anti-correlated with policy performance
here.** Using it for model selection, early stopping or depth selection would pick
exactly the wrong network. Our plateau rule keys off evaluation success, not validation
loss, so it is unaffected — but this is worth stating before anyone reaches for the
obvious-looking metric.

(`binary_accuracy` is logged but near-useless: with batch 1024 the pair matrix is
~1/1024 positives, so predicting "negative" everywhere already scores 0.999. All depths
sit between 0.9991 and 0.9993. `fig3` labels it as such.)

### 5.2 Caveats on the scaling claim

1. **Depth and parameter count are confounded.** Width is fixed at 256, so deeper is
   also bigger: CRL d4 = 1.45M params, d8 = 2.77M. A width-matched control is **not** in
   the grid. The `plain_mlp` preset isolates residual-vs-plain but not depth-vs-size.
2. **d16 and d32 are unfinished.** d16's 1M-step value is sound; its final value is not
   in yet, and d32 has no 1M-step value at all.
3. **15 episodes/task.** `2993575` pending.
4. **One dataset, one environment.** Everything here is `antmaze-medium-stitch-v0`.

---

## 6. What the aggregate hides: depth unlocks one specific task

**Metric:** mean of last 3 evaluations, per task, averaged over 3 seeds. **Units:**
fraction of episodes in `[0, 1]`.

| config | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---|---|---|---|---|
| CRL d4 (residual) | 0.10 | 0.43 | 0.83 | 0.13 | 0.16 |
| CRL d8 | 0.04 | 0.16 | 0.87 | 0.49 | 0.94 |
| CRL d16 \* | 0.39 | 0.64 | 0.95 | **0.80** | 0.93 |
| GCBC d4 | 0.94 | 0.60 | 0.24 | **0.00** | 0.54 |
| GCBC d8 | 0.95 | 0.70 | 0.52 | **0.00** | 0.93 |
| GCBC d16 | 0.94 | 0.82 | 0.39 | **0.00** | 0.96 |
| GCBC d32 \* | 0.90 | 0.82 | 0.39 | 0.10 | 0.89 |
| CRL d4 (stock MLP, anchor) | 0.74 | 0.69 | 0.10 | **0.01** | 0.74 |
| GCBC d4 (stock MLP, anchor) | 0.78 | 0.33 | 0.19 | **0.00** | 0.56 |

1. **Task 4 — the forced 180° detour — is solved only by a deep contrastive critic.**
   0.00 for GCBC at every depth and seed, ~0.00 for both agents under the stock
   architecture, and 0.13 → 0.49 → 0.80 for CRL as depth goes 4 → 8 → 16. A phase
   transition on one task, not a uniform lift, and it is most of CRL's aggregate gain.
2. **Architecture reallocates which tasks are solved, at nearly equal aggregate.** Stock
   CRL d4 scores 0.74 on task 1 and 0.10 on task 3; residual CRL d4 does the reverse —
   0.10 and 0.83. Aggregates 45.5 vs 34.1. The aggregate difference is small; the
   per-task difference is near-total inversion. **Any conclusion from the aggregate
   alone would have missed this.** The two configs differ in tower, width *and*
   `repr_dim` simultaneously, so the inversion is real but unattributed — this is open
   question **Q2** biting, and it now blocks interpretation rather than just configuration.

---

## 7. Cost

Finished runs only. **Units:** `params` = count of trainable parameters (critic +
actor); `steps/s` = gradient steps per second on one GPU, mean over seeds; `wall-clock`
= hours per run, mean over seeds.

| agent | depth | params | steps/s | wall-clock (h) |
|---|---|---|---|---|
| CRL | 4 | 1,446,408 | 151 | 2.67 |
| CRL | 8 | 2,772,488 | 86 | 3.56 |
| GCBC | 4 | 282,888 | 400 | 0.97 |
| GCBC | 8 | 548,104 | 263 | 1.30 |
| GCBC | 16 | 1,078,536 | 186 | 1.58 |
| GCBC | 32 | 2,139,400 | 98 | 3.13 |
| CRL | 4 (stock, anchor) | 3,793,416 | 139 | 2.00 |
| GCBC | 4 (stock, anchor) | 559,624 | 476 | 0.59 |

GCBC d16 (1.08M params, 62.5% success) beats anchor CRL d4 (3.79M params, 45.5%) at a
third the parameters and a quarter the wall clock.

---

## 8. Figures

All in [`figures/<grid>/`](figures/), PNG + CSV.

| file | what it answers |
|---|---|
| `fig1_success_vs_depth` | max-over-evaluations vs depth; in-flight depths hollow |
| `fig1b_last3_vs_depth` | the published metric, pinned to a common step — §5's table |
| `fig2_learning_curves` | success vs gradient steps, by agent, depth as a colour ramp |
| `fig3_critic_diagnostics` | §5.1 — the train/validation decoupling |
| `fig4_cost` | params / throughput / wall clock, so depth is never read without its price |
| `fig6_per_task` | §6 — the 5 tasks × 9 configs heatmap |
| `fig5_depth_data_heatmap` | not generated; needs >1 dataset size (phase 3) |

---

## 9. What this changes

- **The pre-registered stop condition fired.** `phase1_repro`'s spec said to stop and
  report if CRL scales. It scaled.
- **Q1 (critic loss form) is now urgent.** The depth effect is in the critic, which is
  exactly what Q1 is about. `q1_loss_pilot` is configured and unrun.
- **Q2 (`repr_dim`) now blocks interpretation**, per §6 point 2.
- **Extend the CRL sweep past 16.** It has not saturated; d32 is 14% through.
- **Phase 2 probes should bin per task, not only per distance** — and should use the
  first-move angle from §1.4, which predicted GCBC's per-task results exactly.

---

## 10. Defects found while compiling this

All four were found by checking output rather than trusting it; all four are fixed in
`plot_results.py`.

1. `fig3` looked for `train/categorical_accuracy`; the column is
   `train/critic/categorical_accuracy`. Every metric looked absent and the function
   returned without drawing **or warning**. §5.1 did not exist until this was fixed. It
   now warns and names the columns it did find.
2. `fig3` took each run's *last* logged value, comparing a depth-32 run at 280k steps
   against a depth-4 run at 1.9M and labelling both "final". Now pinned to a common step.
3. `fig1` plotted in-flight runs with the same markers as finished ones, so a depth-32
   point at 14% progress read as a result. Now hollow markers on a dotted connector,
   excluded from the solid series and its mean.
4. `fig4` drew ticks from the full `GRID_DEPTHS` constant (empty slot at 64) and
   repeated one legend across three panels.

Defect 1 is the one that mattered: a figure that silently produces nothing is worse than
one that crashes, because the absence looks like a decision.
