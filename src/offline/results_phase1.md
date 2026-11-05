# Phase 1 results — offline GCRL depth scaling on `antmaze-medium-stitch-v0`

Compiled 2026-11-05. Covers grids `phase1_anchor` (finished) and `phase1_repro`
(16 of 24 runs finished, 8 still training).

Figures are in [`figures/`](figures/), copied from
`/scratch/gpfs/vt7993/rl-scaling/experiments/analysis/`. Every PNG has a CSV beside it,
and those CSVs are the source for the tables here — nothing below is typed by hand from
terminal output.

> **Headline.** `phase1_repro`'s own grid spec in `config.py` says: *"Expect CRL flat
> with depth, GCBC rising. If CRL scales here, stop and report — that alone changes the
> project."* **CRL scaled: 34.1 → 49.3 → 74.2 across depths 4 → 8 → 16.**
>
> The 1000-layer paper's own §4.6 reports the opposite for offline CRL on OGBench:
> *"We found little evidence that increasing the network depth of CRL improves
> performance in this offline setting."* **But we are not running their critic loss** —
> ours is OGBench's binary-NCE with dot-product logits, theirs is InfoNCE with negative
> L2 and a logsumexp penalty. Read §2 before §6.

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

## 2. What the 1000-layer paper actually does — and where we differ

Verified 2026-11-05 against arXiv:2503.14858v5 (Wang, Javali, Bortkiewicz, Trzciński,
Eysenbach; NeurIPS 2025) and the project page. Quotes are verbatim from the HTML full
text unless marked.

### 2.1 Yes, there is policy extraction — in both their setup and ours

Their §3: *"The policy πθ(a|s,g) is trained to maximize the critic"*, i.e. maximize
`E[f_φ,ψ(s,a,g)]` where the critic is `f = −‖φ(s,a) − ψ(g)‖₂`. A separate actor network,
trained against the contrastive critic.

**Ours does the same thing.** Our CRL runs use OGBench's `actor_loss='ddpgbc'`:

```python
q_loss   = -q.mean() / stop_gradient(|q|.mean() + 1e-6)   # maximize the critic
bc_loss  = -(alpha * log_prob).mean()                     # alpha = 0.1
actor_loss = q_loss + bc_loss
```

and evaluation rolls out that extracted actor in the environment. **Every success number
in this document is an extracted policy's**, not a critic-only metric. The difference
from theirs is the `bc_loss` term — offline, OGBench regularizes the actor toward the
data, which the online paper does not need.

### 2.2 What the eval metric actually touches: the actor only

Worth stating plainly, because it constrains what §6.1 can claim. OGBench's
`evaluate()` does exactly one thing with the agent:

```python
actor_fn = supply_rng(agent.sample_actions, ...)
...
action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
```

and `CRLAgent.sample_actions` selects **only** the `'actor'` network. **The critic is
never queried at evaluation time.** Neither is the 2-member critic ensemble, whose `min`
appears only inside the actor's training loss. The critic's entire influence on every
success number in this document is mediated through the actor's weights, via the
`q_loss` term.

With `eval_temperature=0.0` and `const_std=True`, `scale_diag = exp(0) * 0 = 0`, so the
rollout is a **deterministic** actor: sampling returns the mean action.

Which parameters each of our three scoring conventions uses:

| convention | parameters evaluated |
|---|---|
| `best_success` | the **live** training parameters at the single best-scoring eval step |
| `last3_eval_mean` | the live parameters at each of the last three eval points — **three different networks**, 100k steps apart, averaged. A noise-reduction device, not one policy's score |
| `reeval@50` | `params_best.pkl`, one saved checkpoint. The pickle holds the whole agent, actor and both critics, but only the actor is exercised |

So "the critic is scaling" is necessarily an inference about the *training signal*. The
only direct measurements of the critic we have are §6.1's diagnostics — InfoNCE top-1
accuracy and the train/validation contrastive losses.

### 2.3 Their evaluation metric is not ours

Their §4.1: *"For evaluation, we measure the number of time steps (out of 1000) that the
agent is near the goal"*, reported as *"the average score over the last five epochs of
training."* That is **dwell time**, not binary success.

We report OGBench's **binary episode success**. So our numbers are not directly
comparable to their headline figures — but they *are* comparable to OGBench's published
table, which is the right comparator for an offline OGBench reproduction. Nothing here
should be read as reproducing their online numbers.

### 2.4 They scale actor and critic jointly — and so do we

Their §4.1: *"the depth refers to the configuration of the actor network and both critic
encoder networks, which are scaled jointly, except for the ablation experiment in
Section 4.4."*

Ours scales jointly too. Measured from our checkpoints:

| depth | CRL critic params | CRL actor params | GCBC total params |
|---|---|---|---|
| 4 | 1,163,520 | 282,888 | 282,888 |
| 16 | 4,346,112 | 1,078,536 | 1,078,536 |

CRL's actor is **parameter-identical** to GCBC's at every depth. That is what licenses
the contrast in §6.1 — but see §6.2 — it is a contrast, not an isolation.

Their §4.4 ("Scaling the Actor vs. Critic Networks", Figure 6) is the isolation, and the
answer is environment-dependent: *"In Arm Push Easy, scaling the critic is more
effective; in Ant Big Maze, the actor matters more. For Humanoid, scaling both is
necessary."* **We have not run this.** It is our `actor_thread` preset, declared in
`conf/experiment/` and not implemented — frozen-critic loading does not exist yet. It is
now the highest-value missing experiment.

### 2.5 Their setting is online; the offline result they report is NEGATIVE

Their §3: *"Our focus is on online RL, where a replay buffer stores the most recent
trajectories."* Main experiments run on JaxGCRL / Brax / MJX locomotion and manipulation,
not OGBench.

On offline, their §4.6, verbatim:

> *"In preliminary experiments, we evaluated depth scaling in the offline
> goal-conditioned setting using OGBench (Park et al., 2024). We found little evidence
> that increasing the network depth of CRL improves performance in this offline
> setting."*

and

> *"In all cases, baseline depth 4 networks often had the highest success. A key
> direction for future work is to see if our method can be adapted to enable scaling in
> the offline setting."*

**We observe the opposite: CRL 34.1 → 49.3 → 74.2 across depths 4 → 8 → 16 (§6).** This
is the project's pre-registered stop-and-report condition, and it contradicts the
paper's own preliminary offline finding. That demands scepticism before it demands
excitement.

For offline **GCBC** they report a positive result on this exact dataset — Figure 14
caption (available only as a summary, not verbatim): *"Our approach successfully scales
depth in offline GCBC on antmaze-medium-stitch (OGBench)."* Our GCBC 53.5 → 65.2 then
flat is broadly consistent with that.

### 2.6 The most likely reason we disagree: we are not running their critic loss

Confirmed from the run config of every CRL run in `phase1_repro`:

| | 1000-layer paper | **what we ran** |
|---|---|---|
| critic loss | InfoNCE | **`binary_nce`** |
| logits | `−‖φ−ψ‖₂` (negative L2) | **`dot`**, `φ·ψ/√d` |
| logsumexp penalty | 0.1 | **0.0** |
| actor | maximize critic | maximize critic **+ BC term**, `alpha=0.1` |
| `repr_dim` | 64 | 64 (matches) |
| setting | online | offline |

Our depth result is for **OGBench's** contrastive loss, not the paper's. So the honest
statement is not "we contradict the paper" but:

> **Offline depth scaling works for CRL on antmaze-medium-stitch with the binary-NCE /
> dot-product critic, which is not the critic the paper scaled.**

That makes open question **Q1** — which critic loss the depth test should use — the
single highest-priority experiment, and it reframes it. It is no longer "pick the right
convention"; it is "is the loss form the reason offline depth scaling works here and not
for them?" Either answer is a result. `q1_loss_pilot` is configured and unrun.

Their offline ablations also included *"scaling critic depth while holding the actor at
4 or 8 layers"* and *"applying cold initialization to the final layers of the critic
encoders"* — the first is our `actor_thread`, the second our `arch=residual_cold`.
Neither has been run.

### 2.7 One reference line in our figures is unverified

`fig1` draws a grey dashed reference at GCBC 0.47 (depth 2) → 0.63 (depth 32), labelled
"1000-layer rebuttal (GCBC)". **Those two numbers do not appear anywhere in the paper
text or project page.** They were recorded here as coming from a rebuttal, which would
be the OpenReview discussion rather than the paper — not checked. The paper's main-text
depths are 4/8/16/32/64 with no depth-2 run. Until someone opens the OpenReview thread,
treat that line as unsourced. It is annotation only and enters no computation.

---

## 3. The two grids

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
and `repr_dim` — the two are *not* a controlled comparison. See §5.

---

## 4. Status

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

## 5. The anchor: did we reproduce OGBench?

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

## 6. The CRL critic is scaling — this is the result

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
then flat at 62.5 and 61.2. (A reference value of 0.63 at depth 32 is recorded in our
figures as coming from the paper's rebuttal; it is **unverified** — see §2.7.)

### 6.1 Is this the critic, or just a bigger network?

Our depth axis moves the **actor and the critic together** (§2.4), so this grid cannot
isolate the two. What it supports is a contrast, not an isolation:

- CRL's actor is **parameter-identical** to GCBC's at every depth (282,888 at d4;
  1,078,536 at d16 — exactly equal, measured from the checkpoints).
- GCBC is that actor **alone**, with no critic. It saturates at depth 8.
- CRL is that same actor **plus** a contrastive critic. It keeps scaling to 16.

So the additional scaling arrives with the critic. That is an inference from the
contrast; the experiment that would settle it is the paper's own §4.4 ablation — hold
actor depth fixed and scale critic depth, and vice versa — which we have **not** run.

The critic's own diagnostics are consistent with it. All depths pinned to **1M gradient steps**; depth 32
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

### 6.2 Caveats on the scaling claim

1. **It contradicts the paper's own offline finding** (§2.5), which is reason to look
   for our error before celebrating. Leading candidate is §2.6: we ran OGBench's
   binary-NCE/dot critic, not the paper's InfoNCE/L2/logsumexp one.
2. **Actor and critic depth are not separated** (§6.1). The paper's §4.4 ablation is the
   missing experiment.
3. **Depth and parameter count are confounded.** Width is fixed at 256, so deeper is
   also bigger: CRL d4 = 1.45M params, d8 = 2.77M. A width-matched control is **not** in
   the grid. The `plain_mlp` preset isolates residual-vs-plain but not depth-vs-size.
4. **d16 and d32 are unfinished.** d16's 1M-step value is sound; its final value is not
   in yet, and d32 has no 1M-step value at all.
5. **15 episodes/task.** `2993575` pending.
6. **One dataset, one environment.** Everything here is `antmaze-medium-stitch-v0`.

---

## 7. What the aggregate hides: depth unlocks one specific task

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

## 8. Cost

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

## 9. Figures

All in [`figures/<grid>/`](figures/), PNG + CSV.

| file | what it answers |
|---|---|
| `fig1_success_vs_depth` | max-over-evaluations vs depth; in-flight depths hollow |
| `fig1b_last3_vs_depth` | the published metric, pinned to a common step — §6's table |
| `fig2_learning_curves` | success vs gradient steps, by agent, depth as a colour ramp |
| `fig3_critic_diagnostics` | §6.1 — the train/validation decoupling |
| `fig4_cost` | params / throughput / wall clock, so depth is never read without its price |
| `fig6_per_task` | §7 — the 5 tasks × 9 configs heatmap |
| `fig5_depth_data_heatmap` | not generated; needs >1 dataset size (phase 3) |

---

## 10. What this changes

- **The pre-registered stop condition fired.** `phase1_repro`'s spec said to stop and
  report if CRL scales. It scaled — and it scaled where the paper reports it does not.
- **Q1 is no longer a convention question, it is the experiment.** We got offline depth
  scaling with OGBench's binary-NCE/dot critic; the paper got none with
  InfoNCE/L2/logsumexp. `q1_loss_pilot` tells us whether the loss form is the reason.
  Either answer is a result.
- **Implement `actor_thread`.** The paper's §4.4 separates actor depth from critic depth
  and finds the answer environment-dependent. Our grid cannot separate them.
- **Run `arch=residual_cold`.** The paper's other offline ablation, also unrun.
- **Q2 (`repr_dim`) now blocks interpretation**, per §7 point 2.
- **Extend the CRL sweep past 16.** It has not saturated; d32 is 14% through.
- **Phase 2 probes should bin per task, not only per distance** — and should use the
  first-move angle from §1.4, which predicted GCBC's per-task results exactly.

---

## 11. Defects found while compiling this

All four were found by checking output rather than trusting it; all four are fixed in
`plot_results.py`.

1. `fig3` looked for `train/categorical_accuracy`; the column is
   `train/critic/categorical_accuracy`. Every metric looked absent and the function
   returned without drawing **or warning**. §6.1 did not exist until this was fixed. It
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
