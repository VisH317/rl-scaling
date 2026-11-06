# interp01 — critic interpretability results

<!-- AUTO:meta -->
**Suite** `interp01` · **generated** 2026-11-05 19:45 · **3 checkpoints**

| | |
|---|---|
| dataset | `antmaze-medium-stitch-v0` |
| grids | phase1_repro |
| states | 16,071 from `antmaze-medium-stitch-v0-val.npz` (val, held out) |
| binning | 4x per cell = 1.00 world units; 285/416 free bins cleared the state floor |
| goals | 8 draws per task (`add_noise_to_goal=True`, so the goal is a distribution) |
| actions | `policy`, `oracle_geodesic`, `oracle_euclid` |
| oracle | `ant` / `params_400000.pkl` — the policy that generated this dataset |
| compute | 0.7 min total across 3 checkpoints |
<!-- /AUTO:meta -->

> **Status:** draft, and **being superseded as of 2026-11-06**. Everything below was
> measured on a **3-depth** ladder (4/8/16), because depth 32 had not finished training
> when the suite ran. It has now finished, and job `2995119` re-runs all four depths
> behind the rescore that picks their seeds. Two things change when it lands:
>
> - **A fourth point at depth 32**, which matters most for §3.2: ρ falls monotonically
>   over 4 → 8 → 16, and whether that continues is the whole question.
> - **`fig3_advantage_field` appears.** It was absent, not broken: it plots
>   Q(policy) − Q(dataset), and the suite requested only `policy`, `oracle_geodesic` and
>   `oracle_euclid` as action sources, so the figure had no data and `figures.py`
>   correctly skipped it. `dataset` has been added to `action_sources`.
>
> The caveats in §6 stand regardless — one seed per depth above all.

> **The finding that most wants the fourth point.** Across 4 → 8 → 16 the critic's value
> field gets **less** aligned with true geodesic distance (ρ 0.632 → 0.532 → 0.462) while
> task success rises 0.43 → 0.57 → 0.85. Being a better distance function is evidently
> not how depth helps this critic. Meanwhile `oracle_gap` collapses
> (0.443 → 0.174 → 0.036): the deeper the critic, the less it prefers its own actor's
> action over the data-generating expert's. Those two together are the interesting
> result, and both are single-seed.

---

## 1. What was measured

Depth helps the offline CRL critic a lot on `antmaze-medium-stitch`: 34.3 → 54.7 → 78.4
(reeval@50, depths 4/8/16). The policy is extracted by DDPG+BC, so **every bit of that
signal arrives through `Q(s, a, g)`**. These evals ask what about `Q` gets better.

Full method: [`critic_heatmap/README.md`](../../src/offline/evals/critic_heatmap/README.md). In one line each:

- **Value landscape** — does `Q` rank maze positions by *geodesic* distance to the goal
  rather than straight-line distance? (`topology_gain`)
- **Action preference** — in the bins where "go around the wall" and "go straight at the
  goal" are different actions, which does the critic score higher?
  (`action_topology_gain`)
- **Extraction** — does the actor's own action beat the maze-correct one under its own
  critic? (`oracle_gap`)

The maze-correct and straight-line actions come from **OGBench's own
direction-conditioned SAC expert — the policy that generated every action in this
dataset** — so they are real actions, not a hand-built controller.

## 2. Checkpoints

One per (agent, depth), chosen by the rule in the suite file. The whole seed set is
shown because the between-seed spread is comparable to some depth gaps.

<!-- AUTO:checkpoints -->
| depth | seed | score | rule | all seeds | spread | run |
|---|---|---|---|---|---|---|
| 4 | 1 | 0.444 | `reeval@50` | s0=0.376, s1=0.444, s2=0.208 | 0.236 | `phase1_repro__crl__d4__n1000000__s1__acdde8ae` |
| 8 | 2 | 0.580 | `reeval@50` | s0=0.544, s1=0.516, s2=0.580 | 0.064 | `phase1_repro__crl__d8__n1000000__s2__43511f71` |
| 16 | 2 | 0.904 | `reeval@50` | s0=0.728, s1=0.720, s2=0.904 | 0.184 | `phase1_repro__crl__d16__n1000000__s2__c674fc85` |
<!-- /AUTO:checkpoints -->

---

## 3. `critic_heatmap`

### 3.1 The value field

![critic value field](fig1_critic_field.png)

`Q(s, π(s,g), g)` per spatial bin, drawn on the environment's own top-down render.
Walls are the white blocks; hatched bins fell below the state floor and enter no metric.
Each panel is normalised on its own 2nd–98th percentile — `Q` is an unnormalised
log-ratio, so **only the shape is comparable across rows**, never the magnitude.

> **Read:** _TODO — what changes down the column?_

### 3.2 Does the field follow the maze?

![topology](fig2_topology.png)

<!-- AUTO:topology -->
| depth | ρ(Q, −geodesic) | ρ(Q, −euclid) | topology_gain | bins | coverage |
|---|---|---|---|---|---|
| 4 | 0.632 | 0.662 | **-0.030** | 285 | 0.69 |
| 8 | 0.531 | 0.527 | **+0.004** | 285 | 0.69 |
| 16 | 0.462 | 0.481 | **-0.019** | 285 | 0.69 |

Units: ρ is dimensionless in [−1, 1]; `topology_gain` is their difference. `coverage` is a control and must be identical across rows.
<!-- /AUTO:topology -->

`topology_gain = ρ(Q, −geodesic) − ρ(Q, −euclid)`. Straight-line distance is nearly free
from the observation (`(x,y)` is dims 0:2 of both `s` and `g`); geodesic distance needs
the walls. The difference is the part of the ranking that cannot be explained by
subtracting coordinates. Positive = the field bends around walls.

> **Read:** _TODO._

### 3.3 Where the field peaks, and whether it climbs

![metrics](fig4_metrics.png)

<!-- AUTO:field -->
| depth | peak_geodesic | top5_geodesic | path_monotone | goal_percentile | q_std | ens_gap |
|---|---|---|---|---|---|---|
| 4 | 4.60 | 4.66 | 0.645 | 0.976 | 5.076 | 3.770 |
| 8 | 4.40 | 4.39 | 0.614 | 0.976 | 4.646 | 4.388 |
| 16 | 4.60 | 3.99 | 0.674 | 0.985 | 4.064 | 4.290 |

Units: the two `*_geodesic` columns are **world units** (0 is perfect); `path_monotone` and `goal_percentile` are fractions; `q_std` and `ens_gap` are **logits** and are not comparable across rows — read `ens_gap` relative to `q_std` in the same row.
<!-- /AUTO:field -->

`path_monotone` is the property DDPG+BC actually needs: the actor does one-step greedy
improvement, so a field can rank well globally and still trap it in a local basin.

> **Read:** _TODO._

### 3.4 Which action does the critic prefer?

![action preference](fig5_action_preference.png)

`Q(maze-correct action) − Q(straight-at-goal action)`, on the bins where those two
directions differ by at least the split angle. Bins where they agree are drawn **pale**
(the question does not apply); bins with no data stay **hatched** (we do not know).

![action tests](fig6_action_tests.png)

<!-- AUTO:action -->
| depth | action_topology_gain | (all bins) | oracle_gap | (split bins) | split bin fraction |
|---|---|---|---|---|---|
| 4 | **-0.073** | -0.016 | +0.443 | +0.625 | 0.40 |
| 8 | **+0.021** | +0.032 | +0.174 | +0.239 | 0.40 |
| 16 | **+0.007** | +0.014 | +0.036 | +0.104 | 0.40 |

Units: **logits**, so sign and trend only. `action_topology_gain` is restricted to bins where the geodesic and straight-line directions disagree by at least 45°; the `(all bins)` column is the same quantity without that restriction, as a check that the restriction concentrates the signal rather than creating it.
<!-- /AUTO:action -->

`oracle_gap` needs the success rate beside it: DDPG+BC maximises `Q`, so `> 0` is the
default and only says extraction worked. **`< 0` = extraction failure** (the actor is
not at its own critic's argmax). **`> 0` with low success = critic failure** (the actor
faithfully maximises a quantity whose maximiser is wrong).

> **Read:** _TODO._

### 3.5 Per task

Task 4 is the one to look at: its start `(6,5)` is a dead end whose only exit points
**180° away** from the goal, and GCBC scores 0.00 on it.

<!-- AUTO:pertask -->
| metric | depth | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---|---|---|---|---|---|
| action_topology_gain | 4 | +0.173 | -0.419 | +0.095 | -0.162 | -0.049 |
| action_topology_gain | 8 | +0.034 | -0.018 | +0.134 | -0.173 | +0.128 |
| action_topology_gain | 16 | -0.024 | +0.045 | -0.111 | +0.021 | +0.103 |
| topology_gain | 4 | -0.068 | -0.101 | +0.083 | -0.034 | -0.029 |
| topology_gain | 8 | -0.042 | -0.049 | +0.110 | -0.027 | +0.030 |
| topology_gain | 16 | -0.072 | -0.049 | +0.109 | -0.023 | -0.061 |
| path_monotone | 4 | 0.556 | 0.629 | 0.583 | 0.750 | 0.706 |
| path_monotone | 8 | 0.583 | 0.571 | 0.667 | 0.600 | 0.647 |
| path_monotone | 16 | 0.583 | 0.600 | 0.833 | 0.650 | 0.706 |

Task starts and goals: **1** (1, 1)→(6, 6); **2** (6, 1)→(1, 6); **3** (5, 3)→(4, 2); **4** (6, 5)→(6, 1); **5** (2, 6)→(1, 1)
<!-- /AUTO:pertask -->

> **Read:** _TODO._

---

## 4. `state_probes`

**Planned, not run.** Design in [`state_probes/README.md`](../../src/offline/evals/state_probes/README.md): do
`φ(s,a)` and `ψ(g)` *linearly* encode position, maze cell and geodesic distance, above a
raw-observation and an untrained-network baseline? It and `critic_heatmap` can disagree,
which is the interesting case — a representation can carry geodesic distance linearly
while the dot product fails to use it.

---

## 5. Reproduce

<!-- AUTO:reproduce -->
```bash
src/eval.py                     # the plan: which checkpoints, which evals
src/eval.py --submit            # queue the job
src/eval.py --status            # per-task state
src/eval.py --figures           # redraw from finished tasks
src/eval.py --report            # regenerate THIS file (prose is preserved)
```

Suite file: [`evals.yaml`](../../src/offline/evals/evals.yaml). Raw output: `/scratch/gpfs/vt7993/rl-scaling/experiments/analysis/evals/interp01`
(`field.npz`, `metrics.json`, `meta.json`, `maze_render.png` per checkpoint).
<!-- /AUTO:reproduce -->

---

## 6. Caveats

1. **One seed per depth.** The spread between seeds at a given depth is in §2 and is
   comparable to some depth gaps. Until `seed_select: all` is run, a per-depth number is
   partly a statement about that seed.
2. **`Q` scale is arbitrary per run.** `topology_gain` is a difference of rank
   correlations and is safe to compare; `action_topology_gain` and `oracle_gap` are in
   **logits** and are comparable in sign and trend only.
3. **One environment.** `antmaze-medium-stitch-v0` only. The 1000-layer paper's own
   offline result (its Figure 14, GCBC) is on this same dataset, so a positive here is
   adjacent to a known positive, not against it.
4. **Not a causal test.** A critic can rank bins correctly and still extract a bad
   policy. `oracle_gap` is the closest thing here to separating the two.

<!-- AUTO:provenance -->
---

<sub>
Figures are drawn on the environment's own top-down render. The world→pixel map is measured, not derived: 9 known body positions, affine fit RMS **0.784 px**, leave-one-out max **2.408 px** on a 480 px image; floor-plane correction `(d−z)/d = 0.975408` with `torso_z = 0.7378`. 
The oracle actions come from `/scratch/gpfs/vt7993/ogbench/experts/ant` (`params_400000.pkl`, trained on `online-ant-xy-v0`). 
Metrics self-check: `python -m offline.evals.critic_heatmap.impl`. Generated by `src/eval.py --report`.
</sub>
<!-- /AUTO:provenance -->
