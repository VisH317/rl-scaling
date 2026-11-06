<!--
  Results template for an eval suite.

  `src/eval.py --report` copies this to `results_<suite>.md` beside it (once, if it
  does not exist), copies the figures in beside it (`results/<suite>/`), and then fills every
  `<!-- AUTO:name -->` ... `<!-- /AUTO:name -->` block with the current numbers.

  Everything OUTSIDE those blocks is yours and survives every regeneration, so write
  the interpretation in the `> **Read:**` stubs and re-run `--report` freely as runs
  finish. Numbers never need to be copied by hand; conclusions never get overwritten.
-->

# {{SUITE}} — critic interpretability results

<!-- AUTO:meta -->
<!-- /AUTO:meta -->

> **Status:** draft. Nothing here is a headline result until the caveats in §6 are
> addressed — one seed per depth above all.

---

## 1. What was measured

Depth helps the offline CRL critic a lot on `antmaze-medium-stitch`: 34.3 → 54.7 → 78.4
(reeval@50, depths 4/8/16). The policy is extracted by DDPG+BC, so **every bit of that
signal arrives through `Q(s, a, g)`**. These evals ask what about `Q` gets better.

Full method: [`critic_heatmap/README.md`](critic_heatmap/README.md). In one line each:

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
<!-- /AUTO:checkpoints -->

---

## 3. `critic_heatmap`

### 3.1 The value field

<!-- AUTO:img_fig1_critic_field -->
<!-- /AUTO:img_fig1_critic_field -->

`Q(s, π(s,g), g)` per spatial bin, drawn on the environment's own top-down render.
Walls are the white blocks; hatched bins fell below the state floor and enter no metric.
Each panel is normalised on its own 2nd–98th percentile — `Q` is an unnormalised
log-ratio, so **only the shape is comparable across rows**, never the magnitude.

> **Read:** _TODO — what changes down the column?_

### 3.2 Does the field follow the maze?

<!-- AUTO:img_fig2_topology -->
<!-- /AUTO:img_fig2_topology -->

<!-- AUTO:topology -->
<!-- /AUTO:topology -->

`topology_gain = ρ(Q, −geodesic) − ρ(Q, −euclid)`. Straight-line distance is nearly free
from the observation (`(x,y)` is dims 0:2 of both `s` and `g`); geodesic distance needs
the walls. The difference is the part of the ranking that cannot be explained by
subtracting coordinates. Positive = the field bends around walls.

> **Read:** _TODO._

### 3.3 Where the field peaks, and whether it climbs

<!-- AUTO:img_fig4_metrics -->
<!-- /AUTO:img_fig4_metrics -->

<!-- AUTO:field -->
<!-- /AUTO:field -->

`path_monotone` is the property DDPG+BC actually needs: the actor does one-step greedy
improvement, so a field can rank well globally and still trap it in a local basin.

> **Read:** _TODO._

### 3.3b Where the policy's action beats the logged one

`Q(policy action) − Q(dataset action)`, per bin. Needs `dataset` in `action_sources`;
without it the figure has no data and `figures.py` skips it rather than drawing an
empty panel.

<!-- AUTO:img_fig3_advantage_field -->
<!-- /AUTO:img_fig3_advantage_field -->

> **Read:** _TODO._

### 3.4 Which action does the critic prefer?

<!-- AUTO:img_fig5_action_preference -->
<!-- /AUTO:img_fig5_action_preference -->

`Q(maze-correct action) − Q(straight-at-goal action)`, on the bins where those two
directions differ by at least the split angle. Bins where they agree are drawn **pale**
(the question does not apply); bins with no data stay **hatched** (we do not know).

<!-- AUTO:img_fig6_action_tests -->
<!-- /AUTO:img_fig6_action_tests -->

<!-- AUTO:action -->
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
<!-- /AUTO:pertask -->

> **Read:** _TODO._

---

## 4. `state_probes`

Do `φ(s,a)` and `ψ(g)` *linearly* encode position, maze cell and geodesic distance,
above a raw-input and an untrained-network baseline? This and `critic_heatmap` can
disagree, which is the interesting case — a representation can carry geodesic distance
linearly while the dot product fails to use it.

Design: [`state_probes/README.md`](../../src/offline/evals/state_probes/README.md).
Which target belongs to which site, and why `raw_obs` differs per site, is in
`impl.py`'s docstring: `φ` and `ψ` never see each other, so a pair property is probed
from their concatenation and a goal property is computed on the goal observation.

<!-- AUTO:img_fig1_probe_vs_depth -->
<!-- /AUTO:img_fig1_probe_vs_depth -->

<!-- AUTO:probes -->
<!-- /AUTO:probes -->

> **Read:** _TODO._

### 4.1 Does the representation explain the value field?

`geodesic_to_task` probe R² against the same checkpoint's `rho_geodesic`. Moving
together would give the heatmap result a representational mechanism; moving apart says
the information is present and the dot product does not use it.

<!-- AUTO:img_fig2_probe_vs_heatmap -->
<!-- /AUTO:img_fig2_probe_vs_heatmap -->

> **Read:** _TODO._

---

## 5. Reproduce

<!-- AUTO:reproduce -->
<!-- /AUTO:reproduce -->

---

## 6. Caveats

1. **One seed per depth.** The spread between seeds at a given depth is in §2 and is
   comparable to some depth gaps. Until `seed_select: all` is run, a per-depth number is
   partly a statement about that seed.
2. **`Q` scale is arbitrary per run.** `topology_gain` is a difference of rank
   correlations and is safe to compare; `action_topology_gain` and `oracle_gap` are in
   **logits** and are comparable in sign and trend only.
3. **Two environments, not a controlled contrast.** `antmaze-medium-stitch-v0` and
   `humanoidmaze-medium-navigate-v0`. The grids behind them differ in dataset,
   `critic_ensemble` **and** `discount`, so a difference between them is not
   attributable to the environment. The 1000-layer paper's own offline result (its
   Figure 14, GCBC) is on antmaze-medium-stitch, so a positive there is adjacent to a
   known positive, not against it.
5. **Figures are one set per grid.** `Q` is an unnormalised log-density ratio whose
   scale is arbitrary per run, so no panel pools the two environments and no colour
   scale is shared across them.
4. **Not a causal test.** A critic can rank bins correctly and still extract a bad
   policy. `oracle_gap` is the closest thing here to separating the two.

<!-- AUTO:provenance -->
<!-- /AUTO:provenance -->
