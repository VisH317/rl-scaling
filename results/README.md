# Results — index

One folder per experiment or eval suite. Each holds **its figures and the document that
reads them**, so a figure is never separated from the sentence interpreting it.

Compiled 2026-11-06.

```
results/
  antmaze_stitch__baseline_arch/            6 runs   done     the reproduction control
  antmaze_stitch__depth/                   24 runs   done     depth helps              <- main result
  antmaze_stitch__depth_ensemble1/          6 runs   3 DONE   is it the ensemble?      -> no
  antmaze_navigate__depth/                 12 runs   6 DONE   is it the data?          -> d4 is at ceiling
  humanoidmaze_navigate__depth_ensemble1/   8 runs   done     depth hurts              <- the reversal
  interp__critic_depth/                    16 evals  done     what does the critic learn?
```

Folder names are `<environment>__<what varies>`. They are **aliases**, not renames: a
run id is a hash over its config and the grid name is part of that config, so the grid
names (`phase1_repro`, `humanoid_ensemble1`, …) are frozen forever and only the folder
a human reads was renamed. The mapping is `rl_scaling/paths.py::RESULTS_ALIAS`.

Nothing here is typed from terminal output. Every table is read from the `.csv` beside
the `.png`; the eval documents fill themselves between `<!-- AUTO -->` markers.
Regenerate with `python src/scripts/plot_results.py --grid <grid>` or
`python src/eval.py --report`.

---

## How to read the numbers

Three different success metrics appear throughout and they are **not interchangeable**.
Every table below names which one it uses.

| metric | what it scores | episodes | units |
|---|---|---|---|
| **`reeval@50`** | `params_best.pkl`, rescored after training | 50/task × 5 tasks = **250** | % of episodes reaching the goal, 0–100 |
| `best_success` | the single best evaluation during training | 15/task × 5 = **75** | % of episodes, 0–100 |
| `last3@1M` | mean of the final 3 evaluations, pinned to **1,000,000 gradient steps** | 15/task × 5 = **75**, ×3 | % of episodes, 0–100 |

- **`reeval@50` is the headline.** It is the only one measured the way OGBench's
  published table is, so it is the only one comparable to 53% (CRL) / 45% (GCBC).
- `best_success` is a **max over ~20 noisy evaluations**, so it is biased upward.
- `last3@1M` is where a run **settles**. The pin to a common step matters: the plateau
  rule stops runs anywhere between 0.9M and 1.9M steps, so an unpinned last-3 would
  measure every run at a different point on its own curve.
- `reeval@50` and `best_success` both score the **best** checkpoint; `last3@1M` scores
  the **end**. When a run degrades late those diverge, and that gap is itself a result
  (see the humanoid grid).

**± is one standard deviation across seeds**, not a standard error and not a confidence
interval. With n = 2 or 3 it is a spread, not an inference.

Other units used in the eval documents: **ρ** is a Spearman rank correlation,
dimensionless in [−1, 1]. **R²** is fraction of variance explained against the test-set
mean, dimensionless, ≤ 1 and free to go negative. **Q** is an unnormalised
log-density-ratio in **logits** whose scale is arbitrary per run — comparable in sign
and trend, never in level.

---

## The one-paragraph version

**Depth helps a lot on one dataset and hurts on the other, and the critic's own metrics
improve in both.** On `antmaze-medium-stitch` the contrastive critic scales strongly —
`reeval@50` of 35.5 → 86.7 across 4 → 32 layers, against OGBench's published 53% — and
nearly all of that gain is one task, the forced 180° detour, going 0.08 → 0.97. On
`humanoidmaze-medium-navigate` the same sweep peaks at depth **8** and then falls, and
where runs *settle* depth 32 collapses to 31.3. In both environments the critic gets
monotonically **better** at its own training objective as depth rises, while its
held-out contrastive loss gets monotonically worse. Probed directly, the deeper critic
linearly encodes maze geometry *less* well, not more — on antmaze at depth 32 it is
worse than an **untrained** network of the same depth — even as success rises.

So depth is buying something real on antmaze; it is not buying "a better distance
function"; and whatever it buys does not transfer. **The caveat governing all of it:**
the two grids differ in **four** ways, not one, so they do not isolate the environment.
Two of the four now have a dedicated experiment (below).

---

## Headline numbers

**Metric: `reeval@50`** — best checkpoint, 250 episodes per point. Percent of episodes,
± 1 SD over seeds. This is the comparable number.

| depth | CRL antmaze-stitch (n=3) | GCBC antmaze-stitch (n=3) | CRL humanoidmaze-navigate (n=2) |
|---|---|---|---|
| 4 | 35.5 ± 9.4 | 54.3 ± 5.2 | 62.0 ± 1.2 |
| 8 | 53.2 ± 4.3 | **65.5 ± 3.1** | **65.0 ± 1.0** |
| 16 | 77.5 ± 7.6 | 64.0 ± 3.1 | 55.0 ± 5.8 |
| 32 | **86.7 ± 2.9** | 60.5 ± 2.6 | 53.0 ± 7.0 |

*Reference: OGBench publishes 53 ± 6 (CRL) and 45 ± 11 (GCBC) for antmaze-medium-stitch.
There is no verified published number for humanoidmaze-medium-navigate, which is why its
figures carry no reference band.*

**Metric: `last3@1M`** — where runs settle, same layout.

| depth | CRL antmaze-stitch | GCBC antmaze-stitch | CRL humanoidmaze-navigate |
|---|---|---|---|
| 4 | 34.1 ± 9.8 | 53.5 ± 6.8 | **64.2 ± 2.4** |
| 8 | 49.3 ± 3.8 | **65.2 ± 1.6** | 63.3 ± 0.7 |
| 16 | 71.4 ± 1.9 | 62.5 ± 2.4 | 54.4 ± 0.7 |
| 32 | **79.3 ± 5.5** | 61.2 ± 5.5 | 31.3 ± 1.1 |

What the two tables together say:

- **CRL antmaze rises throughout** on both metrics, decelerating on `last3@1M`
  (+15.2, +22.1, +7.9 per doubling) but still gaining +9.2 at the last step on
  `reeval@50`.
- **GCBC antmaze saturates at depth 8** and drifts down after, on both.
- **CRL humanoid peaks at depth 8**, not 4, then falls — and the two metrics disagree
  about *how far*. Depth 32 reaches 53.0 from its best checkpoint but settles at 31.3,
  a **21.7-point gap no other cell in either grid shows** (every other is within ±2.2).
  Its failure is mostly *failing to stay*, not failing to learn.

---

## Every figure

Each row is one PNG plus the CSV it was drawn from. The same six figures exist for all
three training grids.

### Training grids

| figure | what it answers | read it for |
|---|---|---|
| `fig1_success_vs_depth` | `best_success` vs depth, per agent | the optimistic view; a max over ~20 noisy points |
| `fig1b_last3_vs_depth` | `last3@1M` vs depth | where runs settle |
| `fig2_learning_curves` | success vs gradient steps, depth as a colour ramp | *when* a run peaks — the humanoid story is here |
| `fig3_critic_diagnostics` | train vs validation contrastive loss, categorical accuracy | the overfitting depth causes in both environments |
| `fig4_cost` | parameters, throughput, wall clock | so depth is never read without its price |
| `fig6_per_task` | the 5 tasks × configs heatmap | what the aggregate hides — task 4 above all |
| `fig5_depth_data_heatmap` | not generated; needs >1 dataset size (phase 3) | — |

### `interp__critic_depth`

Figures are written **one complete set per environment**, suffixed
`__phase1_repro` / `__humanoid_ensemble1`, because `Q`'s scale is arbitrary per run and
the two environments must not share a colour scale or a panel.

| figure | what it answers |
|---|---|
| `fig1_critic_field` | the value field drawn on the real maze, depth × task |
| `fig2_topology` | does the field follow the maze (geodesic) or ignore walls (straight-line)? |
| `fig3_advantage_field` | Q(policy action) − Q(dataset action), per bin |
| `fig4_metrics` | where the field peaks, and whether it climbs along the path |
| `fig5_action_preference` | which of two real expert actions the critic prefers, per bin |
| `fig6_action_tests` | `action_topology_gain` and `oracle_gap` vs depth |
| `fig1_probe_vs_depth` | linear probe R² vs depth, per target, with both controls |
| `fig2_probe_vs_heatmap` | probe R² against the same checkpoint's field ordering |

---

## What each folder is

### [`antmaze_stitch__depth/`](antmaze_stitch__depth/) — the main result · 24 runs · **done**
*(grid `phase1_repro`)* CRL and GCBC on the residual tower from the 1000-layer paper,
depths 4/8/16/32, 3 seeds, `antmaze-medium-stitch-v0`. The most careful document here:
the environment, the five tasks, the maze geometry that predicts the results, and §2's
accounting of where this setup differs from the paper. Read §2 before quoting §6.

### [`humanoidmaze_navigate__depth_ensemble1/`](humanoidmaze_navigate__depth_ensemble1/) — the reversal · 8 runs · **done**
*(grid `humanoid_ensemble1`)* The same sweep on `humanoidmaze-medium-navigate-v0` with a
single critic encoder pair. Depth stops paying. Documents the four-way confound that
stops this being a clean environment comparison, and the full critic analysis.

### [`antmaze_stitch__baseline_arch/`](antmaze_stitch__baseline_arch/) — the control · 6 runs · **done**
*(grid `phase1_anchor`)* Stock OGBench architecture, unmodified, depth 4. Exists so that
a miss means the harness is wrong rather than the idea.

### [`interp__critic_depth/`](interp__critic_depth/) — critic interpretability · 16 evals · **done**
*(suite `interp01`)* `critic_heatmap` and `state_probes`, each on the best seed at every
depth, on **both** environments. Asks what about `Q(s,a,g)` actually improves with
depth. The answer is not "it becomes a better distance function".

### [`antmaze_stitch__depth_ensemble1/`](antmaze_stitch__depth_ensemble1/) — the ensemble is cleared · 3 of 6 runs · **partial**
*(grid `antmaze_ensemble1`)* `phase1_repro` with `critic_ensemble: 2 → 1` and nothing
else, verified by a resolved-config diff. The actor maximises `min` over the ensemble,
so at size 1 there is no pessimism at all. **Depth 4 is final; depth 32 is 38% trained**
(job `2995193`, ~10 h left) and every d32 number in that folder is marked a lower bound.

Provisional answer: **the ensemble is not what made depth pay.** At a matched 735k
steps the depth gain is +40.0 points without pessimism against +47.1 with it, so what
the ensemble buys is a ~13-point level shift at d32 and nothing at all at d4 — where
all three metrics now agree it is a no-op: `reeval@50` 38.9 ± 11.3 against
`phase1_repro`'s 35.5 ± 9.4, `best_success` 43.6 vs 42.7, `last3@1M` 28.0 vs 34.1.
Three deltas inside the seed spread, not even consistent in sign. The late
peak-then-drift that looked like a no-pessimism symptom turns out to be present in
`phase1_repro` too, slightly harder (−11.1 vs −7.1 points).

### [`antmaze_navigate__depth/`](antmaze_navigate__depth/) — d4/d8 done, and they are at ceiling · 6 of 12 runs · **partial**
*(grid `antmaze_navigate1`)* `phase1_repro` with the dataset changed to
`antmaze-medium-navigate-v0` and nothing else. Tests whether the reversal is about the
**data** rather than the body: navigate has 1000 trajectories × 1001 steps against
stitch's 5000 × 201, which is humanoid's structure with the ant's 29-d observation.

**Pack 1 finished and it changes how this grid reads.** `antmaze-medium-navigate` is
*solved at depth 4*:

| metric | d4 (n=3) | d8 (n=3) | same metric, stitch d4 |
|---|---|---|---|
| `best_success` | 99.6 ± 0.6 | 98.7 ± 1.1 | 42.7 ± 11.4 |
| `last3@1M` | 97.2 ± 1.7 | 96.9 ± 0.0 | 34.1 ± 9.8 |

All six runs stopped on the plateau rule at 1.0–1.4M steps. With d4 at 99.6 there is no
headroom left for depth to *help*, so the original reading (“depth still scales → the
data is not the cause”) is no longer available on this grid. What remains is a
**one-sided test for harm**, and that is still worth the d16/d32 runs: if deep networks
fall below the ceiling on ant + navigate data, the humanoid reversal is about the data
structure and the body is incidental. If they stay at ~99, the grid is uninformative
and the dataset confound needs a harder navigate variant (`large`) instead.

Pack 2 (d16, d32) is `2995263`, queued behind `2995193`.

---

## The four-way confound, and what covers it

`phase1_repro` → `humanoid_ensemble1` moved four things at once:

| difference | | | covered by |
|---|---|---|---|
| dataset | stitch | navigate | `antmaze_navigate__depth` (half done — d4/d8 in) |
| `critic_ensemble` | 2 | 1 | `antmaze_stitch__depth_ensemble1` — **closed, not the cause** |
| `actor_p_trajgoal / randomgoal` | 0.5 / 0.5 | 1.0 / 0.0 | **nothing yet** |
| `discount` | 0.99 | 0.995 | **nothing yet** |

The third was found late and may matter most: `p_randomgoal = 0` means the actor is
**never trained on out-of-trajectory goals**, which is exactly the stitching capability
— and task 4, the task that needs it, never works on humanoid at any depth. Both
uncovered differences arrive through `use_published_flags`, which applies OGBench's
published command for whichever dataset is named.

---

## Not here yet

| what | why it matters | cost |
|---|---|---|
| **`actor_p_randomgoal`** — see backlog below | the last uncovered confound | 4 runs |
| early-stop humanoid d32 at its 200k peak | tests whether the collapse is just training too long; checkpoints already exist | free |
| `interp01` on `antmaze_ensemble1` / `antmaze_navigate1` | the probes on the disambiguation grids | 8 evals |
| `state_probes` at `block_k` | per-block representational change; needs activation capture | — |
| `q1_loss_pilot`, `scaling_crl_replication` | **deprioritised** — the paper's contribution is the architecture, and its energy function is believed to be a dot product, so the loss-form difference may not be a real deviation | — |

### Backlog — `actor_p_randomgoal` on humanoidmaze

**Deliberately deferred, not forgotten.** The fourth confound (§ above) is the only one
with no experiment: `humanoid_ensemble1` ran with `actor_p_trajgoal/randomgoal =
1.0/0.0` because that is what OGBench publishes for `humanoidmaze-medium-navigate`,
against `0.5/0.5` on antmaze-stitch. At `p_randomgoal = 0` the actor is **never trained
on an out-of-trajectory goal**, which is the stitching capability itself — and task 4,
the one that needs it, never works on humanoid at any depth.

When it is wanted, it is a copy of `humanoid_ensemble1.yaml` with depths `[4, 32]`,
2 seeds, and these two lines added:

```yaml
actor_p_trajgoal: 0.5       # pinned to antmaze-stitch's values; without these,
actor_p_randomgoal: 0.5     # use_published_flags sets 1.0/0.0 for this dataset
```

4 runs. Read it as: **task 4 becomes non-zero** → the humanoid failure was partly that
stitching was never trained for, and the body is less implicated than it looks;
**task 4 stays at zero** → the capability is absent for a deeper reason and the
goal-sampling difference is a red herring.

Research context and the verified-facts log live in
[`../src/offline/`](../src/offline/): `offline_exp.md` (the brief), `algorithm.md` (what
actually runs), `research_log.md` (verified facts), `impl_log.md` (what was built).
