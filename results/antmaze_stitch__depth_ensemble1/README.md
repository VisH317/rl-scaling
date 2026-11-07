# antmaze_ensemble1 — is the depth result an artefact of the critic ensemble?

Compiled 2026-11-06. **3 of 6 runs finished** — CRL on `antmaze-medium-stitch-v0`,
depth 4 complete at 3 seeds and rescored at 50 episodes/task, depth 32 still training
(≈38%, ~10 h left, job `2995193`).

This document is **partial and says so at every number**. Depth-4 rows are final.
Depth-32 rows are *lower bounds* read off runs that have not stopped. Figures mark
in-flight points with a hollow marker on a dotted line.

Figures are in this folder, PNG + CSV. Every number below is read from those CSVs.

> **Provisional answer: no — the ensemble is not what made depth pay.**
>
> Removing the second critic costs depth 32 real points, but it does not come close to
> removing the depth gain. At the **same gradient step** (735k, the slowest in-flight
> seed's current position):
>
> | | depth 4 | depth 32 | gain |
> |---|---|---|---|
> | `critic_ensemble: 2` (`phase1_repro`) | 39.1 ± 10.9 | 86.2 ± 2.5 | **+47.1** |
> | `critic_ensemble: 1` (this grid) | 32.9 ± 12.0 | 72.9 ± 5.1 | **+40.0** |
>
> *(`best_success` so far, 75 episodes/point, % of episodes. Matched-step because the
> depth-32 runs here are only 38% trained — comparing their current best against
> `phase1_repro`'s final best would understate them.)*
>
> **+40 points of depth gain survives with no pessimism at all.** `phase1_repro`'s
> headline therefore stands: depth, not the ensemble, is doing the work. What the
> ensemble buys is a **level shift at depth** — about 13 points at d32 — and nothing
> measurable at d4, where the two grids agree inside a fifth of a standard deviation
> once both are finished (42.7 ± 11.4 vs 43.6 ± 10.1).
>
> **Treat the d32 number as provisional but not as mid-climb.** The curve already
> peaked at 300–400k and has been flat-to-declining for 400k steps (`fig2`), so 72.9 is
> unlikely to move far. It is still a lower bound and is drawn as one.

---

## 1. What is different from `phase1_repro`

**One knob.** This was verified by diffing the two grids' *resolved* `config.json`, not
by reading the YAML — the lesson from `humanoid_ensemble1`, which documented three
differences and actually had four.

| | `phase1_repro` | `antmaze_ensemble1` |
|---|---|---|
| `critic_ensemble` | 2 | **1** |
| dataset | `antmaze-medium-stitch-v0` | same |
| `discount` | 0.99 | same |
| `actor_p_trajgoal / randomgoal` | 0.5 / 0.5 | same |
| `value_p_trajgoal / randomgoal` | 1.0 / 0.0 | same |
| critic loss / logits | `binary_nce` / dot | same |
| tower / width / `repr_dim` | residual / 256 / 64 | same |
| *(scope, not a knob)* | | |
| agents | CRL + GCBC | CRL only — GCBC has no critic |
| depths | 4, 8, 16, 32 | 4, 32 — the endpoints fix the sign |
| seeds | 3 | 3 |

### Why this is the right single knob

The actor maximises `min` over the critic ensemble (`offline/critic_losses.py`,
`_reduce_q`). At `critic_ensemble: 1` that `min` is a **pass-through**, so there is no
pessimism anywhere in the objective. DDPG+BC then seeks actions where `Q` is high,
which in an offline setting means it actively seeks the critic's *errors* — and a
deeper critic has sharper, more confident errors to find. If that mechanism were what
`humanoid_ensemble1` ran into, depth should stop paying here too. It does not.

---

## 2. Success vs depth

### 2.1 `best_success` — best evaluation during training, 75 episodes/point

Percent of episodes reaching the goal, ± 1 SD over 3 seeds.

| depth | n | mean | SD | seeds | status |
|---|---|---|---|---|---|
| 4 | 3 | 43.6 | 10.1 | 57.3, 33.3, 40.0 | final |
| 32 | 3 | **72.9** | 5.1 | 70.7, 68.0, 80.0 | **lower bound, 38% trained** |

→ `fig1_success_vs_depth.png` / `.csv`

### 2.2 `last3@1M` — where a run settles, pinned to 1,000,000 gradient steps

| depth | n | mean | SD | seeds | status |
|---|---|---|---|---|---|
| 4 | 3 | 28.0 | 10.8 | 39.1, 13.3, 31.6 | final |
| 32 | — | — | — | — | not yet at 1M |

→ `fig1b_last3_vs_depth.png` / `.csv`

### 2.3 `reeval@50` — the headline metric, 250 episodes/point

`params_best.pkl` reloaded and rescored at 50 episodes/task × 5 tasks, the way OGBench's
published table is measured (job `2995669`, `reeval_50ep.json` in this folder).

| depth | n | mean | SD | seeds | status |
|---|---|---|---|---|---|
| 4 | 3 | 38.9 | 11.3 | 54.0, 26.8, 36.0 | final |
| 32 | 1 | 71.6 | — | 71.6 (seed 0 only) | **best checkpoint of a run stopped at 755k steps** |

**On the depth-32 row.** The grid was cancelled at 38% of d32 (see §3). All three
`params_best.pkl` survived, and seed 0 was rescored at 250 episodes afterwards: **71.6**,
against the three-seed training-eval mean of 72.9 ± 5.1. The rescore moves the number by
about a point, so seeds 1 and 2 were not scored — they would have confirmed rather than
informed, and the GPUs were wanted for `humanoid_stitch1`. The checkpoints are on disk
indefinitely, so this can be completed at any time:

```bash
for d in $SCRATCH/rl-scaling/experiments/runs/antmaze_ensemble1__crl__d32__*; do
  python src/scripts/reeval.py --run-dir "$d" --episodes 50
done
```

Against `phase1_repro` d32's **86.7 ± 2.9** this is −15.1 points — consistent with the
matched-step `best_success` gap of −13.3, and the same conclusion: a large deficit at
d32, nowhere near enough to remove the depth gain.

**Against `phase1_repro` d4's 35.5 ± 9.4, this is +3.4 points — a third of a standard
deviation.** On the one metric that is comparable to a published table, removing the
second critic changes nothing measurable at depth 4. Every one of the three metrics now
agrees on that: +0.9 (`best_success`), −6.1 (`last3@1M`), +3.4 (`reeval@50`), all inside
the seed spread and not even consistent in sign.

Both seeds scored below their own training-time best (57.3 → 54.0, 33.3 → 26.8,
40.0 → 36.0), which is the expected direction: `best_success` is a max over ~20 noisy
evaluations and `reeval@50` removes that selection noise from the measurement, though
it still scores the optimistically-chosen checkpoint.

#### Per-task at 50 episodes, depth 4

| | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---|---|---|---|---|
| `critic_ensemble: 2` | 0.167 | 0.553 | 0.773 | 0.080 | 0.200 |
| `critic_ensemble: 1` | 0.147 | 0.333 | 0.580 | 0.053 | **0.833** |

Equal aggregates, **different task mixes.** The single critic trades most of tasks 2
and 3 for task 5. Task 4 — the forced 180° detour that depth unlocks on
`phase1_repro` — is near zero under both, as it is at depth 4 everywhere. Read this as
seed-level reallocation at n=3, not as a property of the knob: the same reshuffling
shows up between architectures at equal aggregate in `antmaze_stitch__depth` §7.

**One unit warning.** `reeval.py` prints its own `last3` column, which is the mean of
the final three evaluations *wherever a run stopped*. That is **not** the `last3@1M` in
§2.2, which is pinned to 1,000,000 gradient steps. The two differ a lot here because
these d4 runs trained to 1.6–2.0M: unpinned 38.7 ± 8.8 against pinned 28.0 ± 10.8. Only
the pinned number is comparable across grids, because the plateau rule stops different
runs at different points on their own curves.

### 2.4 Side by side with `phase1_repro`

Matched on metric and, where the runs are not both finished, on gradient step.

| metric | d4 ens=2 | d4 ens=1 | Δ | d32 ens=2 | d32 ens=1 | Δ |
|---|---|---|---|---|---|---|
| `best_success` @ ≤735k | 39.1 ± 10.9 | 32.9 ± 12.0 | −6.2 | 86.2 ± 2.5 | 72.9 ± 5.1 | **−13.3** |
| `best_success`, final | 42.7 ± 11.4 | 43.6 ± 10.1 | +0.9 | 87.1 ± 3.3 | *(pending)* | — |
| `last3@1M` | 34.1 ± 9.8 | 28.0 ± 10.8 | −6.1 | 79.3 ± 5.5 | *(pending)* | — |

**At depth 4 the ensemble is worth nothing.** The finished `best_success` figures differ by
+0.9 points against a seed spread of ±10–11, and the three per-seed values interleave
(ens=2: 52.0, 49.3, 26.7; ens=1: 57.3, 33.3, 40.0). The `last3@1M` deficit of −6.1 is
likewise well inside one SD.

**At depth 32 it is worth about 13 points** — the only place the knob clearly bites,
and the direction the pessimism argument predicts: the deeper critic is the one whose
errors the actor can exploit, so it is the one that benefits from a `min` over two.

---

## 3. The late decline is *not* caused by removing the ensemble

A natural reading of the humanoid grid is that without pessimism a deep critic peaks
and then comes apart. Measured here, **both** grids do that, and the one *with* the
ensemble does it slightly harder:

| depth 32 | peak | at step | latest / final | at step | gap |
|---|---|---|---|---|---|
| ens=1 seed 0 | 70.7 | 300,000 | 64.0 | 700,000 | −6.7 |
| ens=1 seed 1 | 68.0 | 300,000 | 54.7 | 700,000 | −13.3 |
| ens=1 seed 2 | 80.0 | 400,000 | 78.7 | 700,000 | −1.3 |
| **ens=1 mean** | **72.9** | | | | **−7.1** |
| ens=2 seed 0 | 82.7 | 400,000 | 69.3 | 1,000,000 | −13.3 |
| ens=2 seed 1 | 88.0 | 600,000 | 76.0 | 1,600,000 | −12.0 |
| ens=2 seed 2 | 90.7 | 900,000 | 82.7 | 1,600,000 | −8.0 |
| **ens=2 mean** | **87.1** | | | | **−11.1** |

*(`gap` is latest − peak in percentage points. The ens=1 rows are measured 300–900k
steps earlier than the ens=2 rows, so their gaps have had less time to open — the
comparison is indicative, not matched.)*

So peak-then-drift on deep antmaze runs is a property of the setup, present at
`critic_ensemble: 2`, and **removing pessimism is not its cause**. It is also far
smaller than humanoid depth 32's −21.7 points between best checkpoint and settled
score. Whatever goes wrong on humanoid still needs its own explanation.

→ `fig2_learning_curves.png` — the d32 curve rising to 0.70 by 300k and then flattening
is the clearest single panel in this folder.

---

## 4. Critic diagnostics

Only depth 4 has reached the 1,000,000-step pin, so this is a one-row comparison.

| metric at 1M, depth 4 | ens=2 | ens=1 |
|---|---|---|
| `train/critic/categorical_accuracy` | 0.3691 ± 0.0174 | 0.3203 ± 0.0100 |
| `train/critic/contrastive_loss` | 0.00387 ± 8e-05 | 0.00387 ± 9e-05 |
| `val/critic/contrastive_loss` | 0.00530 ± 5e-05 | 0.00535 ± 5e-05 |
| `train/critic/binary_accuracy` | 0.99911 | 0.99908 |

**The critic's own loss is unchanged to three significant figures**, train and
validation alike, while top-1 retrieval drops 0.369 → 0.320. That pattern is what you
would expect if the metric is computed from the ensemble-reduced logits: two critics at
the same per-critic loss retrieve better than one, because averaging cancels
independent logit noise. It is a measurement effect of the ensemble, not evidence that
the single critic learned a worse representation.

`binary_accuracy` is reported for completeness only. Its base rate is 1 − 1/1024 =
0.99902 on a 1024×1024 logit matrix with a single positive per row, so every value
above is within 0.0005 of chance and none of them mean anything.

→ `fig3_critic_diagnostics.png` / `.csv`

---

## 5. Cost

→ `fig4_cost.png` / `.csv`. Depth 32 trains at ~33 steps/s against depth 4's ~170, so
the 6-run pack is entirely gated by d32: ~2 h for the d4 seeds and ~17 h for the d32
seeds on 2 A100s at 3 runs/GPU.

---

## 6. Per-task breakdown

→ `fig6_per_task.png` / `.csv`. On `phase1_repro`, almost the whole depth gain was one
task — task 4, the forced 180° detour, going 0.08 at d4 to 0.97 at d32. Whether that
reproduces without the ensemble is the most interesting thing left in this grid and
cannot be stated until d32 finishes.

---

## 7. What this does and does not settle

**Settles.** `critic_ensemble` is not the explanation for the `phase1_repro` →
`humanoid_ensemble1` reversal. One of the four confounds is closed.

**Does not settle.**

- The remaining three confounds: dataset (`antmaze_navigate__depth`, running),
  `actor_p_randomgoal` (nothing yet — see the backlog in `../README.md`), and
  `discount` 0.99 → 0.995 (nothing yet).
- Whether the 13-point d32 deficit is final. It is read at 38% of training.
- Whether the per-task structure survives; §6.
- This grid has **no GCBC arm**, by design — GCBC has no critic, so the knob is a
  no-op there. Any GCBC comparison must come from `phase1_repro`.

---

## 8. Reproducing

```bash
python src/run.py experiments/antmaze_ensemble1.yaml            # print the plan
python src/run.py experiments/antmaze_ensemble1.yaml --submit   # queue it
python src/run.py experiments/antmaze_ensemble1.yaml --reeval   # the 250-episode rescore
python src/scripts/plot_results.py --grid antmaze_ensemble1     # redraw + publish here
```

Runs live in `$SCRATCH/rl-scaling/experiments/runs/antmaze_ensemble1__*`; figures are
written to `experiments/analysis/antmaze_ensemble1/` and copied here by `--publish`
(the default).
