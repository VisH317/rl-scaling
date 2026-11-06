# Results — index

Every finished experiment and eval suite, one folder each. A folder holds **its figures
and the document that reads them**, so a figure is never separated from the sentence
interpreting it.

Compiled 2026-11-06. 38 training runs finished, 0 in flight.

```
results/
  phase1_anchor/       6 runs   the reproduction control
  phase1_repro/       24 runs   does depth help on antmaze?           <- the main result
  humanoid_ensemble1/  8 runs   does it survive a new environment?    <- the reversal
  interp01/           3 ckpts   what about the critic actually improves?
```

Nothing in these documents is typed from terminal output: every table is read from the
`.csv` beside the `.png`, and the eval documents fill themselves between `<!-- AUTO -->`
markers. Regenerate with `python src/scripts/plot_results.py --grid <name>` (figures and
tables) or `python src/eval.py --report` (eval suites).

---

## The one-paragraph version

**Depth helps a lot on one dataset and hurts on the other, and the critic's own metrics
improve in both.** On `antmaze-medium-stitch` the contrastive critic scales strongly with
depth — 34.1 → 79.3 across 4 → 32 layers, against OGBench's published 53% — and nearly
all of that gain is one task, the forced 180° detour, going 0.13 → 0.93. On
`humanoidmaze-medium-navigate` the same sweep runs **backwards**: 64.2 → 31.3, with the
deep runs peaking at 200k steps and then losing half their score, and that same detour
task never working at any depth. In both environments the critic gets monotonically
*better* at its own contrastive objective as depth rises (categorical accuracy to 0.81
and 0.86) while its held-out contrastive loss gets monotonically worse. And where the
critic has been probed directly, its value field becomes *less* aligned with true
distance-to-goal as depth rises, even while success rises.

So depth is buying something real on antmaze, it is not buying "a better distance
function", and whatever it buys does not transfer. **The caveat that governs everything
above:** the two grids differ in three ways, not one, so they do not isolate the
environment — [`humanoid_ensemble1/README.md`](humanoid_ensemble1/README.md) §4 names the
4-run experiment that would.

---

## Headline numbers

Last-3-evaluation mean pinned to a common 1,000,000 gradient steps — the convention
OGBench's table uses. Percent of episodes, ± 1 SD over seeds.

| depth | CRL antmaze (n=3) | GCBC antmaze (n=3) | CRL humanoid (n=2) |
|---|---|---|---|
| 4 | 34.1 ± 9.8 | 53.5 ± 6.8 | **64.2 ± 2.4** |
| 8 | 49.3 ± 3.8 | 65.2 ± 1.6 | 63.3 ± 0.7 |
| 16 | 71.4 ± 1.9 | 62.5 ± 2.4 | 54.4 ± 0.7 |
| 32 | **79.3 ± 5.5** | 61.2 ± 5.5 | 31.3 ± 1.1 |

Reference: OGBench publishes 53% (CRL) and 45% (GCBC) for this dataset.

- **CRL antmaze rises throughout, decelerating:** +15.2, +22.1, +7.9 per doubling.
- **GCBC antmaze saturates at depth 8** and drifts down after.
- **CRL humanoid falls throughout,** and the fall is larger than the seed spread at
  every depth.

`reeval@50` — the best checkpoint rescored at OGBench's published 50 episodes/task, and
the only number strictly comparable to their table — is in each folder's
`reeval_50ep.json`. It is currently being re-measured for the two grids that gained runs
(jobs `2995117`, `2995118`); the shape held the last time it was checked
(34.3 → 54.7 → 78.4 for CRL antmaze).

---

## Every figure

Each row is one PNG plus the CSV it was drawn from. The same six grid figures exist for
all three training grids.

### Training grids — `phase1_repro`, `phase1_anchor`, `humanoid_ensemble1`

| figure | what it answers | read it for |
|---|---|---|
| `fig1_success_vs_depth` | max-over-evaluations vs depth, per agent | the optimistic view; a max over ~20 noisy points |
| `fig1b_last3_vs_depth` | the published metric, pinned to a common step | **the headline** |
| `fig2_learning_curves` | success vs gradient steps, depth as a colour ramp | *when* a run peaks — the humanoid story is here |
| `fig3_critic_diagnostics` | train vs validation contrastive loss, categorical accuracy | the overfitting that depth causes in both environments |
| `fig4_cost` | params, throughput, wall clock | so depth is never read without its price |
| `fig6_per_task` | the 5 tasks × configs heatmap | what the aggregate hides — task 4 above all |
| `fig5_depth_data_heatmap` | not generated; needs >1 dataset size (phase 3) | — |

### `interp01` — the critic, probed directly

| figure | what it answers |
|---|---|
| `fig1_critic_field` | the value field on the real maze, depth × task |
| `fig2_topology` | does the field follow the maze (geodesic) or ignore it (straight-line)? |
| `fig3_advantage_field` | Q(policy) − Q(dataset); **appears only once `dataset` is an action source** |
| `fig4_metrics` | where the field peaks, and whether it climbs along the path |
| `fig5_action_preference` | which of two real expert actions the critic prefers, per bin |
| `fig6_action_tests` | `action_topology_gain` and `oracle_gap` vs depth |

---

## What each folder is

### [`phase1_repro/`](phase1_repro/) — the main result · 24 runs
CRL and GCBC on the residual tower from the 1000-layer paper, depths 4/8/16/32, 3 seeds,
`antmaze-medium-stitch-v0`. 651 lines, and the most careful document here: it covers the
environment, the five tasks, the maze geometry that predicts the results, and §2's full
accounting of **where we differ from the paper** — we run binary NCE on dot products
where they run InfoNCE on negative L2 with a logsumexp penalty, and their offline result
is negative where ours is positive. Read §2 before quoting §6.

### [`humanoid_ensemble1/`](humanoid_ensemble1/) — the reversal · 8 runs
The same sweep on `humanoidmaze-medium-navigate-v0` with a single critic encoder pair.
Depth hurts, monotonically. Documents the three-way confound that stops this from being
a clean environment comparison, and names the 4-run experiments that would resolve it.

### [`phase1_anchor/`](phase1_anchor/) — the control · 6 runs
Stock OGBench architecture, unmodified, depth 4. Exists so that a miss means the harness
is wrong rather than the idea. Its analysis lives in `phase1_repro`'s §5.

### [`interp01/`](interp01/) — critic interpretability · 3 checkpoints
The first eval suite. Takes the best seed at each depth and measures what about
`Q(s, a, g)` improves: whether the value field is ordered like geodesic distance, which
of two real expert actions the critic prefers where they disagree, and whether the
actor's own action beats the maze-correct one under its own critic. **Currently being
re-run across all four depths** (job `2995119`) now that depth 32 exists.

---

## Not here yet

| what | why it matters | cost |
|---|---|---|
| `critic_ensemble: 1` on antmaze, d4 + d32 | the cheapest run that would un-confound the reversal | 4 runs |
| `interp01` on `humanoid_ensemble1` | does the *worse*-controlling critic have a *better*-ordered field? | 4 evals |
| `state_probes` | what `ψ(g)` linearly encodes; designed, not implemented | — |
| `q1_loss_pilot` | whether the loss form explains the disagreement with the paper's §4.6 | 4 runs, ~26 GPU-h |
| `scaling_crl_replication` | the paper's own setting | — |

Research context, open questions and the verified-facts log live in
[`../src/offline/`](../src/offline/): `offline_exp.md` (the brief), `algorithm.md` (what
actually runs), `research_log.md` (verified facts), `impl_log.md` (what was built).
