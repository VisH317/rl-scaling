# Experiment Brief: Offline Scaling of Contrastive RL

Nov 4, 2026 · @Vishrut Thoutam

A self-contained plan for testing whether contrastive RL scales with network depth and dataset size offline, written to hand to a coding agent.

## Purpose and how to use this brief

**Goal of the experiment:** find out whether contrastive RL (CRL) gains from network depth in the *offline* setting once dataset size grows with the model, and if not, which part fails (data, actor, or horizon). The end product of this first stage is a small-scale scaling fit (depth × dataset size up to 10M transitions) that justifies one larger training run.

**Context:** this is a project in Benjamin Eysenbach's lab (Princeton), mentored by PhD student Chongyi Zheng. It follows the online result that CRL scales to 1,000+ layers and the open problem that the same recipe gave no gains offline.

### Rules for the coding agent

1. **Verify before assuming.** Repo layouts, config flags, default hyperparameters, dataset names and file sizes in this brief come from papers, not from reading the code. Check each against the actual repository or dataset before relying on it, and report mismatches.
2. **Reproduce before modifying.** Every phase starts by reproducing a known number. Don't change the algorithm until the baseline matches.
3. **Change one thing at a time** and log the full config of every run.
4. **Keep compute visible.** Log wall-clock, gradient steps, samples seen and parameter counts for every run.
5. **Stop and ask** when a choice changes the experiment's meaning: which dataset, which evaluation protocol, how training length scales with data, or anything that looks like it contradicts this brief.
6. **Don't download huge datasets without asking.** Several OGBench and Horizon Reduction datasets are 100M–1B transitions; confirm disk space first.

## Background: what's already known

### Contrastive RL in one paragraph

CRL learns a critic f(s, a, g) = −‖φ(s, a) − ψ(g)‖ (or a dot product) with an InfoNCE loss: for a batch of B transitions, the positive goal for (sᵢ, aᵢ) is a future state from the same trajectory, sampled at a geometric offset, and the other B − 1 goals in the batch are negatives. At the optimum the critic is a log density ratio of reaching g from (s, a). The actor maximizes f(s, π(s, g), g). No rewards, no TD bootstrapping: it is a Monte Carlo, classification-style method. A logsumexp penalty on the logits is standard and important.

### Papers and the facts that matter here

| Paper | Offline-relevant facts |
| --- | --- |
| [1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858) (Wang et al., NeurIPS 2025) | Online CRL scales to 1,024 layers with residual blocks (4 dense layers each), LayerNorm, Swish; removing LayerNorm or using ReLU breaks it. **Offline (rebuttal, OGBench):** CRL gained nothing from depth, even with cold-initialized final layers; GCBC with the same architecture improved from 0.47 to 0.63 (depth 2 → 32) on antmaze-medium-stitch, but collapsed with a standard architecture; BC on expert data collapsed with depth (0.78 → 0.09); QRL stable but no gain. Everything fits on one 80 GB A100. |
| [Stabilizing Contrastive RL from Offline Data](https://arxiv.org/abs/2306.03346) (Zheng et al., 2023) | The offline CRL recipe: LayerNorm everywhere, large batch (2048), cold init of final layers (±10⁻¹²), random-crop augmentation for images, BC term in the actor, small representation (16 dims beat 128/512). Success grew \~3× from 100K to 1M transitions; ResNet-18 overfit vs a 3-layer CNN (28% vs 81%). Subgoal planning hurt by \~30 points |
| [Horizon Reduction Makes RL Scalable](https://arxiv.org/abs/2506.04168) (Park et al., 2025) | Offline GCRL on datasets up to 1B transitions with 4 × 1024 MLPs: no method (flow BC, IQL, CRL, SAC+BC) solved the hardest tasks; plateaus blamed on horizon (TD bias, policy complexity); n-step and hierarchy (SHARSA) help. CRL had better asymptotic performance on humanoidmaze-giant (no TD bias). Easier tasks (cube-double, puzzle-4x4) scale fine. **Confound:** every run used a fixed 5M gradient steps regardless of dataset size. Datasets and code are public |
| [Is Value Learning Really the Main Bottleneck?](https://arxiv.org/abs/2406.09329) (Park et al., NeurIPS 2024) | Offline RL is usually limited by policy extraction and test-time generalization, not value learning. Method: "data-scaling matrices" varying value data and policy data separately. DDPG+BC beats AWR in 15/16 settings (gc-antmaze-large with CRL: AWR 37, DDPG+BC 58). Coverage beats optimality. OPEX: a ← a + β∇ₐQ at test time |
| [Scaling Laws for Imitation Learning](https://arxiv.org/abs/2307.09423) (Tuyls et al., 2024) | BC follows Chinchilla-style compute laws only when data grows with the model (optimal data ∝ C^0.4–0.5). Method template: isoFLOP profiles plus a parametric fit |
| [Offline Actor-Critic Scales](https://arxiv.org/abs/2402.05546) (PAC, Springenberg et al., 2024) | Offline actor-critic with distributional TD scales to \~1B parameters on short-horizon tasks. Training loss didn't predict performance, so fit laws on return. Blending BC and RL losses with α: α ≈ 0.8 best on good data, α = 0 on poor data |

### What the evidence adds up to

Offline CRL's failure to scale is probably not one thing. Data was held fixed while depth grew (data-limited); the actor may exploit critic errors offline (actor-limited); and long horizons thin out CRL's training signal (horizon-limited). GCBC, which has no critic, does scale offline, which makes it the key control.

## Research question and hypotheses

**Question:** does offline CRL improve with depth when dataset size grows with it? If not, which component fails?

| Hypothesis | Mechanism | What we'd see |
| --- | --- | --- |
| **H-data** (data-limited) | Deep networks need more data; the original offline test grew depth on a fixed \~1M dataset | Depth helps at 10M but not at 0.1M; the best depth grows with dataset size |
| **H-actor** (actor-limited) | The critic is fine, but the actor climbs the critic toward actions the data never took, where the critic is wrong | Critic metrics improve with depth, success doesn't; the actor's critic score far exceeds the data's while success lags; anchoring the actor (BC term) or freezing a good critic and training a better actor recovers scaling |
| **H-horizon** (horizon-limited) | CRL's positives for far goals become rare (roughly γ^H), so the critic knows little about long-range goals | Critic accuracy decays with goal distance; failures concentrate on far goals; shorter-horizon tasks scale fine |

The hypotheses can all be partly true. The design separates them: data size is varied directly (H-data), the critic and actor are probed and trained separately (H-actor), and every metric is binned by goal distance (H-horizon).

**Control throughout: GCBC.** Same architecture, same data, no critic. If GCBC scales with depth on a dataset and CRL doesn't, the data supports depth and the problem is in CRL. If neither scales, the data is the limit.

## Plan at a glance

| Phase | What | Output | Check before moving on |
| --- | --- | --- | --- |
| 0 | Get data: OGBench tasks at the needed sizes; later, self-generated JaxGCRL buffers | Datasets subsampled to 0.1, 0.3, 1, 3, 10M transitions | Sizes, trajectory structure and goal sampling verified |
| 1 | Reproduce: online CRL sanity check; offline CRL vs GCBC across depth | Curves matching the 1000-layer paper's online result and offline rebuttal | Online numbers within seed noise of the paper; offline shows CRL flat, GCBC improving |
| 2 | Light probes on phase-1 models (shallow vs deep) | Critic and actor diagnostics | Probes run and give interpretable numbers |
| 3 | Grid: depth × dataset size, CRL and GCBC; start with a corner | Success for every cell, 3 seeds | Trend visible in the corner before filling the grid |
| 4 | Fit scaling curves; full probes across the grid | Fitted law, plots, a proposal for one larger run | Fit is stable under bootstrap over seeds |
| 5 | Actor thread: frozen critics, sweep actor depth, policy class, extraction | Actor-vs-critic bottleneck answer | Runs in parallel with phases 3–4 once critics exist |

Phases 0–2 are the immediate work. Phases 3–4 are sized so the corner of the grid can run first and the rest only if a trend shows.

## Phase 0: data

### Why not the 1B datasets

Horizon Reduction's 1B-transition datasets cover the hardest tasks (cube-octuple, puzzle-4x5/4x6, humanoidmaze-giant), where no method succeeds. That makes the first scaling signal hard to see. Start where methods are known to work, and only move to the hard tasks once the pipeline is proven.

### Primary: OGBench

[OGBench](https://github.com/seohongpark/ogbench) has 85 offline goal-conditioned datasets (dataset types: navigate, stitch, explore, play, noisy; state and pixel versions). Standard sizes are about 1M–5M transitions; 100M-transition versions exist for some tasks, including cube-double/triple/quadruple-play, scene-play and puzzle-3x3/4x4/4x5/4x6-play, plus humanoidmaze-giant-navigate.

| Use | Task | Why |
| --- | --- | --- |
| Reproduction (phase 1) | antmaze-medium-stitch (and antmaze-medium/large-navigate) | The task the 1000-layer rebuttal used for offline GCBC; standard size |
| Scaling grid (phases 3–4) | **cube-double-play** (100M version available) | Known to be solvable and to scale (Horizon Reduction appendix); manipulation rather than another maze |
| Backup grid task | puzzle-4x4-play or scene-play (100M versions) | If cube-double saturates too early |

Use state observations, not pixels, for the first stage.

### Subsampling

From the largest available version, build nested subsets of **0.1, 0.3, 1, 3 and 10M transitions**:

- **Subsample whole trajectories, not transitions,** because CRL samples positives from the future of the same trajectory.
- **Nested:** each smaller set is a subset of the next, so differences come from size, not from which trajectories happened to be drawn.
- **Fixed seed** for the sampling; save the trajectory index lists alongside the data.
- **Record coverage** for each subset (e.g. the fraction of a state-space grid visited, or the number of distinct object configurations), so size and coverage can be told apart later.

### Secondary, later: self-generated JaxGCRL buffers

Dump replay buffers from online CRL runs on JaxGCRL Ant Big Maze (the online scaling project produces these anyway), mixing checkpoints from weak to strong policies. This gives (1) the same task as the online experiments, so offline and online scaling can be compared directly, and (2) control over coverage: buffers from shallow vs deep collectors at matched size. Not needed for phases 0–2.

## Phase 1: reproduction

### 1a. Online sanity check (short)

Run the scaling-CRL code online on JaxGCRL Ant Big Maze at depths 4 and 16 with the paper's defaults. Expected: a large jump between them (the paper reports a critical depth of 8 on this task); the rebuttal reports roughly 2 GPU-hours at depth 4 and 4 at depth 16. Purpose: confirm the architecture, logging and evaluation code are right before going offline. Skip if Chongyi confirms the code is already validated.

### 1b. The offline negative result (the real baseline)

On OGBench antmaze-medium-stitch at its standard size:

- **CRL** at depths 4, 8, 16, 32 with the 1000-layer architecture (residual blocks, LayerNorm, Swish), plus the stable-CRL choices that don't conflict: large batch, cold-initialized final layers, BC term in the actor.
- **GCBC** at the same depths and architecture.
- 3 seeds each; OGBench's standard evaluation (its 5 evaluation goals per task; report success rate).

**Expected:** CRL roughly flat with depth; GCBC rising from about 0.47 to 0.63 (the rebuttal's numbers at depths 2 → 32). If CRL *does* scale here, stop and report: that alone changes the project.

### Which offline CRL implementation

Candidates: OGBench's reference implementations, Horizon Reduction's released code (includes CRL, SHARSA, flow BC, IQL, SAC+BC), or the stable-CRL code. Ask Chongyi which one the lab uses. Whichever is chosen, swap in the 1000-layer residual network and keep everything else at its published defaults.

## Phase 2: probes on the critic and actor

Run on phase-1 checkpoints first (shallow vs deep), then on the full grid in phase 4, where the contrasts across depth and data are the point. All probes use a fixed held-out set of (state, goal) pairs drawn from validation trajectories, **binned by temporal distance** (steps between state and goal along the trajectory).

| # | Probe | How | Hypothesis it tests |
| --- | --- | --- | --- |
| 1 | **Critic ranking accuracy** | For each held-out state, score its true future goal against K random goals; report top-1 accuracy (and InfoNCE loss) per distance bin | Is the critic good, and does it decay with distance? (H-horizon) |
| 2 | **Positive-pair frequency** | Histogram of the temporal distances of positive pairs actually sampled in training | How rare far goals are as training signal (H-horizon) |
| 3 | **Critic exploitation gap** | f(s, π(s, g), g) − f(s, a\_data, g) on held-out pairs, compared with actual success from those states | A large positive gap with low success = the actor is exploiting critic errors (H-actor) |
| 4 | **Action drift** | Distance between actor actions and dataset actions at the same states | Out-of-distribution actions (H-actor) |
| 5 | **Representation health** | Effective rank of φ and ψ embeddings; fraction of dormant units; embedding norms | Collapse or optimization failure at depth |
| 6 | **Value maps** | For the maze task, plot f over positions for a fixed goal (like the 1000-layer paper's Fig. 9) | Whether the critic captures maze topology or just Euclidean distance |
| 7 | **Train vs held-out gap** | Critic accuracy on training vs validation trajectories | Overfitting, expected for deep nets on small data (H-data) |

Save probe outputs as tables per checkpoint so they can be plotted against depth and dataset size later.

## Phases 3–4: depth × data grid and scaling fit

### Grid

| Axis | Values |
| --- | --- |
| Depth (actor and critic together) | 4, 8, 16, 32, 64 (width 256, 1000-layer residual architecture) |
| Dataset size (cube-double-play subsets) | 0.1, 0.3, 1, 3, 10M transitions |
| Algorithm | CRL (with the stable-CRL choices), GCBC |
| Seeds | 3 |

Full grid: 25 cells × 2 algorithms × 3 seeds = 150 runs. **Run the corner first:** depths 4, 16, 64 × data 0.1, 1, 10M = 9 cells × 2 × 3 = 54 runs. Fill the rest only if the corner shows a trend worth fitting.

### Training length: the key protocol decision

Offline, data and compute decouple: a small dataset can be looped over many times. Horizon Reduction trained every run for a fixed 5M gradient steps, so larger datasets got fewer passes per sample, which can fake a data plateau.

**Protocol:** train each cell until evaluation success stops improving, evaluating periodically, with a generous cap that grows with dataset size if runs are still improving at the cap. Record for every run: best success, the step it occurred at, total gradient steps, epochs over the data, and wall-clock. This gives the data-limited curve at each size, with compute recorded so a compute-optimal analysis can be done afterwards. Hold batch size, learning rate and optimizer fixed across the grid (verify the defaults in the chosen code).

### Fits

1. **Heatmap** of success over depth × data, per algorithm.
2. **Best depth per dataset size:** does the optimal depth grow with data? This is the headline plot for H-data.
3. **Parametric fit:**

```latex
\text{Err}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
```

Here N = parameter count (not depth), D = transitions, Err = 1 − success (consider a logistic link, since success is bounded, as PAC did). Bootstrap over seeds for confidence intervals.

4. **Compare CRL with GCBC** on the same axes.
5. **Run the full probe suite** across the grid and plot each probe against depth and data.

### Proposal for a larger run

If a trend fits: use the law to predict success for the 100M cube-double dataset at a deeper network (e.g. depth 128), state the prediction in advance, and propose that single run. If no trend: use the probes to say which hypothesis is responsible, and propose the targeted fix (anchored actor, hierarchy, or shorter-horizon tasks) instead.

## Phase 5: the actor thread

**Idea:** isolate the actor by **freezing a trained critic and training many actors on top of it** (Park et al.'s decoupled setup). One critic run then serves a whole sweep of actors, and any change in success is due to the actor alone.

### Setup

- **Critics:** take trained CRL critics from the grid at two dataset sizes (e.g. 1M and 10M) and at the depth that gave the best critic probe scores (phase 2, probe 1). An actor bottleneck may only show once the critic is good, hence two sizes.
- **Actors trained against each frozen critic**, varying one thing at a time:

| Axis | Values | Why |
| --- | --- | --- |
| Actor depth | 4, 8, 16, 32, 64 | The 1000-layer paper found actor depth mattered more than critic depth on some tasks; its 1,024-layer actor's loss exploded |
| Policy class | Gaussian (default) vs an expressive flow or diffusion policy | Flow BC was a strong offline baseline in Horizon Reduction; multimodal action distributions may need it |
| Extraction method | DDPG+BC (critic gradient + BC anchor, CRL's default), AWR (advantage-weighted BC), BC-weight sweep | DDPG+BC beat AWR in Park et al.; the BC weight sets how far the actor can stray from the data |
| Test-time fix | OPEX: a ← a + β∇ₐf(s, a, g), β swept | Nearly free; tests whether the critic's gradient carries useful information the actor missed |

### What to report

A small matrix per critic: actor depth × extraction method, with success plus probes 3 and 4 (exploitation gap, action drift). If better actors on the *same* frozen critic recover depth gains, H-actor is confirmed. Optionally, the reverse: freeze a good actor initialization and vary critic depth, to complete the value-vs-policy picture.

## Implementation notes

### Code bases (verify each before use)

| Code | Use | Where |
| --- | --- | --- |
| Scaling-CRL (1000-layer paper) | The residual architecture; online sanity check | Linked from the [project page](https://wang-kevin3290.github.io/scaling-crl/); built on JaxGCRL |
| [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) | Online environments and baselines; later, buffer generation | Also on PyPI as `jaxgcrl` |
| [OGBench](https://github.com/seohongpark/ogbench) | Offline datasets, evaluation protocol, reference offline GCRL implementations | Check which agents its reference implementations include (CRL, GCBC expected) |
| Horizon Reduction release | 100M/1B datasets; CRL, SHARSA, flow BC, IQL, SAC+BC implementations at scale | Linked from the [paper](https://arxiv.org/abs/2506.04168) |
| Stable CRL | The offline CRL recipe | Linked from the [paper](https://arxiv.org/abs/2306.03346); ask Chongyi |

All of these are JAX. Prefer one offline codebase for every algorithm in the grid so CRL and GCBC differ only in the loss.

### Architecture

Match the 1000-layer paper: residual blocks of 4 × (Dense → LayerNorm → Swish), residual added after each block; "depth" = total dense layers in the residual stack; width 256 unless stated. Separate towers for φ(s, a) and ψ(g). Count parameters programmatically for every run (N in the fits is parameters, not depth).

### Evaluation

Use OGBench's standard protocol (its 5 evaluation goals per task; confirm episode count per goal in the code). Report success rate, plus the probe tables from phase 2. Evaluate periodically during training so the best checkpoint and its step are known.

### Logging and layout

- One config file per run with every hyperparameter, the dataset subset ID, and the code commit hash.
- Log per run: success over training, best success and its step, gradient steps, epochs over data, wall-clock, peak memory, parameter count.
- Save the final and best checkpoints (probes and the actor thread need them).

```
experiments/
  data/            # subset index lists, coverage stats
  configs/         # one file per run
  runs/<run_id>/   # metrics, checkpoints, probe tables
  analysis/        # fits, plots, notebooks
```

### Compute

The 1000-layer paper ran everything on one 80 GB A100. Offline runs skip simulation, but deep networks on 10M transitions still take hours. **Time one depth-64 run on 10M transitions in phase 1** and multiply out before launching the grid.

## Open decisions, checks and deliverables

### Decisions to confirm (with Vish or Chongyi)

- [ ] Which offline CRL implementation to build on (OGBench, Horizon Reduction, or stable CRL)
- [ ] Grid task: cube-double-play (proposed) vs puzzle-4x4 or scene
- [ ] Whether the online sanity check (1a) is needed or the code is already validated in the lab
- [ ] Training-length protocol details: evaluation interval and the cap on gradient steps
- [ ] GPU budget for phases 1–4

### Things to verify in code or data, not assume

- [ ] Exact OGBench dataset names and which tasks have 100M versions; download sizes
- [ ] Trajectory boundaries in the datasets (needed for whole-trajectory subsampling and positive sampling)
- [ ] Default batch size, learning rate, representation dimension, discount and BC weight in the chosen CRL code
- [ ] Whether the residual architecture drops into the offline code cleanly (towers, normalization, initialization)
- [ ] OGBench evaluation: goals per task, episodes per goal, success definition

### Deliverables

- [ ] Phase 0: nested subsets (0.1–10M) with index lists and coverage stats
- [ ] Phase 1: online sanity plot; offline CRL vs GCBC across depth, matching the rebuttal
- [ ] Phase 2: probe tables for shallow vs deep checkpoints
- [ ] Phases 3–4: corner grid → full grid; heatmaps, best-depth-per-data plot, parametric fit with bootstrap intervals; probe plots across the grid
- [ ] Phase 5: actor matrices on frozen critics
- [ ] A one-page summary with the proposed larger run and its predicted outcome

### Out of scope for now

Pixels, the 1B hardest tasks, quasimetric methods, waypoint planning, and online scaling laws (a separate project, though it will share the architecture and fitting code).

### References

- [1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858)
- [Stabilizing Contrastive RL from Offline Data](https://arxiv.org/abs/2306.03346)
- [Horizon Reduction Makes RL Scalable](https://arxiv.org/abs/2506.04168)
- [Is Value Learning Really the Main Bottleneck in Offline RL?](https://arxiv.org/abs/2406.09329)
- [Scaling Laws for Imitation Learning in Single-Agent Games](https://arxiv.org/abs/2307.09423)
- [Offline Actor-Critic RL Scales to Large Models](https://arxiv.org/abs/2402.05546)
- [Contrastive Learning as Goal-Conditioned RL](https://arxiv.org/abs/2206.07568)
- [OGBench](https://arxiv.org/abs/2410.20092) · [JaxGCRL](https://arxiv.org/abs/2408.11052)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (for repeated-data effects, later)
