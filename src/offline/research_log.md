# Research Log: Offline Scaling of Contrastive RL

Companion to [`offline_exp.md`](offline_exp.md). That document is the **plan** (stable); this one is the
**log** (append-only). Every experiment, verified fact, and decision goes here.

**How to use this log**
- §1 Verified facts — things checked against code/data/cluster. Each row cites where it came from. Amend in place when re-verified; never delete a corrected fact, strike it and note the correction.
- §2 Decisions — numbered, dated, with status. Once `SETTLED`, don't relitigate without a new dated entry.
- §3 Open questions — blocking vs non-blocking.
- §4 Current plan — what's next. Rewritten each session.
- §5 Experiment log — append-only, one entry per run batch. Newest at the bottom.

---

## 1. Verified facts

### 1.1 Cluster (Princeton Stellar)

| Fact | Value | Source |
| --- | --- | --- |
| Login node | `stellar-intel`, no GPU, system Python 3.6.8 | `hostname`, `nvidia-smi` |
| GPU hardware | A100 **40 GB** (not 80 GB) | `sinfo`, RC docs |
| GPU nodes | 1× (8 GPU, 56 core, 1000 GB RAM) + 6× (2 GPU, 128 core, 550 GB RAM) | `sinfo -o "%P %G %D %c %m"` |
| GPU QOS | 7-day limit, **2 running jobs**, **8 GPUs**, 2 nodes per user | `qos` |
| Slurm account | `mae` | `sshare -U` |
| Home quota | 93 GB limit, 23.8 GB used | `checkquota` |
| Scratch quota | `/scratch/gpfs/vt7993`, 1000 GB limit, 17.6 GB used | `checkquota` |
| Download bandwidth | ~40 MB/s from `rail.eecs.berkeley.edu` | timed `curl` |
| Env managers | `uv` and `pixi` on PATH; `anaconda3/*` modules available; no `conda` on PATH | `which`, `module avail` |

> **Corrects the brief:** it assumes an 80 GB A100 (the 1000-layer paper's hardware). Stellar has 40 GB.
> Not a problem at width 256 (~12M params), but it rules out naively copying any memory-bound config.

**Added 2026-11-05, both discovered by hitting them during submission:**

| Fact | Consequence |
| --- | --- |
| `stellar-k09g1`, the only 8-GPU node, is in state `inval` (down). Six 2-GPU nodes remain. | A single-node 4-GPU request is **unschedulable** (`ReqNodeNotAvail`). `gpus_per_job=2` → 6 runs/pack, **12 concurrent, not 24**. Re-check `sinfo -p gpu` before raising. |
| Maintenance is reserved for the second Tuesday monthly (next **2026-11-10, 06:00–14:00**). Slurm will not start a job that cannot finish before a reservation. | A `--time=7-00:00:00` request sits PENDING with `Reserved for maintenance`. `time_limit=24:00:00` schedules immediately. Any run needing >5 days must wait until after the window. |
| EGL rendering **works** on the GPU nodes (resolves Q5). | Videos are available if wanted; `video_episodes=0` remains the default since state-based eval does not need them. |
| GPU QOS throttles *running* jobs only; PENDING jobs are unlimited. | The whole grid can be submitted at once with `afterany` chaining and self-sequences with **no live watcher**. |

**Unverified / to check:** whether compute nodes have outbound internet. If they don't,
`ogbench.make_env_and_datasets` auto-download fails inside a job. Pre-download everything and always
pass explicit `dataset_path=` regardless. Large downloads should use `stellar-vis1` per RC docs.

### 1.2 Data (measured, not assumed)

Downloaded to `/scratch/gpfs/vt7993/ogbench/data`. **Do not** use the `~/.ogbench/data` default — home quota.

| Dataset | Rows | Trajs × states | Transitions | ob / act | Disk |
| --- | --- | --- | --- | --- | --- |
| `antmaze-medium-stitch-v0` | 1,005,000 | 5,000 × 201 | 1,000,000 | 29 / 8 | 245 MB ✅ have |
| `antmaze-medium-stitch-v0-val` | 100,500 | 500 × 201 | 100,000 | 29 / 8 | 25 MB ✅ have |
| `cube-double-play-100m-v0` shard | 1,001,000 | 1,000 × 1001 | 1,000,000 | 37 / 5 | 297 MB ✅ have 000 |
| `cube-double-play-100m-v0` shard `-val` | 100,100 | 100 × 1001 | 100,000 | 37 / 5 | 30 MB ✅ have 000 |

Verified structural facts:
- **Standard OGBench datasets are exactly 1M transitions**, not "1M–5M" as the brief states.
- **Trajectory length is uniform within a dataset.** Subsampling whole trajectories is a reshape, not a
  boundary-detection problem. `terminals` is `bool`, `1` on the last row of each trajectory; the final
  file row is terminal. This closes the brief's open check on trajectory boundaries.
- Rows ≠ transitions: the last state of each trajectory has no successor. 1001 rows → 1000 transitions.
- **100M datasets are sharded directories**, not single files:
  `https://rail.eecs.berkeley.edu/datasets/ogbench/<name>-100m-v0/<name>-v0-{000..099}.npz`, each shard a
  self-contained 1M-transition npz with its own `-val` sibling. 11 such datasets exist; `cube-double-play-100m-v0`
  (the brief's proposed grid task) is confirmed present. Generated with `--seed=$i` per shard → shards are i.i.d.
- Files carry **`qpos` / `qvel`** beyond `observations`/`actions`/`terminals`; `ogbench` strips them unless
  `add_info=True`. ~45% of file size.

### 1.3 What `qpos` / `qvel` are, and why we keep them

MuJoCo generalized coordinates — the raw simulator state. `qpos` is every position DOF (free body = 7:
xyz + quaternion; hinge = 1); `qvel` the matching velocities (free body = 6: linear + angular).

| Task | `qpos` | `qvel` | Relation to `observations` |
| --- | --- | --- | --- |
| `antmaze-medium-stitch` | 15 (torso 7 + 8 hinges) | 14 (6 + 8) | **`observations == concat(qpos, qvel)` exactly** (verified) — pure redundancy |
| `cube-double-play` | 28 (14 arm/gripper + 2 cubes × 7) | 26 (14 + 2 × 6) | 54 ≠ 37 — the observation is a processed view; `qpos` carries strictly more |

**Keep `qpos`/`qvel` in the cube-double subsets.** Three hard dependencies:
1. Cube positions are only cleanly addressable via `qpos` — `ogbench/relabel_utils.py` indexes cube *i* as
   `qpos[14 + 7i : 14 + 7i + 3]`. Phase-0's "distinct object configurations" coverage stat runs on this.
2. **Probe 3 needs them.** The brief compares the critic exploitation gap "with actual success from those
   states" — that means rolling out from held-out *dataset* states, which requires setting the simulator
   state. A 37-dim processed observation cannot; `(qpos, qvel)` can.
3. `-oraclerep` is built from `qpos` (`add_oracle_reps`). Every Horizon Reduction published command uses
   `-oraclerep` env names, so dropping `qpos` forecloses comparison against their numbers.

For antmaze they're reconstructable (split the observation at index 15), so dropping them is safe there —
but at 245 MB there's no reason to.

### 1.4 Environments

Both instantiate headless on a login node (state-based only; a GLFW/X11 warning is harmless).

| Env | ob | act | `max_episode_steps` | Tasks |
| --- | --- | --- | --- | --- |
| `antmaze-medium-stitch-v0` | 29 | 8 | 1000 | `task1`…`task5` |
| `cube-double-play-v0` | 37 | 5 | 500 | `task1_single_pnp`, `task2_double_pnp1`, `task3_double_pnp2`, `task4_swap`, `task5_stack` |

5 evaluation goals per task, confirmed via `env.unwrapped.task_infos`. Horizon Reduction's `main.py`
default is `eval_episodes=15` per task, `eval_interval=250000`, `video_episodes=1`.

**Rendering:** `MUJOCO_GL=egl` failed on the login node (no GPU — expected). Untested on a GPU node.
State-based training needs no rendering; set `--video_episodes=0` to sidestep it entirely.

> **Horizon caveat worth stating before interpreting probes:** `antmaze-medium-stitch` trajectories are
> 201 states, so the **maximum in-trajectory temporal distance for a CRL positive is 200**, while the eval
> horizon is 1000. Probe distance bins cap at 200 on the phase-1 baseline. This is intrinsic to the
> `stitch` dataset type and is arguably the point of it, but it means phase-1 cannot test H-horizon at
> the horizons the task actually demands.

### 1.5 Does OGBench implement stable-CRL? **No — but it is 5 lines away**

OGBench has `impls/agents/crl.py` and `impls/agents/gcbc.py`. There is no stable-CRL agent. However,
OGBench's CRL critic loss *already is* stable-CRL's loss. Verified side by side:

| | scaling-crl (1000-layer, **online**) | OGBench `crl.py` (**offline**) | stable-CRL (Zheng 2023) |
| --- | --- | --- | --- |
| Critic loss | **InfoNCE**: `-mean(diag - logsumexp(axis=1))` | **binary NCE**: `sigmoid_BCE(logits, I)` over B×B | **binary NCE**: `BCEWithLogitsLoss(logits, I)` |
| Logits | `-‖φ-ψ‖₂` (L2) | `φ·ψ / √d` (dot) | `φ·ψ` (dot) |
| Logsumexp penalty | **yes**, coeff 0.1 | **no** | **no** |
| Architecture | residual blocks, Swish, LN | plain MLP, **GELU**, LN *after* activation | plain MLP, LN |
| Depth / width | depth 4 default, width 256 (CLI default) | 3 × 512 | 4 × 1024 ("shallow and wide") |
| Repr dim | **64** (hardcoded `Dense(64)`) | **512** (`latent_dim`) | **16** |
| Final-layer init | `lecun_uniform` everywhere | `default_init(1.0)` | **`init_w = 1e-12`** (cold) |
| Batch | 256 | 1024 | **2048** |
| lr | — | 3e-4 | 3e-4 |
| BC term | — | `ddpgbc`, `alpha = 0.1` | `bc_coef = 0.05` |
| Critic ensemble | 1 | 2 | 2 |
| Augmentation | — | `p_aug = 0.0` | crop, p = 0.5 (images only) |

Sources: `wang-kevin3290/scaling-crl/train.py`; `seohongpark/ogbench/impls/agents/crl.py` + `utils/networks.py`;
`chongyi-zheng/stable_contrastive_rl` (`experiments/train_eval_stable_contrastive_rl.py`,
`rlkit/torch/sac/stable_contrastive_rl_trainer.py`).

Consequences:
- **"Stable-CRL on OGBench" is a small patch to `crl.py`**, not a port: set `latent_dim=16`,
  `batch_size=2048`, `alpha=0.05`, and cold-init the φ/ψ final layers at 1e-12. LayerNorm, lr, the
  2-critic ensemble, the BC term and the loss form are already right.
- **Stable-CRL's own codebase is PyTorch/rlkit**, not JAX — correcting the brief's "All of these are JAX."
  It needs conda, two external env repos (`bullet-manipulation`, `multiworld`) and a Google Drive dataset.
  Not viable as the grid base.
- The 1000-layer paper's **residual architecture does not exist in OGBench.** `utils/networks.py` `MLP`
  is plain feedforward with GELU and no skip connections. It must be written. Exact target
  (from `scaling-crl/train.py`):

  ```
  residual_block(x, width):           # 4 dense layers per block
      identity = x
      4 ×  [ Dense(width, lecun_uniform) → LayerNorm → Swish ]
      return x + identity

  tower(x):
      Dense(width) → LayerNorm → Swish  # stem
      (depth // 4) × residual_block     # "depth" = dense layers in the residual stack
      Dense(repr_dim)                   # no norm, no activation
  ```

### 1.12 What the 1000-layer paper's §4.6 says about **offline** depth scaling

Verified 2026-11-05 against arXiv:2503.14858v5 full text. This is the single most
important fact for the project and it was not in this log before.

> §4.6, verbatim: *"In preliminary experiments, we evaluated depth scaling in the offline
> goal-conditioned setting using OGBench (Park et al., 2024). We found little evidence
> that increasing the network depth of CRL improves performance in this offline
> setting."* ... *"In all cases, baseline depth 4 networks often had the highest success.
> A key direction for future work is to see if our method can be adapted to enable
> scaling in the offline setting."*

**We observe the opposite** (`phase1_repro`, E2): CRL 34.1 → 49.3 → 74.2 across depths
4 → 8 → 16 on antmaze-medium-stitch. The likely reason, confirmed from our run configs:
**we are not running their critic loss.** Ours is `critic_loss=binary_nce`,
`logits=dot`, `logsumexp_penalty=0.0`; theirs is InfoNCE on `−‖φ−ψ‖₂` with a 0.1
logsumexp penalty. This turns **Q1** from a convention question into the experiment.

Other verified facts from the paper:

- **Policy extraction exists in both setups.** Their §3: *"The policy πθ(a|s,g) is
  trained to maximize the critic."* Ours is OGBench `actor_loss='ddpgbc'` =
  `q_loss + bc_loss` with `alpha=0.1`. Every success number we report is an extracted
  policy's, evaluated by rollout.
- **The eval metric never touches a critic.** `evaluate()` calls only
  `agent.sample_actions`, which selects only the `'actor'` network; the 2-member critic
  ensemble's `min` appears only inside the actor's training loss. With
  `eval_temperature=0.0` and `const_std=True` the scale is exactly 0, so the rollout is
  a deterministic actor. Consequence: *"the critic is scaling"* is an inference about
  the training signal, never a direct measurement. The only direct critic measurements
  are the InfoNCE top-1 accuracy and the train/val contrastive losses.
- **Which parameters each scoring convention uses.** `best_success` = live parameters at
  the single best eval step. `last3_eval_mean` = live parameters at each of the last
  three eval points — **three different networks**, 100k steps apart, averaged; a
  noise-reduction device, not one policy's score. `reeval@50` = `params_best.pkl`, one
  saved checkpoint (which holds actor and both critics, but only the actor is run).
- **Their metric is not ours.** §4.1: *"we measure the number of time steps (out of
  1000) that the agent is near the goal"*, averaged over the last five epochs — dwell
  time, not binary success. Our numbers are comparable to OGBench's published table, not
  to their headline figures.
- **Depth scales actor and both critic encoders jointly** (§4.1), as ours does. Their
  §4.4 ablates actor vs critic depth and finds it environment-dependent: *"In Arm Push
  Easy, scaling the critic is more effective; in Ant Big Maze, the actor matters more."*
  **That ablation is our unimplemented `actor_thread`**, and it is now the highest-value
  missing experiment.
- Their other offline ablations: *"scaling critic depth while holding the actor at 4 or
  8 layers"* and *"applying cold initialization to the final layers of the critic
  encoders"* (= our `arch=residual_cold`). Neither run.
- Their main setting is **online** (JaxGCRL / Brax / MJX), not OGBench.
- For offline **GCBC** they report a positive result on this exact dataset (Figure 14,
  available to us only as a summary, not verbatim). Our GCBC 53.5 → 65.2 then flat is
  broadly consistent.

**Correction to this log:** `REBUTTAL_GCBC = {2: 0.47, 32: 0.63}` in `plot_results.py`
is drawn as a reference line labelled "1000-layer rebuttal (GCBC)". Neither number
appears in the paper text or on the project page, and the paper's main-text depths are
4/8/16/32/64 with no depth-2 run. If real they are from the OpenReview thread, which
nobody has opened. **Unsourced until someone does.** It is annotation only.

---

### 1.6 No stable-CRL results exist on OGBench

Checked and did not find any. Stable-CRL (arXiv 2306.03346, June 2023) **predates OGBench** (arXiv
2410.20092, Oct 2024) and evaluates on image-based `bullet-manipulation` Sawyer drawer/pick-place tasks
from the RIG line of work, not on OGBench. OGBench's own benchmark tables report CRL, not stable-CRL.

**So there is no published stable-CRL number on OGBench to reproduce.** Phase 1's reproduction targets
must be (a) OGBench's published CRL number on the chosen task, and (b) the 1000-layer rebuttal's GCBC
depth curve (0.47 → 0.63, depth 2 → 32, `antmaze-medium-stitch`). If you have the rebuttal's CRL-by-depth
numbers or any internal stable-CRL-on-OGBench results, drop them in §5 — they'd change the baseline.


### 1.7 Experiment controls, in full

Everything below is in `src/offline/config.py` and is written into every run's
`config.json`, so a run is reproducible from its own config plus the recorded commit hash.

#### Optimizer — plain constant-LR Adam, deliberately (decision D9)

Verified at `OGBench impls/agents/crl.py:285` and `scaling-crl/train.py:541,556`: both
references use `optax.adam(learning_rate=lr)` with **no schedule, no warmup, no gradient
clipping, no weight decay**.

| Control | Value | Note |
| --- | --- | --- |
| optimizer | `adam` | `adamw` available but unused |
| learning rate | **3e-4, constant** | identical in all three reference codebases |
| actor / critic lr | both 3e-4 | kept separable for phase 5's actor thread |
| β₁, β₂, ε | 0.9, 0.999, 1e-8 | optax defaults, stated explicitly rather than inherited |
| schedule | `constant` | `cosine` / `linear_warmup_cosine` implemented, **off** |
| warmup steps | 0 | |
| gradient clipping | **none** | `clip_by_global_norm` available, off |
| weight decay | 0 | |

This matters for interpreting the result: the 1000-layer paper's stability at extreme
depth came from the **architecture** (LayerNorm + residual connections), not from
optimizer tricks. Warmup and clipping help deep networks *specifically*, so switching
either on would confound the very axis under test. `OptimConfig.deviates_from_baseline()`
returns a list of any departures and is recorded in each run config as
`optim_deviations`, so a deviation can never happen silently.

#### Architecture

| Control | Value |
| --- | --- |
| tower | `residual` (`mlp` = the non-residual control) |
| depth | 4, 8, 16, 32 (phase 1) · 4, 8, 16, 32, 64 (grid) — dense layers in the residual stack, stem and head excluded |
| block size | 4 dense layers per residual block, so depth must be divisible by 4 |
| width | 256 |
| repr dim | 64 (scaling-crl) — **open question Q2** |
| norm | LayerNorm, **pre-activation** (`Dense → LN → Swish`) |
| activation | Swish |
| init | `lecun_uniform` on every Dense; `cold` (1e-12 head) is a separate opt-in |

Depth 2 from the rebuttal is not expressible with 4-layer blocks, so our floor is depth 4.

#### Algorithm

| Control | CRL | GCBC |
| --- | --- | --- |
| batch size | 1024 (`recipe="stable"` → 2048) | 1024 |
| discount | 0.99 | 0.99 (goal sampling only) |
| actor loss | `ddpgbc`, `alpha` = 0.1 (`stable` → 0.05) | — |
| critic loss | `binary_nce` + dot logits, no logsumexp penalty — **open question Q1** | — |
| critic ensemble | 2 | — |
| goal sampling | `value_p_trajgoal` 1.0, `value_geom_sample` True; `actor_p_trajgoal` 1.0, `actor_geom_sample` False | same actor settings |
| const std | True | True |

#### Training length — the brief's key protocol decision

Horizon Reduction trained every run for a fixed 5M steps regardless of dataset size, so
larger datasets got fewer passes per sample, which can manufacture a data plateau.
Instead: **stop on an evaluation plateau, under a cap that grows with dataset size.**

| Dataset size | Step cap | Epochs over data at the cap (batch 1024) |
| --- | --- | --- |
| 0.1M | 500,000 | 5,120 |
| 0.3M | 1,000,000 | 3,413 |
| 1M | 2,000,000 | 2,048 |
| 3M | 3,000,000 | 1,024 |
| 10M | 5,000,000 | 512 |

Plateau rule: stop after `plateau_patience`=5 consecutive evaluations with no improvement
greater than `plateau_min_delta`=0.01. Every run records best success, the step it
occurred at, total gradient steps, epochs over data, samples seen, wall-clock and peak
RSS, so a compute-optimal analysis is possible afterwards even though the protocol is
data-limited. Batch size, lr and optimizer are held fixed across the whole grid.

#### Checkpointing — where and what

Mechanism is OGBench's `utils/flax_utils.py`: `flax.serialization.to_state_dict(agent)`
pickled to `params_<tag>.pkl`. The pickle holds params **and Adam optimizer state**, so a
restore resumes training exactly; it is not a weights-only export.

| What | When | Why |
| --- | --- | --- |
| `params_best.pkl` | overwritten whenever eval success improves | **Phases 2 and 5 depend on this.** The probes characterize a *good* critic and the actor thread freezes one — OGBench's own `main.py` saves only periodically and would lose it. |
| `params_final.pkl` | last step reached | resume a plateau-stopped run |
| `params_<step>.pkl` | every 500k steps, 2 most recent kept | crash recovery |

Per-run layout under `experiments/runs/<run_id>/`: `config.json` (resolved config +
commit hash), `train.csv`, `eval.csv`, `metrics.jsonl`, `status.json` (heartbeat every
5,000 steps), `summary.json` (written once at completion — its presence is what marks a
run done), `driver.log`, `checkpoints/`, `probes/`.

#### Evaluation

| Control | Value |
| --- | --- |
| protocol | OGBench standard: 5 goals per task, `task_id` 1–5 |
| episodes per task | 15 → **75 episodes per evaluation** |
| reported metric | mean success over the 5 tasks, plus per-task breakdown |
| eval temperature | 0.0 (deterministic) |
| eval interval | every 100,000 gradient steps |
| videos | 0 (EGL works, but state-based eval does not need rendering) |
| episode length | antmaze-medium-stitch 1000 · cube-double-play 500 |


### 1.9 Published OGBench flags (what the reference runs actually used)

From `lib/ogbench/impls/hyperparameters.sh`, verbatim:

```
L158  antmaze-medium-stitch-v0 (GCBC)
      main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/gcbc.py
L166  antmaze-medium-stitch-v0 (CRL)
      main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/crl.py \
              --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
```

**`crl.py`'s in-file defaults are NOT what OGBench ran.** The default is
`actor_p_trajgoal=1.0, actor_p_randomgoal=0.0`; the published antmaze-medium-stitch
command uses 0.5/0.5. That matters most on `stitch` datasets, where stitching together
trajectory segments is the entire point and trajectory-only actor goals never pose the
problem. GCBC genuinely uses bare defaults, so only CRL rows change.

These live in `PUBLISHED_FLAGS` in `config.py`, keyed by `(dataset, agent)`, applied in
`run_config()` on top of the agent defaults. Each run config records what was applied
under `published_flags`, so a missing override is visible in the artifact itself.

Other published settings: `train_steps` default 1,000,000; `eval_episodes` 50.

### 1.10 Resolved network shapes, read from a real checkpoint

Verifying that injecting the residual tower did not quietly repurpose OGBench's
`hidden_dims`. From `params_best.pkl` of a CRL depth-4 run on antmaze-medium-stitch
(ob 29, act 8; leading dim 2 is the critic ensemble):

```
modules_critic/phi/ResidualTower_0/Dense_0/kernel              (2, 37, 256)   stem: 29+8 -> width
modules_critic/phi/ResidualTower_0/ResidualBlock_0/Dense_0..3  (2, 256, 256)  4 layers at width
modules_critic/phi/ResidualTower_0/Dense_1/kernel              (2, 256,  64)  head -> repr_dim
modules_actor/actor_net/ResidualTower_0/Dense_0/kernel         (58, 256)      stem: 29+29 -> width
```

So the φ/ψ **head width is 64** (`latent_dim` / `repr_dim`) and the **last hidden layer is
still 256** (`arch.width`). Setting `value_hidden_dims=(256,)` did not change the hidden
width — OGBench builds `MLP((*hidden_dims, latent_dim))` and the shim reads only
`hidden_dims[-1]` as its output dim, so the tuple carries the head width and nothing else.
The checkpoint also confirms the 2-critic ensemble and that Adam state (`mu`/`nu`) is
stored, i.e. checkpoints are resumable.

### 1.8 Goal-sampling distribution — 43.6% of CRL positives are the final state

Measured on `antmaze-medium-stitch-v0` with the real `GCDataset`, 200,000 samples,
`discount=0.99`, trajectory length 201 rows:

| Goal stream | P(goal == trajectory final state) | mean distance | median |
| --- | --- | --- | --- |
| **value goals** (`value_geom_sample=True`) | **0.436** | 56.5 | 44 |
| actor goals (`actor_geom_sample=False`) | 0.024 | 50.2 | 38 |

Mechanism, in `utils/datasets.py` `sample_goals`:
`traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)`. Geometric offsets with
p = 1 − 0.99 have mean 100, well past the ~100-step average remainder of a 201-step
trajectory, and every overshoot is **clipped onto the final state** rather than resampled.
So nearly half of the critic's positive pairs are the same handful of terminal states.

Distance histogram for value goals: `[0,10)` 0.133 · `[10,25)` 0.181 · `[25,50)` 0.228 ·
`[50,100)` 0.274 · `[100,200)` 0.183.

**Consequences for the probes (phase 2):**
- Every distance-binned probe must treat "goal == final state" as its **own bin**, not
  fold it into `[100,200)`. Otherwise 43.6% of the mass — a degenerate, highly repeated
  set of targets — silently dominates the far-distance bins, which are exactly the bins
  H-horizon is about.
- Probe 1 (critic ranking accuracy) will look artificially strong at long distances,
  because the "far goal" it is scoring is often a state the critic has seen thousands of
  times as a positive.
- The effect is **specific to the critic**: actor goals use uniform sampling and pile up
  only 2.4%. So any train/eval asymmetry between critic and actor partly traces to here.
- The fraction is dataset-dependent — measured, not predicted:

  | Dataset | traj len | value: P(final) | value mean dist | actor: P(final) |
  | --- | --- | --- | --- | --- |
  | `antmaze-medium-stitch-v0` | 201 | **0.436** | 56.5 | 0.025 |
  | `cube-double-play-100m-v0` | 1001 | **0.101** | 89.6 | 0.005 |

  So the grid task is 4× less affected than the phase-1 reproduction task. That is itself
  a confound to keep in view: any horizon effect seen on antmaze-medium-stitch is partly
  an artifact of its short trajectories, and will not transfer at the same strength to
  cube-double. **Re-measure per dataset; never reuse 43.6%.**

Reproduce with `src/scripts/goal_distribution.py`.

---

### 1.11 `antmaze-medium-stitch-v0` geometry, and the angle that predicts GCBC

Regenerate with `python src/scripts/env_geometry.py`. Measured, not read off a paper.

**Environment.** OGBench `MazeEnv`, 8x8 cell grid with a solid border, `maze_unit = 4.0`
world units per cell, interior coordinates 0-20 on each axis. Observation 29-D with dims
0-1 the ant's `(x, y)`; action 8-D joint torques in `[-1, 1]`; episode limit 1000 steps;
success is binary per episode. Cell `(i, j)` sits at world `x = 4(j-1)`, `y = 4(i-1)`.

**What `-stitch` means, measured.** 5,000 trajectories x 201 steps. The shortest
maze-path distance between a trajectory's **own** first and last state: mean 3.93 cells,
median 4, p90 4, **max 5**. 97% cover exactly 4 cells. No trajectory in the dataset
solves any of the five tasks end to end -- every task must be composed from pieces.

**The 5 tasks:**

| task | start | goal | cells | detour | stitches | 1st-move angle |
|---|---|---|---|---|---|---|
| 1 | (1,1) | (6,6) | 10 | 1.41 | 2.5 | 45 deg |
| 2 | (6,1) | (1,6) | 10 | 1.41 | 2.5 | 45 deg |
| 3 | (5,3) | (4,2) | 6 | **4.24** | 1.5 | **135 deg** |
| 4 | (6,5) | (6,1) | 10 | 2.50 | 2.5 | **180 deg**, single legal exit |
| 5 | (2,6) | (1,1) | 8 | 1.57 | 2.0 | 11 deg |

`detour` = maze path / straight-line distance. `stitches` = path cells / median
trajectory reach. `1st-move angle` = smallest angle between a progress-making first move
and the straight line to the goal.

**The finding.** GCBC's per-task success is **monotone in the first-move angle** and in
nothing else: 0.96 (11 deg), 0.94 (45), 0.82 (45), 0.39 (135), **0.00 (180)**. Path
length does not order it -- tasks 1, 2 and 4 are all 10 cells and span 0.94 to 0.00.

Task 4's start `(6,5)` is a dead end: `(6,4)`, `(5,5)` and `(7,5)` are walls, so the only
exit is `(6,6)`, directly away from the goal. A goal-conditioned BC policy conditioned on
a goal 16 units west reproduces westward actions and walks into a wall for 1000 steps; it
has no mechanism to represent "go east first". CRL with a deep critic reaches 0.80 there.

**Consequences for phase 2:** probes must bin per task and on this angle, not only on
goal distance. And CRL vs GCBC are **complementary, not ranked** -- CRL is strongest
exactly where GCBC fails (tasks 3, 4) and weakest on task 1, the long well-aligned route
-- so a single aggregate success number on this dataset conveys very little.

See [`results/antmaze_stitch__depth/README.md`](../../results/antmaze_stitch__depth/README.md) sections 1 and 6.

---

## 2. Decisions

| # | Decision | Status | Date | Rationale |
| --- | --- | --- | --- | --- |
| D1 | Keep `qpos`/`qvel` in cube-double subsets | **SETTLED** | 2026-11-04 | Probe 3 and coverage stats require them (§1.3). Cost is negligible. |
| D2 | GCBC = OGBench `impls/agents/gcbc.py` | **SETTLED** | 2026-11-04 | User decision. Horizon Reduction has no plain GCBC (only `gcfbc.py`, flow BC). |
| D3 | Dataset dir = `/scratch/gpfs/vt7993/ogbench/data` | **SETTLED** | 2026-11-04 | Home quota is 93 GB; the `~/.ogbench/data` default would consume it. |
| D4 | Grid task = `cube-double-play-100m-v0` | PROPOSED | — | Brief's proposal; availability now confirmed. |
| D5 | Nested subsets = **shard-aligned** (§2.1) | **SETTLED** | 2026-11-05 | Vish's call. Keeps the 1M cell a standard whole dataset (free reproduction check), needs no new files for 1M/3M/10M, and the seed-spread confound it gives up is negligible across 10 i.i.d. shards. |
| D6 | Codebase = OGBench `impls` + our patches | **SETTLED** | 2026-11-05 | Implemented and verified end-to-end on an A100 (see `impl_log.md` I5–I7). Has both CRL and GCBC (D2); residual tower injected via `patches.py`; agents themselves left unmodified so published defaults stay diffable. |
| D9 | Optimizer stays plain constant-LR Adam | **SETTLED** | 2026-11-05 | Matches both references exactly (§1.7). Warmup/clipping help deep nets specifically, so enabling them would confound the depth axis. |
| D10 | `gpus_per_job=2`, `time_limit=24h` | **SETTLED** | 2026-11-05 | Forced by the cluster, not chosen (§1.1). |
| D7 | Loss form for the depth test (see §3, Q1) | **NEEDS CALL** | — | InfoNCE+L2+logsumexp vs binary NCE+dot. Confounds the depth result. |
| D8 | Representation dim | **NEEDS CALL** | — | 64 (scaling-crl) vs 512 (OGBench) vs 16 (stable-CRL). Interacts with depth. |

### 2.1 D5 — the 1M-cell decision, explained

The grid needs dataset sizes 0.1 / 0.3 / 1 / 3 / 10M, **nested** (each a subset of the next) so that
differences come from size and not from which trajectories got drawn. Source pool = shards 000–009 =
10,000 trajectories × 1000 transitions.

Two constructions, **both of which are genuinely nested** — nesting is not the discriminator:

**Option A — global permutation.** Shuffle all 10,000 trajectory ids once under a fixed seed; take the
first 100 / 300 / 1,000 / 3,000 / 10,000. The 1M cell is then ~100 trajectories drawn from *each* of the
10 shards.

**Option B — shard-aligned.** 0.1M / 0.3M = first 100 / 300 trajectories of shard 000 (after a fixed-seed
shuffle *within* the shard); 1M = all of shard 000; 3M = shards 000–002; 10M = shards 000–009.

What actually differs:

| | Option A | Option B |
| --- | --- | --- |
| Nested? | yes | yes |
| 1M cell is a standard whole dataset? | no (10-shard mixture) | **yes** — enables a free apples-to-apples check against published 1M numbers, i.e. brief rule #2 "reproduce before modifying" |
| Works with HR's `--dataset_dir` + `--num_datasets k` | no — every cell needs a new materialized npz | **yes** for 1M/3M/10M with zero new code |
| Generation-seed spread | shard idiosyncrasies spread evenly across all cells | each size step adds exactly one generation seed; an anomalous shard enters at one size and persists in all larger cells |

The last row is the only real argument for A, and it is weak: shards are i.i.d. by construction
(`--seed=$i`, same scripted policy, same env), and there are 10 of them.

**Recommendation: Option B.** It buys a free pipeline-validation check at the 1M cell, costs no new files
for 1M/3M/10M, and the confound A guards against is negligible. The one thing to pin down: shuffle
*within* shard 000 under a fixed seed before taking the 0.1M/0.3M prefixes, rather than trusting file
order.

Caveat to record either way: the standalone `cube-double-play-v0.npz` (the standard 1M release) is **not**
necessarily byte-identical to shard 000 — both are 1M/1000 trajectories but likely from different
generation seeds. Verify before claiming an exact reproduction.

---

## 3. Open questions

**Blocking (answer changes what we build):**

- **Q1 — Which critic loss does the depth test use?** The 1000-layer depth result was obtained with
  InfoNCE + L2-distance logits + a logsumexp penalty (coeff 0.1). Both offline codebases use binary NCE +
  dot-product logits and **no** logsumexp penalty. If we drop the residual architecture into OGBench and
  keep binary NCE, we are not testing the 1000-layer recipe — we're testing residual blocks under a
  different objective, and a null result wouldn't be attributable. Options: (a) port InfoNCE + logsumexp
  into `crl.py` to match the online recipe exactly, (b) keep binary NCE to match the offline recipe and
  state the deviation, (c) run both at depths 4 and 32 as a 4-run pilot and let the data decide.
  Note the brief's own background paragraph describes InfoNCE + logsumexp, i.e. the **online** code, while
  proposing to build on offline code that does neither.
- **Q4 — Does the humanoid reversal survive `actor_p_randomgoal = 0.5`? (BACKLOG, deferred 2026-11-06.)**
  `phase1_repro` and `humanoid_ensemble1` differ in FOUR ways, not the two their YAML claims:
  dataset, `critic_ensemble` (2→1), `discount` (0.99→0.995) and **`actor_p_trajgoal/randomgoal`
  (0.5/0.5 → 1.0/0.0)**. The last two arrive silently through `use_published_flags`, which applies
  OGBench's published command for whichever dataset is named. Two of the four now have a dedicated
  experiment — `antmaze_ensemble1` (running) and `antmaze_navigate1` (running) — and this one does
  not. It may matter most: at `p_randomgoal = 0` the actor is never trained on an out-of-trajectory
  goal, which is exactly the stitching capability, and task 4 (the forced 180° detour) is at 0.00–0.26
  on humanoid at every depth while going 0.08 → 0.97 with depth on antmaze. 4 runs; the design is in
  `results/README.md` under "Backlog". Deferred by choice, not oversight.

- **Q2 — Representation dim (D8).** Three codebases, three values: 16 (stable-CRL, which explicitly found
  16 ≫ 128/512), 64 (scaling-crl), 512 (OGBench). Stable-CRL's finding was on *images*; whether it
  transfers to 37-dim states is unknown. A narrow bottleneck may also interact with depth, which is the
  axis under test. Suggest pinning one value across the whole grid and noting it, with a small ablation at
  one cell.
- **Q3 — Codebase confirmation with Chongyi (D6).** Also worth asking directly whether any internal
  stable-CRL-on-OGBench results exist (§1.6), and whether the online sanity check (phase 1a) is needed.

**Non-blocking:**

- Q4 — Do compute nodes have internet? (Pre-downloading makes this moot, but good to know.)
- ~~Q5 — Does `MUJOCO_GL=egl` rendering work on a Stellar GPU node?~~ **RESOLVED 2026-11-05: yes**, verified in the smoke test on `stellar-m01g2`. Videos are available; `video_episodes=0` stays the default.
- ~~Q6 — Evaluation interval and gradient-step cap.~~ **RESOLVED 2026-11-05**: see §1.7 "Training length". Revisit if phase-1 runs are still improving at their cap.

---

## 4. Current plan

### 4.0 What is running right now (2026-11-05, ~00:40)

| Job | What | State |
| --- | --- | --- |
| 2993241 | GPU smoke test (9 checks) | RUNNING on `stellar-m01g2`, checks 1–6 PASS |
| 2993251 | pack: CRL+GCBC depth 16, seeds 0–2 | RUNNING on `stellar-m01g3` (6 runs, 3/GPU) |
| 2993246, 2993247, 2993252 | the remaining 18 phase-1 runs | PENDING, gated `afterok:2993241` |

Phase 1 is 24 runs in 4 packs of 6; at most 2 packs run at once (QOS), so it self-sequences
in two waves. Nothing needs a live watcher.

**To check in the morning:**
```bash
cd /home/vt7993/rl-scaling
V=/scratch/gpfs/vt7993/rl-scaling/venv/bin/python
$V src/scripts/monitor.py --grid phase1_repro          # exit 2 = needs a human
$V src/scripts/plot_results.py --grid phase1_repro     # figures + csv tables
$V src/scripts/submit_packs.py --grid phase1_repro --chain   # requeue anything that died
```
Figures land in `experiments/analysis/phase1_repro/`.

### 4.1 Data collection plan — what every run records

Three layers, all machine-readable, so no analysis depends on scraping stdout.

**Per evaluation** (`eval.csv`, every 100k steps):
`eval/success` (mean over 5 tasks), `eval/task{1..5}/success`, `samples_seen`, `epochs`,
`eval_secs`. The per-task breakdown matters because the 5 goals differ in difficulty —
on cube-double, `task5_stack` is much harder than `task1_single_pnp`, so a flat mean can
hide the whole effect of depth.

**Per log interval** (`train.csv`, every 10k steps) — train **and** validation side by
side, which is what makes the overfitting probe (brief probe 7) free:
`train/contrastive_loss`, `train/categorical_accuracy` (InfoNCE top-1),
`train/binary_accuracy`, `train/logits_pos`, `train/logits_neg`, `train/v_mean|max|min`,
`train/actor_loss`, and `val/*` for all of them. Plus `samples_seen`, `epochs`,
`steps_per_s`, `peak_rss_gb`.

**Per run** (`summary.json`, once): `best_success` and `best_step`, `stop_reason`
(`plateau` vs `cap_reached` — if most runs say `cap_reached` the caps are too low and the
data-limited claim is unsupported), `gradient_steps`, `samples_seen`,
`epochs_over_data`, parameter counts per network, `wall_clock_s`, `train_secs`,
`data_load_secs`, `peak_rss_gb`, `devices`, `code_commit`, and the full `data_info`
(file list, rows, trajectories, **valid transitions**, ob/act dims, bytes in RAM).

Why each matters: `params` is `N` in the scaling fit (not depth); `epochs_over_data`
separates data-limited from compute-limited; `stop_reason` validates the protocol;
`data_info.n_transitions` pins `D` unambiguously; `code_commit` makes the run
reproducible.

**Phase 2 probes** (not yet implemented, next on the list) write to
`runs/<run_id>/probes/`, every one **binned by temporal distance** between state and
goal. Caveat already recorded: on antmaze-medium-stitch the maximum in-trajectory
distance is 200 while the eval horizon is 1000, so phase-1 probe bins cap at 200.

### 4.2 Evaluation plan

OGBench's standard protocol, unmodified: 5 goals per task, `task_id` 1–5, 15 episodes
each = **75 episodes per evaluation point**, deterministic actor (`temperature=0`).
Reported metric is the mean success over the 5 tasks, with the per-task breakdown kept.

Evaluation runs every 100k gradient steps, which serves three purposes at once: it drives
the plateau-stopping rule, it gives learning curves rather than a single endpoint, and it
identifies the best checkpoint that phases 2 and 5 need.

Two deliberate choices: **best** success over training is the headline number, not final
(the brief asks for best and its step, and offline runs can degrade late); and
**validation trajectories are a fixed set that never scales with training size**, so
probe numbers stay comparable across grid cells.

The phase-1 success criterion is not "CRL beats GCBC" — it is reproducing a **negative**
result. Expected: CRL roughly flat across depth 4→32, GCBC rising (rebuttal: 0.47→0.63
over depth 2→32). **If CRL does scale here, stop and report** — per the brief, that alone
changes the project.

### 4.3 Visuals plan

Five figures, each with a `.csv` beside it (static PNGs have no hover layer, so the table
is the accessibility channel). Colors are the dataviz reference palette, used unchanged.

| Figure | Form | The question it answers |
| --- | --- | --- |
| `fig1_success_vs_depth` | 2-series line, log₂ depth, SD error bars + per-seed dots | **The headline.** Does depth help, and does CRL differ from GCBC? The rebuttal's values appear as grey annotations, never as a series, so paper numbers are never mistaken for ours. |
| `fig2_learning_curves` | small multiples (one panel per agent), one line per depth from the ordinal blue/orange ramp, min–max band over seeds | Is a flat result real, or just a run stopped too early? Also shows whether deep runs are slower *per step* or genuinely worse. |
| `fig3_critic_diagnostics` | 3 panels, one axis each | Does the critic improve with depth even when success doesn't? That is the H-actor signature. Train-vs-val contrastive loss on the same axes is the H-data overfitting read. |
| `fig4_cost` | 3 panels: params, steps/s, wall-clock vs depth | Depth is never read without its price; also the input to sizing phases 3–4. |
| `fig5_depth_data_heatmap` | sequential one-hue heatmap, direct value labels | Phases 3–4: success over depth × dataset size, per algorithm. Auto-skipped when only one size is present. |

Design rules held throughout: **no dual axes** anywhere; sequential encodings are one hue
light→dark (never a rainbow); legend always present with direct labels for ≤4 series;
recessive hairline grid; text in ink tokens rather than series colors.

Still to build for phase 4: best-depth-per-dataset-size (the headline H-data plot) and the
parametric fit `Err(N,D) = E + A/N^α + B/D^β` with bootstrap-over-seed intervals. Both
need the grid to exist first.

### 4.4 Phase 0 — remaining steps

| Step | Action | Status |
| --- | --- | --- |
| 0.1 | venv at `/scratch/gpfs/vt7993/rl-scaling/venv`, Python 3.11, `jax[cuda12]==0.4.38`; GPU-validated by the smoke test | **done** |
| 0.2 | `src/scripts/download_datasets.py` | **done** |
| 0.3 | All planned downloads complete (4.36 GB) | **done** |
| 0.4 | Manifest script (`dataset_manifest.py`) — partly covered already: `datasets.py` records rows/trajs/transitions/dims per run | todo |
| 0.5 | `build_subsets.py` implementing D5 — only the 0.1M and 0.3M cells need materializing; 1M/3M/10M are whole shards | todo (D5 now settled) |
| 0.6 | Fix the held-out set: **constant across every grid cell**, never scaled with train. Candidates: shard 000's val (100 trajs / 100K transitions) or the union of 10 val shards (1,000 trajs / 1M) | todo |
| 0.7 | Coverage stats per subset: cube-double → cube xyz from `qpos` (distinct configurations + grid occupancy); antmaze → `ob[:2]` xy grid occupancy | todo |
| 0.8 | Write down `D` = **valid transitions**, not rows (§1.2), so the fit's x-axis is unambiguous | todo |
| 0.9 | Repo layout per the brief: `experiments/{data,configs,runs,analysis}` | todo |

**Environment downloads needed — final list.** Total under 4 GB; nothing in "ask first" territory.

| Phase | Dataset | Files | Size | Status |
| --- | --- | --- | --- | --- |
| 1 | `antmaze-medium-stitch-v0` | train + val | 270 MB | ✅ |
| 1 (opt.) | `antmaze-medium-navigate-v0`, `antmaze-large-navigate-v0` | train + val each | ~255 MB each | ✅ |
| 3–4 | `cube-double-play-100m-v0` **shards 000–009 only** | 10 train + 10 val | 3.27 GB | ✅ |
| D5 check | `cube-double-play-v0` (standard 1M release) | train + val | 312 MB | ✅ |
| backup | `puzzle-4x4-play-100m-v0` or `scene-play-100m-v0`, shards 000–009 | — | ~3 GB | deferred |

**All planned data downloaded 2026-11-05** — 28 files, 4.36 GB, at ~40 MB/s, into
`/scratch/gpfs/vt7993/ogbench/data`. Deferred set (the other 90 cube-double shards plus
backup tasks) is 35 GB and only needed for a 100M-scale final run.

The grid tops out at 10M, so **10 of 100 shards suffice** — not the full ~31 GB. Not needed at all:
`cube-octuple`, `puzzle-4x5`/`4x6`, `humanoidmaze-giant`, any `visual-*`, `experts.tar.gz`.

### 4.5 Code work, in dependency order

1. **Residual tower** (§1.5) as a drop-in alongside `MLP` in `utils/networks.py`, with `depth` and `width`
   flags, used by `GCBilinearValue`'s φ/ψ and by `GCActor`. Count parameters programmatically — `N` in
   the scaling fit is parameters, not depth.
2. **Stable-CRL patch** to `crl.py`: `latent_dim`, `batch_size`, `alpha`, cold final-layer init. Gate each
   behind a flag so they can be toggled one at a time (brief rule #3).
3. **Loss switch** in `crl.py` once Q1 is answered: binary-NCE (current) vs InfoNCE + L2 + logsumexp penalty.
4. **Dataset-path plumbing**: accept an explicit subset npz via `dataset_path=`, never auto-download.
5. **Logging** per brief: best success + its step, gradient steps, epochs over data, wall-clock, peak
   memory, parameter count, dataset subset id, code commit hash.

### 4.6 Slurm: the scheduling problem

The QOS cap is the binding constraint: **2 running jobs, 8 GPUs, 2 nodes**. The corner grid alone is
54 runs; the full grid is 150. Naive one-run-per-job means 2 runs at a time.

Three things make this tractable, in order of leverage:

1. **Pack many runs per job.** These networks are tiny — depth 64 × width 256 is ~12M parameters across
   both φ/ψ towers and the actor. A 40 GB A100 is nowhere near the limit. Run several processes per GPU
   with `XLA_PYTHON_CLIENT_MEM_FRACTION` set (JAX preallocates 75% by default — must override or the
   first process takes the card). Budget: 2 jobs × 4 GPUs × 3 runs/GPU ≈ **24 concurrent runs** at the
   QOS cap, versus 2 naively.
2. **Watch RAM, not VRAM.** Each process holds its own dataset copy. 10M cube-double with `qpos`/`qvel` is
   ~3.85 GB in RAM (~1.69 GB without), and `compact_dataset=True` avoids duplicating `next_observations`.
   12 processes × 3.85 GB ≈ 46 GB — fine on a 550 GB or 1000 GB node, but it's the real ceiling on packing
   density, so compute it per cell rather than assuming.
3. **Keep the queue topped up automatically.** A job array (`sbatch --array=0-53%2`) throttles to the
   2-job cap, so the array is the wrong unit — the *pack* is. Use a small submitter that maintains a
   run manifest (one row per grid cell × seed, with status), and on each invocation submits packs for
   pending runs up to the cap, with `--dependency=afterany` chains so a finishing job pulls the next pack.
   Driven either by a cron entry or by `/loop`. The 7-day wall limit is generous relative to 2–5 h runs,
   so prefer long jobs holding many sequential packs over many short jobs.

Set memory explicitly in every script — Stellar's 7500 MB/core default breaks on the AMD GPU nodes
(512/128 = 4 GB/core, so `--mem-per-cpu=4000M`).

**Time one depth-64 × 10M run before launching anything** (brief's instruction) and multiply out.

### 4.7 Immediate next actions

1. **Morning:** read `monitor.py` output; if phase 1 is clean, plot and compare against
   the rebuttal's GCBC curve.
2. Answer **Q1** (critic loss form) — the `q1_loss_pilot` grid is 4 runs and already
   configured; it is the cheapest way to settle whether our depth test is even testing
   the 1000-layer recipe.
3. Answer **Q2** (representation dim: 16 vs 64 vs 512).
4. Build `build_subsets.py` + `dataset_manifest.py` + coverage stats — the remaining
   phase-0 deliverables, needed before phase 3, not before phase 1.
5. Implement the phase-2 probes against the `params_best.pkl` checkpoints phase 1 is
   producing now.

---

## 5. Experiment log

Append one entry per run batch. Template:

**Scoring convention — report all three, they answer different questions:**
`reeval@50` (best checkpoint re-scored at 50 episodes/task — the number comparable to
OGBench's published table), `last3_eval_mean` (mean of the final 3 training evaluations —
the convention the paper scores with), and `best_success` (best single training eval —
optimistic, since it is a max over ~20 noisy points).

```
### E<n> — <short name>  ·  <date>
**Scores:** reeval@50 = · last3 = · train best =
**Question:**
**Config:** agent / depth / width / dataset + subset id / seeds / loss form / repr dim / batch / lr
**Commit:**
**Compute:** GPUs, wall-clock, gradient steps, samples seen, parameter count, peak memory
**Result:** best success + step, per-task breakdown
**Probes:**
**Read:** what this says about H-data / H-actor / H-horizon
**Next:**
```

### E1 — phase1_anchor: reproduce OGBench's published numbers  ·  2026-11-05
**Scores:** reeval@50 = PENDING (job 2993575, queued) · last3 = CRL 45.5 ± 8.1, GCBC 37.2 ± 11.4 · train best = CRL 52.9, GCBC 54.2
**Question:** does our harness reproduce OGBench's antmaze-medium-stitch table (CRL 53 ± 6, GCBC 45 ± 11)?
**Config:** crl / gcbc · stock OGBench MLP 3×512 GELU · `repr_dim`/`latent_dim` 512 · antmaze-medium-stitch-v0 native 1M · seeds 0,1,2 · published flags (`actor_p_trajgoal=0.5`, `actor_p_randomgoal=0.5`, `alpha=0.1`) · batch 1024 · Adam constant lr 3e-4
**Commit:** configs generated at `6e01598`'s tree, `code_commit` recorded as `0009db4` (tree was dirty; we do not record a dirty flag — see defect below)
**Compute:** 1 GPU/run, 2.00 h (CRL) / 0.59 h (GCBC), 1M gradient steps, 1.024e9 samples, 3,793,416 (CRL) / 559,624 (GCBC) params, peak RSS 1.7 GB
**Result:** both ~8 points below published, intervals overlapping. With 3 seeds and SD ≈ 10 pts (SE ≈ 5–6) this cannot resolve an 8-point gap. Per-task: **task 4 is 0.00–0.01 for both agents at every seed**, capping the aggregate at 0.80.
**Probes:** none yet.
**Read:** the published ~0.53 for CRL is *consistent with task 4 being unsolved in the original runs too*. Neither confirms nor refutes the reproduction; the 50-episode re-eval is the deciding measurement.
**Next:** run 2993575. Add a dirty-tree flag to `code_commit`. Fix the `stop_reason: "plateau"` mislabel when `plateau_min_steps == step_cap`.

### E2 — phase1_repro: depth sweep with the residual tower  ·  2026-11-05 (partial)
**Scores:** reeval@50 = PENDING · last3 @ 1M = CRL 34.1 / 49.3 / **74.2** (d4/8/16), GCBC 53.5 / 65.2 / 62.5 / 61.2 (d4/8/16/32) · train best = see `fig1_success_vs_depth.csv`
**Question:** does depth help offline GCRL on a stitching dataset, and does it help CRL and GCBC alike?
**Config:** residual tower, `block_size` 4, width 256, `repr_dim` 64, LayerNorm + Swish, `lecun_uniform` (`variance_scaling(1/3,'fan_in','uniform')`) · depths 4/8/16/32 · seeds 0,1,2 · published flags · batch 1024 · Adam constant lr 3e-4 · cap 2M, plateau min 1M
**Commit:** `0009db4` (d4) / `53177a7` (d8+) — same caveat on dirty trees
**Compute:** 16 of 24 runs finished, 41.2 GPU-h total incl. the anchor. Throughput 151 (CRL d4) → 86 (CRL d8) → 98 (GCBC d32) steps/s.
**Result:** **CRL scales and does not saturate by depth 16** (34 → 49 → 74, a 40-pt swing). **GCBC saturates at depth 8** (53.5 → 65.2, then flat). Per-task: depth unlocks **task 4 for CRL only** (0.13 → 0.49 → 0.80); GCBC is 0.00 on task 4 at every depth and seed. CRL and GCBC solve complementary task subsets — GCBC owns task 1 (0.90–0.95), CRL owns tasks 3 and 4.
**Probes:** critic diagnostics at a common 1M steps — InfoNCE top-1 acc 0.369 → 0.467 → 0.583; train contrastive loss falls 0.00387 → 0.00269; **validation contrastive loss rises 0.00530 → 0.00632**.
**Read:** the depth benefit is specific to the contrastive critic, not to "deeper nets on this data" — a deeper BC policy plateaus, a deeper value function does not. Held-out contrastive loss is **anti-correlated** with success here, so it must not be used for model selection or early stopping. Supports H-horizon: the task depth unlocks is the one requiring stitching.
**Next:** CRL d16/d32 and GCBC d32 still training. Extend the CRL sweep past 16. Depth and parameter count are confounded (width fixed) — a width-matched control is not in the grid. Q2 (`repr_dim`) now blocks interpretation of §3's architecture inversion, not just configuration.

### E2a — phase1_repro update: depths 4/8/16 complete  ·  2026-11-05 13:45
**Scores:** last3 @ 1M = CRL **34.1 / 49.3 / 71.4** (d4/8/16, 3 seeds each, all finished),
GCBC 53.5 / 65.2 / 62.5 / 61.2 (d4/8/16/32, all finished). CRL d32 at ~575k of 2M steps.
**Result:** the CRL trend survives completion and the seed spread *tightens* with depth
(SD 9.8 → 3.8 → 1.9). Critic diagnostics at a common 1M steps: InfoNCE top-1 0.369 →
0.467 → 0.607; train contrastive loss 0.00387 → 0.00258; **val** contrastive loss
0.00530 → 0.00642 — the train/validation decoupling holds. Per-task, task 4 goes
0.13 → 0.49 → 0.76 for CRL, and 0.93 at depth 32 (partial). GCBC reaches 0.12 on task 4
only at depth 32, its first non-zero.
**Correction:** the earlier figure of CRL d16 = 74.2 ± 5.1 was computed when two of its
three seeds were below 1M steps; `last3_mean` filtered evaluations to `step ≤ 1M` but
never checked the run had *reached* 1M. Fixed. Correct value **71.4 ± 1.9**. The same
bug had briefly reported CRL d32 as 0.781 from its 300–500k evaluations.
**Compute:** 72.0 GPU-hours over 27 finished runs. CRL d16 is 5,424,648 params, 49
steps/s, 7.5 h/run.
**Next:** CRL d32 finishes in ~5 h for its 1M value. `reeval@50` (job 2993575) running.

### E3 — reeval@50 for both phase-1 grids  ·  2026-11-05 (job 2993575, COMPLETED 1h27m)
**Scores (reeval@50, best checkpoint, 250 episodes/point — the headline convention):**

| | reeval@50 | last-3 | published |
|---|---|---|---|
| anchor CRL d4 (stock MLP) | **47.5 ± 7.0** | 45.5 ± 8.1 | 53 ± 6 |
| anchor GCBC d4 (stock MLP) | **52.0 ± 5.8** | 37.2 ± 11.4 | 45 ± 11 |
| repro CRL d4 / d8 / d16 | **34.3 / 54.7 / 78.4** | 33.0 / 49.9 / 69.0 | — |
| repro GCBC d4 / d8 / d16 / d32 | **57.3 / 66.3 / 65.5 / 64.4** | 46.5 / 61.9 / 62.4 / 62.1 | — |

**Read:** the anchor reproduction is **clean** at the published episode count — both
agents inside the published bands. The earlier ~8-point shortfall was the 15-episode
evaluator, not the harness (GCBC moves 37.2 → 52.0). General lesson: a 15-episode
evaluator cannot resolve differences under ~10 points.

The CRL depth trend strengthens at 50 episodes: 34.3 → 54.7 → 78.4. GCBC saturates at
depth 8.

**SCOPING CORRECTION (user, 2026-11-05).** These are **not** publishable results and
the "we contradict §4.6" framing was too strong. The paper's Figure 14 already reports
offline **GCBC** scaling with depth on `antmaze-medium-stitch` specifically. We found
CRL scaling on that same dataset — adjacent to a known positive, not against it. The
defensible claim is *"the known antmaze-stitch depth effect is much larger for CRL than
for GCBC"*, and even that rests on one environment. §4.6's negative is about offline CRL
in general; one favourable dataset does not overturn it.
**Next:** other environments, starting with `humanoid_ensemble1`.

### E4 — humanoid_ensemble1  ·  submitted 2026-11-05, jobs 2993926 / 2993927
**Question:** does §E2's depth effect survive a change of environment, or is it about
`antmaze-medium-stitch`?
**Config:** `humanoidmaze-medium-navigate-v0` (ob 69, act 21, **2000**-step episodes),
CRL only, depths 4/8/16/32, **2 seeds**, `critic_ensemble=1` (newly implemented),
binary NCE + dot, repr_dim 64, residual width 256, published flags incl.
**discount 0.995**. 8 runs, ~71 GPU-h estimated, ~12 h wall.
**Why this env:** same 8×8 maze and the same five (start, goal) cell pairs as
antmaze-medium, so research log §1.11's first-move-angle analysis transfers and task 4
is still the forced 180° detour. Two things change at once, deliberately: the body
(humanoid), and `-navigate` instead of `-stitch` — 1001-step trajectories, so far less
stitching is needed. A null here would be informative either way.
**Next after this:** mechanistic interpretability on the depth ladder of critics —
probe `ψ(g)` for maze cell / BFS distance, check whether `φ·ψ` is ordered like true
distance, and look at per-block representational change. See `results/antmaze_stitch__depth/README.md` §10.

**Full write-up with figures and tables: [`results/antmaze_stitch__depth/README.md`](../../results/antmaze_stitch__depth/README.md).**

### E5 — antmaze_ensemble1: is the depth result an artefact of the ensemble?  · 2026-11-06 (partial, job 2995193)
**Scores (depth 4, final, n=3):** reeval@50 = **38.9 ± 11.3** · last3@1M = 28.0 ± 10.8 ·
train best = 43.6 ± 10.1. Depth 32, stopped at 755k steps: `best_success` 72.9 ± 5.1
(n=3), reeval@50 **71.6** (n=1, seed 0 only — see below).
**Question:** `humanoid_ensemble1` moved four things at once; this moves exactly one of
them. Did `phase1_repro`'s depth result depend on having two critics?
**Config:** identical to `phase1_repro` except `critic_ensemble: 2 → 1`, CRL only (GCBC
has no critic), depths {4, 32} only. Verified one-knob by diffing the two grids'
*resolved* `config.json`, not the YAML — the lesson from E4.
**Compute:** 6 runs on 2 A100s at 3 runs/GPU; d4 ~2 h, d32 ~17 h.
**Result — no, the ensemble is not the cause.** Compared at a matched 735k gradient
steps (the slowest in-flight d32 seed's position, because comparing its current best
against `phase1_repro`'s *final* best would understate it):

| | d4 | d32 | depth gain |
|---|---|---|---|
| `critic_ensemble: 2` | 39.1 ± 10.9 | 86.2 ± 2.5 | **+47.1** |
| `critic_ensemble: 1` | 32.9 ± 12.0 | 72.9 ± 5.1 | **+40.0** |

+40 points of depth gain survives with no pessimism anywhere in the objective. The
ensemble buys a ~13-point level shift at d32 and nothing at d4, where all three metrics
disagree in sign and all sit inside the seed spread (+0.9, −6.1, +3.4).
**Two things that did not go as predicted:**
1. **Peak-then-drift is not a no-pessimism symptom.** The obvious story — without a
   `min` over two critics the actor exploits the deep critic's errors and comes apart
   — predicts the decay should appear here. It is present in *both* grids and
   `phase1_repro` does it harder: −11.1 points peak→final at d32 against −7.1 here.
   It is a property of deep antmaze runs at either ensemble size.
2. **`categorical_accuracy` falls 0.369 → 0.320 at d4 while the contrastive loss is
   identical to three significant figures**, train (0.00387) and validation (0.00530 vs
   0.00535) alike. That is a measurement effect, not a worse representation: the metric
   is computed on ensemble-reduced logits, so two critics at equal per-critic loss
   retrieve better than one by cancelling independent logit noise.
**Read:** one of E4's four confounds is closed. `phase1_repro`'s headline stands as
written. Still open: dataset (E6), `actor_p_randomgoal` (§3 Q4), `discount`.
**Stopped 2026-11-06 at 38% of d32 (scancel 2995193).** The three d32 seeds peaked at
300–400k and had been flat-to-declining for 400k steps, so the remaining ~10 h bought
confirmation rather than a result, and the GPUs were wanted for E7. `params_best.pkl`
survived for all three seeds and was rescored at 50 episodes/task afterwards, so the
d32 cell has a comparable headline number — read as *best checkpoint of a run stopped
at 755k steps*, not as a converged d32. (`reeval.py` gained a one-line change for this:
the `summary.json` guard now applies only to a `--grid` sweep, not to an explicitly
named `--run-dir`.) **That rescore was itself stopped after one seed**: seed 0 came back
at 71.6 against the three-seed training-eval mean of 72.9, i.e. the 75 → 250 episode
rescore moves the number by about a point, so seeds 1 and 2 would have confirmed rather
than informed and the slot was wanted for E7's second pack. Checkpoints persist, so it
can be completed whenever.

**Write-up: [`results/antmaze_stitch__depth_ensemble1/README.md`](../../results/antmaze_stitch__depth_ensemble1/README.md).**

### E6 — antmaze_navigate1: is the reversal about the data?  · 2026-11-06 (partial, jobs 2995262 / 2995263)
**Scores (n=3 each):** d4 `best_success` **99.6 ± 0.6**, `last3@1M` 97.2 ± 1.7 ·
d8 `best_success` 98.7 ± 1.1, `last3@1M` 96.9 ± 0.0. d16/d32 not started.
**Question:** `humanoidmaze-medium-navigate` has 1000 trajectories × 2001 steps against
antmaze-stitch's 5000 × 201. `antmaze-medium-navigate` is 1000 × 1001 — humanoid's
data structure with the ant's 29-d observation. Is the reversal about few long
trajectories rather than the body?
**Config:** identical to `phase1_repro` except `dataset`. `actor_p_trajgoal/randomgoal`
pinned to 0.5/0.5 in the YAML, because `use_published_flags` sets 1.0/0.0 for this
dataset and would otherwise have made it a two-knob experiment.
**Result so far — the grid cannot answer the question as posed.** Pack 1 finished and
**navigate is solved at depth 4**: 99.6 against stitch d4's 42.7 on the same metric. All
six runs stopped on the plateau rule at 1.0–1.4M steps. With d4 at the ceiling there is
no headroom for depth to *help*, so "depth still scales → the data is not the cause" is
not available here.
**Read:** what remains is a **one-sided test for harm**, which still earns the d16/d32
runs. If deep networks fall off the ceiling on ant + navigate data, that is the humanoid
signature without the humanoid body, and the data structure is implicated. If they hold
at ~99, this grid is uninformative and the dataset confound needs a harder variant
(`antmaze-large-navigate`) instead.
**Also worth recording on its own:** stitch d4 42.7 vs navigate d4 99.6 on identical
agent, architecture, body and maze. The difference between the two OGBench subsets is
larger than anything depth does in either grid.
**Stopped 2026-11-06 before pack 2 ran (scancel 2995263).** d16/d32 would have cost
~70 GPU-h for a one-sided test on the grid with the least headroom. Closed at 6 of 12
runs by choice; d4 and d8 stand as a finished result. E7 attacks the same question
from the humanoid side instead, where there is room to move in both directions.

### E7 — humanoid_stitch1: the decisive cell of the body × data square  · submitted 2026-11-06, jobs 2995683 / 2995684
**Question:** is the `humanoid_ensemble1` reversal about the humanoid BODY or about
`-navigate` DATA? E6 tried this from the ant side and hit a ceiling. This tries it from
the humanoid side, where d4 = 62.0 and d32 = 53.0 leave headroom in both directions.
**Config:** `humanoidmaze-medium-stitch-v0`, CRL, depths 4/8/16/32, 2 seeds,
`critic_ensemble: 1`, published flags applied (alpha 0.1, discount 0.995,
`actor_p_trajgoal/randomgoal` 0.5/0.5). 8 runs, 71 GPU-h, ~12 h wall.
**The fact that makes this cell worth more than E6's:** OGBench ties the goal split to
the dataset TYPE, not the body. `hyperparameters.sh` publishes `0.5/0.5` for both
`-stitch` sets (L166 antmaze, L296 humanoidmaze) and `1.0/0.0` for both `-navigate`
sets (L257). So this one grid moves the dataset *and* the goal split together — the two
remaining live confounds — while holding body, `discount` and `critic_ensemble` fixed
against E4. It is a weaker isolation than two grids, and it is 8 runs instead of 12.
`discount` stays uncovered because OGBench ties *that* to the body: 0.99 for every
antmaze set, 0.995 for every humanoidmaze set.

| | trajs × len | ob | goal split | discount | depth does |
|---|---|---|---|---|---|
| antmaze-medium-stitch (E2) | 5000 × 201 | 29 | 0.5 / 0.5 | 0.99 | **helps** 35.5 → 86.7 |
| antmaze-medium-navigate (E6) | 1000 × 1001 | 29 | 1.0 / 0.0 | 0.99 | ceiling at d4 (99.6) |
| humanoidmaze-med-navigate (E4) | 1000 × 2001 | 69 | 1.0 / 0.0 | 0.995 | **hurts** 62.0 → 53.0 |
| **humanoidmaze-med-stitch (E7)** | **5000 × 401** | **69** | **0.5 / 0.5** | **0.995** | **?** |

**Dataset verified before submission**, against its own npz rather than assumed from
the generation flags: 5000 × 401 train, 500 × 401 val, ob 69, act 21, 2,005,000 rows.
Added to `VERIFIED_DATASETS`. Generation command is `commands.sh` L60
(`--num_episodes=5000 --max_episode_steps=401`).
**Read:** depth helps → the body is cleared, the reversal is about `-navigate` data.
Depth still hurts → the body is implicated and the question becomes what a 69-d
observation and 21-d action do to contrastive depth scaling. Either way **task 4**
settles §3 Q4 on its own: it never exceeded 0.26 on humanoid at any depth, and this is
the first humanoid grid whose actor ever sees an out-of-trajectory goal.
**Noted while setting this up, not acted on:** `DATASETS["humanoidmaze-medium-navigate-v0"]`
records `traj_len=1001`, but the file is **2001** (verified: 2,001,000 rows / 1000
trajectories). The runs were unaffected — `mode: native` loads the whole file and
`data_info` records the truth read from disk — but `transitions_per_shard` is wrong by
2× for that entry. Not corrected here because `size` feeds the run id, so the fix
would re-key and orphan all 8 E4 runs. Needs a pinned `size` in the grid spec first.

### E8 — actor extraction: separating critic depth from policy depth  · built 2026-11-06, not yet submitted
**Question:** every grid so far moves ONE `depth` knob that scales the critic *and* the
policy, because `patches.ResidualMLPShim` stands in for OGBench's `MLP` at both call
sites — `GCBilinearValue` (phi/psi) and `GCActor` (the policy trunk). Read from
`phase1_repro`'s own configs: d4 is critic 1,163,520 / actor 282,888, d32 is
8,589,568 / 2,139,400 — **the actor grew 7.6× too**. So "antmaze-stitch goes
35.5 → 86.7" is a claim about scaling both networks together, and every headline in
`results/` inherits that ambiguity. GCBC gives partial leverage (actor-only, saturates
at d8) but it is not the same experiment.
**Method:** freeze a trained critic, re-initialise the actor at a pinned `actor_depth`,
train the actor loss alone. No critic is retrained.
**Why reordering is legitimate — verified in `lib/ogbench/impls/agents/crl.py`, not
assumed:**
1. The actor loss never writes to the critic. `actor_loss` L111 calls
   `self.network.select('critic')(...)` with **no** `params=grad_params`, and
   `TrainState.__call__` documents a missing `params` as stop-gradient. Critic
   gradients therefore come only from `contrastive_loss`.
2. The critic never depends on the actor. `contrastive_loss` reads `batch['actions']`
   (the dataset's) and `batch['value_goals']` (future states on the same trajectory).
   No bootstrap on pi, so the critic converges to the behaviour policy's value
   whatever the actor does: `f*(s,a,g) = log[p^beta(g|s,a)/p(g)] + const`.

Joint training differs only in early dynamics — the actor chasing a critic that is
still moving. This is the one-step view of offline RL (Brandfonbrener et al. 2021), and
the decoupling Park et al. 2024 used to separate value learning from policy extraction.
3. `ddpgbc` normalises by its own scale, `-q.mean() / stop_gradient(|q|.mean() + 1e-6)`
   (L115). `interp__critic_depth` found Q's level is arbitrary per run, so this matters:
   every critic presents the actor with the same effective objective, which is what
   makes extraction comparable across depths and environments.

**The failure this is built around.** Dropping the critic loss is not sufficient if the
saved `opt_state` is restored with it — verified present in every `params_best.pkl`
alongside `params` — because Adam's stored momentum keeps integrating and the "frozen"
critic drifts. `offline/extract.py` never reads `opt_state`, labels the critic subtree
`optax.set_to_zero`, and **hashes every critic array before and after training**; a
mismatch raises and the run fails rather than reporting a number.
**Verified end to end before any submission**, not just in the unit self-test: a real
`phase1_repro` d32 checkpoint (8,589,568 critic params) grafted into an agent with a
d4 actor (282,888), 10 real `agent.update` calls, fingerprint `b5c2b9dac263e50a`
unchanged, actor moved, and the logged loss keys are actor-only.
**Grids — one per environment, all `actor_depth: 4`, critic ladder unchanged:**

| experiment | source grid | critic depths | seeds | sources resolvable |
|---|---|---|---|---|
| `extract_antmaze_stitch` | `phase1_repro` | 4/8/16/32 | 3 | 12 / 12 |
| `extract_humanoid_navigate` | `humanoid_ensemble1` | 4/8/16/32 | 2 | 8 / 8 |
| `extract_antmaze_navigate` | `antmaze_navigate1` | 4/8 | 3 | 6 / 6 |
| `extract_humanoid_stitch` | `humanoid_stitch1` | 4/8/16/32 | 2 | blocked on E7 |

**Read:** success still climbing d4 → d32 with a fixed 4-layer policy means the gain is
the **critic's**. A flattening means much of it was the deeper policy and the headline
needs restating. The d4 row is a free control — d4 critic with d4 actor is
`phase1_repro`'s d4 cell in two stages, so any gap there is the two-stage-vs-joint
effect and calibrates every other row. `extract_antmaze_navigate` is a negative
control: at a d4 ceiling of 99.6 there is no headroom, so if extraction does not hold
~99 there, the procedure is losing something and the other three readings are suspect.
**Not built:** the converse sweep (critic pinned at 32, actor 4→32) would isolate the
actor the same way. `actor_depth` is not in `SWEEPABLE`, and adding it needs a
`VISIBLE_AXES` entry so `plot_results` does not silently average two actor depths into
one point.

---

## 6. Changelog

- **2026-11-05 (later)** — Acted on review feedback, all six items. Found and fixed a
  real configuration bug: CRL was using `crl.py`'s bare defaults
  (`actor_p_trajgoal=1.0`) rather than OGBench's published antmaze-medium-stitch command
  (`0.5/0.5`, `hyperparameters.sh` L166) — material on a `stitch` dataset, where
  stitching is the point. Cancelled the 12 in-flight runs, added `PUBLISHED_FLAGS`,
  regenerated all configs. Added the `phase1_anchor` grid (stock OGBench MLP, published
  flags, 1M steps) as the reproduction gate. Plateau rule now needs 1M steps minimum and
  tests a 3-eval moving average. Added `reeval.py` (50 eps/task post hoc) and
  `goal_distribution.py`. Verified resolved network shapes from a real checkpoint (§1.10)
  and measured the goal-sampling pile-up (§1.8).
- **2026-11-05** — D5 settled as shard-aligned; D6 (OGBench impls + patches), D9 (plain
  Adam) and D10 (Slurm shape) settled. Q5 and Q6 resolved. Downloaded all planned data
  (4.36 GB). Built and GPU-verified the full training stack — residual tower with measured
  parameter counts, agent injection, dataset resolution, plateau-stopped training loop,
  pack-based Slurm scheduling, monitor and five figures (see `impl_log.md` I1–I13).
  Launched phase 1 (24 runs, 4 chained packs) behind a 9-check GPU smoke test. Discovered
  and encoded two cluster blockers: the 8-GPU node is down and the maintenance window
  forbids long jobs. Corrected the brief's claim that the 1000-layer paper uses
  stable-CRL's weight init — it uses plain `lecun_uniform` throughout; only LayerNorm is
  shared, and even then with opposite ordering relative to the activation.
- **2026-11-04** — Log created. Verified cluster limits, dataset structure and sizes, `qpos`/`qvel`
  semantics, OGBench/stable-CRL/scaling-crl architecture and loss differences. Established that OGBench
  does not implement stable-CRL and that no stable-CRL results exist on OGBench. Downloaded
  `antmaze-medium-stitch-v0` and `cube-double-play-100m-v0` shard 000. Settled D1–D3; opened Q1, D5, D8.
