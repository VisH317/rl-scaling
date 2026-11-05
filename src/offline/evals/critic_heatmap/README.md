# `critic_heatmap` — can the critic represent the maze?

**Status:** ready · **Applies to:** CRL · **Reads:** `critic`, `actor` · **Cost:** ~3 min/checkpoint

## The question

Depth helps the offline CRL critic a lot on `antmaze-medium-stitch`: success goes
34.3 → 54.7 → 78.4 (reeval@50, depths 4/8/16). The policy is extracted by DDPG+BC, so
all the depth signal arrives through `Q(s, a, g)`. **What about `Q` gets better?**

A value function on a maze has one property it cannot fake: the goal is reachable only
around the walls, so value must decay along the *maze* path, not along the straight
line. This eval measures whether it does, and how that changes with depth.

## What is computed

For every task goal `g` and every spatial bin of the maze,

```
Q(s, π(s,g), g)      s = real states from the dataset that fall in that bin
```

reduced over the critic ensemble with `min`, because that is what the DDPG+BC actor
maximises (`q = min(q1, q2)` in `crl.py`). Two reference fields are measured at the
same states: the **dataset's own action** at that state, and (optionally) a **uniform
random action**. `advantage = policy − dataset` isolates what the policy's choice buys.

### States are real, never synthesised

The obvious way to get a smooth field is to take one reference ant pose and overwrite
dims 0:2 with grid coordinates. That is wrong here: the observation is 29-dim
(x, y, 13 more `qpos`, 14 `qvel`), and a pose/velocity vector pasted onto an arbitrary
`(x, y)` is a state no ant can be in. Off the data manifold a network's output is
unconstrained, so the pretty field would be measuring extrapolation, not value.

Instead, bins are filled from the held-out shard (500 trajectories, 100,500 states).
At `cell_subdiv: 4` the maze has 416 free bins of 1.0 × 1.0 world units, so coverage is
~240 states/bin on average — ample. Bins below `min_states_per_bin` are **masked**:
excluded from every metric and drawn as hatching, not interpolated. Interpolation is a
display option only (`--interpolate` on the figure script), it never enters a number,
and a panel drawn with it says so.

### The goal is a distribution, not a point

`env.reset(options={'task_id': t})` builds the goal observation by stepping a few random
actions and teleporting the body to the goal cell, and `add_noise_to_goal=True` shifts
the goal `xy` by up to ±`unit/4` = ±1.0. So `ψ(g)` differs across resets in both pose
and position. `goal_samples: 8` draws are averaged, and every draw's `xy` is recorded in
`meta.json`. A field computed from a single reset is a field of that reset.

## The metrics

| metric | units | what it means |
|---|---|---|
| `rho_geodesic` | ρ ∈ [−1, 1] | Spearman of `Q` against **−geodesic** distance to the goal. BFS on the bin grid, in world units. |
| `rho_euclid` | ρ | the same against **−straight-line** distance. |
| **`topology_gain`** | Δρ | `rho_geodesic − rho_euclid`. **> 0 only if the field bends around walls.** The headline number. |
| `peak_geodesic` | world units | how far the single highest-value bin is from the goal. 0 is perfect. |
| `top5_geodesic` | world units | mean distance of the top 5 % of bins — robust version of the above. |
| `path_monotone` | fraction | steps along one shortest path from the task's start to its goal where `Q` strictly increases. What an actor hill-climbing on `Q` needs. |
| `goal_percentile` | fraction | percentile rank of the goal bin's own value. 1.0 = the goal bin is the maximum. |
| `coverage` | fraction | free bins that cleared `min_states_per_bin`. Identical across depths by construction — it is a property of the data, and a changed value means the state sample changed. |
| `q_p2`, `q_p98`, `q_mean`, `q_std` | logits | the raw scale. `Q = φᵗψ/√d` is an unnormalised log-density-ratio, so **magnitudes are not comparable across runs** — only the ranking is. Figures normalise per panel for exactly this reason. |
| `ens_gap` | logits | mean \|q1 − q2\| per bin. Large relative to `q_std` means the `min` is doing real work. |

`topology_gain` is signed and bounded, and the self-check pins both ends: a synthetic
field `Q = −geodesic` must score `+`, and `Q = −euclid` must score `−`.

```bash
# the metrics, checked against fields whose answers are known. No GPU, no checkpoint.
$V -m offline.evals.critic_heatmap.impl
```

## Output

`<ANALYSIS_DIR>/evals/<suite>/critic_heatmap/<run_id>/`

| file | contents |
|---|---|
| `field.npz` | `q_policy`, `q_dataset`, `advantage`, `q_*_ens_mean`, `q_*_dispersion_sd`, `ens_gap`, `geodesic`, `euclid` — each `(5, Hf, Wf)` — plus `maze`, `fine`, `count`, `occupied` |
| `metrics.json` | every metric above, per field, per task, plus the across-task mean |
| `meta.json` | run id, depth, seed, param counts, selection rule and the whole seed set, the state files used, every goal `xy` drawn, timing |

`q_*_dispersion_sd` is the spread of `Q` within a bin across both the states in it and
the goal draws — a large value says the bin's single number is hiding structure.

## How to read it

- **`topology_gain` rising with depth** is the result this eval exists to find: deeper
  critics encode maze connectivity, which is what stitching needs.
- **`topology_gain` flat while success rises** means depth is buying something else, and
  the next place to look is the actor's conditioning, not the critic's geometry.
- **`peak_geodesic` → 0 with depth** would say the shallow critic's value field peaks in
  the wrong room — a specific, visible failure.
- A field that is high-value *through* a wall is the picture of a critic that learned
  `euclid`. Task 4 is the case to look at: its start `(6,5)` is a dead end whose only
  exit points 180° away from the goal, and GCBC scores 0.00 on it.

## Caveats

1. **One seed per depth.** The seed is picked by `reeval@50` and the whole seed set is
   recorded in `meta.json`. At depth 4 the spread is 0.24 (0.208 / 0.376 / 0.444) — wider
   than some depth gaps — so a per-depth field is as much a property of that seed as of
   depth until `seed_select: all` is run.
2. **`Q` scale is arbitrary per run.** Never compare raw values across panels; compare
   ranks and shapes. Binary NCE fixes no overall constant.
3. **The ensemble `min` is not the only reading.** `q_*_ens_mean` is stored alongside;
   if the two disagree about the field's shape, `ens_gap` says where.
4. **This is not a causal test.** A critic can rank bins correctly and still extract a
   bad policy. The heatmap explains the critic; it does not explain the success rate.
