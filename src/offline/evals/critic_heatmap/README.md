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

Notation: `b` ranges over usable bins (free, with at least `min_states_per_bin` real
states); `Q_b` is the mean of `Q(s, π(s,g), g)` over that bin's states and the goal
draws; `d_b` is the geodesic (maze-path) distance from bin `b` to the goal in world
units; `e_b` is the straight-line distance; `ρ` is Spearman rank correlation.

### `rho_geodesic` — ρ(Q, −d) · range [−1, 1] · higher is better

Does the value field decrease as you get further from the goal *along the maze*? Rank
correlation, not Pearson, because `Q` is an unnormalised log-ratio with no reason to be
linear in distance — only the ordering is meaningful. **+1** = the field ranks every bin
exactly by how far it is from the goal. **0** = no spatial information. **Negative** =
the field is *anti*-correlated with distance, which would be a broken critic, not a
weak one.

Ties matter here, which is why distance is computed by BFS on the **bin** grid (416 free
bins) rather than the cell grid (26 cells): cell distance takes ~8 distinct values, so
most bin pairs would be ties and ρ would be compressed toward 0 for every run equally.

### `rho_euclid` — ρ(Q, −e) · range [−1, 1]

The same against straight-line distance, **ignoring walls**. Not a quality measure on
its own: on an open arena the two would be identical. Its job is to be subtracted.

### `topology_gain` — `rho_geodesic − rho_euclid` · the headline

> **The one number this eval exists to produce.**

`e` is available from the observation almost for free: `(x, y)` is dims 0:2, the goal's
`(x, y)` is dims 0:2 of `g`, and a bilinear form can approximate `−‖x − x_g‖` without
knowing anything about the maze. `d` requires knowing where the walls are. So the
*difference* isolates the part of the ranking that cannot be explained by "subtract the
coordinates".

- **`> 0`** — the field bends around walls. The critic has maze structure in it.
- **`≈ 0`** — indistinguishable; the maze may simply not separate the two distances much
  for that goal (check per-task, not just the mean).
- **`< 0`** — the field tracks the straight line *better* than the maze path: a critic
  that assigns high value across a wall. This is the specific failure that breaks
  stitching, because the actor hill-climbs into the wall.

Signed, bounded, and pinned at both ends by the self-check: a synthetic `Q = −d` must
score positive and `Q = −e` must score negative. Magnitudes are small by construction
(both ρ are large and similar), so read the *sign and trend across depth*, not the
absolute size. Observed on the real depth-4 checkpoint: **−0.044**.

### `peak_geodesic` — world units · 0 is perfect

Geodesic distance from the goal to the single highest-value bin. Answers "if the actor
teleported to the argmax of `Q`, where would it land?". A large value is a specific,
visible failure: the field peaks in the wrong room. Sensitive to one outlier bin by
construction — that is the point; `top5_geodesic` is the robust companion.

### `top5_geodesic` — world units · 0 is perfect

Mean geodesic distance over the top 5 % of bins by value. Immune to a single noisy bin.
If `peak_geodesic` is large and `top5_geodesic` is small, the peak is noise; if both are
large, the field genuinely peaks in the wrong place.

### `path_monotone` — fraction in [0, 1] · higher is better

Walk one shortest path from the task's start bin to its goal bin and count the steps
where `Q` strictly increases. **This is the property DDPG+BC actually needs**: the actor
does one-step greedy improvement, so a field can have excellent global rank correlation
and still contain local basins the actor gets stuck in. Chance is ≈ 0.5 for an
unstructured field (the self-check pins a constant field at exactly 0.0, since no step
*strictly* increases).

Bins with too little data break the chain and are **skipped, not interpolated**, so
`path_bins` and `path_total_bins` are reported beside it: if they differ a lot, the
fraction is over a short, broken path and should be read with suspicion.

### `goal_percentile` — fraction in [0, 1] · 1.0 is perfect

Percentile rank of the goal bin's own value among usable bins, ties counted. 1.0 means
the goal bin is the maximum. Weaker than `peak_geodesic` (it does not say *where* a
wrong peak is) but it is the quantity most directly comparable across tasks, because it
is normalised by the number of bins rather than by maze size.

### `coverage` — fraction in [0, 1] · a control, not a result

Share of free bins that cleared `min_states_per_bin`. **Identical across depths by
construction** — it is a property of the data and the binning, not of the network. If it
differs between two runs in the same suite, the state sample differed and no other
number is comparable. Treat a change here as a bug signal.

### `n_bins`, `path_bins`, `path_total_bins` — counts

How much the metric above was actually computed on. `n_bins < 8` returns NaN for
everything rather than a confident number from six points.

### `q_p2`, `q_p98`, `q_mean`, `q_std` — logits, raw scale

The untransformed range. `Q = φ(s,a)ᵀψ(g)/√d` is an unnormalised log-density-ratio:
binary NCE fixes no overall constant, so **magnitudes are not comparable between runs**
and a "bigger `Q`" means nothing. These exist for two purposes: to spot a collapsed
field (`q_std ≈ 0`, meaning the critic outputs nearly the same value everywhere, which
makes every ρ above meaningless noise), and to set the per-panel color scale.

### `ens_gap` — logits · read relative to `q_std`

Mean `|q₁ − q₂|` per bin across the critic ensemble. The actor uses `min(q₁, q₂)`, so
this says how much that `min` is doing. `ens_gap ≪ q_std` means the two members agree
and "the critic" is a fair singular; `ens_gap ~ q_std` means they disagree about the
field and every statement about "the critic" needs to name a member. Stored as a field,
so you can see *where* they disagree — typically low-coverage regions.

### The same metrics on the other fields

Every metric is computed for `q_policy`, `q_dataset`, `advantage`, the ensemble-mean
variants and `ens_gap`, not just the headline field:

- **`q_dataset`** — the same states scored with the action the dataset actually took.
  If `q_policy` and `q_dataset` have the same topology_gain, the field's structure is a
  property of `(s, g)` and the policy's action choice adds nothing spatial.
- **`advantage` = `q_policy − q_dataset`** — where the learned action beats the logged
  one. `ρ(advantage, −d) > 0` means the policy's edge is largest near the goal;
  negative means it is largest far away, which is the more interesting case for
  stitching.
- **`q_*_dispersion_sd`** — spread of `Q` within a bin, across both the states in it and
  the goal draws. A large value says the bin's single number is hiding structure: the
  critic is using more of the 29-dim state than position.

## Drawn on the actual maze

`render_maze: true` (default) renders the environment's own top-down frame and places
the field on it, rather than on a schematic rebuilt from `maze_map`.

The world → pixel map is **measured, not derived**. OGBench's maze camera is
`elevation=-90` at `lookat=(2(W−3), 2(H−3), 0)`, `distance=5(W−2)`, so any horizontal
plane projects by an affine map — but rather than trust a derivation through fovy and
azimuth sign conventions, `common.render_top_down` teleports the body to 9 known free
cells, renders each, and least-squares fits the affine to where the body actually
appeared. Checked leave-one-out. Measured on `antmaze-medium-stitch-v0`: **fit RMS
0.75–0.87 px, leave-one-out max 1.9 px** on a 480 px image spanning 32 world units —
about 0.1 world units.

One exact correction is applied: the fit is made at the body's height (`torso_z` ≈ 0.7,
recorded), while the field lives on the floor at `z = 0`. Pixel offset from the
principal point scales as `1/(d − z)`, so the floor map is the fitted one scaled by
`(d − z)/d` about the principal point — which is itself the pixel of the lookat. The
scale factor is recorded as `floor_scale` (≈ 0.976). Verified once by overlaying the
predicted wall footprints on the render: the rendered-wall / predicted-wall IoU peaks at
a **(0, 0) pixel shift**, i.e. the floor map is correctly centred.

The background is the **median over the calibration frames**, which removes the body.
It is then desaturated and lifted toward the surface, because the photograph is context
and the field is the data; the field keeps the full sequential ramp at 0.88 alpha.

Known artefact: the walls are 3-D boxes, so their **tops splay outward** from their floor
footprint under the perspective camera. The overlay is aligned to the floor plane, where
the states are, so wall tops overhang the field slightly at wall edges. The caption says
so.

If no GL backend is available the render is skipped, the reason is recorded in
`meta.json` as `render_error`, and the figures fall back to the schematic — same
geometry, same transform code path, walls drawn as rectangles instead.

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
