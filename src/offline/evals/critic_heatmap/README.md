# `critic_heatmap` — can the critic represent the maze?

**Status:** ready · **Applies to:** CRL · **Reads:** `critic`, `actor` · **Cost:** ~5 min/checkpoint

## The question

Depth helps the offline CRL critic a lot on `antmaze-medium-stitch`: success goes
34.3 → 54.7 → 78.4 (reeval@50, depths 4/8/16). The policy is extracted by DDPG+BC, so
all the depth signal arrives through `Q(s, a, g)`. **What about `Q` gets better?**

A value function on a maze has one property it cannot fake: the goal is reachable only
around the walls, so value must decay along the *maze* path, not along the straight
line. This eval measures whether it does, and how that changes with depth.

## What is computed

For every task goal `g` and every spatial bin of the maze, `Q(s, a, g)` over real
states `s` in that bin, reduced over the critic ensemble with `min` because that is
what the DDPG+BC actor maximises (`q = min(q1, q2)` in `crl.py`) — for **three choices
of action**:

| source | action | the hypothesis it embodies |
|---|---|---|
| `policy` | `a = π(s, g)` | what the actor actually does |
| `oracle_geodesic` | `a = expert(s, u_geo)` | move toward the next shortest-path cell |
| `oracle_euclid` | `a = expert(s, u_euc)` | move straight at the goal, through walls |

Also available: `dataset` (the logged action) and `random` (a floor).

### The oracle is not a heuristic

`expert` is **OGBench's own direction-conditioned SAC policy — the one that generated
every action in this dataset.** `data_gen_scripts/generate_locomaze.py` runs

```python
subgoal_xy, _ = env.get_oracle_subgoal(env.get_xy(), env.cur_goal_xy)
dir = normalise(subgoal_xy - env.get_xy())
action = expert(concat(ob[2:], dir), temperature=0) + N(0, 0.2)
```

so feeding it a direction of our choosing yields the action the data generator *would
have taken* to move that way. No hand-built controller, no transplanting an action from
one state to another, no synthesised state. The checkpoint is `experts/ant`
(`params_400000.pkl`) from `experts.tar.gz`, and `meta.json` records which one was used.

`u_geo` reproduces `get_oracle_subgoal`'s rule exactly — BFS from the goal cell, step to
the strictly-closer 4-neighbour — and is **cross-checked against the live env**:
on task 4 (`start (6,5) → goal (6,1)`) our direction and `env.get_oracle_subgoal`'s
agree to 1e-6, and the geodesic and straight-line directions there differ by **180°**.
That is the task GCBC scores 0.00 on.

Verified that the expert actually drives the body: commanding `+x` for 50 steps moves
the ant `(+5.42, −0.18)`; commanding `+y` moves it `(−0.69, +4.29)`.

### Ordering constraint

The expert is loaded and all its actions computed **before** the run's own agent.
`install_residual_arch` replaces `utils.networks.MLP` for the whole process and Flax
re-runs `setup()` on every apply, so an expert applied after the patch would rebuild
itself as a residual tower and fail on a parameter-structure mismatch.
`common.load_expert` raises if the patch is already installed, rather than failing
later with a confusing shape error.

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

### `action_topology_gain` — logits · the sharp test

`mean( Q[oracle_geodesic] − Q[oracle_euclid] )`, restricted to bins where the two
directions **disagree by at least `direction_split_deg`** (default 45°).

This is the action-level version of `topology_gain`, and it is the stronger of the two
because it is measured exactly where the two hypotheses make opposite predictions. In a
bin where the straight line points into a wall, one of these two actions is right and
the other is wrong; the critic has to choose. **Positive = it prefers the maze-correct
action there.**

The restriction matters: in the bins where the two directions nearly agree the two
actions are nearly the same action, so their difference is noise. Averaging over all
bins would dilute the signal with bins that cannot speak to it.
`action_topology_gain_all_bins` is reported too, as a check that the restriction is
doing what it claims; `split_bin_fraction` says how many bins qualified.

Unlike the ρ-based `topology_gain`, this is in **logits**, so it is not comparable
across runs in magnitude — only its sign and its trend with depth.

### `oracle_gap` — logits · read it WITH the success rate

`mean( Q[policy] − Q[oracle_geodesic] )`. **The sign alone decides nothing**, and this
is the metric most likely to be misread:

- DDPG+BC trains the actor to maximise `Q`, so **`> 0` is the default** and by itself
  only says that extraction worked — the actor found a high-`Q` action.
- **`< 0`** means the actor is *not* at its own critic's argmax: an **extraction
  failure**. The critic prefers the maze-correct action and the actor is not taking it.
- **`> 0` together with low task success** means the critic's preferred action is not
  the maze-correct one: a **critic failure**. The actor is faithfully maximising a
  quantity whose maximiser is wrong.

The two failure modes look identical in the success rate and are distinguishable only
here. `oracle_gap_split_bins` restricts to the disagreeing bins, where the distinction
is sharpest.

### `*_norm` and `q_scale` — dimensionless · the forms to quote across runs

Q is an unnormalised log-density-ratio. Its level *and* its spread are arbitrary per
run, which this suite measured directly: `q_scale` varies substantially between
depths and between environments. A difference of two Q values, in logits, therefore
cannot be compared across runs — a bigger number may only mean a critic with a wider
output range.

`q_scale` is the **per-task standard deviation of a reference Q field** over the usable
bins, reference chosen as the first of `q_policy`, `q_oracle_geodesic`, `q_dataset`
that is present and named in the output. It is computed over *all* usable bins, not
over each metric's own mask, so every normalised metric is divided by the same number
and they stay comparable to each other.

| metric | units | compare across runs? |
|---|---|---|
| `action_topology_gain`, `oracle_gap` | logits | **no** — sign and trend only |
| `action_topology_gain_norm`, `oracle_gap_norm` | SDs of that run's Q | **yes** |
| `q_scale` | logits | it IS the scale; useful on its own |

Normalisation is per task, then averaged, because the spread of Q genuinely differs
between the five tasks and one pooled SD would flatten that. A task whose scale is
below 1e-9 is dropped rather than divided — a flat critic has no meaningful unit, and
0/0 would otherwise arrive looking like a real number. `n_tasks_scored` records how
many survived.

The self-check asserts the defining property directly: multiplying every Q field by 37
multiplies `q_scale` and the raw metrics by 37.0 and leaves the `_norm` values
unchanged to 4 decimal places.

### The same metrics on the other fields

Every metric is computed for `q_policy`, `q_dataset`, `advantage`, the ensemble-mean
variants and `ens_gap`, not just the headline field:

- **`q_oracle_geodesic` / `q_oracle_euclid`** — the two oracle fields in their own
  right. If `q_oracle_geodesic` has a much higher `rho_geodesic` than `q_policy`, the
  critic's spatial structure is clearer under the correct action than under the
  actor's, which again separates critic quality from extraction quality.
- **`q_dataset`** (optional) — the logged action, i.e. the same expert aimed at *that
  trajectory's* goal rather than at ours. Useful as a "typical on-manifold action"
  reference. `advantage = q_policy − q_dataset` is then where the learned action beats
  the logged one.
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
| `field.npz` | `q_<source>` for each action source, plus `q_*_ens_mean`, `q_*_dispersion_sd`, the difference fields (`oracle_geo_minus_euclid`, `policy_minus_oracle`, `advantage`), `ens_gap`, `dir_angle_deg`, `geodesic`, `euclid` — each `(5, Hf, Wf)` — plus `maze`, `fine`, `count`, `occupied` |
| `metrics.json` | every metric above, per field, per task, plus the across-task mean; `_cross` holds the between-source metrics (`action_topology_gain`, `oracle_gap`, their `_norm` companions, `q_scale`, `split_bin_fraction`) |
| `meta.json` | run id, depth, seed, param counts, selection rule and the whole seed set, the state files used, every goal `xy` drawn, the expert checkpoint, the render calibration, timing |

`q_*_dispersion_sd` is the spread of `Q` within a bin across both the states in it and
the goal draws — a large value says the bin's single number is hiding structure.

### Figures

| figure | what |
|---|---|
| `fig1_critic_field` | the `q_policy` field, depth × task, on the rendered maze |
| `fig2_topology` | `rho_geodesic` / `rho_euclid` / `topology_gain` vs depth |
| `fig4_metrics` | `peak_geodesic` / `path_monotone` / `goal_percentile` vs depth |
| `fig5_action_preference` | `Q[geodesic] − Q[straight]`, diverging, **only on the disagreeing bins** — the others are drawn pale, because the question does not apply there, which is a different thing from having no data (hatched) |
| `fig6_action_tests` | `action_topology_gain` and `oracle_gap` vs depth, raw (top row, logits) and scale-free (bottom row), mean ± 1 SD over seeds |
| `fig3_advantage_field` | only if `dataset` is in `action_sources` |

## How to read it

- **`topology_gain` and `action_topology_gain` rising with depth** is the result this
  eval exists to find: deeper critics encode maze connectivity, which is what stitching
  needs. The second is the stronger claim, because it is measured where the two
  hypotheses disagree.
- **`action_topology_gain` rising while `topology_gain` is flat** would say the critic's
  *action* preferences carry the maze even though its *state* value landscape does not
  — which is the thing the actor consumes, so it would be the better explanation of the
  success curve.
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
   `oracle_gap` is the closest thing here to separating the two, and it still needs the
   success rate beside it.
5. **The oracle is an expert for the *environment*, not for the task.** It was trained
   to follow a commanded direction, not to solve these five goals. Feeding it the
   geodesic direction makes it act optimally *given that direction is right*; it says
   nothing about whether an even better action exists.
6. **`action_topology_gain` and `oracle_gap` are in logits**, so the RAW forms are
   comparable across depths only in sign and trend, never in magnitude. Since
   2026-11-06 each also has a `_norm` companion divided by `q_scale`; quote those for
   anything cross-depth, cross-seed or cross-environment. See the section below.
