# `state_probes` — what do the critic's representations linearly encode?

**Status:** PLANNED (nothing here has been run) · **Applies to:** CRL · **Reads:** `critic`
· **Estimated cost:** ~5 min/checkpoint, ~0.4 GPU-h for a 4-depth ladder

## The question

`critic_heatmap` asks whether the *value* `Q = φ(s,a)ᵀψ(g)/√d` ranks states by maze
distance. This asks the layer underneath: whether `φ` and `ψ` **linearly encode** the
quantities that ranking would need. A bilinear critic can only express what a dot
product of its two representations can express, so if `ψ(g)` does not linearly carry
"where in the maze this goal is", no amount of actor extraction can recover it.

The two evals answer different questions and can disagree, which is the interesting
case: a representation can carry geodesic distance linearly while the dot product fails
to use it (bad `φ`/`ψ` alignment), or the field can rank correctly while neither
representation is linearly decodable (the information is there non-linearly).

## Sites

| site | what it is | shape |
|---|---|---|
| `psi_goal` | `ψ(g)`, the goal tower's output | `(N, repr_dim)` per ensemble member |
| `phi_state_action` | `φ(s, a)`, the state-action tower's output | `(N, repr_dim)` per member |
| `block_k` *(later)* | the residual stream after block *k* | `(N, width)` |

`repr_dim` is **64 for the whole depth sweep**, so probe capacity is constant across
depths by construction — the comparison is not confounded by representation width. The
anchor runs use 512 and must not be put on the same axis as the sweep.

`block_k` needs intermediate-activation capture, which `offline/networks.py` does not
expose today. That is phase 2 of this eval and is why `sites` is a list.

## Targets

| target | kind | why |
|---|---|---|
| `xy` | regression | **A leak check, not a result.** `(x, y)` is dims 0:2 of the observation, so any representation that merely preserves its input scores R² ≈ 1. If `ψ` scores *low* here, it has actively discarded position — itself a finding. |
| `maze_cell` | 26-way classification | Coarse position, with a non-trivial chance baseline (the largest-class frequency, not 1/26). |
| **`geodesic_to_task`** | regression, 5 columns | BFS distance from the state to each task goal, in world units. **The headline.** This is the quantity that is *not* in the observation and that stitching needs. |
| `steps_to_goal` | regression | Steps remaining until the state that was sampled as this trajectory's goal. The quantity CRL's geometric goal sampling makes `Q` a monotone function of, under the idealised objective. |
| `torso_z` | regression | Control: a body coordinate with no spatial-planning content. |
| `joint_vel_norm` | regression | Control: should **not** improve with depth. If it does, deeper towers are simply better linear bases and every other number needs re-reading. |

## Method — the parts that are easy to get wrong

**Split by trajectory, never by state.** Consecutive states in an OGBench trajectory are
near-duplicates at 0.05 s apart. A random state split puts a state's own neighbours in
the training set and inflates held-out R² toward 1.0 for every site, every depth and
every baseline at once — which looks like a clean result. `split: trajectory` holds out
100 whole trajectories.

**Choose λ inside the training trajectories.** Grouped 5-fold CV over the 400 training
trajectories, on the λ grid in `eval.yaml`. Touching the test trajectories to pick λ
turns the held-out score into a training score.

**Standardise features, not targets.** Report R² against the test-set mean, so the
number is "fraction of variance explained" and 0 means "no better than the mean".

**Classification without scikit-learn.** `sklearn` is not in the venv and does not need
to be: ridge is a closed-form `lstsq` solve, and `maze_cell` is a multinomial logistic
fit in a few lines of JAX + optax, both of which are already dependencies. Do not add
a dependency for this.

## The three baselines — the result is the comparison, not the number

An R² of 0.85 means nothing on its own.

1. **`raw_obs`** — the identical probe fit on the raw 29-dim observation. The
   representation has to **beat** this to have added anything. For `xy` it cannot (see
   above); for `geodesic_to_task` it is a genuinely hard baseline, because geodesic
   distance is a non-linear function of position with walls in it.
2. **`random_init`** — the same architecture at the same depth with **untrained**
   parameters. Deeper random networks are different random bases, and a deep random
   projection can be a better linear basis than a shallow one. Without this control,
   "probe accuracy rises with depth" may be a statement about architecture, not about
   learning. This is the control most likely to overturn a positive result.
3. **`label_shuffle`** — targets permuted across trajectories. Must come out at chance.
   It catches leakage in the split and bugs in the fit, and costs one extra solve.

## Expected output

`<ANALYSIS_DIR>/evals/<suite>/state_probes/<run_id>/`

- `probes.json` — one record per (site, target, baseline): test R² or accuracy, chance
  level, chosen λ, CV curve, n train/test trajectories.
- `figures/fig1_probe_vs_depth.png` — R² against depth, one panel per target; the two
  baselines as reference lines on each panel, not as separate panels, so the gap is
  read directly.
- `figures/fig2_probe_vs_heatmap.png` — `geodesic_to_task` probe R² against the
  `topology_gain` from `critic_heatmap` at the same checkpoint. One point per depth.
  If these two move together, the heatmap result has a representational mechanism.

## Open questions before implementing

1. **Which ensemble member?** The critic is a 2-member ensemble by default and the actor
   uses `min(q1, q2)`. Probe both and report both; if they differ, the "the critic"
   framing is wrong.
2. **Goals for `psi_goal`.** The 5 task goals give only 5 distinct `ψ` vectors — far too
   few to fit a probe. Use the goal distribution the critic was *trained* on
   (`value_p_trajgoal=1.0`, geometric with γ), which is what `GCDataset` samples, and
   record that choice: probing `ψ` on goals the critic never saw measures extrapolation.
3. **`steps_to_goal` is defined only relative to a sampled goal**, so it needs the goal
   sampler in the loop rather than a fixed per-state label.
