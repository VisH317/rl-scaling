# The algorithm, exactly as we run it

Offline goal-conditioned CRL and GCBC on OGBench. Every line below is from
`lib/ogbench/impls/agents/{crl,gcbc}.py`, `utils/networks.py`, `utils/datasets.py` and
our own `config.json`. Deviations from OGBench's defaults are marked **bold**.

> **There is a policy.** The actor is trained at *every* gradient step, jointly with the
> critic, under one combined loss and one optimizer. There is no separate extraction
> phase. At evaluation the actor alone selects actions — the critic is not queried,
> because its content has already been distilled into the actor through the `q_loss`
> term. "No critic at eval" means "the critic's job is done by then", not "no policy".

---

## 1. Notation

`s` state (29-D), `a` action (8-D), `g` goal state, `B` batch size, `d` = `repr_dim`.
A batch is `B` transitions `(s, a, s')` sampled i.i.d. from the fixed dataset, each
paired with a **value goal** `g_v` and an **actor goal** `g_a` drawn as in §4.

## 2. Networks

Three towers, no target networks, no EMA.

| tower | input | output | ensemble |
|---|---|---|---|
| `φ` | `[s, a]` (37-D) | `R^d` | 2 |
| `ψ` | `g` (29-D) | `R^d` | 2 |
| actor | `[s, g]` (58-D) | mean of a diagonal Gaussian over `a` | 1 |

Critic (`GCBilinearValue`, `value_exp=False`):

```
Q(s, a, g) = φ(s,a)ᵀ ψ(g) / √d
```

Each tower is `depth` dense layers then a linear head to `d`. **We replace OGBench's
plain MLP with the residual tower** (`block_size=4`, `Dense → LayerNorm → Swish`,
identity added per block, `variance_scaling(1/3,'fan_in','uniform')` init). The anchor
grid keeps OGBench's stock `3×512` GELU MLP.

Actor (`GCActor`, `state_dependent_std=False`, `const_std=True`):

```
π(a|s,g) = N( μ_θ(s,g),  I )          log σ ≡ 0, clipped to [-5, 2]
a ~ clip(π, -1, 1)                    final layer init scale 1e-2
```

## 3. Losses

**Critic — symmetric binary NCE over the B×B pair matrix.** `L = φΨᵀ/√d`,
`L[i,j] = Q(sᵢ, aᵢ, g_v,ⱼ)`. The diagonal is the positive pairs:

```
𝓛_critic = mean_{e,i,j}  BCEWithLogits( L[i,j,e],  𝟙[i=j] )
```

Averaged over the 2 ensemble members. **No logsumexp penalty.** There is **no TD
backup** anywhere — `discount` enters only through goal sampling (§4), never a
bootstrap. This is what makes CRL self-supervised rather than dynamic-programming.

**Actor — DDPG + BC** (`actor_loss='ddpgbc'`). `â ~ π(·|s, g_a)`, clipped:

```
q         = min( Q₁(s, â, g_a),  Q₂(s, â, g_a) )
𝓛_q       = −E[q] / stopgrad(E|q| + 1e-6)        # scale-invariant
𝓛_bc      = −α · E[ log π(a | s, g_a) ]          # a = the DATASET action
𝓛_actor   = 𝓛_q + 𝓛_bc
```

`𝓛_q` pushes the policy toward actions the critic rates highly; `𝓛_bc` pins it to the
data, which offline RL needs and the online 1000-layer paper does not have. The
normalization by `E|q|` matters: contrastive logits have no fixed scale, so an
unnormalized `𝓛_q` would silently reweight against `𝓛_bc` as the critic trains.

**Total.** `𝓛 = 𝓛_critic + 𝓛_actor`, one `optax.adam(3e-4)` over one `ModuleDict`
holding both. Critic and actor gradients are computed in the same backward pass.
(`value_loss` exists only for `actor_loss='awr'`; with `ddpgbc` it is literally `0.0`
and no `value` network is built.)

**GCBC** is the ablation: the actor only, `𝓛 = −E[log π(a|s, g_a)]`. No critic, no `Q`,
same goal sampling, same architecture for the actor.

## 4. Goal sampling (`GCDataset`)

Per sampled index `i`, with `T(i)` the final index of `i`'s trajectory:

```
geometric:  Δ ~ Geom(1 − γ),  g = s[ min(i + Δ, T(i)) ]
uniform:    u ~ U[0,1],       g = s[ round( (i+1)·u + T(i)·(1−u) ) ]
random:     g = s[ j ],  j uniform over the whole dataset
```

| | value goal `g_v` | actor goal `g_a` |
|---|---|---|
| future-in-trajectory | 1.0, **geometric** | **0.5**, uniform |
| random state | 0.0 | **0.5** |
| current state | 0.0 | 0.0 |

**Bold = OGBench's published `antmaze-medium-stitch` command**, not `crl.py`'s bare
defaults (which are `actor_p_trajgoal=1.0`, `actor_p_randomgoal=0.0`). Using the
defaults was a real bug here, caught in review.

Both branches **clip** rather than resample: `min(i + Δ, T(i))`. With `γ = 0.99` and
201-step trajectories this piles mass on the trajectory's last state — measured at
**43.6%** of value goals and 2.4% of actor goals. Probes must bin that case separately.

## 5. Training loop

```
for step in 1 … step_cap:
    batch ← sample B transitions + goals            # i.i.d., no replay ordering
    𝓛 ← 𝓛_critic + 𝓛_actor
    θ ← Adam(θ, ∇θ 𝓛, lr=3e-4)                      # ONE optimizer, both towers
    if step % 100_000 == 0:  evaluate, checkpoint if best, test plateau
```

No target network, no Polyak averaging, no LR schedule, no gradient clipping, no weight
decay, no data augmentation (`p_aug=0`), no frame stacking.

## 6. Evaluation

`evaluate()` calls **only** `agent.sample_actions`, which selects only the `actor`
module. The critic and its ensemble are never queried.

```
a_t = μ_θ(s_t, g)                 # temperature 0 ⟹ scale_diag = exp(0)·0 = 0
                                  # ⟹ deterministic: sampling returns the mean
```

Rollout until the 1000-step limit; success is binary per episode; the reported number is
the mean over 5 tasks × `eval_episodes` each. Three scoring conventions, which evaluate
**different parameters**:

| convention | parameters |
|---|---|
| `best_success` | live params at the single best-scoring eval step |
| `last3_eval_mean` | live params at each of the last 3 eval points — **3 different networks**, 100k steps apart, averaged |
| `reeval@50` | `params_best.pkl`, one checkpoint; the headline, comparable to OGBench's table |

---

## 7. Hyperparameters

Everything needed to reproduce. **Bold** = differs from `crl.py`'s bare defaults.

### Optimization — identical for every run in phases 1–4

| | value | why fixed |
|---|---|---|
| optimizer | Adam, `b1=0.9`, `b2=0.999`, `eps=1e-8` | OGBench and scaling-crl both |
| learning rate | `3e-4`, **constant** | warmup/decay help deep nets *specifically*, confounding the depth axis |
| LR schedule | none | ditto |
| gradient clipping | none (`max_grad_norm=0`) | ditto |
| weight decay | 0 | ditto |
| separate actor/critic LR | no — one optimizer over both | `crl.py` builds one `optax.adam` |
| batch size | 1024 | |
| grad steps | cap 1M (anchor) / **2M** (repro), plateau floor 1M | |

A loud warning fires in `launch.py` if any of the first five is changed, naming the
confound.

### Agent

| | ours | `crl.py` default | 1000-layer paper |
|---|---|---|---|
| critic loss | `binary_nce` | `binary_nce` | **InfoNCE** |
| logits | `dot`, `φ·ψ/√d` | `dot` | **`−‖φ−ψ‖₂`** |
| logsumexp penalty | 0.0 | 0.0 | **0.1** |
| critic ensemble | 2, combined by `min` | 2 | 1 |
| actor loss | `ddpgbc` | `ddpgbc` | maximize critic, no BC |
| `α` (BC coefficient) | **0.1** | 0.1 | — |
| `const_std` | True | True | |
| `γ` (goal sampling only) | 0.99 | 0.99 | |
| `value_p_trajgoal / randomgoal` | 1.0 / 0.0, geometric | same | |
| `actor_p_trajgoal / randomgoal` | **0.5 / 0.5**, uniform | 1.0 / 0.0 | |
| `p_aug`, frame stack | 0, none | same | |

The first three rows are **open question Q1** and the likely reason our offline depth
result disagrees with the paper's §4.6. See `results_phase1.md` §2.6.

### Architecture

| | `phase1_repro` | `phase1_anchor` | 1000-layer | stable-CRL |
|---|---|---|---|---|
| tower | residual | stock OGBench MLP | residual | plain MLP |
| depth | **4 / 8 / 16 / 32** | 3 (fixed) | 4 … 1024 | 4 |
| width | **256** | 512 | 256 | 1024 |
| `block_size` | 4 | — | 4 | — |
| activation | **Swish** | GELU | Swish | |
| norm | LayerNorm **inside** block | LayerNorm **after** GELU | inside | yes |
| `repr_dim` (`latent_dim`) | **64** | 512 | 64 | **16** |
| init | **`variance_scaling(1/3,'fan_in','uniform')`** | Flax default | same | **`1e-12`** (cold) |
| actor hidden dims | same tower as critic | `(512,512,512)` | | |

`repr_dim` is **open question Q2**; it is currently unresolved and blocks attributing
§7's architecture inversion in `results_phase1.md`.

### Data and evaluation

| | value |
|---|---|
| dataset | `antmaze-medium-stitch-v0`, native 1M transitions (5,000 × 201 rows) |
| validation split | the shipped `-val.npz`, 500 trajectories |
| seeds | 0, 1, 2 |
| eval interval | every 100,000 gradient steps |
| eval episodes | **15/task** during training (75/point); **50/task** for `reeval@50` (250/point) |
| eval temperature | 0.0 (deterministic) |
| episode limit | 1000 steps |
| plateau rule | stop when 3-eval moving average has not improved by `0.01` for 5 evals **and** `step ≥ 1M`. Patience is *not* reset at the floor, so a run that plateaued early stops at ≈1.0–1.1M |
| checkpoints | best (on eval success) + final + every 500k, keep 2 |
| heartbeat | `status.json` every 5,000 steps |

---

## 8. What this algorithm does *not* do

Worth stating, because each is a thing people assume is present:

- **No TD learning.** No Bellman backup, no target network. `γ` only shapes the goal
  distribution. Depth therefore cannot be helping by stabilizing bootstrapping.
- **No separate policy-extraction phase.** Critic and actor co-train from step 1. The
  experiment that *would* separate them is `actor_thread` (freeze critic, vary actor
  depth), configured and **unimplemented**.
- **No reward.** Success is measured at evaluation only; it never enters training.
- **No LR schedule, warmup, clipping or weight decay.** Deliberate — see §7.
- **No image encoder.** State-based throughout; `qpos`/`qvel` are kept for probes.
