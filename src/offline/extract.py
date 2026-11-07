"""Two-stage CRL: train a critic, freeze it, then extract a policy from it.

WHY THIS IS THE SAME ALGORITHM, NOT A NEW ONE
---------------------------------------------
Joint training is how `crl.py` is organised, not something offline CRL needs. Two
properties, both verified against `lib/ogbench/impls/agents/crl.py` rather than assumed:

1. **The actor loss never writes to the critic.** `CRLAgent.actor_loss` (L111) calls
   `self.network.select('critic')(...)` with **no** `params=grad_params`, and
   `TrainState.__call__` treats a missing `params` as "use the stored ones, stop the
   gradient". So critic gradients come only from `contrastive_loss`.
2. **The critic never depends on the actor.** `contrastive_loss` (L24-40) reads
   `batch['actions']` -- the *dataset's* actions -- and `batch['value_goals']`, which are
   future states from the same trajectory. There is no bootstrap on pi's actions, so the
   critic converges to the behaviour policy's value whatever the actor does:

       f*(s,a,g) = log [ p^beta(g | s,a) / p(g) ] + const

Reordering therefore changes only the early dynamics -- jointly, the actor chases a
critic that is still moving. This is the one-step view of offline RL
(Brandfonbrener et al. 2021), and the decoupling Park et al. 2024 used to separate value
learning from policy extraction.

WHY WE WANT IT
--------------
Every grid run so far scales the actor *and* the critic with one `depth` knob, because
`patches.ResidualMLPShim` stands in for OGBench's `MLP` at both call sites. So
"depth 32 reaches 86.7" is a statement about a 32-layer critic AND a 32-layer policy.
Freezing a trained critic and re-extracting at a pinned `actor_depth` separates them
without retraining a single critic.

THE GOTCHA THIS MODULE EXISTS TO HANDLE
---------------------------------------
Dropping the critic loss is *not* enough on its own if the saved optimiser state is
restored with it: Adam's stored momentum would keep integrating and the "frozen" critic
would drift. We never restore `opt_state` -- we build a fresh one -- and on top of that
we label the critic subtree `set_to_zero`, so the updates are identically zero by
construction rather than by argument. `critic_fingerprint` then asserts it, before and
after, from the actual arrays.

A convenient property for comparing across runs: DDPG+BC normalises by its own scale,
`-q.mean() / stop_gradient(|q|.mean() + 1e-6)` (L115). `interp__critic_depth` found Q's
level is arbitrary per run, but every critic presents the actor with the same effective
objective, so extraction is a fair comparison across depths and environments.
"""

from __future__ import annotations

import glob
import hashlib
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

#: Top-level parameter groups held constant during extraction. `modules_value` only
#: exists under `actor_loss='awr'`; it is a critic by another name and is frozen too.
FROZEN_PREFIXES = ("modules_critic", "modules_value")


def is_frozen(name: str) -> bool:
    return name.startswith(FROZEN_PREFIXES)


def critic_fingerprint(params) -> str:
    """A content hash of every frozen group, for the before/after assertion.

    Hashes raw bytes in a sorted, fully-qualified key order, so it is sensitive to any
    change in value, shape or structure -- not just to a summary statistic that a drift
    could preserve.
    """
    h = hashlib.sha256()
    flat = jax.tree_util.tree_flatten_with_path(
        {k: v for k, v in params.items() if is_frozen(k)}
    )[0]
    for path, leaf in sorted(flat, key=lambda kv: jax.tree_util.keystr(kv[0])):
        a = np.asarray(leaf)
        h.update(jax.tree_util.keystr(path).encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def graft_critic(params, source_params) -> dict:
    """Return `params` with every frozen group replaced by the source checkpoint's.

    Shapes are checked per leaf. A mismatch here means the source critic was trained at
    a different depth/width/repr_dim than the one being rebuilt, which is a
    configuration error, not something to broadcast around.
    """
    out = dict(params)
    for name in params:
        if not is_frozen(name):
            continue
        if name not in source_params:
            raise KeyError(
                f"source checkpoint has no '{name}'; it was trained with a different "
                f"agent configuration (source groups: {sorted(source_params)})"
            )
        tgt_shapes = jax.tree_util.tree_map(jnp.shape, params[name])
        src_shapes = jax.tree_util.tree_map(jnp.shape, source_params[name])
        if tgt_shapes != src_shapes:
            raise ValueError(
                f"'{name}' shape mismatch between the rebuilt agent and the source "
                f"checkpoint -- the critic arch does not match. Rebuild the agent with "
                f"the source run's own arch (actor_depth may differ, nothing else)."
            )
        out[name] = source_params[name]
    return out


def frozen_critic_tx(params, lr: float):
    """Adam on the actor, exact zero on the critic.

    `optax.set_to_zero()` is belt-and-braces: with the critic loss dropped its
    gradients are already zero, and a *fresh* Adam state would leave zero gradients
    alone. This makes the guarantee structural, so it survives anyone later adding a
    schedule, weight decay, or a stray critic term to the loss.
    """
    labels = {
        k: jax.tree_util.tree_map(lambda _, kk=k: ("freeze" if is_frozen(kk) else "train"), v)
        for k, v in params.items()
    }
    return optax.multi_transform(
        {"train": optax.adam(learning_rate=lr), "freeze": optax.set_to_zero()},
        labels,
    )


def install_actor_only_loss(agent_class) -> bool:
    """Patch `total_loss` to drop the critic term. Process-lifetime, like the others.

    Returns False if already installed, so repeated calls are safe.
    """
    if getattr(agent_class, "_actor_only_installed", False):
        return False
    original = agent_class.total_loss

    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {f"actor/{k}": v for k, v in actor_info.items()}
        return actor_loss, info

    agent_class.total_loss = total_loss
    agent_class._actor_only_loss_original = original
    agent_class._actor_only_installed = True
    return True


def find_source_run(runs_dir: str, grid: str, agent: str, depth: int, seed: int) -> str:
    """The one finished run of `grid` at this (agent, depth, seed).

    Matched by directory name rather than by recomputing the source run id, so an
    extraction config stays valid without importing the source grid's YAML, and a
    source grid that has not run yet fails here with a readable message instead of
    silently resolving to nothing.
    """
    pattern = os.path.join(runs_dir, f"{grid}__{agent}__d{depth}__*__s{seed}__*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if len(dirs) != 1:
        raise FileNotFoundError(
            f"expected exactly 1 source run for {grid} {agent} d{depth} s{seed}, "
            f"found {len(dirs)}: {[os.path.basename(d) for d in dirs]}"
        )
    return dirs[0]


def load_source_params(run_dir: str, checkpoint: str = "best") -> dict:
    """The frozen groups from a source run's checkpoint.

    Only `params` is read. The pickle also carries `opt_state`, and restoring THAT is
    the failure this module is built around: Adam's stored momentum would keep
    integrating and move a critic that every other signal says is frozen.
    """
    path = os.path.join(run_dir, "checkpoints", f"params_{checkpoint}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    with open(path, "rb") as f:
        blob = pickle.load(f)
    params = blob["agent"]["network"]["params"]
    frozen = {k: v for k, v in params.items() if is_frozen(k)}
    if not frozen:
        raise ValueError(f"{path} holds no critic groups (has {sorted(params)})")
    return frozen


def prepare(agent, source_params, lr: float, agent_class=None):
    """Graft the critic in, freeze it, and hand back (agent, fingerprint).

    The caller must compare the returned fingerprint against
    `critic_fingerprint(agent.network.params)` after training.
    """
    from utils.flax_utils import TrainState  # noqa: PLC0415  (OGBench, on sys.path)

    params = graft_critic(agent.network.params, source_params)
    tx = frozen_critic_tx(params, lr)
    network = TrainState.create(agent.network.model_def, params, tx=tx)
    agent = agent.replace(network=network)
    if agent_class is not None:
        install_actor_only_loss(agent_class)
    return agent, critic_fingerprint(params)


# ---------------------------------------------------------------------------
# Self-test -- run by src/check.py.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Train a toy two-group model and assert the frozen group is bit-identical."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    params = {
        "modules_critic": {"w": jax.random.normal(k1, (4, 4))},
        "modules_actor": {"w": jax.random.normal(k2, (4, 4))},
    }
    before = critic_fingerprint(params)

    tx = frozen_critic_tx(params, lr=1e-1)
    opt_state = tx.init(params)

    # A loss that pulls BOTH groups hard, so a leaky freeze cannot hide.
    def loss_fn(p):
        return sum(jnp.sum(v["w"] ** 2) for v in p.values())

    for _ in range(5):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    after = critic_fingerprint(params)
    assert before == after, "frozen critic moved under a gradient that targets it"
    assert not jnp.allclose(params["modules_actor"]["w"],
                            jax.random.normal(k2, (4, 4))), "actor did not train"

    # The shape guard must fire rather than silently broadcasting.
    try:
        graft_critic(params, {"modules_critic": {"w": jnp.zeros((2, 2))}})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("graft_critic accepted a shape mismatch")

    # A missing group must be named, not KeyError'd opaquely elsewhere.
    try:
        graft_critic(params, {})
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("graft_critic accepted a source with no critic")

    print("extract: frozen critic bit-identical after 5 steps; actor moved; "
          "guards fire")


if __name__ == "__main__":
    _selftest()
