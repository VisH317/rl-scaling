"""Contrastive critic losses, including the one the 1000-layer paper actually used.

OGBench's `crl.py` hardcodes **binary NCE on dot-product logits**. The 1000-layer paper
uses **forward InfoNCE on negative-L2 logits with a logsumexp penalty**. Until this
module existed, our configs carried `critic_loss`/`logits`/`logsumexp_penalty` fields
that nothing read: setting `critic_loss="infonce"` silently ran binary NCE. That is the
same silent-no-op class of bug as the `fig3` metric keys and the step pin, and it would
have made a 1000-layer replication quietly meaningless.

Installed the same way as the residual tower: a process-lifetime monkeypatch of
`CRLAgent.contrastive_loss`, because Flax re-enters the method on every `apply`.

Logit forms
-----------
    dot      logits[i,j] = phi_i . psi_j / sqrt(d)      OGBench, stable-CRL
    neg_l2   logits[i,j] = -||phi_i - psi_j||_2         1000-layer paper

`neg_l2` is computed by expansion, |a-b|^2 = |a|^2 + |b|^2 - 2 a.b, which is O(B^2)
instead of the O(B^2 d) a naive broadcast would need. At B=1024, d=64, e=2 the naive
form allocates ~0.5 GB per forward; this one allocates ~8 MB.

Losses
------
    binary_nce   mean over all B^2 entries of BCE(logits, 1[i=j]).
                 No normalization axis, so "forward/backward/symmetric" does not apply;
                 it is exactly invariant under transposing the logit matrix.

    infonce      forward: -mean_i( logits[i,i] - logsumexp_j logits[i,j] ),
                 normalizing over goals for each (s,a).

Logsumexp penalty
-----------------
`+ lambda * mean_i( (logsumexp_j logits[i,j])^2 )`, applied only with `infonce`.

This is not cosmetic. The row softmax is invariant to adding any c(s,a) to a whole row,
so forward InfoNCE pins the density ratio only up to a function of (s,a) -- and the
actor's `max_a Q(s,a,g)` compares across exactly that freedom. The penalty anchors the
per-row normalizer and removes the invariance. Running `infonce` with lambda = 0 would
therefore likely break actor extraction for reasons that have nothing to do with depth,
which is why `run.py` refuses that combination unless it is asked for explicitly.
(That reading of the penalty's role is ours; the paper states the coefficient, 0.1,
not the justification.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

#: Set by `install_critic_loss`. Read on every call, so it must stay process-global.
_ACTIVE: dict | None = None

DEFAULTS = dict(critic_loss="binary_nce", logits="dot", logsumexp_penalty=0.0)


def compute_logits(phi: jnp.ndarray, psi: jnp.ndarray, mode: str) -> jnp.ndarray:
    """(e, B, d), (e, B, d) -> (B, B, e)."""
    if mode == "dot":
        return jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])
    if mode == "neg_l2":
        cross = jnp.einsum("eik,ejk->eij", phi, psi)
        sq = (jnp.sum(phi ** 2, -1)[:, :, None]
              + jnp.sum(psi ** 2, -1)[:, None, :]
              - 2.0 * cross)
        # Clamp before sqrt: the expansion can go slightly negative in fp32.
        dist = jnp.sqrt(jnp.maximum(sq, 1e-12))
        return jnp.transpose(-dist, (1, 2, 0))
    raise ValueError(f"unknown logits mode: {mode!r}")


def contrastive_objective(logits: jnp.ndarray, eye: jnp.ndarray, spec: dict):
    """(B, B, e) -> scalar loss, plus the penalty term for logging."""
    kind = spec["critic_loss"]
    if kind == "binary_nce":
        per = jax.vmap(
            lambda lg: optax.sigmoid_binary_cross_entropy(logits=lg, labels=eye),
            in_axes=-1, out_axes=-1,
        )(logits)
        return jnp.mean(per), jnp.array(0.0)
    if kind == "infonce":
        # Forward: normalize over goals (axis 1) for each (s, a).
        lse = jax.scipy.special.logsumexp(logits, axis=1)          # (B, e)
        diag = jnp.einsum("iie->ie", logits)                       # (B, e)
        loss = -jnp.mean(diag - lse)
        penalty = jnp.mean(lse ** 2)
        lam = float(spec["logsumexp_penalty"])
        return loss + lam * penalty, penalty
    raise ValueError(f"unknown critic_loss: {kind!r}")


def _patched_contrastive_loss(self, batch, grad_params, module_name="critic"):
    """Replacement for `CRLAgent.contrastive_loss`.

    The statistics block is kept byte-for-byte compatible with OGBench's so that
    `train.csv` columns do not change meaning between loss forms.
    """
    spec = _ACTIVE or DEFAULTS
    batch_size = batch["observations"].shape[0]
    actions = batch["actions"] if module_name == "critic" else None

    v, phi, psi = self.network.select(module_name)(
        batch["observations"], batch["value_goals"],
        actions=actions, info=True, params=grad_params,
    )
    if len(phi.shape) == 2:            # non-ensemble
        phi, psi = phi[None, ...], psi[None, ...]

    logits = compute_logits(phi, psi, spec["logits"])
    eye = jnp.eye(batch_size)
    loss, penalty = contrastive_objective(logits, eye, spec)

    v = jnp.exp(v)
    flat = jnp.mean(logits, axis=-1)
    correct = jnp.argmax(flat, axis=1) == jnp.argmax(eye, axis=1)
    return loss, {
        "contrastive_loss": loss,
        "logsumexp_penalty_term": penalty,
        "v_mean": v.mean(), "v_max": v.max(), "v_min": v.min(),
        # NOTE: with neg_l2 every logit is <= 0, so a ">0" threshold labels everything
        # negative and this reads exactly 1 - 1/B. Meaningless there; see algorithm.md.
        "binary_accuracy": jnp.mean((flat > 0) == eye),
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": jnp.sum(flat * eye) / jnp.sum(eye),
        "logits_neg": jnp.sum(flat * (1 - eye)) / jnp.sum(1 - eye),
        "logits": flat.mean(),
    }


def install_critic_loss(spec: dict) -> bool:
    """Install the critic loss for the REST OF THE PROCESS. Returns True if patched.

    A no-op (returns False) when the spec is OGBench's own binary_nce + dot, so the
    default path keeps running OGBench's unmodified code and stays diffable.
    """
    global _ACTIVE
    resolved = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if k in spec}}
    if (resolved["critic_loss"] == "binary_nce" and resolved["logits"] == "dot"
            and float(resolved["logsumexp_penalty"]) == 0.0):
        _ACTIVE = None
        return False

    import agents.crl as crl_mod
    _ACTIVE = resolved
    crl_mod.CRLAgent.contrastive_loss = _patched_contrastive_loss
    return True


def _selftest() -> None:
    """Checks the properties the docstring claims. Run: python -m offline.critic_losses"""
    import numpy as np
    rng = np.random.default_rng(0)
    e, B, d = 2, 8, 5
    phi = jnp.asarray(rng.normal(size=(e, B, d)))
    psi = jnp.asarray(rng.normal(size=(e, B, d)))
    eye = jnp.eye(B)

    # neg_l2 by expansion == naive broadcast
    fast = compute_logits(phi, psi, "neg_l2")
    naive = jnp.transpose(
        -jnp.sqrt(jnp.sum((phi[:, :, None, :] - psi[:, None, :, :]) ** 2, -1)), (1, 2, 0))
    assert jnp.allclose(fast, naive, atol=1e-5), "neg_l2 expansion mismatch"

    # dot matches OGBench's einsum exactly
    dot = compute_logits(phi, psi, "dot")
    assert jnp.allclose(dot, jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(d))

    # binary_nce is transpose-invariant; infonce is not
    bn = dict(critic_loss="binary_nce", logits="dot", logsumexp_penalty=0.0)
    inf = dict(critic_loss="infonce", logits="dot", logsumexp_penalty=0.0)
    lT = jnp.transpose(dot, (1, 0, 2))
    assert jnp.allclose(contrastive_objective(dot, eye, bn)[0],
                        contrastive_objective(lT, eye, bn)[0]), "binary_nce not symmetric"
    assert not jnp.allclose(contrastive_objective(dot, eye, inf)[0],
                            contrastive_objective(lT, eye, inf)[0]), \
        "infonce should differ forward vs backward"

    # the penalty actually enters the loss
    a = contrastive_objective(dot, eye, inf)[0]
    b = contrastive_objective(dot, eye, {**inf, "logsumexp_penalty": 0.1})[0]
    assert float(b) > float(a), "logsumexp penalty had no effect"

    # a perfect critic drives infonce toward 0
    sharp = jnp.transpose(jnp.stack([eye * 50.0] * e), (1, 2, 0))
    assert float(contrastive_objective(sharp, eye, inf)[0]) < 1e-3

    print("critic_losses: all checks passed")
    print(f"  binary_nce(dot)   = {float(contrastive_objective(dot, eye, bn)[0]):.6f}")
    print(f"  infonce(dot)      = {float(contrastive_objective(dot, eye, inf)[0]):.6f}")
    print(f"  infonce(neg_l2)   = {float(contrastive_objective(fast, eye, inf)[0]):.6f}")


if __name__ == "__main__":
    _selftest()


# ---------------------------------------------------------------------------
# Critic ensemble size
# ---------------------------------------------------------------------------
# OGBench hardcodes a 2-member ensemble: `GCBilinearValue.setup` calls
# `ensemblize(MLP, 2)` for each of phi and psi, and `CRLAgent.actor_loss` then does
# `q1, q2 = critic(...)`. The 1000-layer paper uses one member. Supporting that needs
# two coordinated patches -- changing only the first makes the actor unpack a
# length-1 axis and crash, which is why `run.py` refused `critic_ensemble: 1` until
# both existed.
#
# `contrastive_loss` needs no change: it already promotes a 2-D phi to 3-D, and with
# axis_size=1 the arrays simply carry a leading 1.

_ENSEMBLE_N: int | None = None
_orig_ensemblize = None


def _reduce_q(qs):
    """(E, B) -> (B) by min; passes a bare (B,) through.

    `min` over the ensemble is OGBench's own choice (`jnp.minimum(q1, q2)`), kept so a
    2-member run is bit-identical to unpatched OGBench.
    """
    return jnp.min(qs, axis=0) if qs.ndim > 1 else qs


def _patched_actor_loss(self, batch, grad_params, rng=None):
    """`ddpgbc` only, rewritten to reduce over the ensemble axis generically."""
    cfg = self.config
    if cfg["actor_loss"] != "ddpgbc":
        raise NotImplementedError(
            f"ensemble patch supports actor_loss='ddpgbc', not {cfg['actor_loss']!r}")
    dist = self.network.select("actor")(
        batch["observations"], batch["actor_goals"], params=grad_params)
    q_actions = jnp.clip(dist.mode() if cfg["const_std"] else dist.sample(seed=rng),
                         -1, 1)
    qs = self.network.select("critic")(
        batch["observations"], batch["actor_goals"], q_actions)
    q = _reduce_q(qs)

    q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
    log_prob = dist.log_prob(batch["actions"])
    bc_loss = -(cfg["alpha"] * log_prob).mean()
    actor_loss = q_loss + bc_loss
    return actor_loss, {
        "actor_loss": actor_loss, "q_loss": q_loss, "bc_loss": bc_loss,
        "q_mean": q.mean(), "q_abs_mean": jnp.abs(q).mean(),
        "bc_log_prob": log_prob.mean(),
        "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
        "std": jnp.mean(dist.scale_diag),
    }


def install_critic_ensemble(n: int) -> bool:
    """Set the critic ensemble size for the REST OF THE PROCESS. True if patched.

    A no-op at n == 2, so the default path keeps running OGBench's own code.
    Must be called BEFORE the agent is constructed -- it changes how phi/psi are built.
    """
    global _ENSEMBLE_N, _orig_ensemblize
    if n == 2:
        _ENSEMBLE_N = None
        return False
    if n < 1:
        raise ValueError(f"critic_ensemble must be >= 1, got {n}")

    import agents.crl as crl_mod
    import utils.networks as nets

    if _orig_ensemblize is None:
        _orig_ensemblize = nets.ensemblize
    _ENSEMBLE_N = int(n)

    def ensemblize(cls, num_qs, out_axes=0, **kw):
        # `num_qs` is hardcoded 2 at the call site; override it, keep everything else.
        return _orig_ensemblize(cls, _ENSEMBLE_N, out_axes=out_axes, **kw)

    # GCBilinearValue calls `ensemblize` as a module-global, resolved at call time,
    # so rebinding it on the module is enough.
    nets.ensemblize = ensemblize
    crl_mod.CRLAgent.actor_loss = _patched_actor_loss
    return True
