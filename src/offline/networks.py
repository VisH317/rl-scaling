"""The 1000-layer paper's residual tower, as a drop-in for OGBench's `MLP`.

Transcribed from `wang-kevin3290/scaling-crl/train.py` (verified 2026-11-04). OGBench's
`impls/utils/networks.py` `MLP` differs in three ways that all matter at depth:

    1. no residual connections at all;
    2. GELU, not Swish;
    3. LayerNorm AFTER the activation (`Dense -> GELU -> LN`), where scaling-crl
       normalizes BEFORE it (`Dense -> LN -> Swish`).

`depth` counts dense layers in the residual stack only -- the stem and the output head are
excluded -- matching the paper's convention. So depth 64 is 16 blocks of 4, plus stem and head.

Init: scaling-crl uses `variance_scaling(1/3, 'fan_in', 'uniform')` (lecun_uniform) on every
Dense, including the head. Stable-CRL's `init_w = 1e-12` cold head init is NOT part of that
recipe -- the 1000-layer rebuttal tested it as a separate ablation -- so it lives behind
`init_scheme='cold'` and must be turned on deliberately.

NOTE: the field is `init_scheme`, not `init` -- `init` is `nn.Module.init()` and a dataclass
field of that name silently shadows it. `ArchConfig.init` maps to it in `make_tower`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling

#: scaling-crl's initializer, used on every Dense in the tower.
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
#: stable-CRL's cold init for the final head only.
COLD_INIT_SCALE = 1e-12

ACTIVATIONS: dict[str, Callable] = {
    "swish": nn.swish,
    "relu": nn.relu,
    "gelu": nn.gelu,
}


def cold_init(scale: float = COLD_INIT_SCALE):
    """Uniform in [-scale, scale], matching stable-CRL's `init_w`."""

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return init


class ResidualBlock(nn.Module):
    """`block_size` x (Dense -> Norm -> Activation), then add the identity.

    Pre-activation normalization, matching scaling-crl. The identity is added after the
    final activation of the block (not before it), which is what the reference does.
    """

    width: int
    block_size: int = 4
    norm: str = "layer_norm"
    activation: str = "swish"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        identity = x
        for _ in range(self.block_size):
            x = nn.Dense(self.width, kernel_init=lecun_uniform,
                         bias_init=nn.initializers.zeros)(x)
            if self.norm == "layer_norm":
                x = nn.LayerNorm()(x)
            x = act(x)
        return x + identity


class ResidualTower(nn.Module):
    """Stem -> (depth // block_size) residual blocks -> linear head.

    With `out_dim=None` the head is omitted and the tower returns width-dimensional
    features, which is what an actor trunk wants (OGBench's `GCActor` adds its own mean
    and log-std heads on top).
    """

    depth: int
    width: int = 256
    out_dim: int | None = 64
    block_size: int = 4
    norm: str = "layer_norm"
    activation: str = "swish"
    init_scheme: str = "lecun_uniform"

    @nn.compact
    def __call__(self, x):
        if self.depth % self.block_size != 0:
            raise ValueError(
                f"depth {self.depth} must be divisible by block_size {self.block_size}"
            )
        act = ACTIVATIONS[self.activation]
        # Stem.
        x = nn.Dense(self.width, kernel_init=lecun_uniform,
                     bias_init=nn.initializers.zeros)(x)
        if self.norm == "layer_norm":
            x = nn.LayerNorm()(x)
        x = act(x)
        # Residual stack.
        for _ in range(self.depth // self.block_size):
            x = ResidualBlock(
                width=self.width, block_size=self.block_size,
                norm=self.norm, activation=self.activation,
            )(x)
        # Head.
        if self.out_dim is not None:
            head_init = cold_init() if self.init_scheme == "cold" else lecun_uniform
            x = nn.Dense(self.out_dim, kernel_init=head_init,
                         bias_init=nn.initializers.zeros)(x)
        return x


class PlainTower(nn.Module):
    """Non-residual control, reproducing the "standard architecture" that collapsed with
    depth in the 1000-layer rebuttal. Same norm/activation placement as ResidualTower so
    the residual connections are the *only* difference."""

    depth: int
    width: int = 256
    out_dim: int | None = 64
    block_size: int = 4  # ignored; kept for interface parity
    norm: str = "layer_norm"
    activation: str = "swish"
    init_scheme: str = "lecun_uniform"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for _ in range(self.depth + 1):  # +1 for the stem, to match parameter count
            x = nn.Dense(self.width, kernel_init=lecun_uniform,
                         bias_init=nn.initializers.zeros)(x)
            if self.norm == "layer_norm":
                x = nn.LayerNorm()(x)
            x = act(x)
        if self.out_dim is not None:
            head_init = cold_init() if self.init_scheme == "cold" else lecun_uniform
            x = nn.Dense(self.out_dim, kernel_init=head_init,
                         bias_init=nn.initializers.zeros)(x)
        return x


TOWERS = {"residual": ResidualTower, "mlp": PlainTower}

#: Distinct from None, which means "no output head".
_UNSET: Any = object()


def make_tower(arch: dict | Any, out_dim: int | None | Any = _UNSET) -> nn.Module:
    """Build a tower from an `ArchConfig` or its dict form.

    `out_dim` overrides `arch.repr_dim`. Pass `out_dim=None` for a headless trunk that
    returns width-dimensional features; omit it to use `repr_dim`.
    """
    d = arch if isinstance(arch, dict) else arch.__dict__
    cls = TOWERS[d.get("tower", "residual")]
    return cls(
        depth=d["depth"],
        width=d["width"],
        out_dim=d["repr_dim"] if out_dim is _UNSET else out_dim,
        block_size=d.get("block_size", 4),
        norm=d.get("norm", "layer_norm"),
        activation=d.get("activation", "swish"),
        init_scheme=d.get("init", "lecun_uniform"),
    )


# ---------------------------------------------------------------------------
# Parameter counting -- N in the scaling fit is parameters, not depth.
# ---------------------------------------------------------------------------

def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def tower_params(depth: int, width: int, in_dim: int, out_dim: int | None = 64,
                 **kw) -> int:
    """Instantiate and count, so the number is measured rather than derived."""
    tower = make_tower(
        {"depth": depth, "width": width, "repr_dim": out_dim, **kw}, out_dim=out_dim
    )
    p = tower.init(jax.random.PRNGKey(0), jnp.zeros((1, in_dim)))
    return count_params(p)


def analytic_tower_params(depth: int, width: int, in_dim: int,
                          out_dim: int | None = 64, norm: str = "layer_norm") -> int:
    """Closed form, as a cross-check on `tower_params`."""
    ln = 2 * width if norm == "layer_norm" else 0
    stem = in_dim * width + width + ln
    body = depth * (width * width + width + ln)
    head = (width * out_dim + out_dim) if out_dim is not None else 0
    return stem + body + head


def _selftest() -> None:
    """Verify shapes, the analytic count, and that deep towers stay finite at init."""
    import itertools

    print(f"{'depth':>6} {'width':>6} {'repr':>5} {'params':>12} {'analytic':>12}  ok")
    for depth, width in itertools.product((4, 8, 16, 32, 64), (256,)):
        in_dim, out_dim = 42, 64
        measured = tower_params(depth, width, in_dim, out_dim)
        expected = analytic_tower_params(depth, width, in_dim, out_dim)
        print(f"{depth:>6} {width:>6} {out_dim:>5} {measured:>12,} {expected:>12,}  "
              f"{'yes' if measured == expected else 'MISMATCH'}")
        assert measured == expected, (depth, width, measured, expected)

    # Forward pass stays finite and well-scaled at the deepest setting.
    for depth in (4, 64):
        t = make_tower({"depth": depth, "width": 256, "repr_dim": 64})
        x = jax.random.normal(jax.random.PRNGKey(1), (8, 42))
        p = t.init(jax.random.PRNGKey(0), x)
        y = t.apply(p, x)
        assert y.shape == (8, 64), y.shape
        assert jnp.all(jnp.isfinite(y)), f"depth {depth}: non-finite output"
        print(f"depth {depth:>3}: out std {float(y.std()):.4f}  "
              f"absmax {float(jnp.abs(y).max()):.4f}")

    # Cold init must actually be cold, and only on the head.
    t = make_tower({"depth": 8, "width": 256, "repr_dim": 16, "init": "cold"})
    p = t.init(jax.random.PRNGKey(0), jnp.zeros((1, 42)))
    y = t.apply(p, jax.random.normal(jax.random.PRNGKey(1), (8, 42)))
    assert jnp.abs(y).max() < 1e-9, f"cold init not cold: {jnp.abs(y).max()}"
    print(f"cold head: absmax {float(jnp.abs(y).max()):.3e}  (expect <1e-9)")

    # A trunk with no head returns width-dim features.
    t = make_tower({"depth": 4, "width": 256, "repr_dim": 64}, out_dim=None)
    p = t.init(jax.random.PRNGKey(0), jnp.zeros((1, 42)))
    assert t.apply(p, jnp.zeros((3, 42))).shape == (3, 256)
    print("headless trunk: ok")
    print("\nall self-tests passed")


if __name__ == "__main__":
    _selftest()
