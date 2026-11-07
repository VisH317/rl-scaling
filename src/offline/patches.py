"""Inject the 1000-layer residual tower into OGBench's agents.

OGBench's agents are left unmodified so their published defaults stay visible and
diffable (brief rule #2). The only change is a swap of the network primitive:
`impls/utils/networks.py` `MLP` is replaced, for the duration of agent construction,
by a shim that builds a `ResidualTower` instead.

The swap is exact because OGBench calls `MLP` in only two shapes:

    GCBilinearValue.setup   MLP((*hidden_dims, latent_dim), activate_final=False,
                                layer_norm=...)          -> phi / psi tower WITH a head
    GCActor.setup           MLP(hidden_dims, activate_final=True)
                                                         -> actor trunk, NO head
                                                            (GCActor adds mean_net itself)

so `activate_final` cleanly discriminates the two cases:
  * `activate_final=False` -> tower with an output head of width `hidden_dims[-1]`,
    which is `latent_dim` -- i.e. the representation dimension.
  * `activate_final=True`  -> headless trunk returning `width` features.

`ensemblize` wraps the shim with `nn.vmap`, which is agnostic to what it wraps, so the
2-critic ensemble keeps working.
"""

from __future__ import annotations

import contextlib
from typing import Any, Sequence

import flax.linen as nn

from offline.networks import ResidualTower, count_params, make_tower

#: Architecture used by the shim. Set only inside `residual_arch(...)`.
_ACTIVE_ARCH: dict | None = None


class ResidualMLPShim(nn.Module):
    """Drop-in for OGBench's `MLP` that builds a residual tower.

    Field names and defaults mirror `impls/utils/networks.py` `MLP` so that every
    call site keeps working unchanged.
    """

    hidden_dims: Sequence[int]
    activations: Any = None
    activate_final: bool = False
    kernel_init: Any = None
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        arch = _ACTIVE_ARCH
        if arch is None:
            raise RuntimeError(
                "ResidualMLPShim used outside residual_arch(...); agent construction "
                "must happen inside that context."
            )
        # activate_final=True is the actor trunk: no head, GCActor adds its own.
        # It is also the ONLY signal distinguishing the two call sites, since
        # GCBilinearValue and GCActor both just call MLP(...). `arch["actor_depth"]`
        # rides on that: when set, the policy trunk is built at a fixed depth while
        # the critic towers still follow `arch["depth"]`.
        is_actor_trunk = self.activate_final
        out_dim = None if is_actor_trunk else int(self.hidden_dims[-1])
        if is_actor_trunk and arch.get("actor_depth") is not None:
            arch = {**arch, "depth": int(arch["actor_depth"])}
        tower = make_tower(arch, out_dim=out_dim)
        return tower(x)


def _patch_targets():
    """Every module holding a reference to `MLP`.

    `from utils.networks import MLP` binds a separate name per importing module, so
    patching `utils.networks.MLP` alone would miss the agents.
    """
    import utils.networks as ognet

    targets = [ognet]
    for name in ("agents.crl", "agents.gcbc"):
        try:
            targets.append(__import__(name, fromlist=["MLP"]))
        except Exception:  # noqa: BLE001  -- agent not present; nothing to patch
            pass
    return targets


def install_residual_arch(arch: dict) -> bool:
    """Swap OGBench's `MLP` for the residual shim for the REST OF THE PROCESS.

    This must NOT be scoped to agent construction. Flax re-runs `setup()` on every
    `apply`, so a module tree built with the shim and applied with the original `MLP`
    produces a parameter-structure mismatch (`ScopeParamNotFoundError` on
    `vmap(phi)/Dense_0/kernel`). One run per process, one arch per run, so a
    process-lifetime patch is the correct scope.

    Returns True if the patch was installed, False on explicit opt-out.
    """
    global _ACTIVE_ARCH

    if arch.get("tower") == "ogbench_mlp":
        # Opt-out: keep OGBench's own MLP, for reproducing its published numbers exactly.
        return False

    _ACTIVE_ARCH = arch
    for m in _patch_targets():
        if getattr(m, "MLP", None) is not None:
            setattr(m, "MLP", ResidualMLPShim)
    return True


@contextlib.contextmanager
def residual_arch(arch: dict):
    """Scoped form of `install_residual_arch`, for tests only.

    Do not use this around agent construction in training -- see the note above.
    """
    global _ACTIVE_ARCH
    saved = [(m, getattr(m, "MLP", None)) for m in _patch_targets()]
    try:
        install_residual_arch(arch)
        yield
    finally:
        for m, old in saved:
            if old is not None:
                setattr(m, "MLP", old)
        _ACTIVE_ARCH = None


def agent_param_report(agent) -> dict:
    """Parameter counts per network. `N` in the scaling fit is parameters, not depth."""
    params = agent.network.params
    out: dict[str, int] = {}
    for key, sub in params.items():
        out[key] = count_params(sub)
    out["total"] = count_params(params)
    return out


def build_config(run_cfg: dict, base_config) -> dict:
    """Overlay our run config onto an OGBench agent config dict.

    Only keys OGBench already defines are written, so a typo raises here rather than
    being silently ignored at train time.
    """
    cfg = dict(base_config)
    agent = run_cfg["agent"]
    arch = run_cfg["arch"]
    optim = run_cfg["optim"]

    mapping = {
        "lr": optim["lr"],
        "batch_size": agent["batch_size"],
        "discount": agent["discount"],
        "alpha": agent.get("alpha"),
        "actor_loss": agent.get("actor_loss"),
        "const_std": agent.get("const_std"),
        "latent_dim": arch["repr_dim"],
        "layer_norm": arch["norm"] == "layer_norm",
        "value_p_trajgoal": agent.get("value_p_trajgoal"),
        "value_p_randomgoal": agent.get("value_p_randomgoal"),
        "value_geom_sample": agent.get("value_geom_sample"),
        "actor_p_trajgoal": agent.get("actor_p_trajgoal"),
        "actor_p_randomgoal": agent.get("actor_p_randomgoal"),
        "actor_geom_sample": agent.get("actor_geom_sample"),
    }
    for k, v in mapping.items():
        if v is None or k not in cfg:
            continue
        cfg[k] = v

    if arch.get("tower") == "ogbench_mlp":
        # Anchor runs: leave OGBench's own architecture completely untouched, so a
        # missed target can only be data/flags, never our networks.
        dims = tuple(arch.get("ogbench_hidden_dims", (512, 512, 512)))
        if "value_hidden_dims" in cfg:
            cfg["value_hidden_dims"] = dims
        if "actor_hidden_dims" in cfg:
            cfg["actor_hidden_dims"] = dims
        if "latent_dim" in cfg:
            cfg["latent_dim"] = arch["repr_dim"]
        return cfg

    # For the residual tower, `hidden_dims` carries ONLY the head width: OGBench calls
    # MLP((*hidden_dims, latent_dim)) and the shim takes hidden_dims[-1] as its output
    # dim, so the last element is latent_dim and everything before it is unused. The
    # tower's own depth and width come from `arch`. A 1-tuple makes that explicit --
    # and means the stock 512-wide hidden layers are NOT silently applied.
    width = arch["width"]
    if "value_hidden_dims" in cfg:
        cfg["value_hidden_dims"] = (width,)
    if "actor_hidden_dims" in cfg:
        cfg["actor_hidden_dims"] = (width,)
    return cfg
