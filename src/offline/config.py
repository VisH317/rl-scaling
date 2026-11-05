"""Central configuration for the offline CRL scaling experiments.

Every number here is either (a) verified against code/data -- see `research_log.md` section 1 --
or (b) an explicit experiment choice. Nothing is a silent default. Where a value disagrees across
the three reference codebases (scaling-crl, OGBench impls, stable-CRL), all three are recorded and
the one we use is marked.

Run `python -m offline.config` to print a resolved summary and sanity-check the grids.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Scratch root. Home quota on Stellar is 93 GB; never put datasets under ~.
SCRATCH_ROOT = os.environ.get("RL_SCALING_SCRATCH", "/scratch/gpfs/vt7993")

DATA_DIR = os.path.join(SCRATCH_ROOT, "ogbench", "data")
SUBSET_DIR = os.path.join(SCRATCH_ROOT, "ogbench", "subsets")
EXPERIMENTS_DIR = os.path.join(SCRATCH_ROOT, "rl-scaling", "experiments")

RUNS_DIR = os.path.join(EXPERIMENTS_DIR, "runs")
CONFIGS_DIR = os.path.join(EXPERIMENTS_DIR, "configs")
ANALYSIS_DIR = os.path.join(EXPERIMENTS_DIR, "analysis")
MANIFEST_DIR = os.path.join(EXPERIMENTS_DIR, "data")

DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"


# ---------------------------------------------------------------------------
# Dataset registry  (verified 2026-11-04 against the live index + downloaded files)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """One OGBench dataset.

    `n_shards > 1` means the dataset lives in a subdirectory as `<stem>-{000..NNN}.npz`,
    each shard a self-contained 1M-transition npz with its own `-val` sibling.
    """

    name: str
    env_name: str
    #: States per trajectory *as stored*. Transitions per trajectory is this minus 1.
    traj_len: int
    n_train_trajs: int
    n_val_trajs: int
    ob_dim: int
    act_dim: int
    #: 1 for a single-file dataset; 100 for the -100m- sharded sets.
    n_shards: int = 1
    #: Approximate bytes per train shard, for disk budgeting.
    train_bytes: int = 0
    val_bytes: int = 0

    @property
    def is_sharded(self) -> bool:
        return self.n_shards > 1

    @property
    def transitions_per_traj(self) -> int:
        return self.traj_len - 1

    @property
    def transitions_per_shard(self) -> int:
        return self.n_train_trajs * self.transitions_per_traj

    def shard_stem(self, shard: int | None = None) -> str:
        """Filename stem (no extension, no -val)."""
        if not self.is_sharded:
            return self.name
        base = self.name.replace("-100m-v0", "-v0")
        return f"{base}-{shard:03d}"

    def rel_paths(self, shard: int | None = None) -> tuple[str, str]:
        """(train, val) paths relative to DATA_DIR."""
        stem = self.shard_stem(shard)
        subdir = f"{self.name}/" if self.is_sharded else ""
        return f"{subdir}{stem}.npz", f"{subdir}{stem}-val.npz"


DATASETS: dict[str, DatasetSpec] = {
    # --- Phase 1 reproduction: the task the 1000-layer rebuttal used for offline GCBC.
    "antmaze-medium-stitch-v0": DatasetSpec(
        name="antmaze-medium-stitch-v0",
        env_name="antmaze-medium-stitch-v0",
        traj_len=201, n_train_trajs=5000, n_val_trajs=500,
        ob_dim=29, act_dim=8,
        train_bytes=245_000_000, val_bytes=25_000_000,
    ),
    "antmaze-medium-navigate-v0": DatasetSpec(
        name="antmaze-medium-navigate-v0",
        env_name="antmaze-medium-navigate-v0",
        traj_len=1001, n_train_trajs=1000, n_val_trajs=100,
        ob_dim=29, act_dim=8,
        train_bytes=232_000_000, val_bytes=23_000_000,
    ),
    "antmaze-large-navigate-v0": DatasetSpec(
        name="antmaze-large-navigate-v0",
        env_name="antmaze-large-navigate-v0",
        traj_len=1001, n_train_trajs=1000, n_val_trajs=100,
        ob_dim=29, act_dim=8,
        train_bytes=232_000_000, val_bytes=23_000_000,
    ),
    # --- Phases 3-4 scaling grid.
    "cube-double-play-100m-v0": DatasetSpec(
        name="cube-double-play-100m-v0",
        env_name="cube-double-play-v0",
        traj_len=1001, n_train_trajs=1000, n_val_trajs=100,
        ob_dim=37, act_dim=5,
        n_shards=100,
        train_bytes=297_000_000, val_bytes=30_000_000,
    ),
    #: The standard 1M release. Downloaded only to check whether it is identical to
    #: shard 000 of the 100m set -- see research_log.md section 2.1 caveat.
    "cube-double-play-v0": DatasetSpec(
        name="cube-double-play-v0",
        env_name="cube-double-play-v0",
        traj_len=1001, n_train_trajs=1000, n_val_trajs=100,
        ob_dim=37, act_dim=5,
        train_bytes=284_000_000, val_bytes=28_000_000,
    ),
    # --- Backup grid tasks (not downloaded by default; see DOWNLOAD_PLAN).
    "puzzle-4x4-play-100m-v0": DatasetSpec(
        name="puzzle-4x4-play-100m-v0",
        env_name="puzzle-4x4-play-v0",
        traj_len=1001, n_train_trajs=1000, n_val_trajs=100,
        ob_dim=-1, act_dim=-1,  # unverified; fill from the manifest once downloaded
        n_shards=100,
        train_bytes=260_000_000, val_bytes=26_000_000,
    ),
    "scene-play-100m-v0": DatasetSpec(
        name="scene-play-100m-v0",
        env_name="scene-play-v0",
        traj_len=1001, n_train_trajs=1000, n_val_trajs=100,
        ob_dim=-1, act_dim=-1,  # unverified
        n_shards=100,
        train_bytes=270_000_000, val_bytes=27_000_000,
    ),
}

#: Keys of DATASETS whose structure has been verified against a downloaded file.
VERIFIED_DATASETS = ("antmaze-medium-stitch-v0", "cube-double-play-100m-v0")


# ---------------------------------------------------------------------------
# What to download
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadItem:
    dataset: str
    #: None for single-file datasets; otherwise an inclusive shard range.
    shards: tuple[int, int] | None = None
    reason: str = ""

    def shard_list(self) -> list[int | None]:
        if self.shards is None:
            return [None]
        return list(range(self.shards[0], self.shards[1] + 1))


#: The grid tops out at 10M transitions, so 10 of 100 cube-double shards suffice
#: (3.3 GB instead of ~31 GB). Everything here totals ~4.3 GB.
DOWNLOAD_PLAN: tuple[DownloadItem, ...] = (
    DownloadItem("antmaze-medium-stitch-v0", None, "phase 1 reproduction (rebuttal's task)"),
    DownloadItem("antmaze-medium-navigate-v0", None, "phase 1 secondary"),
    DownloadItem("antmaze-large-navigate-v0", None, "phase 1 secondary"),
    DownloadItem("cube-double-play-100m-v0", (0, 9), "phases 3-4 grid, up to 10M transitions"),
    DownloadItem("cube-double-play-v0", None, "cross-check vs shard 000 (D5 caveat)"),
)

#: Available but deliberately not fetched. Revisit when the corner grid shows a trend.
DOWNLOAD_DEFERRED: tuple[DownloadItem, ...] = (
    DownloadItem("cube-double-play-100m-v0", (10, 99), "only needed for a 100M-scale final run"),
    DownloadItem("puzzle-4x4-play-100m-v0", (0, 9), "backup grid task if cube-double saturates"),
    DownloadItem("scene-play-100m-v0", (0, 9), "backup grid task if cube-double saturates"),
)


# ---------------------------------------------------------------------------
# Nested subsets  (decision D5: shard-aligned, settled 2026-11-05)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubsetPlan:
    """Nested trajectory subsets of a sharded dataset.

    Shard-aligned construction (D5): sizes at or below one shard are a fixed-seed
    permutation prefix *within* shard 000; sizes above take whole shards 000..k-1.
    This is nested, keeps the 1M cell a standard whole dataset (so it can be checked
    against published numbers), and lets 1M/3M/10M be consumed as plain shard lists
    with no new files.
    """

    dataset: str
    #: Target transition counts. Must each be a whole number of trajectories.
    sizes: tuple[int, ...] = (100_000, 300_000, 1_000_000, 3_000_000, 10_000_000)
    #: Fixed seed for the within-shard permutation. Never change without a new subset id.
    seed: int = 0
    #: Keep qpos/qvel (decision D1). Required by probe 3, coverage stats and -oraclerep.
    keep_sim_state: bool = True
    #: Validation is held FIXED across every grid cell -- never scaled with train size.
    val_shards: tuple[int, ...] = (0,)

    def resolve(self) -> list["SubsetSpec"]:
        spec = DATASETS[self.dataset]
        per_traj = spec.transitions_per_traj
        per_shard = spec.transitions_per_shard
        out: list[SubsetSpec] = []
        for size in self.sizes:
            if size % per_traj != 0:
                raise ValueError(
                    f"{self.dataset}: size {size} is not a whole number of trajectories "
                    f"({per_traj} transitions each)"
                )
            n_trajs = size // per_traj
            if size < per_shard:
                # Within shard 000: permutation prefix.
                mode = "prefix"
                shards = (0,)
                n_trajs_in_prefix = n_trajs
            else:
                if size % per_shard != 0:
                    raise ValueError(
                        f"{self.dataset}: size {size} exceeds one shard but is not a whole "
                        f"number of shards ({per_shard} transitions each)"
                    )
                mode = "whole_shards"
                shards = tuple(range(size // per_shard))
                n_trajs_in_prefix = 0
            out.append(SubsetSpec(
                plan=self, size=size, n_trajs=n_trajs, mode=mode,
                shards=shards, prefix_trajs=n_trajs_in_prefix,
            ))
        return out


@dataclass(frozen=True)
class SubsetSpec:
    plan: SubsetPlan
    size: int
    n_trajs: int
    mode: Literal["prefix", "whole_shards"]
    shards: tuple[int, ...]
    prefix_trajs: int

    @property
    def subset_id(self) -> str:
        """Stable identifier recorded in every run config."""
        m = "M" if self.size >= 1_000_000 else "K"
        n = self.size // (1_000_000 if m == "M" else 1_000)
        return f"{self.plan.dataset}__{n}{m}__seed{self.plan.seed}"

    @property
    def materialize(self) -> bool:
        """Whole-shard subsets need no new files; prefixes do."""
        return self.mode == "prefix"


SUBSET_PLANS: dict[str, SubsetPlan] = {
    "cube-double-play-100m-v0": SubsetPlan(dataset="cube-double-play-100m-v0"),
}


# ---------------------------------------------------------------------------
# Architecture  (verified against wang-kevin3290/scaling-crl/train.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchConfig:
    """The 1000-layer paper's residual tower.

        stem:   Dense(width) -> LayerNorm -> Swish
        body:   (depth // block_size) x residual_block
        head:   Dense(repr_dim)                       # no norm, no activation

        residual_block(x):
            identity = x
            block_size x [ Dense(width) -> LayerNorm -> Swish ]
            return x + identity

    `depth` counts dense layers in the residual stack only (stem and head excluded),
    matching the paper's convention, so it must be divisible by `block_size`.
    """

    depth: int
    width: int = 256                    # scaling-crl CLI default; the brief's choice
    repr_dim: int = 64                  # scaling-crl: 64 | OGBench: 512 | stable-CRL: 16  (see Q2)
    block_size: int = 4                 # 4 dense layers per residual block
    norm: Literal["layer_norm", "none"] = "layer_norm"
    activation: Literal["swish", "relu", "gelu"] = "swish"
    #: "lecun_uniform" matches scaling-crl everywhere. "cold" applies 1e-12 to the
    #: final head only (stable-CRL's init_w) and is an ORTHOGONAL knob -- the
    #: 1000-layer base recipe does not use it.
    init: Literal["lecun_uniform", "cold"] = "lecun_uniform"
    #: Residual towers are the point of the experiment; "mlp" reproduces the
    #: "standard architecture" control that collapsed with depth in the rebuttal.
    tower: Literal["residual", "mlp"] = "residual"

    def __post_init__(self) -> None:
        if self.tower == "residual" and self.depth % self.block_size != 0:
            raise ValueError(
                f"depth {self.depth} must be divisible by block_size {self.block_size}"
            )


#: Depths for the phase-1 depth sweep and the phase 3-4 grid. All divisible by 4.
#: NOTE: the rebuttal's shallowest point was depth 2, which a 4-layer residual block
#: cannot express -- that point must have been a plain MLP. Our floor is depth 4.
PHASE1_DEPTHS = (4, 8, 16, 32)
GRID_DEPTHS = (4, 8, 16, 32, 64)
CORNER_DEPTHS = (4, 16, 64)


# ---------------------------------------------------------------------------
# Algorithms and losses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Algorithm hyperparameters.

    Defaults are OGBench impls' published values, with stable-CRL's deltas available
    via `recipe="stable"`. Verified source values:

                        scaling-crl   OGBench crl.py   stable-CRL
        batch_size            256           1024          2048
        lr                      -           3e-4          3e-4
        repr dim               64            512            16
        BC coefficient          -      alpha=0.1    bc_coef=0.05
        critic ensemble         1              2             2
        logsumexp penalty     0.1           none          none
        logits              -||phi-psi||  phi.psi/sqrt(d)  phi.psi
        critic loss        InfoNCE       binary NCE     binary NCE
    """

    agent: Literal["crl", "gcbc"]
    #: "ogbench" = published OGBench defaults. "stable" = stable-CRL's deltas
    #: (repr_dim 16, batch 2048, bc 0.05, cold head init).
    recipe: Literal["ogbench", "stable"] = "ogbench"
    #: OPEN QUESTION Q1. The 1000-layer depth result used infonce+l2+logsumexp;
    #: both offline codebases use binary_nce+dot. Keeping both switchable so the
    #: pilot can settle it instead of us guessing.
    critic_loss: Literal["binary_nce", "infonce"] = "binary_nce"
    logits: Literal["dot", "neg_l2"] = "dot"
    logsumexp_penalty: float = 0.0       # scaling-crl uses 0.1 with infonce
    batch_size: int = 1024
    lr: float = 3e-4
    discount: float = 0.99
    #: BC coefficient in DDPG+BC (OGBench's `alpha`).
    alpha: float = 0.1
    actor_loss: Literal["ddpgbc", "awr"] = "ddpgbc"
    critic_ensemble: int = 2
    const_std: bool = True
    #: Goal sampling. OGBench crl.py defaults; actor_geom_sample=False is deliberate.
    value_p_trajgoal: float = 1.0
    value_p_randomgoal: float = 0.0
    value_geom_sample: bool = True
    actor_p_trajgoal: float = 1.0
    actor_p_randomgoal: float = 0.0
    actor_geom_sample: bool = False

    def resolved(self) -> "AgentConfig":
        """Apply the `recipe` preset on top of the current values."""
        if self.recipe != "stable":
            return self
        return dataclasses.replace(self, batch_size=2048, alpha=0.05)

    def arch_overrides(self) -> dict:
        """Architecture changes implied by the recipe."""
        if self.recipe == "stable":
            return {"repr_dim": 16, "init": "cold"}
        return {}


#: Loss variants for the Q1 pilot: 2 losses x 2 depths, CRL only.
Q1_PILOT_LOSSES = (
    AgentConfig(agent="crl", critic_loss="binary_nce", logits="dot", logsumexp_penalty=0.0),
    AgentConfig(agent="crl", critic_loss="infonce", logits="neg_l2", logsumexp_penalty=0.1),
)


# ---------------------------------------------------------------------------
# Optimizer  (verified: OGBench impls/agents/crl.py:285 and scaling-crl/train.py:541,556)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimConfig:
    """Optimizer controls.

    VERIFIED BASELINE -- both reference codebases use **plain Adam at a constant learning
    rate**, with no schedule, no warmup, no gradient clipping and no weight decay:

        OGBench  impls/agents/crl.py:285   `optax.adam(learning_rate=config['lr'])`
        scaling-crl train.py:541,556       `optax.adam(learning_rate=args.{actor,critic}_lr)`
                                           actor_lr = critic_lr = 3e-4, constant

    That matters for interpreting the depth result: the 1000-layer paper's stability at
    extreme depth came from the architecture (LayerNorm + residual connections), NOT from
    optimizer tricks. So the DEFAULTS HERE ARE DELIBERATELY PLAIN, and every knob below is
    off. Turning any on is a deviation from both baselines and must be logged as such --
    it would confound the depth axis, since warmup and clipping help deep nets specifically.

    Both codebases use one optimizer per network with the same lr. We keep separate actor
    and critic lrs so phase 5 (the actor thread) can vary them independently.
    """

    #: Adam, constant. 3e-4 in all three reference codebases.
    lr: float = 3e-4
    actor_lr: float | None = None       # None -> use `lr`
    critic_lr: float | None = None      # None -> use `lr`
    optimizer: Literal["adam", "adamw"] = "adam"
    #: optax.adam defaults, stated explicitly rather than inherited silently.
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    #: OFF in both baselines. adamw only.
    weight_decay: float = 0.0
    #: OFF in both baselines. Schedule shape applied to `lr` if enabled.
    schedule: Literal["constant", "cosine", "linear_warmup_cosine"] = "constant"
    warmup_steps: int = 0
    #: Final lr as a fraction of peak, for the decay schedules.
    end_lr_fraction: float = 0.1
    #: OFF in both baselines. `optax.clip_by_global_norm` if > 0.
    max_grad_norm: float = 0.0

    def resolved_lrs(self) -> tuple[float, float]:
        return (self.actor_lr or self.lr, self.critic_lr or self.lr)

    def deviates_from_baseline(self) -> list[str]:
        """Anything here that departs from the verified reference setup."""
        out = []
        if self.schedule != "constant":
            out.append(f"schedule={self.schedule} (both baselines: constant)")
        if self.warmup_steps:
            out.append(f"warmup_steps={self.warmup_steps} (both baselines: 0)")
        if self.max_grad_norm:
            out.append(f"max_grad_norm={self.max_grad_norm} (both baselines: unclipped)")
        if self.weight_decay:
            out.append(f"weight_decay={self.weight_decay} (both baselines: 0)")
        if self.optimizer != "adam":
            out.append(f"optimizer={self.optimizer} (both baselines: adam)")
        a, c = self.resolved_lrs()
        if a != 3e-4 or c != 3e-4:
            out.append(f"lr actor={a} critic={c} (both baselines: 3e-4)")
        return out


OPTIM = OptimConfig()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointConfig:
    """What gets written to disk, when, and what is kept.

    Mechanism is OGBench's `utils/flax_utils.py`:
        save_agent(agent, save_dir, step) -> pickle of
            {'agent': flax.serialization.to_state_dict(agent)}
        written to <save_dir>/params_<step>.pkl
        restore_agent(agent, restore_path, step) reads it back.

    The pickle holds params AND Adam optimizer state, so a restore resumes training
    exactly; it is not a weights-only export.

    Layout per run (under RUNS_DIR/<run_id>/):
        config.json          the resolved run config, including the code commit hash
        train.csv            per-log-interval training metrics
        eval.csv             per-eval-interval evaluation metrics
        metrics.jsonl        append-only union of the above, for the monitor
        status.json          heartbeat: state, step, best success, wall-clock, host
        checkpoints/
            params_best.pkl  the best-evaluation checkpoint (overwritten on improvement)
            params_final.pkl the last step reached
            params_<step>.pkl periodic, pruned to `keep_periodic` most recent
        probes/              probe tables, written in phase 2/4

    Phases 2 and 5 both need the BEST checkpoint (the probes characterize a good critic,
    and the actor thread freezes one), which is why `save_best` is on by default --
    OGBench's main.py only saves periodically and would lose it.
    """

    #: Always keep the best-evaluation checkpoint. Phases 2 and 5 depend on it.
    save_best: bool = True
    #: Always keep the final step, so a plateau-stopped run can be resumed.
    save_final: bool = True
    #: Periodic snapshots, for recovering from a crash mid-run.
    periodic_interval: int = 500_000
    #: Keep only this many periodic snapshots (best and final are exempt).
    keep_periodic: int = 2
    #: A depth-64 run is ~4.3M params per tower; a pickle with Adam state is ~50 MB.
    #: 150 runs x (best + final + 2 periodic) is well within scratch quota.
    compress: bool = False
    #: Heartbeat written every N gradient steps so the monitor can spot a hung run.
    heartbeat_interval: int = 5_000


CHECKPOINT = CheckpointConfig()


# ---------------------------------------------------------------------------
# Training protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Training length and evaluation.

    The brief's key protocol point: Horizon Reduction trained every run for a fixed
    5M gradient steps regardless of dataset size, so larger datasets got fewer passes
    per sample, which can manufacture a data plateau. Here the cap GROWS with dataset
    size and runs stop on a plateau of evaluation success rather than at a fixed step.
    """

    #: Gradient-step cap as a function of dataset size. Keys are transition counts.
    step_caps: dict = field(default_factory=lambda: {
        100_000: 500_000,
        300_000: 1_000_000,
        1_000_000: 2_000_000,
        3_000_000: 3_000_000,
        10_000_000: 5_000_000,
    })
    eval_interval: int = 100_000
    log_interval: int = 10_000
    save_interval: int = 500_000
    #: OGBench protocol: 5 evaluation goals per task. HR's main.py default is 15.
    eval_episodes: int = 15
    eval_temperature: float = 0.0
    #: 0 avoids MuJoCo rendering entirely (EGL on the GPU nodes is untested, Q5).
    video_episodes: int = 0
    #: Early stop: no improvement in best success over this many evaluations.
    plateau_patience: int = 5
    plateau_min_delta: float = 0.01
    seeds: tuple[int, ...] = (0, 1, 2)

    def step_cap(self, size: int) -> int:
        if size in self.step_caps:
            return self.step_caps[size]
        # Linear interpolation in log-size for sizes not listed.
        keys = sorted(self.step_caps)
        if size <= keys[0]:
            return self.step_caps[keys[0]]
        if size >= keys[-1]:
            return self.step_caps[keys[-1]]
        for lo, hi in zip(keys, keys[1:]):
            if lo <= size <= hi:
                f = (size - lo) / (hi - lo)
                return int(self.step_caps[lo] + f * (self.step_caps[hi] - self.step_caps[lo]))
        raise AssertionError("unreachable")


TRAIN = TrainConfig()


# ---------------------------------------------------------------------------
# Experiment grids
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    """One batch of runs: the cross product of depths x sizes x agents x seeds."""

    name: str
    dataset: str
    depths: tuple[int, ...]
    #: None means "the dataset's native size" (phase 1 uses the standard 1M release).
    sizes: tuple[int, ...] | None
    agents: tuple[AgentConfig, ...]
    seeds: tuple[int, ...] = TRAIN.seeds
    arch: ArchConfig | None = None
    note: str = ""

    def n_runs(self) -> int:
        n_sizes = len(self.sizes) if self.sizes else 1
        return len(self.depths) * n_sizes * len(self.agents) * len(self.seeds)

    def cells(self):
        sizes = self.sizes if self.sizes else (None,)
        for depth, size, agent, seed in itertools.product(
            self.depths, sizes, self.agents, self.seeds
        ):
            yield depth, size, agent, seed


_CRL = AgentConfig(agent="crl")
_GCBC = AgentConfig(agent="gcbc")

GRIDS: dict[str, GridSpec] = {
    # --- Phase 1b: the offline negative result. The real baseline.
    "phase1_repro": GridSpec(
        name="phase1_repro",
        dataset="antmaze-medium-stitch-v0",
        depths=PHASE1_DEPTHS,
        sizes=None,
        agents=(_CRL, _GCBC),
        note="Expect CRL flat with depth, GCBC rising (rebuttal: 0.47 -> 0.63, depth 2 -> 32). "
             "If CRL scales here, stop and report -- that alone changes the project.",
    ),
    # --- Q1 pilot: does the critic loss form change the depth verdict?
    "q1_loss_pilot": GridSpec(
        name="q1_loss_pilot",
        dataset="antmaze-medium-stitch-v0",
        depths=(4, 32),
        sizes=None,
        agents=Q1_PILOT_LOSSES,
        seeds=(0,),
        note="4 runs. binary_nce+dot (offline codebases) vs infonce+neg_l2+logsumexp "
             "(the recipe the 1000-layer depth result was actually obtained with).",
    ),
    # --- Phase 3 corner: run this before filling the grid.
    "phase3_corner": GridSpec(
        name="phase3_corner",
        dataset="cube-double-play-100m-v0",
        depths=CORNER_DEPTHS,
        sizes=(100_000, 1_000_000, 10_000_000),
        agents=(_CRL, _GCBC),
        note="9 cells x 2 algorithms x 3 seeds = 54 runs. Fill the full grid only if a "
             "trend shows here.",
    ),
    # --- Phase 3-4 full grid.
    "phase3_full": GridSpec(
        name="phase3_full",
        dataset="cube-double-play-100m-v0",
        depths=GRID_DEPTHS,
        sizes=(100_000, 300_000, 1_000_000, 3_000_000, 10_000_000),
        agents=(_CRL, _GCBC),
        note="25 cells x 2 algorithms x 3 seeds = 150 runs.",
    ),
}


# ---------------------------------------------------------------------------
# Slurm  (Stellar: 2 running jobs, 8 GPUs, 2 nodes per user; A100 40 GB)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlurmConfig:
    account: str = "mae"
    partition: str = "gpu"
    #: QOS caps. Exceeding these holds the job rather than failing it.
    max_running_jobs: int = 2
    max_gpus: int = 8
    max_nodes: int = 2
    #: 2, not 4. Stellar's only 8-GPU node (stellar-k09g1) was in state `inval` on
    #: 2026-11-05, leaving six nodes with 2 GPUs each, so a single-node 4-GPU request
    #: is unschedulable ("ReqNodeNotAvail"). 2 GPUs x 1 node x 2 jobs also stays inside
    #: the QOS 2-node cap. Re-check `sinfo -p gpu` before raising this.
    gpus_per_job: int = 2
    #: Runs co-located on one GPU. These networks are tiny (~12M params at depth 64,
    #: width 256), so VRAM is not the constraint -- host RAM is, since each process
    #: holds its own dataset copy.
    runs_per_gpu: int = 3
    #: JAX preallocates 75% of the card by default; must be overridden when packing.
    xla_mem_fraction: float = 0.28
    cpus_per_run: int = 4
    #: Stellar's 7500M/core default breaks on the AMD GPU nodes (512/128 = 4 GB/core).
    mem_per_cpu: str = "4000M"
    #: NOT the 7-day QOS maximum. Stellar reserves the second Tuesday of each month
    #: 06:00-14:00 for maintenance, and Slurm will not start a job that cannot finish
    #: before a reservation -- a 7-day request sits PENDING with
    #: "ReqNodeNotAvail, Reserved for maintenance" until after the window. Phase-1 runs
    #: are a few hours each and a pack runs them concurrently, so 24h schedules
    #: immediately and still leaves generous headroom. Raise it only when a run
    #: genuinely needs longer AND the next maintenance window is far enough away.
    time_limit: str = "24:00:00"

    @property
    def runs_per_pack(self) -> int:
        return self.gpus_per_job * self.runs_per_gpu

    @property
    def max_concurrent_runs(self) -> int:
        return self.max_running_jobs * self.runs_per_pack


SLURM = SlurmConfig()


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def run_config(
    grid: str, depth: int, size: int | None, agent: AgentConfig, seed: int,
    arch: ArchConfig | None = None,
) -> dict:
    """Fully resolved config for one run. This dict IS the run's identity."""
    spec = GRIDS[grid]
    agent = agent.resolved()
    base = arch or spec.arch or ArchConfig(depth=depth)
    base = dataclasses.replace(base, depth=depth, **agent.arch_overrides())
    ds = DATASETS[spec.dataset]
    cfg = {
        "grid": grid,
        "dataset": spec.dataset,
        "env_name": ds.env_name,
        "subset_id": _subset_id(spec.dataset, size),
        "size": size if size is not None else ds.transitions_per_shard,
        "seed": seed,
        "arch": dataclasses.asdict(base),
        "agent": dataclasses.asdict(agent),
        "train": {
            k: v for k, v in dataclasses.asdict(TRAIN).items() if k != "step_caps"
        },
        "step_cap": TRAIN.step_cap(size if size is not None else ds.transitions_per_shard),
        "optim": dataclasses.asdict(OPTIM),
        "checkpoint": dataclasses.asdict(CHECKPOINT),
        "optim_deviations": OPTIM.deviates_from_baseline(),
    }
    cfg["run_id"] = _run_id(cfg)
    return cfg


def _subset_id(dataset: str, size: int | None) -> str:
    if size is None:
        return f"{dataset}__native"
    plan = SUBSET_PLANS.get(dataset)
    if plan is None:
        return f"{dataset}__{size}"
    for s in plan.resolve():
        if s.size == size:
            return s.subset_id
    raise KeyError(f"no subset of size {size} planned for {dataset}")


def _run_id(cfg: dict) -> str:
    payload = json.dumps({k: v for k, v in cfg.items() if k != "run_id"}, sort_keys=True)
    h = hashlib.sha1(payload.encode()).hexdigest()[:8]
    a = cfg["agent"]
    return (
        f"{cfg['grid']}__{a['agent']}__d{cfg['arch']['depth']}"
        f"__n{cfg['size']}__s{cfg['seed']}__{h}"
    )


# ---------------------------------------------------------------------------
# Summary / self-check
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def download_bytes(plan=DOWNLOAD_PLAN) -> int:
    total = 0
    for item in plan:
        spec = DATASETS[item.dataset]
        n = len(item.shard_list())
        total += n * (spec.train_bytes + spec.val_bytes)
    return total


def main() -> None:
    print("=" * 76)
    print("Offline CRL scaling -- resolved configuration")
    print("=" * 76)
    print(f"\ndata    {DATA_DIR}\nsubsets {SUBSET_DIR}\nruns    {RUNS_DIR}")

    print(f"\n-- download plan ({_fmt_bytes(download_bytes())}) " + "-" * 30)
    for item in DOWNLOAD_PLAN:
        spec = DATASETS[item.dataset]
        n = len(item.shard_list())
        rng = "" if item.shards is None else f" shards {item.shards[0]:03d}-{item.shards[1]:03d}"
        size = _fmt_bytes(n * (spec.train_bytes + spec.val_bytes))
        print(f"  {item.dataset}{rng}  {size:>9}   {item.reason}")
    print(f"  deferred: {_fmt_bytes(download_bytes(DOWNLOAD_DEFERRED))}")

    print("\n-- nested subsets (D5: shard-aligned) " + "-" * 36)
    for plan in SUBSET_PLANS.values():
        for s in plan.resolve():
            tag = "materialize" if s.materialize else f"shards {s.shards[0]:03d}-{s.shards[-1]:03d}"
            print(f"  {s.size/1e6:>5.1f}M  {s.n_trajs:>6} trajs  {s.mode:<13} {tag}")

    print("\n-- grids " + "-" * 65)
    for g in GRIDS.values():
        print(f"  {g.name:<16} {g.n_runs():>4} runs   depths={g.depths}  "
              f"sizes={'native' if not g.sizes else tuple(f'{x/1e6:g}M' for x in g.sizes)}")

    print("\n-- slurm " + "-" * 65)
    print(f"  {SLURM.runs_per_pack} runs/pack ({SLURM.gpus_per_job} GPUs x "
          f"{SLURM.runs_per_gpu}/GPU), {SLURM.max_running_jobs} packs concurrent "
          f"-> {SLURM.max_concurrent_runs} runs in flight")
    for g in ("phase1_repro", "phase3_corner", "phase3_full"):
        packs = -(-GRIDS[g].n_runs() // SLURM.runs_per_pack)
        print(f"  {g:<16} {GRIDS[g].n_runs():>4} runs = {packs:>3} packs")

    print("\n-- example run config " + "-" * 52)
    cfg = run_config("phase1_repro", depth=16, size=None, agent=_CRL, seed=0)
    print(f"  run_id    {cfg['run_id']}")
    print(f"  subset_id {cfg['subset_id']}")
    print(f"  step_cap  {cfg['step_cap']:,}")
    print()


if __name__ == "__main__":
    main()
