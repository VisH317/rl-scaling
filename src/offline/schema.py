"""Hydra structured-config schema for the offline CRL scaling experiments.

Why a separate module from `config.py`:

* `config.py` stays the authoritative record of *what the reference codebases do* --
  every value there is annotated with where it came from. It is readable as a document.
* OmegaConf cannot represent some of the types `config.py` uses (`tuple`, bare `dict`,
  `tuple | None`), so the Hydra schema restates them in OmegaConf-friendly form.

That duplication is a drift risk, so `test_schema_matches_config()` asserts the two
agree on every shared field. It runs in the smoke test and in `launch.py --check`.
If you change a default, change it in `config.py` first; the test will tell you what
else to update.

Every field here is overridable from the command line:

    python src/scripts/launch.py arch.depth=64 agent=crl_stable optim.lr=1e-4
    python src/scripts/launch.py experiment=phase1_repro train.plateau_patience=8
    python src/scripts/launch.py -m arch.depth=4,16,64 agent=crl,gcbc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@dataclass
class PathsConf:
    """Where everything lives. Never point data at $HOME -- 93 GB quota."""

    scratch_root: str = "/scratch/gpfs/vt7993"
    data_dir: str = "${paths.scratch_root}/ogbench/data"
    subset_dir: str = "${paths.scratch_root}/ogbench/subsets"
    experiments_dir: str = "${paths.scratch_root}/rl-scaling/experiments"
    runs_dir: str = "${paths.experiments_dir}/runs"
    configs_dir: str = "${paths.experiments_dir}/configs"
    analysis_dir: str = "${paths.experiments_dir}/analysis"
    slurm_dir: str = "${paths.experiments_dir}/slurm"
    manifest: str = "${paths.experiments_dir}/run_manifest.json"
    venv_python: str = "${paths.scratch_root}/rl-scaling/venv/bin/python"
    repo: str = "/home/vt7993/rl-scaling"


@dataclass
class DatasetConf:
    """Which dataset, and the structural facts measured from the real files."""

    name: str = MISSING
    env_name: str = MISSING
    traj_len: int = MISSING          # states per trajectory AS STORED
    n_train_trajs: int = MISSING
    n_val_trajs: int = MISSING
    ob_dim: int = MISSING
    act_dim: int = MISSING
    n_shards: int = 1                # >1 means a sharded -100m- directory
    train_bytes: int = 0
    val_bytes: int = 0
    #: None = the dataset's native size. Otherwise a nested-subset size in transitions.
    size: Optional[int] = None


@dataclass
class ArchConfig:
    """Network architecture.

    `depth` counts dense layers in the residual stack only (stem and head excluded),
    matching the 1000-layer paper, so it must be divisible by `block_size`.
    """

    depth: int = 4
    width: int = 256
    #: Representation dim. scaling-crl 64 | OGBench 512 | stable-CRL 16. Open question Q2.
    repr_dim: int = 64
    block_size: int = 4
    norm: str = "layer_norm"         # layer_norm | none
    activation: str = "swish"        # swish | relu | gelu
    #: lecun_uniform matches scaling-crl everywhere. "cold" applies stable-CRL's 1e-12
    #: to the head ONLY and is NOT part of the 1000-layer recipe -- opt in deliberately.
    init: str = "lecun_uniform"      # lecun_uniform | cold
    #: residual = the 1000-layer tower; mlp = same norm/activation, no skips (the
    #: collapsing control); ogbench_mlp = leave OGBench's own MLP untouched (anchor).
    tower: str = "residual"
    ogbench_hidden_dims: list = field(default_factory=lambda: [512, 512, 512])


@dataclass
class AgentConf:
    """Algorithm. Defaults are OGBench's published values."""

    agent: str = MISSING             # crl | gcbc
    recipe: str = "ogbench"          # ogbench | stable
    #: Open question Q1. The 1000-layer depth result used infonce + neg_l2 + logsumexp;
    #: both offline codebases use binary_nce + dot with no penalty.
    critic_loss: str = "binary_nce"  # binary_nce | infonce
    logits: str = "dot"              # dot | neg_l2
    logsumexp_penalty: float = 0.0   # scaling-crl uses 0.1 with infonce
    batch_size: int = 1024
    discount: float = 0.99
    alpha: float = 0.1               # BC coefficient in DDPG+BC
    actor_loss: str = "ddpgbc"       # ddpgbc | awr
    critic_ensemble: int = 2
    const_std: bool = True
    value_p_curgoal: float = 0.0
    value_p_trajgoal: float = 1.0
    value_p_randomgoal: float = 0.0
    value_geom_sample: bool = True
    actor_p_curgoal: float = 0.0
    actor_p_trajgoal: float = 1.0
    actor_p_randomgoal: float = 0.0
    actor_geom_sample: bool = False
    #: Apply the published per-(dataset, agent) flags from hyperparameters.sh. Turning
    #: this off reverts to the agent file's bare defaults, which is NOT what OGBench ran.
    use_published_flags: bool = True


@dataclass
class OptimConf:
    """Both reference codebases use plain constant-LR Adam: no schedule, no warmup,
    no clipping, no weight decay. Every knob below is off by default, deliberately --
    warmup and clipping help deep nets specifically and would confound the depth axis.
    """

    optimizer: str = "adam"          # adam | adamw
    lr: float = 3e-4
    actor_lr: Optional[float] = None   # None -> lr
    critic_lr: Optional[float] = None  # None -> lr
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    schedule: str = "constant"       # constant | cosine | linear_warmup_cosine
    warmup_steps: int = 0
    end_lr_fraction: float = 0.1
    max_grad_norm: float = 0.0       # 0 = unclipped


@dataclass
class TrainConf:
    """Training length and the plateau rule."""

    #: Gradient-step cap per dataset size. Overrides the size-derived cap when
    #: `fixed_step_cap` is set (the anchor grid uses 1M to match OGBench).
    step_caps: dict = field(default_factory=lambda: {
        100_000: 500_000, 300_000: 1_000_000, 1_000_000: 2_000_000,
        3_000_000: 3_000_000, 10_000_000: 5_000_000,
    })
    fixed_step_cap: Optional[int] = None
    log_interval: int = 10_000
    eval_interval: int = 100_000
    #: Plateau stops after this many evaluations with no gain in the moving average,
    #: but never before `plateau_min_steps`.
    plateau_patience: int = 5
    plateau_min_delta: float = 0.01
    plateau_window: int = 3
    plateau_min_steps: int = 1_000_000
    seeds: list = field(default_factory=lambda: [0, 1, 2])


@dataclass
class EvalConf:
    """Evaluation protocol. OGBench is 5 goals per task, task_id 1..5."""

    #: During training. OGBench's published commands use 50; we use 15 for cheap curves
    #: and re-score the best checkpoint at `published_episodes` post hoc (reeval.py).
    episodes: int = 15
    published_episodes: int = 50
    temperature: float = 0.0
    gaussian: Optional[float] = None
    #: 0 avoids rendering entirely. EGL does work on the Stellar GPU nodes.
    video_episodes: int = 0
    video_frame_skip: int = 3
    n_tasks: int = 5


@dataclass
class CheckpointConf:
    """OGBench's flax pickle holds params AND Adam state, so restores resume exactly."""

    save_best: bool = True           # phases 2 and 5 need the BEST critic
    save_final: bool = True
    periodic_interval: int = 500_000
    keep_periodic: int = 2
    heartbeat_interval: int = 5_000


@dataclass
class SlurmConf:
    """Stellar GPU QOS: 2 running jobs, 8 GPUs, 2 nodes per user."""

    account: str = "mae"
    partition: str = "gpu"
    max_running_jobs: int = 2
    max_gpus: int = 8
    max_nodes: int = 2
    #: 2, not 4: the only 8-GPU node (stellar-k09g1) was down on 2026-11-05, so a
    #: single-node 4-GPU request is unschedulable. Re-check `sinfo -p gpu`.
    gpus_per_job: int = 2
    runs_per_gpu: int = 3
    #: JAX preallocates 75% of the card by default; must be lowered when packing.
    xla_mem_fraction: float = 0.28
    cpus_per_run: int = 4
    mem_per_cpu: str = "4000M"
    #: NOT the 7-day QOS max: Stellar reserves the 2nd Tuesday monthly and will not
    #: start a job that cannot finish before the reservation.
    time_limit: str = "24:00:00"
    stagger_seconds: int = 20
    deadline_margin_s: int = 1200
    max_attempts: int = 3


@dataclass
class SubsetConf:
    """Nested subsets, shard-aligned (decision D5)."""

    sizes: list = field(default_factory=lambda: [100_000, 300_000, 1_000_000,
                                                 3_000_000, 10_000_000])
    seed: int = 0
    keep_sim_state: bool = True      # qpos/qvel: probe 3 + coverage + -oraclerep
    val_shards: list = field(default_factory=lambda: [0])


@dataclass
class SweepConf:
    """What to expand into runs. Empty lists mean 'use the single value above'."""

    name: str = "adhoc"
    depths: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    agents: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    note: str = ""
    #: Expected results, for the anchor grid. Free text, recorded in the manifest.
    targets: str = ""


@dataclass
class RootConf:
    """Root schema. The defaults LIST lives in conf/config.yaml, not here -- declaring
    it in both places makes Hydra reject the composition ("paths appears more than once
    in the final defaults list")."""

    paths: PathsConf = field(default_factory=PathsConf)
    dataset: DatasetConf = field(default_factory=DatasetConf)
    arch: ArchConfig = field(default_factory=ArchConfig)
    agent: AgentConf = field(default_factory=AgentConf)
    optim: OptimConf = field(default_factory=OptimConf)
    train: TrainConf = field(default_factory=TrainConf)
    eval: EvalConf = field(default_factory=EvalConf)
    checkpoint: CheckpointConf = field(default_factory=CheckpointConf)
    slurm: SlurmConf = field(default_factory=SlurmConf)
    subset: SubsetConf = field(default_factory=SubsetConf)
    experiment: SweepConf = field(default_factory=SweepConf)
    #: Write configs and manifest but do not submit.
    dry_run: bool = False
    #: Submit after generating. "chain" submits every pack with afterany dependencies.
    submit: str = "none"             # none | chain | cap
    #: sbatch dependency gate for the first wave, e.g. a smoke-test job id.
    after: Optional[str] = None


def register() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConf)
    for group, node in (
        ("paths", PathsConf), ("dataset", DatasetConf), ("arch", ArchConfig),
        ("agent", AgentConf), ("optim", OptimConf), ("train", TrainConf),
        ("eval", EvalConf), ("checkpoint", CheckpointConf), ("slurm", SlurmConf),
        ("subset", SubsetConf), ("experiment", SweepConf),
    ):
        cs.store(group=f"schema/{group}", name=group, node=node)
    return cs


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------

def test_schema_matches_config() -> list[str]:
    """Assert the Hydra defaults equal `config.py`'s. Returns a list of mismatches."""
    from offline import config as C

    problems: list[str] = []

    def cmp(label: str, hydra_obj: Any, legacy_obj: Any, skip: tuple = ()) -> None:
        for f in hydra_obj.__dataclass_fields__:
            if f in skip or not hasattr(legacy_obj, f):
                continue
            a = getattr(hydra_obj, f)
            b = getattr(legacy_obj, f)
            if isinstance(a, list) and isinstance(b, tuple):
                a, b = list(a), list(b)
            if isinstance(b, tuple):
                b = list(b)
            if a != b:
                problems.append(f"{label}.{f}: hydra={a!r} config.py={b!r}")

    cmp("arch", ArchConfig(), C.ArchConfig(depth=4))
    cmp("optim", OptimConf(), C.OPTIM)
    cmp("checkpoint", CheckpointConf(), C.CHECKPOINT)
    cmp("slurm", SlurmConf(), C.SLURM)
    cmp("train", TrainConf(), C.TRAIN,
        skip=("fixed_step_cap", "seeds"))
    if list(TrainConf().seeds) != list(C.TRAIN.seeds):
        problems.append(f"train.seeds: hydra={TrainConf().seeds} config.py={C.TRAIN.seeds}")
    if EvalConf().published_episodes != C.PUBLISHED_EVAL_EPISODES:
        problems.append("eval.published_episodes != config.PUBLISHED_EVAL_EPISODES")
    cmp("agent", AgentConf(agent="crl"), C.AgentConfig(agent="crl"),
        skip=("use_published_flags",))
    cmp("subset", SubsetConf(), C.SubsetPlan(dataset="cube-double-play-100m-v0"))
    cmp("paths", PathsConf(), type("P", (), {"scratch_root": C.SCRATCH_ROOT})(),
        skip=tuple(f for f in PathsConf.__dataclass_fields__ if f != "scratch_root"))
    return problems


if __name__ == "__main__":
    import sys
    bad = test_schema_matches_config()
    if bad:
        print("SCHEMA DRIFT between schema.py and config.py:")
        for b in bad:
            print("  ", b)
        sys.exit(1)
    print("schema.py matches config.py on every shared field")
