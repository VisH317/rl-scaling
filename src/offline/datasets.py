"""Resolve a run's subset id to concrete files and load it.

Three cases, all driven by `config.SubsetPlan` (decision D5, shard-aligned):

  native         a single standard OGBench release   -> one npz
  whole_shards   1M / 3M / 10M of a 100m dataset     -> shards 000..k-1, no new files
  prefix         0.1M / 0.3M                         -> a materialized subset npz
                                                        (written by build_subsets.py)

Loading notes
-------------
* `compact_dataset=True` keeps `valids` and omits `next_observations`, which halves the
  observation memory. OGBench's `Dataset.create` infers next observations by shifting
  indices, so nothing downstream changes.
* `add_info=False`: training never needs `qpos`/`qvel`, and they are ~45% of the bytes.
  The probe and coverage scripts read them straight from the npz instead.
* Multi-shard loads preallocate and fill rather than `np.concatenate`, so peak memory
  is 1x the final array rather than 2x.
* Validation is a FIXED set, independent of training size (`SubsetPlan.val_shards`),
  so probe numbers are comparable across grid cells.
"""

from __future__ import annotations

import os

import numpy as np

from offline import config as C

_KEYS = ("observations", "actions", "terminals")


def _subset_npz_path(subset_id: str) -> str:
    return os.path.join(C.SUBSET_DIR, f"{subset_id}.npz")


def resolve_paths(run_cfg: dict) -> tuple[list[str], list[str], dict]:
    """Return (train_paths, val_paths, info) for a run config."""
    dataset = run_cfg["dataset"]
    spec = C.DATASETS[dataset]
    subset_id = run_cfg["subset_id"]
    info: dict = {"dataset": dataset, "subset_id": subset_id}

    if subset_id.endswith("__native"):
        train_rel, val_rel = spec.rel_paths(0 if spec.is_sharded else None)
        info["mode"] = "native"
        return (
            [os.path.join(C.DATA_DIR, train_rel)],
            [os.path.join(C.DATA_DIR, val_rel)],
            info,
        )

    plan = C.SUBSET_PLANS[dataset]
    sub = next(s for s in plan.resolve() if s.subset_id == subset_id)
    val_paths = [
        os.path.join(C.DATA_DIR, spec.rel_paths(sh)[1]) for sh in plan.val_shards
    ]
    info["mode"] = sub.mode
    info["val_shards"] = list(plan.val_shards)

    if sub.materialize:
        p = _subset_npz_path(subset_id)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"subset {subset_id} not materialized at {p}; "
                f"run: python src/scripts/build_subsets.py"
            )
        return [p], val_paths, info

    info["shards"] = list(sub.shards)
    return (
        [os.path.join(C.DATA_DIR, spec.rel_paths(sh)[0]) for sh in sub.shards],
        val_paths,
        info,
    )


def load_npz_list(paths: list[str]) -> dict:
    """Load and concatenate npz files in compact form, preallocating the result.

    Compact form: `terminals` marks the last *valid* row of each trajectory and
    `valids` masks the final state, matching `ogbench.load_dataset(compact_dataset=True)`.
    """
    shapes: list[tuple[int, dict]] = []
    for p in paths:
        with np.load(p) as f:
            shapes.append((f["observations"].shape[0],
                           {k: (f[k].shape[1:], f[k].dtype) for k in _KEYS}))
    total = sum(n for n, _ in shapes)
    _, meta = shapes[0]

    out: dict = {}
    for k in _KEYS:
        tail, dtype = meta[k]
        dtype = np.float32 if k != "observations" else np.float32
        out[k] = np.empty((total, *tail), dtype=dtype)

    off = 0
    for p, (n, _) in zip(paths, shapes):
        with np.load(p) as f:
            for k in _KEYS:
                out[k][off:off + n] = f[k][...].astype(np.float32, copy=False)
        off += n
    assert off == total

    # Compact transform, applied per the concatenated whole. Because every shard ends
    # on a terminal row, shifting across the concatenation boundary is safe.
    terminals = out["terminals"]
    out["valids"] = (1.0 - terminals).astype(np.float32)
    shifted = np.concatenate([terminals[1:], [1.0]])
    out["terminals"] = np.minimum(terminals + shifted, 1.0).astype(np.float32)
    return out


def dataset_info(d: dict, paths: list[str]) -> dict:
    """Structural facts recorded in every run's config, so `D` is never ambiguous."""
    terminals = d["terminals"]
    valids = d["valids"]
    n_rows = int(terminals.shape[0])
    n_trajs = int(np.sum(valids < 0.5))
    return {
        "n_files": len(paths),
        "files": [os.path.basename(p) for p in paths],
        "n_rows": n_rows,
        "n_trajs": n_trajs,
        # D for the scaling fit: valid transitions, NOT stored rows.
        "n_transitions": int(np.sum(valids > 0.5)),
        "traj_len_rows": n_rows // n_trajs if n_trajs else None,
        "ob_dim": int(d["observations"].shape[1]),
        "act_dim": int(d["actions"].shape[1]),
        "bytes_in_ram": int(sum(v.nbytes for v in d.values())),
    }


def build_datasets(run_cfg: dict):
    """Returns (env, train_raw, val_raw, info). `*_raw` are dicts for `Dataset.create`."""
    import ogbench

    train_paths, val_paths, info = resolve_paths(run_cfg)
    for p in train_paths + val_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing dataset file: {p}")

    env = ogbench.make_env_and_datasets(run_cfg["env_name"], env_only=True)

    train = load_npz_list(train_paths)
    val = load_npz_list(val_paths)

    # Clip actions off the boundary, as Horizon Reduction does; tanh-based log-probs
    # are undefined at exactly +/-1 and the scripted policies saturate.
    eps = 1e-5
    for d in (train, val):
        np.clip(d["actions"], -1 + eps, 1 - eps, out=d["actions"])

    info["train"] = dataset_info(train, train_paths)
    info["val"] = dataset_info(val, val_paths)
    info["n_transitions"] = info["train"]["n_transitions"]
    return env, train, val, info
