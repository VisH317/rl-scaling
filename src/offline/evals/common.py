"""Shared machinery for every eval: checkpoint loading, maze geometry, run selection.

Three things live here because every eval needs them and none of them should be
re-derived per eval:

`load_agent`      rebuild the exact agent a run trained and restore its parameters.
`MazeGeometry`    the maze map, the ij<->xy convention, and a fine-grained geodesic.
`select_runs`     "the best seed at this depth", with the rule recorded, not implied.

Known overlap: `scripts/reeval.py` has its own copy of the agent-rebuild sequence. It
also builds a `GCDataset` because it then rolls the policy out; these evals never need
one, and loading the 1M-row train set costs ~40 s and ~1.2 GB for an example batch of
one row. The two should be merged once reeval is not on the critical path for the
headline numbers -- until then this file is the one the evals use and reeval is left
exactly as it was measured.
"""

from __future__ import annotations

import dataclasses
import glob
import importlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
_IMPLS = os.path.join(os.path.dirname(_SRC), "lib", "ogbench", "impls")
for _p in (_SRC, _IMPLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline import config as C  # noqa: E402

EVALS_DIR = _HERE
SUITE_FILE = os.path.join(EVALS_DIR, "evals.yaml")

#: Where every eval writes. One directory per suite, then per eval, then per run.
EVALS_OUT = os.path.join(C.ANALYSIS_DIR, "evals")


# ---------------------------------------------------------------------------
# Eval discovery
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EvalSpec:
    name: str
    title: str
    status: str                      # "ready" | "planned"
    applies_to: tuple               # agent names
    requires: tuple                 # network modules the eval reads
    env_family: tuple               # "locomaze", "cube", ...  "*" for any
    module: str                      # module in this package exposing run(task)
    figures: str | None
    params: dict
    resources: dict
    dir: str

    @property
    def readme(self) -> str:
        return os.path.join(self.dir, "README.md")

    def impl(self):
        return importlib.import_module(f"offline.evals.{self.name}.{self.module}")

    def figure_module(self):
        if not self.figures:
            return None
        return importlib.import_module(f"offline.evals.{self.name}.{self.figures}")


_SPEC_KEYS = {"name", "title", "status", "applies_to", "requires", "env_family",
              "module", "figures", "params", "resources", "doc"}


def discover() -> dict:
    """Every eval directory under this package, keyed by name.

    A directory without an `eval.yaml` is not an eval (so `__pycache__` and notes
    directories are ignored). A malformed one is an error, not a skip: an eval that
    silently fails to register is an eval you think you ran.
    """
    import yaml

    out = {}
    for path in sorted(glob.glob(os.path.join(EVALS_DIR, "*", "eval.yaml"))):
        d = os.path.dirname(path)
        folder = os.path.basename(d)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - _SPEC_KEYS
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}")
        if raw.get("name") != folder:
            raise ValueError(f"{path}: name={raw.get('name')!r} must equal the folder "
                             f"name {folder!r} -- the folder is the id")
        if raw.get("status") not in ("ready", "planned"):
            raise ValueError(f"{path}: status must be 'ready' or 'planned'")
        if not os.path.exists(os.path.join(d, raw.get("doc", "README.md"))):
            raise ValueError(f"{path}: doc {raw.get('doc', 'README.md')!r} is missing; "
                             f"every eval must describe itself")
        out[folder] = EvalSpec(
            name=folder, title=raw.get("title", folder), status=raw["status"],
            applies_to=tuple(raw.get("applies_to", ["crl"])),
            requires=tuple(raw.get("requires", [])),
            env_family=tuple(raw.get("env_family", ["*"])),
            module=raw.get("module", "impl"), figures=raw.get("figures"),
            params=dict(raw.get("params", {})),
            resources=dict(raw.get("resources", {})), dir=d,
        )
    return out


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

SELECT_RULES = ("reeval", "best_success", "last3", "all")


def _reeval_table(grid: str, episodes: int = None) -> dict:
    episodes = episodes or C.PUBLISHED_EVAL_EPISODES
    p = os.path.join(C.ANALYSIS_DIR, grid, f"reeval_{episodes}ep.json")
    if not os.path.exists(p):
        return {}
    return {r["run_id"]: r for r in json.load(open(p))}


def _eval_csv_tail(run_dir: str, n: int = 3):
    """(mean of the last n evals, last step) from `eval.csv`, for in-flight runs."""
    import csv as _csv
    p = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(p):
        return None, None
    rows = [r for r in _csv.DictReader(open(p)) if r.get("eval/success") not in (None, "")]
    if not rows:
        return None, None
    vals = [float(r["eval/success"]) for r in rows[-n:]]
    step = int(float(rows[-1].get("step", 0)))
    return sum(vals) / len(vals), step


def _run_score(run_dir: str, rule: str, reeval: dict, require_finished: bool):
    """(score, rule_used) for one run, or (None, reason) if it cannot be scored."""
    run_id = os.path.basename(run_dir)
    summary_p = os.path.join(run_dir, "summary.json")
    finished = os.path.exists(summary_p)

    if rule == "last3":
        r = reeval.get(run_id, {})
        v = r.get("last3_eval_mean")
        if v is not None:
            return float(v), "last3"
    if rule in ("reeval", "all"):
        r = reeval.get(run_id)
        if r is not None:
            return float(r["reeval_success"]), "reeval@%d" % r["episodes_per_task"]
        # Fall through to the training-time number, but SAY which one was used: a
        # silent fallback makes a 15-episode score look like a 50-episode one.
    if finished:
        s = json.load(open(summary_p))
        v = s.get("best_success")
        if v is not None:
            return float(v), "best_success@%d" % int(s.get("eval_episodes", 0) or 0)
        return None, "finished but no best_success"
    if require_finished:
        return None, "unfinished (no summary.json)"
    v, step = _eval_csv_tail(run_dir)
    if v is None:
        return None, "unfinished, no evals logged yet"
    return float(v), "in_flight@%s" % f"{step:,}"


def select_runs(grids, agents, depths, rule="reeval", require_finished=True) -> list:
    """One record per (grid, agent, depth) point, naming the run that was picked.

    Returns dicts with `run_id`, `score`, `rule_used`, `all_seeds` and `spread` -- the
    whole seed set, not just the winner. Picking a best seed out of three is a real
    choice; the runner-up and the spread have to travel with it, or a per-depth figure
    looks like a property of depth when it may be a property of one seed.

    With `require_finished: false`, a run that is still training is scored from the
    tail of its `eval.csv` and the rule is recorded as `in_flight@<step>`. That makes
    the half-trained checkpoint usable and impossible to mistake for a finished one.
    """
    if rule not in SELECT_RULES:
        raise ValueError(f"unknown seed_select {rule!r}; one of {SELECT_RULES}")
    out = []
    for grid in grids:
        reeval = _reeval_table(grid)
        for agent in agents:
            for depth in depths:
                pat = os.path.join(C.RUNS_DIR, f"{grid}__{agent}__d{depth}__*")
                cands = sorted(glob.glob(pat))
                scored, skipped = [], []
                for d in cands:
                    score, why = _run_score(d, rule, reeval, require_finished)
                    if score is None:
                        skipped.append([os.path.basename(d), why])
                    else:
                        scored.append((score, d, why))
                if not scored:
                    out.append(dict(grid=grid, agent=agent, depth=depth, run_id=None,
                                    reason=("no scorable run" if cands
                                            else "no run directory"),
                                    skipped=skipped))
                    continue
                scored.sort(key=lambda t: (-t[0], t[1]))
                vals = [s for s, _, _ in scored]
                seeds = {os.path.basename(d).split("__s")[1].split("__")[0]: round(s, 4)
                         for s, d, _ in scored}
                picks = scored if rule == "all" else scored[:1]
                for score, d, used in picks:
                    out.append(dict(
                        grid=grid, agent=agent, depth=depth,
                        run_id=os.path.basename(d), run_dir=d,
                        score=round(score, 4), rule_used=used,
                        n_candidates=len(cands), n_scored=len(scored),
                        spread=round(max(vals) - min(vals), 4),
                        runner_up=round(vals[1], 4) if len(vals) > 1 else None,
                        all_seeds=seeds, skipped=skipped,
                    ))
    return out


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_run_config(run_dir: str) -> dict:
    p = os.path.join(run_dir, "config.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{run_dir} has no config.json")
    return json.load(open(p))


def load_agent(run_dir: str, checkpoint: str = "best"):
    """Rebuild the agent this run trained and restore `params_<checkpoint>.pkl`.

    Returns (agent, env, og_cfg, run_cfg). The architecture patches are
    process-lifetime by design (see `offline/patches.py`), so ONE run per process:
    loading a second checkpoint of a different depth in the same process would apply
    the first run's tower and fail on a parameter-structure mismatch.
    """
    import agents as og_agents
    from utils.flax_utils import restore_agent

    from offline import datasets as D
    from offline.critic_losses import install_critic_ensemble, install_critic_loss
    from offline.patches import agent_param_report, build_config, install_residual_arch

    run_cfg = load_run_config(run_dir)
    ckpt = os.path.join(run_dir, "checkpoints", f"params_{checkpoint}.pkl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt}")

    import ogbench
    env = ogbench.make_env_and_datasets(run_cfg["env_name"], env_only=True)

    agent_name = run_cfg["agent"]["agent"]
    agent_module = importlib.import_module(f"agents.{agent_name}")
    og_cfg = build_config(run_cfg, agent_module.get_config())

    install_residual_arch(run_cfg["arch"])
    install_critic_loss(run_cfg["agent"])
    install_critic_ensemble(int(run_cfg["agent"].get("critic_ensemble", 2)))

    # One row is all `create` reads -- it uses shapes and dtypes to initialise. Taking
    # it from the val npz avoids loading the train set for a single example.
    _, val_paths, _ = D.resolve_paths(run_cfg)
    with np.load(val_paths[0]) as f:
        ex_obs = np.asarray(f["observations"][:1], dtype=np.float32)
        ex_act = np.asarray(f["actions"][:1], dtype=np.float32)

    agent = og_agents.agents[agent_name].create(run_cfg["seed"], ex_obs, ex_act, og_cfg)
    agent = restore_agent(agent, os.path.join(run_dir, "checkpoints"), checkpoint)
    params = agent_param_report(agent)
    return agent, env, og_cfg, dict(run_cfg, _params=params)


def load_states(run_cfg: dict, source: str = "val") -> dict:
    """The raw state/action arrays an eval measures on.

    `val` is the held-out shard -- the honest choice for anything that claims to read
    off a learned representation rather than a memorised one. `train` is the same data
    the critic was fit on, and is here for the train/val gap, not as a default.
    """
    from offline import datasets as D

    train_paths, val_paths, _ = D.resolve_paths(run_cfg)
    paths = {"val": val_paths, "train": train_paths}[source]
    d = D.load_npz_list(paths)
    return dict(observations=d["observations"], actions=d["actions"],
                valids=d["valids"], terminals=d["terminals"], source=source,
                files=[os.path.basename(p) for p in paths])


# ---------------------------------------------------------------------------
# Maze geometry
# ---------------------------------------------------------------------------

class MazeGeometry:
    """The maze map plus the two conversions every spatial eval needs.

    OGBench's convention (`ogbench/locomaze/maze.py`):

        ij_to_xy((i, j)) = (j * unit - offset_x,  i * unit - offset_y)
        xy_to_ij((x, y)) = (int((y + offset_y + unit/2) / unit),
                            int((x + offset_x + unit/2) / unit))

    with `offset_x == offset_y == unit == 4.0` for every maze, so cell (1, 1) sits at
    the world origin. Everything here is derived from the live env object rather than
    hardcoded, because `maze_unit` is a constructor argument.

    A *bin* is a `subdiv x subdiv` subdivision of a maze cell. Distances are computed
    by BFS on the bin grid, not the cell grid: cell distance on a medium maze takes
    ~8 distinct values, which is too coarse to rank against a continuous Euclidean
    distance without most pairs being ties.
    """

    def __init__(self, env, subdiv: int = 4):
        u = env.unwrapped
        self.maze = np.asarray(u.maze_map)
        self.unit = float(getattr(u, "_maze_unit", 4.0))
        self.offset_x = float(getattr(u, "_offset_x", 4.0))
        self.offset_y = float(getattr(u, "_offset_y", 4.0))
        self.subdiv = int(subdiv)
        self.goal_tol = float(getattr(u, "_goal_tol", 0.5))
        self.noise = float(getattr(u, "_noise", 1.0))
        self.add_noise_to_goal = bool(getattr(u, "_add_noise_to_goal", True))
        self.H, self.W = self.maze.shape
        self.fine = np.repeat(np.repeat(self.maze, self.subdiv, 0), self.subdiv, 1)
        self.bin_size = self.unit / self.subdiv
        self.tasks = [dict(task_id=t + 1,
                           init_ij=tuple(int(v) for v in info["init_ij"]),
                           goal_ij=tuple(int(v) for v in info["goal_ij"]),
                           init_xy=[float(v) for v in info["init_xy"]],
                           goal_xy=[float(v) for v in info["goal_xy"]])
                      for t, info in enumerate(u.task_infos)]

    # --- conversions -------------------------------------------------------
    def ij_to_xy(self, ij) -> np.ndarray:
        return np.array([ij[1] * self.unit - self.offset_x,
                         ij[0] * self.unit - self.offset_y], dtype=np.float64)

    def xy_to_bin(self, xy) -> np.ndarray:
        """(N, 2) world xy -> (N, 2) integer bin index on the fine grid."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        bi = np.floor((xy[:, 1] + self.offset_y + 0.5 * self.unit) / self.bin_size)
        bj = np.floor((xy[:, 0] + self.offset_x + 0.5 * self.unit) / self.bin_size)
        bi = np.clip(bi, 0, self.H * self.subdiv - 1).astype(np.int64)
        bj = np.clip(bj, 0, self.W * self.subdiv - 1).astype(np.int64)
        return np.stack([bi, bj], 1)

    def bin_centre_xy(self) -> np.ndarray:
        """(Hf, Wf, 2) world xy of every fine bin's centre."""
        bi = np.arange(self.H * self.subdiv)
        bj = np.arange(self.W * self.subdiv)
        y = (bi + 0.5) * self.bin_size - self.offset_y - 0.5 * self.unit
        x = (bj + 0.5) * self.bin_size - self.offset_x - 0.5 * self.unit
        return np.stack(np.broadcast_arrays(y[:, None], x[None, :]), -1)[..., ::-1]

    def cell_of_bin(self, bi, bj):
        return bi // self.subdiv, bj // self.subdiv

    # --- distances ---------------------------------------------------------
    def geodesic(self, xy) -> np.ndarray:
        """Shortest free-space distance in WORLD UNITS from every bin to `xy`.

        4-connected BFS on the bin grid. Unreachable (walled) bins are `inf`.
        """
        src = self.xy_to_bin(xy)[0]
        occ = self.fine
        dist = np.full(occ.shape, np.inf)
        if occ[src[0], src[1]]:
            # The goal's own bin is inside a wall only if `xy` was clipped; fall back
            # to the nearest free bin rather than returning an all-inf field.
            free = np.argwhere(~occ.astype(bool))
            src = free[np.argmin(np.abs(free - src).sum(1))]
        dist[src[0], src[1]] = 0.0
        queue = [tuple(src)]
        head = 0
        while head < len(queue):
            i, j = queue[head]
            head += 1
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if (0 <= ni < occ.shape[0] and 0 <= nj < occ.shape[1]
                        and not occ[ni, nj] and not np.isfinite(dist[ni, nj])):
                    dist[ni, nj] = dist[i, j] + 1.0
                    queue.append((ni, nj))
        return dist * self.bin_size

    def euclid(self, xy) -> np.ndarray:
        c = self.bin_centre_xy()
        return np.linalg.norm(c - np.asarray(xy, dtype=np.float64), axis=-1)

    def optimal_path(self, start_xy, goal_xy) -> np.ndarray:
        """Bin indices along one shortest path start -> goal, descending the geodesic."""
        d = self.geodesic(goal_xy) / self.bin_size
        cur = tuple(self.xy_to_bin(start_xy)[0])
        if not np.isfinite(d[cur]):
            return np.zeros((0, 2), dtype=np.int64)
        path = [cur]
        while d[cur] > 0:
            nbrs = [(cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]),
                    (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)]
            nbrs = [n for n in nbrs
                    if 0 <= n[0] < d.shape[0] and 0 <= n[1] < d.shape[1]
                    and np.isfinite(d[n])]
            nxt = min(nbrs, key=lambda n: d[n])
            if d[nxt] >= d[cur]:
                break
            cur = nxt
            path.append(cur)
        return np.array(path, dtype=np.int64)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman(a, b) -> float:
    """Rank correlation, NaN-safe, ties averaged. scipy is present; this keeps the
    tie handling and the NaN mask in one place so every eval reports the same thing."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata
    ra, rb = rankdata(a[m]), rankdata(b[m])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Top-down render, with a verified world -> pixel map
# ---------------------------------------------------------------------------

def render_top_down(env, geo: "MazeGeometry", width: int = 480, n_calib: int = 9):
    """A clean top-down image of the maze plus the affine that places world xy on it.

    OGBench's maze camera is `elevation=-90` (straight down) at `lookat=(2(W-3),
    2(H-3), 0)` and `distance=5(W-2)`, so any horizontal plane projects to the image
    by an affine map. Rather than derive it from fovy and azimuth and hope the sign
    conventions match, it is MEASURED: the body is teleported to `n_calib` known free
    cells, each pose is rendered, and the affine is least-squares fitted to where the
    body actually appeared. The fit is then checked leave-one-out, which is a purely
    geometric check -- no assumption about what anything in the image looks like.

    One correction is applied and it is exact arithmetic, not a fudge. The fit is made
    at the body's height `z` (0.795 for the ant, constant), while the value field lives
    on the floor at z=0. Under a camera `d` above the lookat, pixel offset from the
    principal point scales as `1/(d - z)`, so the floor map is the fitted one scaled by
    `(d - z)/d` about the principal point -- which is itself the pixel of the lookat.

    The background is the MEDIAN over the calibration frames, so the body is removed.

    Returns None if the environment cannot render (no GL backend), with the reason.
    Caveat worth knowing when reading the figures: the walls are 3D boxes, so their
    TOPS splay outward from the floor footprint under the perspective camera. The
    overlay is aligned to the floor plane, where the states are; wall tops therefore
    overhang the field slightly at wall edges. Verified once on
    antmaze-medium-stitch-v0: the rendered-wall / predicted-wall IoU peaks at a
    (0, 0) pixel shift, i.e. the floor-plane map is correctly centred.
    """
    try:
        import mujoco
    except Exception as e:                                     # noqa: BLE001
        return None, f"mujoco not importable: {type(e).__name__}"
    u = env.unwrapped
    cam = getattr(u, "custom_camera", None)
    if cam is None or not hasattr(cam, "distance"):
        return None, "env has no custom top-down camera"
    try:
        R = mujoco.Renderer(u.model, width=width, height=width)
    except Exception as e:                                     # noqa: BLE001
        # The offscreen framebuffer is capped by the model XML (<global offheight>).
        try:
            R = mujoco.Renderer(u.model)
            width = R.height
        except Exception as e2:                                # noqa: BLE001
            return None, f"renderer unavailable: {type(e2).__name__}: {e2}"

    try:
        env.reset(options=dict(task_id=1))   # the body must exist before set_xy
    except Exception as e:                                     # noqa: BLE001
        return None, f"reset failed: {type(e).__name__}: {e}"

    free = [(i, j) for i in range(geo.H) for j in range(geo.W) if not geo.maze[i, j]]
    pts = [free[k] for k in np.linspace(0, len(free) - 1, min(n_calib, len(free)))
           .astype(int)]
    frames, xys, zs = [], [], []
    try:
        for ij in pts:
            xy = geo.ij_to_xy(ij)
            u.set_xy(xy)
            mujoco.mj_forward(u.model, u.data)
            R.update_scene(u.data, camera=cam)
            frames.append(R.render().astype(np.float64))
            xys.append(xy)
            zs.append(float(u.data.qpos[2]))
    except Exception as e:                                     # noqa: BLE001
        return None, f"render failed: {type(e).__name__}: {e}"

    frames = np.stack(frames)
    bg = np.median(frames, axis=0)
    uv = []
    for f in frames:
        d = np.abs(f - bg).sum(-1)
        m = d > np.percentile(d, 99.7)
        ii, jj = np.nonzero(m)
        w = d[m]
        uv.append([(jj * w).sum() / w.sum(), (ii * w).sum() / w.sum()])
    uv, xys = np.array(uv), np.array(xys)

    def _fit(X, Y):
        return np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)[0]

    A = _fit(xys, uv)
    rms = float(np.sqrt((((np.c_[xys, np.ones(len(xys))] @ A) - uv) ** 2).sum(1).mean()))
    loo = []
    for k in range(len(xys)):
        keep = [i for i in range(len(xys)) if i != k]
        Ak = _fit(xys[keep], uv[keep])
        loo.append(float(np.linalg.norm(np.r_[xys[k], 1] @ Ak - uv[k])))

    M, t = A[:2].T, A[2]
    z, dist = float(np.mean(zs)), float(cam.distance)
    lookat = np.asarray(cam.lookat, dtype=float)[:2]
    p0 = M @ lookat + t                      # principal point: the lookat's pixel
    Mf = M * (dist - z) / dist               # same map, on the floor plane
    tf = p0 - Mf @ lookat
    return dict(
        image=bg.astype(np.uint8),
        affine=[[Mf[0, 0], Mf[0, 1], tf[0]], [Mf[1, 0], Mf[1, 1], tf[1]]],
        width=int(width), height=int(width), fit_rms_px=round(rms, 3),
        loo_max_px=round(float(max(loo)), 3), n_calib=len(xys),
        torso_z=round(z, 4), camera_distance=dist, lookat=lookat.tolist(),
        floor_scale=round((dist - z) / dist, 6),
    ), None
