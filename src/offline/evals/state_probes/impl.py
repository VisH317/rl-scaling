"""Linear probes on the critic's representations.

The design, and the reasoning behind every choice here, is in README.md beside this
file. What the code adds to that document:

WHICH TARGET BELONGS TO WHICH SITE. `Q = phi(s,a) . psi(g)`. Neither factor sees the
other, so a target that is a property of the PAIR cannot be probed from either one
alone, and a target that is a property of the GOAL must be computed on the goal
observation rather than the state. The README's single target list silently assumed
otherwise. The mapping used here, recorded in `probes.json` as `site_targets`:

    phi_state_action   xy, maze_cell, geodesic_to_task, torso_z, joint_vel_norm
                       -- all functions of s alone
    psi_goal           xy, maze_cell, geodesic_to_task
                       -- the same three, computed ON THE GOAL observation
    phi_psi_concat     steps_to_goal
                       -- a pair property; [phi | psi] is the smallest site that
                          could carry it, and it is the quantity the geometric goal
                          sampler makes Q a monotone function of

VELOCITY BLOCK. `joint_vel_norm` needs to know where the velocities start, which
differs per body (29-d ant vs 69-d humanoid). It is read from the MuJoCo model rather
than hardcoded: the locomaze observation is `[x, y, qpos[2:], qvel]`, so its width is
`nq + nv` and the velocities are the trailing `nv`. The constructor asserts that
identity, so a body whose observation is assembled differently fails loudly instead of
probing the wrong columns.

COST. Everything after the representations is numpy on a 64-wide feature matrix, so the
GPU is used only to run the towers once per site.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from offline.evals import common

#: Targets that are functions of a single observation, and so can be computed for a
#: state or for a goal by passing the right array.
_OBS_TARGETS = ("xy", "maze_cell", "geodesic_to_task", "torso_z", "joint_vel_norm")

SITE_TARGETS = {
    "phi_state_action": ("xy", "maze_cell", "geodesic_to_task",
                         "torso_z", "joint_vel_norm"),
    "psi_goal": ("xy", "maze_cell", "geodesic_to_task"),
    "phi_psi_concat": ("steps_to_goal",),
}

_KIND = {"xy": "regression", "maze_cell": "classification",
         "geodesic_to_task": "regression", "steps_to_goal": "regression",
         "torso_z": "regression", "joint_vel_norm": "regression"}


# ---------------------------------------------------------------------------
# Trajectory bookkeeping
# ---------------------------------------------------------------------------

def trajectory_bounds(n_rows: int, traj_len: int):
    """(traj_id per row, [start, end] per trajectory).

    OGBench stores trajectories concatenated at a fixed length, so this is exact
    rather than inferred from terminals -- and `n_rows % traj_len == 0` is asserted,
    because a silent off-by-one here would scramble the trajectory split that the
    whole method rests on.
    """
    if traj_len <= 1 or n_rows % traj_len:
        raise ValueError(f"{n_rows} rows is not a whole number of {traj_len}-row "
                         f"trajectories")
    n_traj = n_rows // traj_len
    tid = np.repeat(np.arange(n_traj), traj_len)
    bounds = np.stack([np.arange(n_traj) * traj_len,
                       (np.arange(n_traj) + 1) * traj_len - 1], axis=1)
    return tid, bounds


def sample_goals(idx, tid, bounds, discount, rng):
    """A future index in the SAME trajectory, geometric with the critic's discount.

    This is the value-goal distribution `GCDataset` trains the critic on at
    `value_p_trajgoal = 1.0`. Probing psi on anything else would measure
    extrapolation to goals the critic never saw, which is a different question and
    one this eval does not claim to answer.
    """
    ends = bounds[tid[idx], 1]
    offs = rng.geometric(1.0 - discount, size=idx.shape[0])
    return np.minimum(idx + offs, ends)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

class TargetBuilder:
    def __init__(self, env, geo, ob_dim):
        self.geo = geo
        m = env.unwrapped.model
        self.nq, self.nv = int(m.nq), int(m.nv)
        if self.nq + self.nv != ob_dim:
            raise ValueError(
                f"observation is {ob_dim}-d but the model has nq={self.nq}, "
                f"nv={self.nv} (expected nq+nv for a locomaze [x, y, qpos[2:], qvel] "
                f"observation). Refusing to guess where the velocities start.")
        # Free cells, in a fixed order, so class ids are stable across depths.
        free = np.argwhere(geo.maze == 0)
        self.cell_index = {(int(i), int(j)): k for k, (i, j) in enumerate(free)}
        self.n_cells = len(free)
        self.task_goal_ij = self._task_goals(geo)
        self.task_bfs = [geo.cell_bfs(ij) for ij in self.task_goal_ij]

    @staticmethod
    def _task_goals(geo):
        return [t["goal_ij"] for t in geo.tasks]

    def cell_of(self, xy):
        """(N, 2) world xy -> (N, 2) maze cell (i, j).

        Via the bin grid, which is the only vectorised conversion MazeGeometry has
        (`ij_to_xy` goes the other way and takes one pair). At subdiv 1 a bin IS a
        cell, which is why this eval builds its geometry with subdiv=1.
        """
        b = self.geo.xy_to_bin(xy)
        return np.stack(self.geo.cell_of_bin(b[:, 0], b[:, 1]), axis=1)

    def build(self, name, obs):
        """`(y, kind)` for one obs-shaped target."""
        if name == "xy":
            return obs[:, :2].astype(np.float64), "regression"
        if name == "torso_z":
            return obs[:, 2:3].astype(np.float64), "regression"
        if name == "joint_vel_norm":
            v = obs[:, self.nq:]
            return np.linalg.norm(v, axis=1, keepdims=True).astype(np.float64), "regression"
        ij = self.cell_of(obs[:, :2])
        if name == "maze_cell":
            y = np.array([self.cell_index.get((int(a), int(b)), -1) for a, b in ij])
            return y, "classification"
        if name == "geodesic_to_task":
            cols = []
            for bfs in self.task_bfs:
                d = bfs[ij[:, 0], ij[:, 1]].astype(np.float64)
                d[~np.isfinite(d)] = np.nan
                cols.append(d * self.geo.unit)
            return np.stack(cols, axis=1), "regression"
        raise ValueError(f"unknown target {name!r}")


# ---------------------------------------------------------------------------
# Probes -- ridge regression and multinomial logistic, both with grouped CV
# ---------------------------------------------------------------------------

def grouped_folds(groups, k, rng):
    """k folds that never split a trajectory across the boundary."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, k)
    return [np.isin(groups, c) for c in chunks]


def _shuffle_by_trajectory(y, groups, rng):
    """Give each trajectory another trajectory's label sequence.

    Not a plain row permutation: that destroys the within-trajectory structure as
    well as the feature-label link, so it would score ~0 even if the split were
    leaking. Reassigning whole blocks keeps each trajectory's labels as internally
    correlated as they really are, and still breaks their tie to the features -- which
    is what makes this a test of the SPLIT and not just of the solver. Blocks differ
    in length after subsampling, so the donor is indexed modulo its own length.
    """
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    src = np.empty(len(groups), dtype=np.int64)
    for i, g in enumerate(uniq):
        tgt, donor = idx_by_g[g], idx_by_g[uniq[perm[i]]]
        src[tgt] = donor[np.arange(len(tgt)) % len(donor)]
    return y[src]


def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge(Xtr, Ytr, Xte, lam):
    """Closed form, with the intercept handled by centring rather than a bias column
    (so `lam` never penalises the mean)."""
    ym = Ytr.mean(0)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ (Ytr - ym))
    return Xte @ W + ym


def r2_score(y, yhat):
    """Fraction of variance explained, against the TEST mean, averaged over columns.

    Against the test mean, so 0 means 'no better than predicting the mean of the thing
    being predicted' and a negative number means actively worse -- which is a real
    outcome for a probe and must not be clipped away.
    """
    ss_res = np.nansum((y - yhat) ** 2, axis=0)
    ss_tot = np.nansum((y - np.nanmean(y, axis=0)) ** 2, axis=0)
    ok = ss_tot > 1e-12
    if not ok.any():
        return float("nan")
    return float(np.mean(1.0 - ss_res[ok] / ss_tot[ok]))


def fit_regression(X, y, groups, train, lambdas, folds, rng):
    inner = grouped_folds(groups[train], folds, rng)
    Xtr_all, ytr_all = X[train], y[train]
    best, best_lam, curve = -np.inf, lambdas[0], {}
    for lam in lambdas:
        sc = []
        for mask in inner:
            if mask.all() or not mask.any():
                continue
            a, b = _standardise(Xtr_all[~mask], Xtr_all[mask])
            sc.append(r2_score(ytr_all[mask], ridge(a, ytr_all[~mask], b, lam)))
        curve[f"{lam:g}"] = round(float(np.mean(sc)), 4) if sc else None
        if sc and np.mean(sc) > best:
            best, best_lam = float(np.mean(sc)), lam
    a, b = _standardise(X[train], X[~train])
    return dict(score=round(r2_score(y[~train], ridge(a, y[train], b, best_lam)), 4),
                metric="r2", lam=best_lam, cv_best=round(best, 4), cv=curve)


def _logistic(Xtr, ytr, Xte, n_cls, lam, steps=250, lr=0.1):
    import jax
    import jax.numpy as jnp
    import optax

    Xtr_j, Xte_j = jnp.asarray(Xtr), jnp.asarray(Xte)
    Y = jax.nn.one_hot(jnp.asarray(ytr), n_cls)
    W = jnp.zeros((Xtr.shape[1], n_cls))
    b = jnp.zeros((n_cls,))
    opt = optax.adam(lr)
    state = opt.init((W, b))

    def loss_fn(params):
        W, b = params
        logits = Xtr_j @ W + b
        ce = optax.softmax_cross_entropy(logits, Y).mean()
        return ce + lam * jnp.sum(W ** 2) / Xtr.shape[0]

    @jax.jit
    def step(params, state):
        g = jax.grad(loss_fn)(params)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state

    params = (W, b)
    for _ in range(steps):
        params, state = step(params, state)
    W, b = params
    return np.asarray(jnp.argmax(Xte_j @ W + b, axis=1))


def fit_classification(X, y, groups, train, lambdas, folds, rng, n_cls):
    inner = grouped_folds(groups[train], folds, rng)
    Xtr_all, ytr_all = X[train], y[train]
    best, best_lam, curve = -np.inf, lambdas[0], {}
    for lam in lambdas:
        sc = []
        for mask in inner:
            if mask.all() or not mask.any():
                continue
            a, b = _standardise(Xtr_all[~mask], Xtr_all[mask])
            pred = _logistic(a, ytr_all[~mask], b, n_cls, lam)
            sc.append(float((pred == ytr_all[mask]).mean()))
        curve[f"{lam:g}"] = round(float(np.mean(sc)), 4) if sc else None
        if sc and np.mean(sc) > best:
            best, best_lam = float(np.mean(sc)), lam
    a, b = _standardise(X[train], X[~train])
    pred = _logistic(a, y[train], b, n_cls, best_lam)
    acc = float((pred == y[~train]).mean())
    # Chance is the largest class in the TEST set, not 1/n_cells: the maze is visited
    # very unevenly, so 1/n_cells would flatter every probe equally.
    _, counts = np.unique(y[~train], return_counts=True)
    return dict(score=round(acc, 4), metric="accuracy", lam=best_lam,
                cv_best=round(best, 4), cv=curve,
                chance=round(float(counts.max() / counts.sum()), 4))


# ---------------------------------------------------------------------------

def representations(agent, obs, acts, goals, max_batch=4096):
    """`phi(s,a)` and `psi(g)`, ensemble axis kept separate. -> (E, N, d) each."""
    import jax.numpy as jnp

    net = agent.network

    def one(o, a, g):
        _, phi, psi = net.select("critic")(o, g, actions=a, info=True)
        if phi.ndim == 2:                     # non-ensemble
            phi, psi = phi[None], psi[None]
        return np.asarray(phi), np.asarray(psi)

    ph, ps = [], []
    for i in range(0, len(obs), max_batch):
        sl = slice(i, i + max_batch)
        a, b = one(jnp.asarray(obs[sl]), jnp.asarray(acts[sl]), jnp.asarray(goals[sl]))
        ph.append(a)
        ps.append(b)
    return np.concatenate(ph, axis=1), np.concatenate(ps, axis=1)


def run(task: dict) -> dict:
    p = dict(task["params"])
    out_dir = task["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(int(p.get("seed", 0)))

    import ogbench
    run_cfg = common.load_run_config(task["run_dir"])
    env = ogbench.make_env_and_datasets(run_cfg["env_name"], env_only=True)
    geo = common.MazeGeometry(env, subdiv=1)

    st = common.load_states(run_cfg, source=p.get("state_source", "val"))
    obs_all, act_all = st["observations"], st["actions"]
    traj_len = int(p.get("traj_len") or _infer_traj_len(run_cfg, len(obs_all)))
    tid, bounds = trajectory_bounds(len(obs_all), traj_len)
    n_traj = len(bounds)

    # The configured split is a CAP. The humanoid val shard holds 100 trajectories
    # against a request for 500, and silently training on 400 of 100 would either
    # crash or, worse, overlap. Scale to the same ratio and record what was used.
    want_tr, want_te = int(p["n_train_trajs"]), int(p["n_test_trajs"])
    if want_tr + want_te > n_traj:
        frac = want_tr / (want_tr + want_te)
        want_tr = max(1, int(round(n_traj * frac)))
        want_te = n_traj - want_tr
    order = rng.permutation(n_traj)
    tr_traj, te_traj = order[:want_tr], order[want_tr:want_tr + want_te]

    # Subsample states within each kept trajectory: neighbours 0.05 s apart are near
    # duplicates, so every 20th row costs almost no information and keeps the solves
    # small.
    stride = int(p.get("state_stride", 20))
    keep = np.concatenate([
        np.arange(bounds[t, 0], bounds[t, 1] + 1, stride)
        for t in np.concatenate([tr_traj, te_traj])])
    rng.shuffle(keep)
    cap = int(p.get("max_states", 20000))
    if len(keep) > cap:
        keep = keep[:cap]
    is_train = np.isin(tid[keep], tr_traj)

    discount = float(run_cfg["agent"].get("discount", 0.99))
    gidx = sample_goals(keep, tid, bounds, discount, rng)

    obs, acts, gobs = obs_all[keep], act_all[keep], obs_all[gidx]
    groups = tid[keep]
    steps_to_goal = (gidx - keep).astype(np.float64)[:, None]

    tb = TargetBuilder(env, geo, obs.shape[1])
    print(f"  {len(keep)} states from {n_traj} trajectories "
          f"({want_tr} train / {want_te} test), stride {stride}, discount {discount}")

    agent, env, og_cfg, run_cfg2 = common.load_agent(task["run_dir"],
                                                     task.get("checkpoint", "best"),
                                                     env=env)
    phi, psi = representations(agent, obs, acts, gobs, int(p.get("max_batch", 4096)))
    n_ens = phi.shape[0]

    # The random-init control: the SAME architecture and width at the SAME depth, never
    # trained. A deeper random tower is a different random basis and can be a better
    # one, so without this a probe score that rises with depth is not yet evidence of
    # learning. Built after the patches, so it is genuinely the same network.
    import agents as og_agents
    rnd = og_agents.agents[run_cfg["agent"]["agent"]].create(
        int(run_cfg["seed"]) + 977, obs[:1], acts[:1], og_cfg)
    rphi, rpsi = representations(rnd, obs, acts, gobs, int(p.get("max_batch", 4096)))

    # `raw_obs` is per SITE, and that is not a detail. A tower's baseline has to be
    # the raw input THAT TOWER sees, or the comparison is rigged: probing psi_goal for
    # the goal's position against a baseline of the STATE observation asks the baseline
    # a question it cannot answer (it scored 0.38 where the honest baseline scores
    # ~1.0) and would have reported the representation as adding information it does
    # not add.
    raw_for = {
        "phi_state_action": np.concatenate([obs, acts], axis=1).astype(np.float64),
        "psi_goal": gobs.astype(np.float64),
        "phi_psi_concat": np.concatenate([obs, acts, gobs], axis=1).astype(np.float64),
    }
    feature_sets = {}
    for site in SITE_TARGETS:
        fs = {"raw_obs": raw_for[site]}
        for e in range(n_ens):
            trained = {"phi_state_action": phi[e], "psi_goal": psi[e],
                       "phi_psi_concat": np.concatenate([phi[e], psi[e]], axis=1)}[site]
            untrained = {"phi_state_action": rphi[e], "psi_goal": rpsi[e],
                         "phi_psi_concat": np.concatenate([rphi[e], rpsi[e]], axis=1)}[site]
            fs[f"{site}[{e}]"] = trained.astype(np.float64)
            fs[f"random_init:{site}[{e}]"] = untrained.astype(np.float64)
        feature_sets[site] = fs

    lambdas = [float(x) for x in p["ridge_lambdas"]]
    folds = int(p["cv_folds"])
    records = []
    for site, tnames in SITE_TARGETS.items():
        for tname in tnames:
            if tname == "steps_to_goal":
                y, kind = steps_to_goal, "regression"
            else:
                src = gobs if site == "psi_goal" else obs
                y, kind = tb.build(tname, src)
            for fs_name, X in feature_sets[site].items():
                for shuffled in (False, True):
                    if shuffled and (fs_name.startswith("random_init")
                                     or fs_name == "raw_obs"):
                        continue            # one shuffle control is enough
                    yy = _shuffle_by_trajectory(y, groups, rng) if shuffled else y
                    m = ~np.isnan(yy).any(axis=1) if yy.ndim > 1 else (yy >= 0)
                    tr = is_train & m
                    te = (~is_train) & m
                    if tr.sum() < 50 or te.sum() < 20:
                        continue
                    sel = tr | te
                    if kind == "regression":
                        r = fit_regression(X[sel], yy[sel], groups[sel], tr[sel],
                                           lambdas, folds, rng)
                    else:
                        r = fit_classification(X[sel], yy[sel], groups[sel], tr[sel],
                                               lambdas, folds, rng, tb.n_cells)
                    r.update(site=site, target=tname, features=fs_name,
                             shuffled=shuffled, n_train=int(tr.sum()),
                             n_test=int(te.sum()), dim=int(X.shape[1]))
                    records.append(r)
                    tag = fs_name + ("  [shuffled]" if shuffled else "")
                    print(f"    {tname:<18}{tag:<34}{r['metric']} {r['score']:+.4f}")

    meta = dict(
        eval="state_probes", run_id=run_cfg["run_id"], grid=run_cfg["grid"],
        agent=run_cfg["agent"]["agent"], depth=run_cfg["arch"]["depth"],
        seed=run_cfg["seed"], dataset=run_cfg["dataset"],
        repr_dim=(run_cfg["agent"].get("repr_dim") or og_cfg.get("latent_dim")),
        critic_ensemble=n_ens, checkpoint=task.get("checkpoint", "best"),
        state_source=p.get("state_source", "val"), discount=discount,
        n_states=int(len(keep)), n_traj=int(n_traj), n_train_trajs=int(want_tr),
        n_test_trajs=int(want_te), state_stride=stride,
        n_cells=int(tb.n_cells), nq=tb.nq, nv=tb.nv,
        site_targets={k: list(v) for k, v in SITE_TARGETS.items()},
        secs=round(time.time() - t0, 1))
    json.dump(dict(meta=meta, records=records),
              open(os.path.join(out_dir, "probes.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(f"  wrote {len(records)} probe records in {meta['secs']}s")
    return meta


def _infer_traj_len(run_cfg, n_rows):
    from offline import datasets as D
    _, val_paths, _ = D.resolve_paths(run_cfg)
    with np.load(val_paths[0]) as f:
        if "terminals" in f:
            t = np.asarray(f["terminals"]).reshape(-1)
            idx = np.flatnonzero(t)
            if len(idx) >= 2:
                return int(idx[1] - idx[0])
    raise ValueError("could not infer trajectory length; set params.traj_len")


# ---------------------------------------------------------------------------

def _selftest() -> int:
    """No GPU, no checkpoint: the parts that can be wrong silently."""
    rng = np.random.default_rng(0)
    fails = []

    tid, bounds = trajectory_bounds(20, 5)
    assert tid[0] == 0 and tid[-1] == 3 and bounds[2].tolist() == [10, 14]
    try:
        trajectory_bounds(21, 5)
        fails.append("trajectory_bounds accepted a ragged length")
    except ValueError:
        pass

    # A probe that SHOULD work: y is a linear function of X.
    n_traj, per = 60, 40
    groups = np.repeat(np.arange(n_traj), per)
    X = rng.normal(size=(n_traj * per, 12))
    w = rng.normal(size=(12, 1))
    y = X @ w + 0.05 * rng.normal(size=(n_traj * per, 1))
    train = groups < 45
    r = fit_regression(X, y, groups, train, [1e-3, 1e-1, 1.0], 5, rng)
    if r["score"] < 0.95:
        fails.append(f"linear target should be recovered, got r2={r['score']}")

    # The same probe with labels permuted across trajectories must collapse to ~0.
    perm = rng.permutation(n_traj)
    y_sh = y[np.concatenate([np.where(groups == perm[g])[0][:per] for g in range(n_traj)])]
    r0 = fit_regression(X, y_sh, groups, train, [1e-3, 1e-1, 1.0], 5, rng)
    if abs(r0["score"]) > 0.15:
        fails.append(f"shuffled labels should score ~0, got r2={r0['score']}")

    # r2 is against the TEST mean: predicting that mean scores exactly 0.
    yy = rng.normal(size=(100, 1))
    if abs(r2_score(yy, np.full_like(yy, yy.mean()))) > 1e-9:
        fails.append("r2 of the mean predictor is not 0")
    # and a worse-than-mean prediction must be allowed to go negative
    if r2_score(yy, -yy) > 0:
        fails.append("r2 did not go negative on an anti-correlated prediction")

    # Grouped folds never split a trajectory.
    for mask in grouped_folds(groups, 5, rng):
        for g in np.unique(groups):
            if len(np.unique(mask[groups == g])) != 1:
                fails.append("a fold split a trajectory")
                break

    # Geometric goal sampling stays inside the trajectory.
    tid2, b2 = trajectory_bounds(1000, 100)
    idx = rng.integers(0, 1000, size=500)
    g = sample_goals(idx, tid2, b2, 0.99, rng)
    if not (np.all(g >= idx) and np.all(tid2[g] == tid2[idx])):
        fails.append("sampled goals left their own trajectory")

    for f in fails:
        print("  FAIL", f)
    print("state_probes selftest:", "FAILED" if fails else "ok")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
