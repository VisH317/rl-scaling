"""Critic value field over the maze, at one checkpoint.

For every task goal g and every spatial bin of the maze, this measures Q(s, a, g) over
real states s in that bin, for THREE choices of action:

    policy            a = pi(s, g)             what the actor would actually do
    oracle_geodesic   a = expert(s, u_geo)     move toward the next shortest-path cell
    oracle_euclid     a = expert(s, u_euc)     move straight at the goal, through walls

`expert` is not a heuristic. It is OGBench's own direction-conditioned SAC policy --
the one that GENERATED every action in this dataset (`data_gen_scripts/
generate_locomaze.py`), fed a direction of our choosing. `u_geo` reproduces that
generator's rule exactly (`maze.py: get_oracle_subgoal`), verified against the live env.

Three questions, one per comparison:

  1. Does the field rank space by MAZE distance rather than straight-line distance?
     `topology_gain = rho(Q, -geodesic) - rho(Q, -euclid)`, positive only if the field
     bends around walls. A property of the value landscape.
  2. Does the critic prefer the maze-correct ACTION where the straight line is wrong?
     `action_topology_gain = mean(Q[oracle_geodesic] - Q[oracle_euclid])` over bins
     where the two directions disagree by more than `direction_split_deg`. This is the
     local, action-level version of (1), and it is the sharper test: it is measured
     exactly where the two hypotheses make opposite predictions.
  3. How does the critic rank the policy's action against the maze-correct one?
     `oracle_gap = mean(Q[policy] - Q[oracle_geodesic])`. Read it WITH the success
     rate, because the sign alone decides nothing: DDPG+BC trains the actor to
     maximise Q, so `> 0` is the default and only says extraction worked. `< 0` means
     the actor is not at its own critic's argmax -- an extraction failure. `> 0`
     together with LOW task success means the critic's preferred action is not the
     maze-correct one -- a critic failure. The two are distinguishable only here.

States are real, never synthesised. Overwriting dims 0:2 of one reference ant pose with
grid coordinates would give a prettier dense field at the cost of evaluating the critic
on 29-dim states that no ant can be in -- off the data manifold, which is exactly where
a network's behaviour is unconstrained and uninterpretable. Bins with too few real
states are masked, and the mask is drawn.

Run via `src/eval.py`; `python -m offline.evals.critic_heatmap.impl` runs the metric
self-check on a synthetic ground-truth field (no GPU, no checkpoint).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from offline.evals import common

ACTION_SOURCES = ("policy", "oracle_geodesic", "oracle_euclid",
                  "dataset", "random")


# ---------------------------------------------------------------------------
# Metrics -- pure functions of a field, so they can be tested without a GPU
# ---------------------------------------------------------------------------

def field_metrics(q, geo, geodesic, euclid, start_xy, goal_xy, count,
                  min_states) -> dict:
    """Every number this eval reports about one (task, field) pair.

    `q` is (Hf, Wf) with NaN where the bin has no data. Nothing here looks at the
    network; a synthetic field with known structure pins the expected values.
    """
    free = np.isfinite(geodesic)
    usable = free & (count >= min_states) & np.isfinite(q)
    n = int(usable.sum())
    out = dict(n_bins=n,
               coverage=float(usable.sum() / max(free.sum(), 1)))
    if n < 8:
        return dict(out, rho_geodesic=float("nan"), rho_euclid=float("nan"),
                    topology_gain=float("nan"), peak_geodesic=float("nan"),
                    top5_geodesic=float("nan"), path_monotone=float("nan"),
                    goal_percentile=float("nan"), q_p2=float("nan"),
                    q_p98=float("nan"))

    qv, gv, ev = q[usable], geodesic[usable], euclid[usable]
    rho_g = common.spearman(qv, -gv)
    rho_e = common.spearman(qv, -ev)

    # Where the field peaks, measured in how far that bin is from the goal.
    flat = np.where(usable, q, -np.inf)
    pk = np.unravel_index(np.argmax(flat), flat.shape)
    k = max(1, int(round(0.05 * n)))
    top_idx = np.argsort(qv)[-k:]

    # Monotonicity along one shortest path: the property an actor doing hill-climbing
    # on Q needs. Bins with no data break the chain, so they are skipped, not filled.
    path = geo.optimal_path(start_xy, goal_xy)
    seq = [q[i, j] for i, j in path
           if usable[i, j]] if len(path) else []
    steps = [(b - a) for a, b in zip(seq, seq[1:])]
    mono = float(np.mean([s > 0 for s in steps])) if steps else float("nan")

    gb = geo.xy_to_bin(goal_xy)[0]
    goal_q = q[gb[0], gb[1]]
    # Percentile RANK with ties counted, so the best possible field scores exactly
    # 1.0 rather than (n-1)/n.
    pct = (float((qv <= goal_q).mean()) if np.isfinite(goal_q) else float("nan"))

    return dict(
        out,
        rho_geodesic=round(rho_g, 4), rho_euclid=round(rho_e, 4),
        topology_gain=round(rho_g - rho_e, 4),
        peak_geodesic=round(float(geodesic[pk]), 3),
        top5_geodesic=round(float(np.mean(gv[top_idx])), 3),
        path_monotone=round(mono, 4) if mono == mono else float("nan"),
        path_bins=int(len(seq)), path_total_bins=int(len(path)),
        goal_percentile=round(pct, 4) if pct == pct else float("nan"),
        q_p2=round(float(np.percentile(qv, 2)), 4),
        q_p98=round(float(np.percentile(qv, 98)), 4),
        q_mean=round(float(qv.mean()), 4), q_std=round(float(qv.std()), 4),
    )


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def _pick_states(flat_bin, rng, per_bin):
    """Up to `per_bin` row indices per occupied bin, sampled without replacement."""
    order = rng.permutation(flat_bin.shape[0])
    b = flat_bin[order]
    srt = np.argsort(b, kind="stable")
    b = b[srt]
    rows = order[srt]
    uniq = np.unique(b)
    starts = np.searchsorted(b, uniq, side="left")
    ends = np.searchsorted(b, uniq, side="right")
    return np.concatenate([rows[s:min(s + per_bin, e)] for s, e in zip(starts, ends)])


def _goal_observations(env, task_id, n, seed):
    """`n` goal observations for one task, as the evaluator itself would get them.

    `env.reset(options={'task_id': t})` builds the goal observation by teleporting the
    body to the goal cell after a few random actions, and `add_noise_to_goal` shifts
    the goal xy by up to +/- unit/4. So the goal is a DISTRIBUTION, not a point: psi(g)
    varies with pose and with xy across resets. Averaging over draws is what makes the
    field a property of the task rather than of one reset.
    """
    np.random.seed(seed)
    obs = []
    for _ in range(n):
        _, info = env.reset(options=dict(task_id=task_id))
        obs.append(np.asarray(info["goal"], dtype=np.float32))
    return np.stack(obs)


def _oracle_actions(dataset, obs, geo, tasks, goal_xys, n_goal, seed):
    """Expert actions for the geodesic and straight-line directions, per task per draw.

    Computed BEFORE the run's own agent is loaded, and that ordering is load-bearing:
    `install_residual_arch` replaces `utils.networks.MLP` for the process, and Flax
    re-runs `setup()` on every apply, so an expert applied after the patch would
    rebuild itself as a residual tower. `common.load_expert` refuses if the patch is
    already installed rather than failing later with a shape error.
    """
    expert, meta = common.load_expert(dataset, obs.shape[1], _ACT_DIM[0], seed)
    xy = np.asarray(obs[:, :2], dtype=np.float64)
    out, angles = {}, {}
    for ti, t in enumerate(tasks):
        for gk in range(n_goal):
            gxy = np.asarray(goal_xys[ti][gk], dtype=np.float64)
            ug = geo.oracle_direction(xy, t["goal_ij"], gxy)
            ue = geo.straight_direction(xy, gxy)
            out[(ti, gk, "oracle_geodesic")] = common.expert_actions(
                expert, obs, ug.astype(np.float32), seed)
            out[(ti, gk, "oracle_euclid")] = common.expert_actions(
                expert, obs, ue.astype(np.float32), seed)
            angles[(ti, gk)] = np.degrees(np.arccos(
                np.clip((ug * ue).sum(1), -1.0, 1.0)))
    return out, angles, meta


_ACT_DIM = [8]   # filled in by run() before _oracle_actions is called


def run(task: dict) -> dict:
    import jax
    import jax.numpy as jnp

    p = dict(task["params"])
    out_dir = task["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    import ogbench
    run_cfg = common.load_run_config(task["run_dir"])
    env = ogbench.make_env_and_datasets(run_cfg["env_name"], env_only=True)
    subdiv = int(p["cell_subdiv"])
    geo = common.MazeGeometry(env, subdiv=subdiv)
    Hf, Wf = geo.fine.shape
    rng = np.random.default_rng(int(p.get("seed", 0)))
    seed = int(p.get("seed", 0))

    sources = list(p.get("action_sources", ["policy", "oracle_geodesic",
                                            "oracle_euclid"]))
    bad = set(sources) - set(ACTION_SOURCES)
    if bad:
        raise ValueError(f"unknown action_source(s) {sorted(bad)}; "
                         f"known: {list(ACTION_SOURCES)}")

    # A top-down render of the ACTUAL maze, plus the measured world -> pixel map, so
    # the figures can lay the field on the environment instead of a schematic of it.
    # Done before the expensive part: a GL failure should surface in seconds.
    render_meta, render_err = None, None
    if p.get("render_maze", True):
        r, render_err = common.render_top_down(env, geo)
        if r is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.image as mpimg
            mpimg.imsave(os.path.join(out_dir, "maze_render.png"), r["image"])
            render_meta = {k: v for k, v in r.items() if k != "image"}
            print(f"  rendered maze: fit rms {r['fit_rms_px']} px, "
                  f"leave-one-out max {r['loo_max_px']} px")
        else:
            print(f"  no maze render ({render_err}); figures fall back to the schematic")

    st = common.load_states(run_cfg, p.get("state_source", "val"))
    obs_all, act_all = st["observations"], st["actions"]
    valid = st["valids"] > 0.5                   # the last row of a traj has no action
    bins = geo.xy_to_bin(obs_all[:, :2])
    flat_all = bins[:, 0] * Wf + bins[:, 1]
    idx_valid = np.flatnonzero(valid)
    keep = idx_valid[_pick_states(flat_all[idx_valid], rng, int(p["states_per_bin"]))]
    obs_np = np.asarray(obs_all[keep], dtype=np.float32)
    data_act_np = np.asarray(act_all[keep], dtype=np.float32)
    flat = flat_all[keep]
    count = np.bincount(flat, minlength=Hf * Wf).reshape(Hf, Wf)
    occupied = np.bincount(flat_all[idx_valid], minlength=Hf * Wf).reshape(Hf, Wf)
    N = obs_np.shape[0]
    _ACT_DIM[0] = data_act_np.shape[1]

    n_goal = int(p["goal_samples"])
    tasks = geo.tasks
    goals = [_goal_observations(env, t["task_id"], n_goal, seed + ti)
             for ti, t in enumerate(tasks)]
    goal_xys = [g[:, :2].astype(np.float64) for g in goals]

    # --- the oracle, BEFORE any architecture patch is installed ----------------
    oracle_acts, dir_angles, expert_meta = {}, {}, None
    if any(s.startswith("oracle_") for s in sources):
        oracle_acts, dir_angles, expert_meta = _oracle_actions(
            run_cfg["dataset"], obs_np, geo, tasks, goal_xys, n_goal, seed)
        print(f"  oracle: {expert_meta['name']} expert "
              f"({expert_meta['checkpoint']}), the policy that generated this dataset")

    # --- now the run's own agent ----------------------------------------------
    agent, env, og_cfg, run_cfg = common.load_agent(
        task["run_dir"], task["checkpoint"], env=env)
    obs = jnp.asarray(obs_np)
    data_act = jnp.asarray(data_act_np)
    n_ens = int(run_cfg["agent"].get("critic_ensemble", 2))
    net = agent.network
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def _policy(o, g):
        dist = net.select("actor")(o, g, temperature=0.0)
        return jnp.clip(dist.mode(), -1.0, 1.0)

    @jax.jit
    def _critic(o, g, a):
        v, phi, psi = net.select("critic")(o, g, actions=a, info=True)
        if v.ndim == 1:                           # non-ensemble critic
            v, phi, psi = v[None], phi[None], psi[None]
        return v, jnp.linalg.norm(phi, axis=-1), jnp.linalg.norm(psi, axis=-1)

    chunk = int(p.get("max_batch", 4096))
    acc = {s: np.zeros((len(tasks), Hf, Wf)) for s in sources}
    acc_mean = {s: np.zeros((len(tasks), Hf, Wf)) for s in sources}
    acc_sq = {s: np.zeros((len(tasks), Hf, Wf)) for s in sources}
    ens_gap = np.zeros((len(tasks), Hf, Wf))
    angle_acc = np.zeros((len(tasks), Hf, Wf))
    psi_norm = np.zeros((len(tasks), n_ens))
    phi_norm = np.zeros((len(tasks), n_ens))

    def _scatter(vals):
        return np.bincount(flat, weights=np.asarray(vals, dtype=np.float64),
                           minlength=Hf * Wf).reshape(Hf, Wf)

    for ti, tinfo in enumerate(tasks):
        for gk in range(n_goal):
            g = jnp.asarray(np.broadcast_to(goals[ti][gk], (N, goals[ti].shape[1])))
            acts = {}
            if "policy" in sources:
                acts["policy"] = jnp.concatenate(
                    [_policy(obs[i:i + chunk], g[i:i + chunk])
                     for i in range(0, N, chunk)])
            if "dataset" in sources:
                acts["dataset"] = data_act
            if "random" in sources:
                key, sub = jax.random.split(key)
                acts["random"] = jax.random.uniform(
                    sub, data_act.shape, minval=-1.0, maxval=1.0)
            for s in sources:
                if s.startswith("oracle_"):
                    acts[s] = jnp.asarray(oracle_acts[(ti, gk, s)])
            if dir_angles:
                angle_acc[ti] += _scatter(dir_angles[(ti, gk)])
            for s in sources:
                vs, phs, pss = [], [], []
                for i in range(0, N, chunk):
                    v, ph, ps = _critic(obs[i:i + chunk], g[i:i + chunk],
                                        acts[s][i:i + chunk])
                    vs.append(np.asarray(v))
                    phs.append(np.asarray(ph))
                    pss.append(np.asarray(ps))
                v = np.concatenate(vs, axis=1)            # (e, N)
                q_min = v.min(0)                          # what the actor maximises
                acc[s][ti] += _scatter(q_min)
                acc_mean[s][ti] += _scatter(v.mean(0))
                acc_sq[s][ti] += _scatter(q_min ** 2)
                if s == sources[0]:
                    if n_ens > 1:
                        ens_gap[ti] += _scatter(np.abs(v[0] - v[1]))
                    phi_norm[ti] += np.concatenate(phs, axis=1).mean(1) / n_goal
                    psi_norm[ti] += np.concatenate(pss, axis=1).mean(1) / n_goal

    # Normalise: every bin accumulated count[bin] * n_goal samples.
    denom = np.where(count > 0, count, np.nan)[None] * n_goal
    fields = {}
    geodesic = np.stack([geo.geodesic(t["goal_xy"]) for t in tasks])
    euclid = np.stack([geo.euclid(t["goal_xy"]) for t in tasks])
    min_states = int(p["min_states_per_bin"])

    for s in sources:
        fields[f"q_{s}"] = acc[s] / denom
        fields[f"q_{s}_ens_mean"] = acc_mean[s] / denom
        var = acc_sq[s] / denom - (acc[s] / denom) ** 2
        fields[f"q_{s}_dispersion_sd"] = np.sqrt(np.maximum(var, 0.0))

    # Differences between action sources: each one answers a different question, so
    # they are named for the question and not for the arithmetic.
    pairs = [("policy", "dataset", "advantage"),
             ("policy", "oracle_geodesic", "policy_minus_oracle"),
             ("oracle_geodesic", "oracle_euclid", "oracle_geo_minus_euclid")]
    for a, b, name in pairs:
        if a in sources and b in sources:
            fields[name] = fields[f"q_{a}"] - fields[f"q_{b}"]
    fields["ens_gap"] = ens_gap / denom if n_ens > 1 else np.full_like(ens_gap, np.nan)
    fields["dir_angle_deg"] = (angle_acc / denom if dir_angles
                               else np.full_like(angle_acc, np.nan))
    fields["geodesic"] = geodesic
    fields["euclid"] = euclid

    metrics = {}
    for name, arr in list(fields.items()):
        if name in ("geodesic", "euclid", "dir_angle_deg"):
            continue
        per_task = []
        for ti, tinfo in enumerate(tasks):
            m = field_metrics(arr[ti], geo, geodesic[ti], euclid[ti],
                              geo.ij_to_xy(tinfo["init_ij"]), tinfo["goal_xy"],
                              count, min_states)
            m["task_id"] = tinfo["task_id"]
            per_task.append(m)
        agg = {k: round(float(np.nanmean([m[k] for m in per_task])), 4)
               for k in per_task[0] if k != "task_id"}
        metrics[name] = dict(mean=agg, per_task=per_task)

    cross = cross_metrics(fields, geo, geodesic, count, min_states, tasks,
                          float(p.get("direction_split_deg", 45.0)))
    metrics["_cross"] = cross

    meta = dict(
        eval="critic_heatmap", run_id=run_cfg["run_id"], grid=run_cfg["grid"],
        agent=run_cfg["agent"]["agent"], depth=run_cfg["arch"]["depth"],
        seed=run_cfg["seed"], repr_dim=run_cfg["arch"]["repr_dim"],
        critic_ensemble=n_ens, critic_loss=run_cfg["agent"]["critic_loss"],
        logits=run_cfg["agent"]["logits"], checkpoint=task["checkpoint"],
        dataset=run_cfg["dataset"], params=run_cfg.get("_params"),
        selection=task.get("selection"),
        state_source=st["source"], state_files=st["files"],
        n_states_used=int(N), n_states_available=int(valid.sum()),
        subdiv=subdiv, bin_size=geo.bin_size, unit=geo.unit,
        goal_samples=n_goal, goal_xy_draws=[g.tolist() for g in goal_xys],
        add_noise_to_goal=geo.add_noise_to_goal, goal_tol=geo.goal_tol,
        action_sources=sources, expert=expert_meta,
        direction_split_deg=float(p.get("direction_split_deg", 45.0)),
        min_states_per_bin=min_states,
        render=render_meta, render_error=render_err,
        free_bins=int(np.isfinite(geodesic[0]).sum()),
        bins_with_data=int((count > 0).sum()),
        bins_usable=int((count >= min_states).sum()),
        tasks=tasks, secs=round(time.time() - t0, 1),
    )
    np.savez_compressed(os.path.join(out_dir, "field.npz"),
                        maze=geo.maze, fine=geo.fine, count=count,
                        occupied=occupied, **fields)
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    headline = metrics.get("q_policy", {}).get("mean", {})
    print(f"  d{meta['depth']} s{meta['seed']}  "
          f"rho_geo={headline.get('rho_geodesic')}  "
          f"topology_gain={headline.get('topology_gain')}  "
          f"action_topology_gain={cross.get('action_topology_gain', {}).get('mean')}  "
          f"oracle_gap={cross.get('oracle_gap', {}).get('mean')}  ({meta['secs']}s)")
    return dict(meta=meta, metrics=metrics)


def cross_metrics(fields, geo, geodesic, count, min_states, tasks,
                  split_deg: float) -> dict:
    """Comparisons BETWEEN action sources, which is where the sharp tests live.

    `action_topology_gain` is restricted to bins where the geodesic and straight-line
    directions actually disagree by more than `split_deg`. Everywhere else the two
    oracle actions are nearly the same action and their difference is noise, so
    averaging over all bins would dilute the signal with bins that cannot speak to it.
    """
    out = {}
    ang = fields.get("dir_angle_deg")

    def _summarise(arr, mask_by_angle):
        per_task, n_bins = [], []
        for ti in range(len(tasks)):
            free = np.isfinite(geodesic[ti])
            m = free & (count >= min_states) & np.isfinite(arr[ti])
            if mask_by_angle and ang is not None:
                m = m & np.isfinite(ang[ti]) & (ang[ti] >= split_deg)
            vals = arr[ti][m]
            per_task.append(round(float(vals.mean()), 4) if vals.size >= 8
                            else float("nan"))
            n_bins.append(int(m.sum()))
        return dict(mean=round(float(np.nanmean(per_task)), 4), per_task=per_task,
                    n_bins=n_bins)

    if "oracle_geo_minus_euclid" in fields:
        out["action_topology_gain"] = _summarise(
            fields["oracle_geo_minus_euclid"], True)
        out["action_topology_gain_all_bins"] = _summarise(
            fields["oracle_geo_minus_euclid"], False)
    if "policy_minus_oracle" in fields:
        out["oracle_gap"] = _summarise(fields["policy_minus_oracle"], False)
        out["oracle_gap_split_bins"] = _summarise(fields["policy_minus_oracle"], True)
    if ang is not None and np.isfinite(ang).any():
        free = np.isfinite(geodesic) & (count[None] >= min_states)
        out["split_bin_fraction"] = round(
            float(np.nansum((ang >= split_deg) & free) / max(np.sum(free), 1)), 4)
        out["direction_split_deg"] = split_deg
    return out


# ---------------------------------------------------------------------------
# Self-check: the metrics, against fields whose answers are known
# ---------------------------------------------------------------------------

def _selftest() -> int:
    class _FakeEnv:
        class unwrapped:
            maze_map = np.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]])
            _maze_unit, _offset_x, _offset_y = 4.0, 4.0, 4.0
            _goal_tol, _noise, _add_noise_to_goal = 0.5, 1.0, True
            task_infos = [dict(init_ij=(1, 1), goal_ij=(3, 3),
                               init_xy=(0.0, 0.0), goal_xy=(8.0, 8.0))]

    geo = common.MazeGeometry(_FakeEnv(), subdiv=4)
    t = geo.tasks[0]
    gd, ed = geo.geodesic(t["goal_xy"]), geo.euclid(t["goal_xy"])
    count = np.where(np.isfinite(gd), 99, 0)
    ok = True

    def check(label, got, want, tol=1e-6):
        nonlocal ok
        good = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}: {got} (want {want})")

    # 1. The ideal critic: Q = -geodesic.
    m = field_metrics(-gd, geo, gd, ed, t["init_xy"], t["goal_xy"], count, 1)
    check("ideal rho_geodesic", m["rho_geodesic"], 1.0)
    check("ideal peak_geodesic", m["peak_geodesic"], 0.0)
    check("ideal path_monotone", m["path_monotone"], 1.0)
    check("ideal goal_percentile", m["goal_percentile"], 1.0)
    print(f"        topology_gain {m['topology_gain']} (must be > 0)")
    ok &= m["topology_gain"] > 0

    # 2. A critic that only knows straight-line distance: topology_gain must be < 0.
    m2 = field_metrics(-ed, geo, gd, ed, t["init_xy"], t["goal_xy"], count, 1)
    print(f"  {'PASS' if m2['topology_gain'] < 0 else 'FAIL'}  "
          f"euclid-only topology_gain {m2['topology_gain']} (must be < 0)")
    ok &= m2["topology_gain"] < 0

    # 3. A constant field: no ranking information at all.
    m3 = field_metrics(np.where(np.isfinite(gd), 1.0, np.nan), geo, gd, ed,
                       t["init_xy"], t["goal_xy"], count, 1)
    check("flat q_std", m3["q_std"], 0.0)
    check("flat path_monotone", m3["path_monotone"], 0.0)

    # 4. Masking: a bin below min_states must not enter any metric.
    sparse = np.where(np.isfinite(gd), 2, 0)
    sparse[5, 5] = 99        # a FREE fine bin: cell (1, 1) spans bins 4..7
    m4 = field_metrics(-gd, geo, gd, ed, t["init_xy"], t["goal_xy"], sparse, 10)
    check("masked n_bins", float(m4["n_bins"]), 1.0)

    print("\n" + ("critic_heatmap metrics: ALL PASS" if ok else "SELF-CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
