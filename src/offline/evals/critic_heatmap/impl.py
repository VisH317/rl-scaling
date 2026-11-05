"""Critic value field over the maze, at one checkpoint.

For every task goal g and every spatial bin of the maze, this measures

    Q(s, pi(s, g), g)        s drawn from real states that fall in that bin

so the field is the value the actor's own greedy action attains, not the value of an
arbitrary action. Two reference fields are measured at the same states for contrast:
the dataset's own action, and a uniform random action. `advantage = policy - dataset`.

What makes this a test rather than a picture: a critic that has learned the maze should
rank bins by the *geodesic* distance to the goal, not by the straight-line distance.
`topology_gain = rho(Q, -geodesic) - rho(Q, -euclid)` is positive only if the value
field bends around walls. Everything else here is secondary to that number.

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

ACTION_SOURCES = ("policy", "dataset", "random")


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
    starts = np.searchsorted(b, np.unique(b), side="left")
    ends = np.searchsorted(b, np.unique(b), side="right")
    keep = np.concatenate([rows[s:min(s + per_bin, e)] for s, e in zip(starts, ends)])
    return keep


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


def run(task: dict) -> dict:
    import jax
    import jax.numpy as jnp

    p = dict(task["params"])
    out_dir = task["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    agent, env, og_cfg, run_cfg = common.load_agent(task["run_dir"], task["checkpoint"])
    subdiv = int(p["cell_subdiv"])
    geo = common.MazeGeometry(env, subdiv=subdiv)
    Hf, Wf = geo.fine.shape
    rng = np.random.default_rng(int(p.get("seed", 0)))

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
    obs = jnp.asarray(obs_all[keep])
    data_act = jnp.asarray(act_all[keep])
    flat = flat_all[keep]
    count = np.bincount(flat, minlength=Hf * Wf).reshape(Hf, Wf)
    occupied = np.bincount(flat_all[idx_valid], minlength=Hf * Wf).reshape(Hf, Wf)

    n_ens = int(run_cfg["agent"].get("critic_ensemble", 2))
    net = agent.network
    key = jax.random.PRNGKey(int(p.get("seed", 0)))

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

    sources = [s for s in p.get("action_sources", ["policy", "dataset"])]
    bad = set(sources) - set(ACTION_SOURCES)
    if bad:
        raise ValueError(f"unknown action_source(s) {sorted(bad)}")

    n_goal = int(p["goal_samples"])
    tasks = geo.tasks
    chunk = int(p.get("max_batch", 4096))
    N = obs.shape[0]

    # (source, task) -> accumulators over goal draws, in bin space
    acc = {s: np.zeros((len(tasks), Hf, Wf)) for s in sources}
    acc_mean = {s: np.zeros((len(tasks), Hf, Wf)) for s in sources}
    acc_sq = {s: np.zeros((len(tasks), Hf, Wf)) for s in sources}
    ens_gap = np.zeros((len(tasks), Hf, Wf))
    psi_norm = np.zeros((len(tasks), n_ens))
    phi_norm = np.zeros((len(tasks), n_ens))
    goal_xys = []

    def _scatter(vals):
        """(N,) per-state values -> (Hf, Wf) sums and counts."""
        s = np.bincount(flat, weights=np.asarray(vals, dtype=np.float64),
                        minlength=Hf * Wf).reshape(Hf, Wf)
        return s

    for ti, tinfo in enumerate(tasks):
        goals = _goal_observations(env, tinfo["task_id"], n_goal,
                                   int(p.get("seed", 0)) + ti)
        goal_xys.append(goals[:, :2].tolist())
        for gk in range(n_goal):
            g = jnp.asarray(np.broadcast_to(goals[gk], (N, goals.shape[1])))
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
                vs, phs, pss = [], [], []
                for i in range(0, N, chunk):
                    v, ph, ps = _critic(obs[i:i + chunk], g[i:i + chunk],
                                        acts[s][i:i + chunk])
                    vs.append(np.asarray(v))
                    phs.append(np.asarray(ph))
                    pss.append(np.asarray(ps))
                v = np.concatenate(vs, axis=1)            # (e, N)
                q_min = v.min(0)                          # what the actor maximises
                q_mean = v.mean(0)
                acc[s][ti] += _scatter(q_min)
                acc_mean[s][ti] += _scatter(q_mean)
                acc_sq[s][ti] += _scatter(q_min ** 2)
                if s == sources[0]:
                    if n_ens > 1:
                        ens_gap[ti] += _scatter(np.abs(v[0] - v[1]))
                    phi_norm[ti] += np.concatenate(phs, axis=1).mean(1) / n_goal
                    psi_norm[ti] += np.concatenate(pss, axis=1).mean(1) / n_goal

    # Normalise: every bin accumulated count[bin] * n_goal samples.
    denom = np.where(count > 0, count, np.nan)[None] * n_goal
    fields, metrics = {}, {}
    geodesic = np.stack([geo.geodesic(t["goal_xy"]) for t in tasks])
    euclid = np.stack([geo.euclid(t["goal_xy"]) for t in tasks])
    min_states = int(p["min_states_per_bin"])

    for s in sources:
        fields[f"q_{s}"] = acc[s] / denom
        fields[f"q_{s}_ens_mean"] = acc_mean[s] / denom
        var = acc_sq[s] / denom - (acc[s] / denom) ** 2
        fields[f"q_{s}_dispersion_sd"] = np.sqrt(np.maximum(var, 0.0))
    if "policy" in sources and "dataset" in sources:
        fields["advantage"] = fields["q_policy"] - fields["q_dataset"]
    fields["ens_gap"] = ens_gap / denom if n_ens > 1 else np.full_like(ens_gap, np.nan)
    fields["geodesic"] = geodesic
    fields["euclid"] = euclid

    for name, arr in list(fields.items()):
        if name in ("geodesic", "euclid"):
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
        goal_samples=n_goal, goal_xy_draws=goal_xys,
        add_noise_to_goal=geo.add_noise_to_goal, goal_tol=geo.goal_tol,
        action_sources=sources, min_states_per_bin=min_states,
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
    headline = metrics["q_policy"]["mean"] if "q_policy" in metrics else {}
    print(f"  d{meta['depth']} s{meta['seed']}  "
          f"rho_geo={headline.get('rho_geodesic')}  "
          f"topology_gain={headline.get('topology_gain')}  "
          f"peak_geo={headline.get('peak_geodesic')}  "
          f"coverage={headline.get('coverage')}  ({meta['secs']}s)")
    return dict(meta=meta, metrics=metrics)


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
