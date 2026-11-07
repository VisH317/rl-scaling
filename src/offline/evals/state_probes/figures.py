"""Figures for `state_probes`.

    fig1_probe_vs_depth   held-out probe score vs depth, one panel per target, with
                          the two controls drawn as reference lines IN each panel
    fig2_probe_vs_heatmap  geodesic probe score against the same checkpoint's
                          `topology_gain` from `critic_heatmap`, one point per depth

Colour, per the data-viz method: depth is the x axis, not a colour, so the only
categorical distinction inside a panel is learned-vs-control, which gets the two
reference hues plus a neutral for the shuffle. A probe score is a signed quantity with
a meaningful zero (0 = no better than predicting the mean), so every regression panel
pins 0 and draws it.

One complete set per grid, never one set across grids: the representations come from
different environments with different observation spaces, and the depth trend within
an environment is the thing being read.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_results as P  # noqa: E402

LEARNED = P.SERIES["crl"]
CONTROL = P.SERIES["gcbc"]
NEUTRAL = P.MUTED


def load(suite_dir: str, eval_name: str = "state_probes") -> list:
    out = []
    for d in sorted(glob.glob(os.path.join(suite_dir, eval_name, "*"))):
        f = os.path.join(d, "probes.json")
        if os.path.exists(f):
            out.append(json.load(open(f)))
    out.sort(key=lambda r: (r["meta"]["grid"], r["meta"]["depth"]))
    return out


def _pick(records, site, target, features_pred, shuffled=False):
    for r in records:
        if (r["site"] == site and r["target"] == target
                and r["shuffled"] == shuffled and features_pred(r["features"])):
            return r
    return None


def _primary(f):      # the trained representation, ensemble member 0
    return not f.startswith("random_init") and f != "raw_obs" and f.endswith("[0]")


def _random(f):
    return f.startswith("random_init") and f.endswith("[0]")


SITE_OF = {"xy": "phi_state_action", "maze_cell": "phi_state_action",
           "geodesic_to_task": "phi_state_action", "body_height": "phi_state_action",
           "joint_vel_norm": "phi_state_action", "steps_to_goal": "phi_psi_concat"}

#: Short enough not to collide at three panels per row. The longer gloss belongs in
#: the document, not wrapped across a neighbouring axis.
TITLES = {
    "geodesic_to_task": "geodesic distance to goals *",
    "xy": "(x, y)  -- leak check",
    "maze_cell": "maze cell  (accuracy)",
    "steps_to_goal": "steps to sampled goal",
    "body_height": "body height  (control)",
    "joint_vel_norm": "joint velocity norm  (control)",
}


def fig_probe_vs_depth(recs, out_dir, suffix="", site_override=None):
    targets = [t for t in TITLES if any(r["target"] == t for rec in recs
                                        for r in rec["records"])]
    if not targets:
        return None
    n = len(targets)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow),
                             facecolor=P.SURFACE, squeeze=False)
    depths = sorted({rec["meta"]["depth"] for rec in recs})
    rows = []
    for k, tname in enumerate(targets):
        ax = axes[k // ncol][k % ncol]
        ax.set_facecolor(P.SURFACE)
        site = site_override or SITE_OF[tname]
        # Average over every seed at a depth. This used to be `next(...)`, which took
        # the FIRST record and silently ignored the others -- harmless while the suite
        # ran one seed per depth, wrong the moment `seed_select: all` landed, and
        # invisible either way because the chart looked identical.
        series = {"learned": [], "random_init": [], "shuffled": [], "raw_obs": []}
        spread = {k: [] for k in series}
        per_seed = []
        for d in depths:
            at_d = [r for r in recs if r["meta"]["depth"] == d]
            picks = {
                "learned": lambda R: _pick(R, site, tname, _primary),
                "random_init": lambda R: _pick(R, site, tname, _random),
                "shuffled": lambda R: _pick(R, site, tname, _primary, True),
                "raw_obs": lambda R: _pick(R, site, tname,
                                           lambda f: f == "raw_obs"),
            }
            for name, fn in picks.items():
                vals = [v for v in (_g(fn(r["records"])) for r in at_d)
                        if v is not None]
                series[name].append(float(np.mean(vals)) if vals else None)
                spread[name].append(float(np.std(vals)) if vals else None)
            for r in at_d:
                for name, fn in picks.items():
                    v = _g(fn(r["records"]))
                    if v is not None:
                        per_seed.append((d, r["meta"]["seed"], name, v))
        metric = "accuracy" if tname == "maze_cell" else "r2"
        for name, colour, style, lw in (
                ("learned", LEARNED, "solid", 2.0),
                ("raw_obs", P.INK, (0, (1, 2)), 1.4),
                ("random_init", CONTROL, (0, (5, 4)), 1.4),
                ("shuffled", NEUTRAL, (0, (1, 6)), 1.0)):
            ys = series[name]
            if all(y is None for y in ys):
                continue
            keep = [i for i, y in enumerate(ys) if y is not None]
            xs = [depths[i] for i in keep]
            vs = [ys[i] for i in keep]
            sds = [spread[name][i] or 0.0 for i in keep]
            # linestyle=, never the third positional arg: a dash-pattern tuple is
            # not a format string and matplotlib rejects it.
            ax.plot(xs, vs, linestyle=style, color=colour, linewidth=lw,
                    marker="o" if name == "learned" else None, markersize=6,
                    label=name, zorder=4 if name == "learned" else 2)
            if name == "learned" and any(sd > 0 for sd in sds):
                # Only the learned series gets a band; four bands would be mud.
                ax.fill_between(xs, [v - sd for v, sd in zip(vs, sds)],
                                [v + sd for v, sd in zip(vs, sds)],
                                color=colour, alpha=0.16, linewidth=0, zorder=3)
            for i, (d, y) in enumerate(zip(xs, vs)):
                rows.append([suffix.strip("_") or "", tname, site, name, d,
                             metric, round(y, 4), round(sds[i], 4),
                             sum(1 for pd, _, pn, _ in per_seed
                                 if pd == d and pn == name), ""])
        # One row per seed as well, so the CSV keeps what the band summarises.
        for d, seed, name, v in per_seed:
            rows.append([suffix.strip("_") or "", tname, site, name, d,
                         metric, round(v, 4), "", 1, seed])
        ax.axhline(0, color=P.AXIS, linewidth=0.8, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        P.style_axes(ax, xlabel="depth",
                     ylabel="accuracy" if metric == "accuracy" else "held-out R^2",
                     title=TITLES[tname])
        ax.set_title(TITLES[tname], color=P.INK, fontsize=10, loc="left")
        if k == 0:
            P.legend_if_any(ax, loc="best", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    ttl = "What the critic's representations linearly encode"
    if suffix:
        ttl += f"  [{recs[0]['meta']['dataset']}]"
    fig.suptitle(ttl, color=P.INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(0.5, 0.005,
             "* the headline target.  Held out by TRAJECTORY. `learned` must beat "
             "BOTH controls to mean anything: raw_obs is the same probe on the raw "
             "input THAT TOWER sees, random_init the same architecture untrained. "
             "Lines are the mean over seeds; the band on `learned` is 1 SD. Rows with "
             "a `seed` are the individual runs behind it.",
             color=P.MUTED, fontsize=8, ha="center")
    return P.save(fig, out_dir, f"fig1_probe_vs_depth{suffix}", rows,
                  ["grid", "target", "site", "series", "depth", "metric", "score",
                   "sd_over_seeds", "n_seeds", "seed"])


def _g(r):
    return None if r is None else float(r["score"])


def fig_probe_vs_heatmap(recs, out_dir, suite_dir, suffix=""):
    """Does the representational story explain the value-field story?"""
    pts, rows = [], []
    for rec in recs:
        rid = rec["meta"]["run_id"]
        m = os.path.join(suite_dir, "critic_heatmap", rid, "metrics.json")
        if not os.path.exists(m):
            continue
        hm = json.load(open(m))
        tg = hm.get("q_policy", {}).get("mean", {})
        pr = _pick(rec["records"], "phi_state_action", "geodesic_to_task", _primary)
        if pr is None or "rho_geodesic" not in tg:
            continue
        pts.append((rec["meta"]["depth"], float(pr["score"]),
                    float(tg["rho_geodesic"])))
        rows.append([rec["meta"]["depth"], round(float(pr["score"]), 4),
                     round(float(tg["rho_geodesic"]), 4), rid])
    if len(pts) < 2:
        return None
    pts.sort()
    fig, (ax,) = P.new_fig(w=6.0, h=4.4)
    d, probe, rho = zip(*pts)
    ax.plot(probe, rho, linestyle="-", color=P.AXIS, linewidth=1.0, zorder=2)
    ax.scatter(probe, rho, s=70, color=LEARNED, zorder=3, linewidths=0)
    for di, x, y in pts:
        ax.annotate(f"d{di}", (x, y), textcoords="offset points", xytext=(8, 4),
                    color=P.INK, fontsize=9)
    P.style_axes(ax,
                 xlabel="probe: held-out R^2 for geodesic distance from phi(s,a)",
                 ylabel="value field: Spearman rho vs -geodesic distance",
                 title="Does a better-decodable representation give a "
                       "better-ordered value field?")
    fig.text(0.5, -0.06,
             "One point per depth, same checkpoint in both axes. Moving together "
             "would give the heatmap result a representational mechanism; moving "
             "apart says the information is present but the dot product does not "
             "use it.", color=P.MUTED, fontsize=8, ha="center")
    return P.save(fig, out_dir, f"fig2_probe_vs_heatmap{suffix}", rows,
                  ["depth", "probe_r2_geodesic", "rho_geodesic", "run_id"])


def make_figures(suite_dir: str, out_dir: str = None, interpolate: bool = False) -> list:
    out_dir = out_dir or os.path.join(suite_dir, "figures")
    allrecs = load(suite_dir)
    if not allrecs:
        print(f"  no state_probes results under {suite_dir}")
        return []
    grids = sorted({r["meta"]["grid"] for r in allrecs})
    made = []
    for g in grids:
        sub = [r for r in allrecs if r["meta"]["grid"] == g]
        sfx = f"__{g}" if len(grids) > 1 else ""
        made.append(fig_probe_vs_depth(sub, out_dir, sfx))
        made.append(fig_probe_vs_heatmap(sub, out_dir, suite_dir, sfx))
    return [m for m in made if m]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite_dir")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    for p in make_figures(a.suite_dir, a.out_dir):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
