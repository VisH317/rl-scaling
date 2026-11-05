"""Figures for `critic_heatmap`.

Four outputs, each with the `.csv` beside it -- these are static PNGs with no hover
layer, so the table IS the accessibility channel.

    fig1_critic_field       depth x task small multiples of Q(s, pi(s,g), g)
    fig2_topology           rho_geodesic / rho_euclid / topology_gain vs depth
    fig3_advantage_field    depth x task, Q(policy) - Q(dataset), diverging
    fig4_metrics            peak_geodesic / path_monotone / goal_percentile vs depth

Color, per the data-viz method: `Q` is continuous magnitude with no meaningful zero, so
it gets ONE hue light->dark (the reference palette's sequential blue, steps 100-700).
`advantage` has a meaningful zero and a sign, so it gets the documented diverging pair
(blue <-> red) with the neutral gray midpoint; the red arm's steps are derived to mirror
the blue arm's lightness, since the reference palette tabulates only the blue ramp.
Never a rainbow: a hue-cycling map would invent boundaries in a smooth field.

`Q = phi(s,a)^T psi(g) / sqrt(d)` is an unnormalised log-density-ratio, so its SCALE is
arbitrary per run. Every panel is therefore normalised on its own 2nd-98th percentile
and prints that raw range, and the colorbar is labelled "per panel". Sharing one scale
across depths would be a chart that looks comparable and is not.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

import plot_results as P                              # noqa: E402  (one palette source)

matplotlib.rcParams["hatch.linewidth"] = 0.5

#: Sequential blue, reference palette steps 100 -> 700. Continuous magnitude, so the
#: lightest step may recede toward the surface.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Diverging blue <-> red, neutral gray midpoint (`#f0efec`, light surface).
DIV_BR = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#8f2423"]

CMAP_Q = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_ADV = LinearSegmentedColormap.from_list("div_br", DIV_BR)
WALL = "#c3c2b7"       # AXIS neutral: visibly not data
NODATA = "#e8e7e0"     # masked free bins, with hatching on top


def load(suite_dir: str, eval_name: str = "critic_heatmap") -> list:
    """Every finished result in a suite directory, ordered by (agent, depth)."""
    out = []
    for d in sorted(glob.glob(os.path.join(suite_dir, eval_name, "*"))):
        f, m, me = (os.path.join(d, "field.npz"), os.path.join(d, "metrics.json"),
                    os.path.join(d, "meta.json"))
        if not all(os.path.exists(x) for x in (f, m, me)):
            continue
        out.append(dict(dir=d, meta=json.load(open(me)),
                        metrics=json.load(open(m)), npz=f))
    out.sort(key=lambda r: (r["meta"]["agent"], r["meta"]["depth"]))
    return out


def _panel(ax, field, geodesic, count, meta, task, *, cmap, vmin, vmax,
           interpolate=False, show_path=True):
    """One maze panel. Walls are drawn, masked bins are hatched, neither is data."""
    unit, sub = meta["unit"], meta["subdiv"]
    maze = np.asarray(meta["_maze"])
    H, W = maze.shape
    ox = oy = unit  # OGBench: offset_x == offset_y == unit
    ext = [-(ox + unit / 2), W * unit - ox - unit / 2,
           -(oy + unit / 2), H * unit - oy - unit / 2]

    free = np.isfinite(geodesic)
    usable = free & (count >= meta["min_states_per_bin"]) & np.isfinite(field)
    shown = np.where(usable, field, np.nan)
    if interpolate:
        shown = _fill_nearest(shown, free)

    # Masked bins: flat neutral plus hatching, so "no data" is distinguishable from
    # "low value" without relying on color alone (CVD, print, forced-colors).
    bs = unit / sub
    masked = np.argwhere(free & ~usable)
    for bi, bj in masked[:600]:
        ax.add_patch(Rectangle((bj * bs - ox - unit / 2, bi * bs - oy - unit / 2),
                               bs, bs, facecolor=NODATA, edgecolor=P.MUTED, lw=0.0,
                               hatch="///", zorder=1))
    im = ax.imshow(shown, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear" if interpolate else "nearest", zorder=2)
    for i in range(H):
        for j in range(W):
            if maze[i, j]:
                ax.add_patch(Rectangle(
                    (j * unit - ox - unit / 2, i * unit - oy - unit / 2), unit, unit,
                    facecolor=WALL, edgecolor="none", zorder=3))

    gx, gy = task["goal_xy"]
    sx, sy = task["init_xy"]
    if show_path:
        path = np.asarray(task.get("_path", []))
        if len(path):
            xs = (path[:, 1] + 0.5) * bs - ox - unit / 2
            ys = (path[:, 0] + 0.5) * bs - oy - unit / 2
            ax.plot(xs, ys, color=P.INK, lw=1.0, alpha=0.45, zorder=4,
                    solid_capstyle="round")
    ax.plot([sx], [sy], marker="o", ms=7, mfc="none", mec=P.INK, mew=1.4, zorder=5)
    ax.plot([gx], [gy], marker="*", ms=13, mfc=P.SURFACE, mec=P.INK, mew=1.1, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(P.AXIS)
        s.set_linewidth(0.8)
    return im


def _fill_nearest(a, free):
    """Nearest-neighbour fill of masked bins. DISPLAY ONLY -- never fed to a metric."""
    out = a.copy()
    known = np.argwhere(np.isfinite(a))
    if not len(known):
        return out
    todo = np.argwhere(free & ~np.isfinite(a))
    for i, j in todo:
        d = np.abs(known - np.array([i, j])).sum(1)
        out[i, j] = a[tuple(known[np.argmin(d)])]
    return out


def _attach_geometry(rec):
    """Rebuild the maze map and optimal paths from the npz, so figures need no env."""
    z = np.load(rec["npz"])
    rec["z"] = {k: z[k] for k in z.files}
    rec["meta"]["_maze"] = rec["z"]["maze"].tolist()
    # The path is cheap to recompute from the stored geodesic field.
    sub, unit = rec["meta"]["subdiv"], rec["meta"]["unit"]
    for ti, t in enumerate(rec["meta"]["tasks"]):
        d = rec["z"]["geodesic"][ti] / (unit / sub)
        start = (int((t["init_xy"][1] + unit + unit / 2) / (unit / sub)),
                 int((t["init_xy"][0] + unit + unit / 2) / (unit / sub)))
        path, cur = [], start
        if np.isfinite(d[cur]):
            path = [cur]
            while d[cur] > 0:
                nb = [(cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]),
                      (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)]
                nb = [n for n in nb if 0 <= n[0] < d.shape[0] and 0 <= n[1] < d.shape[1]
                      and np.isfinite(d[n])]
                nxt = min(nb, key=lambda n: d[n])
                if d[nxt] >= d[cur]:
                    break
                cur = nxt
                path.append(cur)
        t["_path"] = path
    return rec


# ---------------------------------------------------------------------------
# Figure 1 / 3 -- the fields
# ---------------------------------------------------------------------------

def _field_figure(recs, out_dir, *, key, name, cmap, diverging, title, sublabel,
                  interpolate=False):
    if not recs:
        return None
    n_task = len(recs[0]["meta"]["tasks"])
    fig, axes = plt.subplots(len(recs), n_task,
                             figsize=(2.05 * n_task + 1.3, 2.15 * len(recs) + 0.9),
                             facecolor=P.SURFACE, squeeze=False)
    rows = []
    for ri, rec in enumerate(recs):
        m, z = rec["meta"], rec["z"]
        if key not in z:
            continue
        for ti, task in enumerate(m["tasks"]):
            ax = axes[ri][ti]
            fld, geod, cnt = z[key][ti], z["geodesic"][ti], z["count"]
            usable = (np.isfinite(geod) & (cnt >= m["min_states_per_bin"])
                      & np.isfinite(fld))
            vals = fld[usable]
            if vals.size < 8:
                ax.set_facecolor(P.SURFACE)
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        color=P.MUTED, fontsize=8)
                ax.set_xticks([]), ax.set_yticks([])
                continue
            if diverging:
                hi = float(np.percentile(np.abs(vals), 98))
                vmin, vmax = -hi, hi
            else:
                vmin = float(np.percentile(vals, 2))
                vmax = float(np.percentile(vals, 98))
            _panel(ax, fld, geod, cnt, m, task, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolate=interpolate)
            met = rec["metrics"][key]["per_task"][ti]
            ax.set_title(f"[{vmin:+.2f}, {vmax:+.2f}]", color=P.MUTED, fontsize=7.5,
                         loc="right", pad=3)
            ax.text(0.03, 0.03, sublabel(met), transform=ax.transAxes, fontsize=7.5,
                    color=P.INK_2, va="bottom", ha="left",
                    bbox=dict(facecolor=P.SURFACE, edgecolor="none", alpha=0.82,
                              pad=1.6))
            if ri == 0:
                ax.text(0.5, 1.22, f"task {task['task_id']}", transform=ax.transAxes,
                        ha="center", color=P.INK, fontsize=10)
            if ti == 0:
                ax.set_ylabel(f"depth {m['depth']}\nseed {m['seed']}", color=P.INK_2,
                              fontsize=9.5)
            rows.append([m["depth"], m["seed"], m["run_id"], task["task_id"],
                         round(vmin, 4), round(vmax, 4), met.get("rho_geodesic"),
                         met.get("rho_euclid"), met.get("topology_gain"),
                         met.get("peak_geodesic"), met.get("path_monotone"),
                         met.get("goal_percentile"), met.get("coverage")])

    fig.suptitle(title, color=P.INK, fontsize=13, x=0.012, ha="left", y=1.005)
    note = ("walls grey; bins below the state floor hatched; o start, * goal; "
            "thin line = one shortest path. Each panel on its own 2nd-98th percentile "
            "(printed above it) -- Q is an unnormalised log-ratio, so scale is not "
            "comparable across runs.")
    if interpolate:
        note = "INTERPOLATED for display (nearest-fill + bilinear). " + note
    fig.text(0.012, -0.015, note, color=P.MUTED, fontsize=8, ha="left", va="top",
             wrap=True)
    # tight_layout FIRST, then place the bar outside the grid -- added before, it is
    # laid out as another axes and lands on top of the last column.
    fig.tight_layout()
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([1.012, 0.32, 0.008, 0.36])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["low", "0" if diverging else "mid", "high"],
                          color=P.INK_2, fontsize=8)
    cb.ax.tick_params(length=2, width=0.8, colors=P.MUTED)
    cb.outline.set_visible(False)
    cb.ax.set_title("per\npanel", color=P.MUTED, fontsize=7.5, pad=6)
    header = ["depth", "seed", "run_id", "task", "vmin", "vmax", "rho_geodesic",
              "rho_euclid", "topology_gain", "peak_geodesic", "path_monotone",
              "goal_percentile", "coverage"]
    P.save(fig, out_dir, name, table_rows=rows, table_header=header)
    return os.path.join(out_dir, f"{name}.png")


# ---------------------------------------------------------------------------
# Figure 2 / 4 -- metrics against depth
# ---------------------------------------------------------------------------

def _metric_lines(recs, out_dir, name, title, panels, key="q_policy"):
    if not recs:
        return None
    depths = [r["meta"]["depth"] for r in recs]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5),
                             facecolor=P.SURFACE, squeeze=False)
    rows = []
    for pi, (metrics, label, ylim) in enumerate(panels):
        ax = axes[0][pi]
        for si, m in enumerate(metrics):
            ys = [r["metrics"][key]["mean"].get(m) for r in recs]
            color = P.SERIES["crl"] if si == 0 else P.SERIES["gcbc"]
            ax.plot(depths, ys, marker="o", ms=8, lw=2.0, color=color, label=m,
                    clip_on=False, zorder=3)
            if len(depths) and ys[-1] is not None:
                # Direct-label the last point only. `annotation_clip=False` plus the
                # x-margin below keeps it inside the panel instead of over the next
                # panel's y-axis.
                ax.annotate(f"{ys[-1]:.2f}", (depths[-1], ys[-1]),
                            textcoords="offset points", xytext=(8, 0), fontsize=8.5,
                            color=P.INK_2, va="center", ha="left",
                            annotation_clip=False)
        if len(metrics) == 1 and metrics[0] == "topology_gain":
            ax.axhline(0.0, color=P.AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
        # The title names the measure; repeating it on the y-axis is noise.
        P.style_axes(ax, xlabel="depth (residual blocks)", title=label)
        ax.set_xscale("log", base=2)
        ax.margins(x=0.16)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        if ylim:
            ax.set_ylim(*ylim)
        if len(metrics) >= 2:
            P.legend_if_any(ax, loc="best", fontsize=8.5)
    for r in recs:
        mm = r["metrics"][key]["mean"]
        rows.append([r["meta"]["depth"], r["meta"]["seed"], r["meta"]["run_id"],
                     r["meta"]["selection"].get("score")
                     if r["meta"].get("selection") else None]
                    + [mm.get(k) for k in ("rho_geodesic", "rho_euclid",
                                           "topology_gain", "peak_geodesic",
                                           "top5_geodesic", "path_monotone",
                                           "goal_percentile", "coverage", "q_std")])
    fig.suptitle(title, color=P.INK, fontsize=13, x=0.012, ha="left")
    fig.tight_layout()
    header = (["depth", "seed", "run_id", "select_score", "rho_geodesic", "rho_euclid",
               "topology_gain", "peak_geodesic", "top5_geodesic", "path_monotone",
               "goal_percentile", "coverage", "q_std"])
    P.save(fig, out_dir, name, table_rows=rows, table_header=header)
    return os.path.join(out_dir, f"{name}.png")


def make_figures(suite_dir: str, out_dir: str = None, interpolate: bool = False) -> list:
    out_dir = out_dir or os.path.join(suite_dir, "figures")
    recs = [_attach_geometry(r) for r in load(suite_dir)]
    if not recs:
        print(f"  no critic_heatmap results under {suite_dir}")
        return []
    made = []
    made.append(_field_figure(
        recs, out_dir, key="q_policy", name="fig1_critic_field", cmap=CMAP_Q,
        diverging=False, interpolate=interpolate,
        title="Critic value field under the policy's own action, by depth and task",
        sublabel=lambda m: f"rho {m['rho_geodesic']:+.2f}  gain {m['topology_gain']:+.2f}"))
    made.append(_metric_lines(
        recs, out_dir, "fig2_topology", "Does the value field follow the maze?",
        panels=[(["rho_geodesic", "rho_euclid"], "Spearman rho vs -distance", (-1, 1)),
                (["topology_gain"], "topology_gain = rho_geo - rho_euclid", None)]))
    if any("advantage" in r["z"] for r in recs):
        made.append(_field_figure(
            recs, out_dir, key="advantage", name="fig3_advantage_field", cmap=CMAP_ADV,
            diverging=True, interpolate=interpolate,
            title="Q(policy action) - Q(dataset action), by depth and task",
            sublabel=lambda m: f"mean {m['q_mean']:+.3f}"))
    made.append(_metric_lines(
        recs, out_dir, "fig4_metrics", "Where the field peaks, and whether it climbs",
        panels=[(["peak_geodesic"], "distance of the peak bin to goal (units)", None),
                (["path_monotone"], "fraction of path steps where Q rises", (0, 1)),
                (["goal_percentile"], "percentile rank of the goal bin", (0, 1))]))
    return [m for m in made if m]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite_dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--interpolate", action="store_true",
                    help="nearest-fill masked bins for display; never enters a metric")
    a = ap.parse_args()
    for p in make_figures(a.suite_dir, a.out_dir, a.interpolate):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
