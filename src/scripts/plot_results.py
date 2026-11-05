#!/usr/bin/env python3
"""Figures for the offline CRL scaling experiments.

Colors come from the dataviz reference palette, used unchanged: categorical slots 1
(blue, CRL) and 2 (orange, GCBC), plus the documented blue/orange sequential ramps for
depth. `palette.md` certifies the first three categorical slots all-pairs in both modes,
so the two-series charts here are covered; `scripts/validate_palette.js` needs node,
which is not installed on Stellar, and is not required for the unmodified reference set.

Every figure also writes a `.csv` beside the `.png`. These are static images with no
hover layer, so the table IS the accessibility channel -- do not drop it.

Usage:
    python src/scripts/plot_results.py --grid phase1_repro
    python src/scripts/plot_results.py --grid phase1_repro --out-dir /tmp/figs
    python src/scripts/plot_results.py --grid phase3_corner --heatmap
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402

# ---------------------------------------------------------------------------
# Palette (dataviz reference instance, light mode)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = {"crl": "#2a78d6", "gcbc": "#eb6834"}          # slots 1, 2
SERIES_LABEL = {"crl": "CRL", "gcbc": "GCBC"}
#: Blue ramp steps 250..700 for CRL depths, orange-side equivalents for GCBC.
#: Ordinal use, so the lightest step is no lighter than step 250 (clears 2:1).
RAMP = {
    "crl": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    "gcbc": ["#f5a583", "#f08354", "#eb6834", "#c04f24", "#8f3a1a"],
}

#: Reference points from the 1000-layer paper's rebuttal (antmaze-medium-stitch, GCBC).
#: Annotations, NOT a series -- they come from a paper, not from these runs.
REBUTTAL_GCBC = {2: 0.47, 32: 0.63}


def legend_if_any(ax, **kw):
    """matplotlib warns when legend() finds no labelled artists; skip it instead."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = ax.legend(handles, labels, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK_2)


def style_axes(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def new_fig(w=7.2, h=4.6, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def save(fig, out_dir, name, table_rows=None, table_header=None):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(png, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    msg = f"  {png}"
    if table_rows:
        csv_p = os.path.join(out_dir, f"{name}.csv")
        with open(csv_p, "w", newline="") as f:
            w = csv.writer(f)
            if table_header:
                w.writerow(table_header)
            w.writerows(table_rows)
        msg += f"  (+ {os.path.basename(csv_p)})"
    print(msg)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(grid: str) -> list[dict]:
    """Every finished or in-progress run for a grid."""
    runs = []
    for d in sorted(glob.glob(os.path.join(C.RUNS_DIR, f"{grid}__*"))):
        cfg_p, sum_p = os.path.join(d, "config.json"), os.path.join(d, "summary.json")
        if not os.path.exists(cfg_p):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)
        rec = {
            "run_dir": d,
            "agent": cfg["agent"]["agent"],
            "depth": cfg["arch"]["depth"],
            "size": cfg["size"],
            "seed": cfg["seed"],
            "params": (cfg.get("params") or {}).get("total"),
            "done": os.path.exists(sum_p),
        }
        if rec["done"]:
            with open(sum_p) as f:
                rec["summary"] = json.load(f)
        rec["eval"] = read_csv(os.path.join(d, "eval.csv"))
        rec["train"] = read_csv(os.path.join(d, "train.csv"))
        runs.append(rec)
    return runs


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rec = {}
            for k, v in row.items():
                try:
                    rec[k] = float(v) if v not in ("", None) else None
                except (TypeError, ValueError):
                    rec[k] = v
            out.append(rec)
    return out


def best_success(rec: dict) -> float | None:
    if rec.get("done"):
        return rec["summary"].get("best_success")
    vals = [r["eval/success"] for r in rec["eval"] if r.get("eval/success") is not None]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# Figure 1 -- the headline: success vs depth
# ---------------------------------------------------------------------------

def fig_success_vs_depth(runs, out_dir, grid):
    fig, (ax,) = new_fig()
    rows, header = [], ["agent", "depth", "n_seeds", "mean_success", "std", "seeds"]
    any_data = False
    depths_seen: set[int] = set()

    for agent in ("crl", "gcbc"):
        sub = [r for r in runs if r["agent"] == agent]
        depths = sorted({r["depth"] for r in sub})
        xs, ys, es = [], [], []
        for d in depths:
            vals = [v for v in (best_success(r) for r in sub if r["depth"] == d)
                    if v is not None]
            if not vals:
                continue
            xs.append(d)
            depths_seen.add(d)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
            rows.append([agent, d, len(vals), round(np.mean(vals), 4),
                         round(np.std(vals), 4),
                         ";".join(f"{v:.3f}" for v in vals)])
            # Per-seed points, so the spread is visible and not only summarized.
            ax.scatter([d] * len(vals), vals, s=18, color=SERIES[agent],
                       alpha=0.35, zorder=3, linewidths=0, label=None)
        if not xs:
            continue
        any_data = True
        ax.errorbar(xs, ys, yerr=es, color=SERIES[agent], linewidth=2.0,
                    marker="o", markersize=7, capsize=3, capthick=1.0,
                    elinewidth=1.0, zorder=4, label=SERIES_LABEL[agent])
        # Direct label at the last point (2 series, so <=4 -> direct-label both).
        # Nudge vertically so the label clears the grey reference dash at the same x.
        dy = 11 if agent == "gcbc" else -11
        ax.annotate(SERIES_LABEL[agent], (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(9, dy),
                    color=SERIES[agent], fontsize=10, va="center")

    if not any_data:
        ax.text(0.5, 0.5, "no results yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=12)
        depths_seen = set(C.GRID_DEPTHS)

    # Rebuttal reference points -- annotation, not a series. Drawn as a faint dashed
    # connector plus end dashes so they read as "from a paper" at a glance.
    rb = sorted(REBUTTAL_GCBC.items())
    ax.plot([d for d, _ in rb], [v for _, v in rb], color=MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=1, alpha=0.85)
    for d, v in rb:
        ax.scatter([d], [v], marker="_", s=240, color=MUTED, zorder=2, linewidths=1.8)
    ax.annotate("1000-layer rebuttal (GCBC)", (rb[0][0], rb[0][1]),
                textcoords="offset points", xytext=(2, 12),
                color=MUTED, fontsize=8.5, ha="left")

    # Ticks from the depths actually on screen, so there is no empty slot at the end.
    ticks = sorted({d for d in C.GRID_DEPTHS if d in depths_seen}
                   | set(REBUTTAL_GCBC) | depths_seen)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(d) for d in ticks])
    ax.set_xlim(min(ticks) / 1.45, max(ticks) * 1.45)
    ax.set_ylim(-0.03, 1.0)
    style_axes(ax, xlabel="Network depth (dense layers in the residual stack)",
               ylabel="Best evaluation success rate",
               title=f"{grid}: does depth help?")
    legend_if_any(ax, loc="upper left", fontsize=9.5)
    # Figure coords, below the x label, so it cannot collide with the tick labels.
    fig.text(0.5, -0.08,
             "Mean over seeds; error bars 1 SD; faint dots individual seeds. "
             "Grey dashed line is the published reference, not runs from this grid.",
             color=MUTED, fontsize=8, ha="center")
    save(fig, out_dir, "fig1_success_vs_depth", rows, header)


# ---------------------------------------------------------------------------
# Figure 2 -- learning curves, small multiples by agent
# ---------------------------------------------------------------------------

def fig_learning_curves(runs, out_dir, grid):
    agents = [a for a in ("crl", "gcbc") if any(r["agent"] == a for r in runs)]
    if not agents:
        return
    fig, axes = plt.subplots(1, len(agents), figsize=(5.4 * len(agents), 4.4),
                             facecolor=SURFACE, squeeze=False, sharey=True)
    rows, header = [], ["agent", "depth", "seed", "step", "success"]

    for ax, agent in zip(axes[0], agents):
        sub = [r for r in runs if r["agent"] == agent]
        depths = sorted({r["depth"] for r in sub})
        ramp = RAMP[agent]
        for i, d in enumerate(depths):
            color = ramp[min(i, len(ramp) - 1)]
            curves = []
            for r in (x for x in sub if x["depth"] == d):
                pts = [(e["step"], e["eval/success"]) for e in r["eval"]
                       if e.get("eval/success") is not None]
                if pts:
                    curves.append(pts)
                    for s, v in pts:
                        rows.append([agent, d, r["seed"], int(s), round(v, 4)])
            if not curves:
                continue
            n = min(len(c) for c in curves)
            steps = np.array([p[0] for p in curves[0][:n]])
            vals = np.array([[p[1] for p in c[:n]] for c in curves])
            mean = vals.mean(axis=0)
            ax.plot(steps, mean, color=color, linewidth=2.0, zorder=3,
                    label=f"depth {d}")
            if vals.shape[0] > 1:
                ax.fill_between(steps, vals.min(axis=0), vals.max(axis=0),
                                color=color, alpha=0.14, linewidth=0, zorder=2)
            ax.annotate(f"d{d}", (steps[-1], mean[-1]), textcoords="offset points",
                        xytext=(6, 0), color=color, fontsize=8.5, va="center")
        style_axes(ax, xlabel="Gradient steps",
                   ylabel="Eval success" if agent == agents[0] else "",
                   title=SERIES_LABEL[agent])
        ax.set_ylim(-0.03, 1.0)
        legend_if_any(ax, fontsize=8.5, loc="upper left")

    fig.suptitle(f"{grid}: learning curves by depth", color=INK, fontsize=12.5,
                 x=0.01, ha="left", y=1.02)
    save(fig, out_dir, "fig2_learning_curves", rows, header)


# ---------------------------------------------------------------------------
# Figure 3 -- critic diagnostics vs depth (CRL only); one axis per panel
# ---------------------------------------------------------------------------

CRITIC_METRICS = [
    ("train/categorical_accuracy", "InfoNCE top-1 accuracy", "Is the critic learning?"),
    ("train/contrastive_loss", "Contrastive loss", "Fit on the training batch"),
    ("val/contrastive_loss", "Validation contrastive loss", "Held-out fit (H-data)"),
]


def fig_critic_diagnostics(runs, out_dir, grid):
    sub = [r for r in runs if r["agent"] == "crl" and r["train"]]
    if not sub:
        return
    present = [m for m in CRITIC_METRICS
               if any(t.get(m[0]) is not None for r in sub for t in r["train"])]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.0),
                             facecolor=SURFACE, squeeze=False)
    rows, header = [], ["metric", "depth", "n_seeds", "mean_final", "std"]

    for ax, (key, label, subtitle) in zip(axes[0], present):
        depths = sorted({r["depth"] for r in sub})
        xs, ys, es = [], [], []
        for d in depths:
            finals = []
            for r in (x for x in sub if x["depth"] == d):
                vals = [t[key] for t in r["train"] if t.get(key) is not None]
                if vals:
                    finals.append(vals[-1])
            if not finals:
                continue
            xs.append(d)
            ys.append(float(np.mean(finals)))
            es.append(float(np.std(finals)))
            rows.append([key, d, len(finals), round(np.mean(finals), 5),
                         round(np.std(finals), 5)])
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=SERIES["crl"], linewidth=2.0,
                        marker="o", markersize=7, capsize=3, elinewidth=1.0, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(C.GRID_DEPTHS))
        ax.set_xticklabels([str(x) for x in C.GRID_DEPTHS])
        style_axes(ax, xlabel="Depth", ylabel=label, title=subtitle)

    fig.suptitle(f"{grid}: CRL critic diagnostics at the final logged step",
                 color=INK, fontsize=12.5, x=0.01, ha="left", y=1.03)
    save(fig, out_dir, "fig3_critic_diagnostics", rows, header)


# ---------------------------------------------------------------------------
# Figure 4 -- compute cost, so depth is never read without its price
# ---------------------------------------------------------------------------

def fig_cost(runs, out_dir, grid):
    done = [r for r in runs if r.get("done")]
    if not done:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), facecolor=SURFACE, squeeze=False)
    rows = []
    header = ["agent", "depth", "params", "steps_per_s", "wall_clock_h", "peak_rss_gb"]

    panels = [
        ("params", "Parameters", lambda s, r: r.get("params") or s["params"]["total"]),
        ("throughput", "Gradient steps / second",
         lambda s, r: s["gradient_steps"] / max(s["train_secs"], 1e-9)),
        ("wall", "Wall-clock (hours)", lambda s, r: s["wall_clock_s"] / 3600),
    ]
    for ax, (_, label, fn) in zip(axes[0], panels):
        for agent in ("crl", "gcbc"):
            sub = [r for r in done if r["agent"] == agent]
            depths = sorted({r["depth"] for r in sub})
            xs, ys = [], []
            for d in depths:
                vals = [fn(r["summary"], r) for r in sub if r["depth"] == d]
                if vals:
                    xs.append(d)
                    ys.append(float(np.mean(vals)))
            if xs:
                ax.plot(xs, ys, color=SERIES[agent], linewidth=2.0, marker="o",
                        markersize=7, zorder=3, label=SERIES_LABEL[agent])
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(C.GRID_DEPTHS))
        ax.set_xticklabels([str(x) for x in C.GRID_DEPTHS])
        style_axes(ax, xlabel="Depth", ylabel=label, title=label)
        legend_if_any(ax, fontsize=9)

    for r in done:
        s = r["summary"]
        rows.append([r["agent"], r["depth"], s["params"]["total"],
                     round(s["gradient_steps"] / max(s["train_secs"], 1e-9), 1),
                     round(s["wall_clock_s"] / 3600, 2), s.get("peak_rss_gb")])

    fig.suptitle(f"{grid}: what each depth costs", color=INK, fontsize=12.5,
                 x=0.01, ha="left", y=1.04)
    save(fig, out_dir, "fig4_cost", rows, header)


# ---------------------------------------------------------------------------
# Figure 5 -- depth x data heatmap (phases 3-4). Sequential: one hue, light->dark.
# ---------------------------------------------------------------------------

def fig_heatmap(runs, out_dir, grid):
    from matplotlib.colors import LinearSegmentedColormap

    blues = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", blues)

    agents = [a for a in ("crl", "gcbc") if any(r["agent"] == a for r in runs)]
    if not agents:
        return
    depths = sorted({r["depth"] for r in runs})
    sizes = sorted({r["size"] for r in runs})
    if len(sizes) < 2:
        print("  (heatmap skipped: only one dataset size present)")
        return

    fig, axes = plt.subplots(1, len(agents), figsize=(5.6 * len(agents), 4.4),
                             facecolor=SURFACE, squeeze=False)
    rows, header = [], ["agent", "depth", "size", "n_seeds", "mean_success"]

    for ax, agent in zip(axes[0], agents):
        grid_vals = np.full((len(depths), len(sizes)), np.nan)
        for i, d in enumerate(depths):
            for j, n in enumerate(sizes):
                vals = [v for v in (best_success(r) for r in runs
                                    if r["agent"] == agent and r["depth"] == d
                                    and r["size"] == n) if v is not None]
                if vals:
                    grid_vals[i, j] = float(np.mean(vals))
                    rows.append([agent, d, n, len(vals), round(np.mean(vals), 4)])
        im = ax.imshow(grid_vals, cmap=cmap, vmin=0, vmax=1, origin="lower",
                       aspect="auto")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{n/1e6:g}M" for n in sizes])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([str(d) for d in depths])
        ax.grid(False)
        # Direct value labels: ink tokens, switched for contrast on dark cells.
        for i in range(len(depths)):
            for j in range(len(sizes)):
                if not np.isnan(grid_vals[i, j]):
                    ax.text(j, i, f"{grid_vals[i, j]:.2f}", ha="center", va="center",
                            fontsize=9,
                            color="#ffffff" if grid_vals[i, j] > 0.55 else INK)
        style_axes(ax, xlabel="Dataset size (transitions)", ylabel="Depth",
                   title=SERIES_LABEL[agent])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Best success", color=INK_2, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_visible(False)

    fig.suptitle(f"{grid}: success over depth x dataset size", color=INK,
                 fontsize=12.5, x=0.01, ha="left", y=1.03)
    save(fig, out_dir, "fig5_depth_data_heatmap", rows, header)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="phase1_repro", choices=list(C.GRIDS))
    ap.add_argument("--out-dir")
    ap.add_argument("--heatmap", action="store_true",
                    help="also draw the depth x data heatmap (needs >1 dataset size)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(C.ANALYSIS_DIR, args.grid)
    runs = load_runs(args.grid)
    n_done = sum(1 for r in runs if r.get("done"))
    print(f"{args.grid}: {len(runs)} runs found, {n_done} finished")
    if not runs:
        print("nothing to plot yet")
        return 0

    fig_success_vs_depth(runs, out_dir, args.grid)
    fig_learning_curves(runs, out_dir, args.grid)
    fig_critic_diagnostics(runs, out_dir, args.grid)
    fig_cost(runs, out_dir, args.grid)
    if args.heatmap or len({r["size"] for r in runs}) > 1:
        fig_heatmap(runs, out_dir, args.grid)
    print(f"figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
