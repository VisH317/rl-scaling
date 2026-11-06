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
from rl_scaling import paths as P  # noqa: E402

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

#: Reference points recorded as coming from the 1000-layer paper's rebuttal
#: (antmaze-medium-stitch, GCBC). Annotations, NOT a series -- they are not runs of ours.
#:
#: UNVERIFIED (checked 2026-11-05): neither number appears in arXiv:2503.14858v5's text
#: or on the project page, and the paper's main-text depths are 4/8/16/32/64 with no
#: depth-2 run. If they are real they come from the OpenReview discussion, which nobody
#: has opened. Treat the dashed line as unsourced until someone does. It enters no
#: computation -- it is drawn and annotated only.
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


def publish(grid: str, out_dir: str) -> int:
    """Copy a grid's figures into `results/<grid>/`, beside the document that reads
    them.

    The figures are GENERATED into scratch (they are outputs, and scratch is where
    outputs live) but READ from the repo, so the document renders on GitHub and a
    figure is versioned with the sentence interpreting it. Copying is the step that
    used to be done by hand, which is how `src/offline/figures/` came to hold a stale
    subset.
    """
    import shutil
    dest = P.results_dir(grid)
    n = 0
    # reeval_*.json comes with them: it is the 50-episodes/task table the documents
    # quote their headline numbers from, so it belongs beside them, not only in scratch.
    for pat in ("*.png", "*.csv", "reeval_*.json"):
        for src in sorted(glob.glob(os.path.join(out_dir, pat))):
            shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
            n += 1
    return n


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
            "dataset": cfg.get("dataset"),
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
    """Max over all evaluations. Optimistically biased: it is a max over ~10-20 noisy
    75-episode estimates, so it sits above where the policy actually settles. Use it to
    answer "how good did this ever get", never to compare against a published table."""
    if rec.get("done"):
        return rec["summary"].get("best_success")
    vals = [r["eval/success"] for r in rec["eval"] if r.get("eval/success") is not None]
    return max(vals) if vals else None


def eval_series(rec: dict) -> list[tuple[int, float]]:
    return [(int(r["step"]), float(r["eval/success"])) for r in rec["eval"]
            if r.get("eval/success") is not None and r.get("step") is not None]


def last3_mean(rec: dict, up_to_step: int | None = None) -> float | None:
    """Mean of the final three evaluations -- the convention the OGBench paper reports,
    and the only one of our metrics comparable to its table.

    `up_to_step` pins the measurement to a common step. That matters because the plateau
    rule stops runs at different steps, so an unpinned last-3 is measured at a different
    point for every run and a depth comparison built on it is not like-for-like.
    """
    series = eval_series(rec)
    if not series:
        return None
    if up_to_step is not None and max(st for st, _ in series) < up_to_step:
        # The run has not REACHED the pin yet. Returning its last three evaluations
        # here would report, say, a 575k-step value under the label "at 1M" -- the
        # exact mislabel this function exists to prevent.
        return None
    vals = [v for st, v in series if up_to_step is None or st <= up_to_step]
    if len(vals) < 3:
        return None
    return float(np.mean(vals[-3:]))


def is_partial(rec: dict) -> bool:
    """True for a run still training. Its numbers are lower bounds, not results."""
    return not rec.get("done")


def progress(rec: dict) -> tuple[int, int] | None:
    st = os.path.join(rec["run_dir"], "status.json")
    if not os.path.exists(st):
        return None
    try:
        with open(st) as f:
            j = json.load(f)
        return int(j.get("step") or 0), int(j.get("step_cap") or 0)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Figure 1 -- the headline: success vs depth
# ---------------------------------------------------------------------------

def fig_success_vs_depth(runs, out_dir, grid):
    fig, (ax,) = new_fig()
    any_data = False
    depths_seen: set[int] = set()

    rows, header = [], ["agent", "depth", "n_seeds", "mean_success", "std", "seeds",
                        "status"]
    for agent in ("crl", "gcbc"):
        sub = [r for r in runs if r["agent"] == agent]
        depths = sorted({r["depth"] for r in sub})
        xs, ys, es = [], [], []
        part_xs, part_ys, part_es = [], [], []
        for d in depths:
            at_d = [r for r in sub if r["depth"] == d]
            vals = [v for v in (best_success(r) for r in at_d) if v is not None]
            if not vals:
                continue
            depths_seen.add(d)
            # A depth with ANY run still training is a lower bound, not a result. It
            # gets a hollow marker and a dotted connector so it cannot be misread as
            # finished, and it is never averaged into the solid series.
            partial = any(is_partial(r) for r in at_d)
            tgt = (part_xs, part_ys, part_es) if partial else (xs, ys, es)
            tgt[0].append(d)
            tgt[1].append(float(np.mean(vals)))
            tgt[2].append(float(np.std(vals)))
            rows.append([agent, d, len(vals), round(np.mean(vals), 4),
                         round(np.std(vals), 4),
                         ";".join(f"{v:.3f}" for v in vals),
                         "in-flight (lower bound)" if partial else "finished"])
            # Per-seed points, so the spread is visible and not only summarized.
            ax.scatter([d] * len(vals), vals, s=18, color=SERIES[agent],
                       alpha=0.35, zorder=3, linewidths=0, label=None)
        if not (xs or part_xs):
            continue
        any_data = True
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=SERIES[agent], linewidth=2.0,
                        marker="o", markersize=7, capsize=3, capthick=1.0,
                        elinewidth=1.0, zorder=4, label=SERIES_LABEL[agent])
        if part_xs:
            # Join the last finished depth to the in-flight ones so the trend reads,
            # but dotted, and hollow markers.
            join_x = ([xs[-1]] + part_xs) if xs else part_xs
            join_y = ([ys[-1]] + part_ys) if xs else part_ys
            ax.plot(join_x, join_y, color=SERIES[agent], linewidth=1.6,
                    linestyle=(0, (2, 2)), zorder=4)
            ax.errorbar(part_xs, part_ys, yerr=part_es, fmt="o", markersize=7,
                        markerfacecolor=SURFACE, markeredgecolor=SERIES[agent],
                        markeredgewidth=1.8, ecolor=SERIES[agent], capsize=3,
                        capthick=1.0, elinewidth=1.0, zorder=5, linestyle="none")
            xs, ys = (xs + part_xs), (ys + part_ys)
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
                textcoords="offset points", xytext=(0, -16),
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
             "Mean over seeds; error bars 1 SD; faint dots individual seeds. Hollow "
             "markers on a dotted line are runs still training -- lower bounds, not "
             "results.\nGrey dashed line is the published reference, not runs from "
             "this grid. Y axis is the MAX over evaluations, which runs above where a "
             "policy settles.",
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
        end_labels = []          # (y_at_last_step, text, color) -- de-collided below
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
            end_labels.append([float(steps[-1]), float(mean[-1]), f"d{d}", color])

        # Direct labels only -- a legend duplicated these and landed on top of a curve.
        # Nudge any two labels that would print within 0.035 of each other apart.
        end_labels.sort(key=lambda t: t[1])
        for j in range(1, len(end_labels)):
            gap = end_labels[j][1] - end_labels[j - 1][1]
            if gap < 0.035:
                end_labels[j][1] = end_labels[j - 1][1] + 0.035
        for x_end, y_lbl, text, color in end_labels:
            ax.annotate(text, (x_end, y_lbl), textcoords="offset points",
                        xytext=(6, 0), color=color, fontsize=9, va="center",
                        fontweight="medium")
        style_axes(ax, xlabel="Gradient steps",
                   ylabel="Eval success" if agent == agents[0] else "",
                   title=SERIES_LABEL[agent])
        ax.set_ylim(-0.03, 1.0)
        ax.margins(x=0.08)

    fig.suptitle(f"{grid}: learning curves by depth", color=INK, fontsize=12.5,
                 x=0.01, ha="left", y=1.04)
    fig.text(0.01, -0.07,
             "Line is the mean over seeds, band the min-max envelope. Darker = deeper. "
             "Each depth is truncated to its SHORTEST seed, so a curve ends where its "
             "earliest-stopping seed did.", color=MUTED, fontsize=8, ha="left")
    save(fig, out_dir, "fig2_learning_curves", rows, header)


# ---------------------------------------------------------------------------
# Figure 3 -- critic diagnostics vs depth (CRL only); one axis per panel
# ---------------------------------------------------------------------------

#: NOTE: these keys must match `train.csv` headers exactly. They are written by
#: OGBench's agents as `train/critic/<name>` -- an earlier version of this list omitted
#: the `critic/` segment, every metric looked absent, and `fig_critic_diagnostics`
#: returned without drawing or warning. Hence the loud warning below.
CRITIC_METRICS = [
    ("train/critic/categorical_accuracy", "InfoNCE top-1 accuracy",
     "Is the critic learning?"),
    # Kept but caveated: with batch 1024 the pair matrix is ~1/1024 positives, so
    # "always negative" already scores ~0.999. The panel's dynamic range is noise.
    ("train/critic/binary_accuracy", "Binary NCE accuracy",
     "Near-saturated: read with care"),
    ("train/critic/contrastive_loss", "Contrastive loss",
     "Fit on the training batch"),
    ("val/critic/contrastive_loss", "Validation contrastive loss",
     "Held-out fit (generalization gap)"),
]


def fig_critic_diagnostics(runs, out_dir, grid):
    sub = [r for r in runs if r["agent"] == "crl" and r["train"]]
    if not sub:
        return
    present = [m for m in CRITIC_METRICS
               if any(t.get(m[0]) is not None for r in sub for t in r["train"])]
    missing = [m[0] for m in CRITIC_METRICS if m not in present]
    if missing:
        # Silence here once cost us the whole figure. Say so.
        cols = sorted({k for r in sub for t in r["train"] for k in t
                       if "critic" in k})
        print(f"  WARNING fig3: no data for {missing}; "
              f"critic columns present are {cols or 'none'}")
    if not present:
        print("  WARNING fig3: skipped -- not one critic metric matched a CSV column")
        return
    # The step every depth is compared at, and the depths that have not reached it.
    def reached(r):
        return max((t.get("step") or 0) for t in r["train"]) if r["train"] else 0
    diag_step = COMMON_STEP
    by_depth = {}
    for r in sub:
        by_depth.setdefault(r["depth"], []).append(reached(r))
    omitted = ", ".join(f"d{d}" for d, v in sorted(by_depth.items())
                        if max(v) < diag_step)
    ticks = sorted(d for d, v in by_depth.items() if max(v) >= diag_step)
    if not ticks:
        diag_step = max(max(v) for v in by_depth.values())
        omitted = ""
        ticks = sorted(by_depth)

    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.0),
                             facecolor=SURFACE, squeeze=False)
    rows, header = [], ["metric", "depth", "n_seeds", f"mean_at_{diag_step}", "std"]

    for ax, (key, label, subtitle) in zip(axes[0], present):
        depths = sorted({r["depth"] for r in sub})
        xs, ys, es = [], [], []
        for d in depths:
            if d not in ticks:          # has not reached diag_step; omitted, not faked
                continue
            finals = []
            for r in (x for x in sub if x["depth"] == d):
                # Pinned to a common step. Taking each run's LAST logged value would
                # compare a depth-32 run at 280k steps against a depth-4 run at 1.9M.
                vals = [t[key] for t in r["train"]
                        if t.get(key) is not None
                        and (t.get("step") or 0) <= diag_step]
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
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
        style_axes(ax, xlabel="Depth", ylabel=label, title=subtitle)

    fig.suptitle(f"{grid}: CRL critic diagnostics at {diag_step // 1000}k steps",
                 color=INK, fontsize=12.5, x=0.01, ha="left", y=1.03)
    fig.text(0.01, -0.06,
             f"All depths measured at the same {diag_step // 1000}k gradient steps. "
             f"Depths omitted for not reaching it: {omitted or 'none'}.",
             color=MUTED, fontsize=8, ha="left")
    save(fig, out_dir, "fig3_critic_diagnostics", rows, header)


# ---------------------------------------------------------------------------
# Figure 4 -- compute cost, so depth is never read without its price
# ---------------------------------------------------------------------------

def fig_cost(runs, out_dir, grid):
    done = [r for r in runs if r.get("done")]
    if not done:
        return
    cost_ticks = sorted({r["depth"] for r in done})
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
        ax.set_xticks(cost_ticks)
        ax.set_xticklabels([str(x) for x in cost_ticks])
        ax.set_xlim(min(cost_ticks) / 1.4, max(cost_ticks) * 1.4)
        style_axes(ax, xlabel="Depth", ylabel=label, title=label)
        if ax is axes[0][0]:            # one legend for the row, not three
            legend_if_any(ax, fontsize=9)

    for r in done:
        s = r["summary"]
        rows.append([r["agent"], r["depth"], s["params"]["total"],
                     round(s["gradient_steps"] / max(s["train_secs"], 1e-9), 1),
                     round(s["wall_clock_s"] / 3600, 2), s.get("peak_rss_gb")])

    fig.suptitle(f"{grid}: what each depth costs", color=INK, fontsize=12.5,
                 x=0.01, ha="left", y=1.10)
    fig.text(0.01, -0.13,
             "Finished runs only, so a depth still training is absent here even when "
             "it appears in the success figures.", color=MUTED, fontsize=8, ha="left")
    fig.subplots_adjust(top=0.84, wspace=0.32)
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


# ---------------------------------------------------------------------------
# Figure 1b -- the comparable headline: last-3-eval mean, pinned to a common step
# ---------------------------------------------------------------------------

#: OGBench's published numbers, keyed BY DATASET (mean of last 3 evals, 50 eps/task).
#:
#: Keyed by dataset because it was previously a bare {agent: value} pair holding the
#: antmaze-medium-stitch numbers, which were then drawn on every grid -- including
#: humanoid_ensemble1, where they are a reference from a different environment and
#: invite exactly the wrong comparison. A grid whose dataset is not listed here gets no
#: band and says so in its caption. Add an entry only with a source: these two are
#: verified in research_log.md (CRL 53 +/- 6, GCBC 45 +/- 11).
PUBLISHED_BY_DATASET = {
    "antmaze-medium-stitch-v0": {"crl": (0.53, 0.06), "gcbc": (0.45, 0.11)},
}


def published_for(runs) -> dict:
    """The published reference for these runs' dataset, or {} if none is verified."""
    ds = {r.get("dataset") for r in runs if r.get("dataset")}
    return PUBLISHED_BY_DATASET.get(next(iter(ds)), {}) if len(ds) == 1 else {}

#: The step every run in a depth comparison must be measured at. The plateau rule stops
#: runs between 0.9M and 1.9M steps, so an unpinned last-3 compares different points on
#: different curves. 1M is the largest step essentially every run reaches.
COMMON_STEP = 1_000_000


def fig_last3_vs_depth(runs, out_dir, grid):
    """Success vs depth using the metric the published table uses.

    Differs from fig1 in two ways that matter: the y value is the mean of the last three
    evaluations rather than the max over all of them, and it is measured at a fixed step
    for every run. A run still training is included if it has already passed that step --
    its value there is final regardless of what it does later.
    """
    fig, (ax,) = new_fig()
    rows = []
    header = ["agent", "depth", "n_seeds", f"last3_mean_at_{COMMON_STEP}", "std",
              "seeds", "n_excluded_not_yet_at_step"]
    any_data, depths_seen, excluded = False, set(), []

    for agent in ("crl", "gcbc"):
        sub = [r for r in runs if r["agent"] == agent]
        xs, ys, es = [], [], []
        for d in sorted({r["depth"] for r in sub}):
            at_d = [r for r in sub if r["depth"] == d]
            vals, short = [], 0
            for r in at_d:
                v = last3_mean(r, up_to_step=COMMON_STEP)
                if v is None:
                    short += 1
                    continue
                vals.append(v)
            if short:
                excluded.append(f"{SERIES_LABEL[agent]} d{d}: {short}")
            if not vals:
                continue
            depths_seen.add(d)
            xs.append(d); ys.append(float(np.mean(vals))); es.append(float(np.std(vals)))
            rows.append([agent, d, len(vals), round(float(np.mean(vals)), 4),
                         round(float(np.std(vals)), 4),
                         ";".join(f"{v:.3f}" for v in vals), short])
            ax.scatter([d] * len(vals), vals, s=18, color=SERIES[agent],
                       alpha=0.35, zorder=3, linewidths=0)
        if not xs:
            continue
        any_data = True
        ax.errorbar(xs, ys, yerr=es, color=SERIES[agent], linewidth=2.0, marker="o",
                    markersize=7, capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
        ax.annotate(SERIES_LABEL[agent], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(9, 0), color=SERIES[agent], fontsize=10, va="center")

    if not any_data:
        ax.text(0.5, 0.5, "no results yet", transform=ax.transAxes, ha="center",
                va="center", color=MUTED, fontsize=12)
        depths_seen = set(C.GRID_DEPTHS)

    # Published bands as shaded reference, not series -- and only for the dataset they
    # were actually measured on.
    published = published_for(runs)
    for i, (agent, (mu, sd)) in enumerate(sorted(published.items())):
        ax.axhspan(mu - sd, mu + sd, color=SERIES[agent], alpha=0.07, zorder=0)
        ax.axhline(mu, color=SERIES[agent], linewidth=1.0, linestyle=(0, (5, 4)),
                   alpha=0.55, zorder=1)
        ax.annotate(f"published {SERIES_LABEL[agent]} {mu:.2f} +- {sd:.2f}",
                    (min(depths_seen), mu), textcoords="offset points",
                    xytext=(2, 5 if agent == "crl" else -12), color=SERIES[agent],
                    fontsize=8, alpha=0.9)

    ticks = sorted(depths_seen)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(d) for d in ticks])
    ax.set_xlim(min(ticks) / 1.45, max(ticks) * 1.6)
    ax.set_ylim(-0.03, 1.0)
    style_axes(ax, xlabel="Network depth (dense layers in the residual stack)",
               ylabel=f"Last-3-eval mean at {COMMON_STEP // 1000}k steps",
               title=f"{grid}: depth vs the published metric")
    cap = ("Mean over seeds; error bars 1 SD; faint dots individual seeds. Measured at "
           f"a fixed {COMMON_STEP // 1000}k steps for every run, so depths are "
           "like-for-like.")
    ds = {r.get("dataset") for r in runs if r.get("dataset")}
    if published:
        cap += ("\nShaded bands are OGBench's published numbers for this dataset at 50 "
                "episodes/task; these runs evaluate with 15, so they are noisier.")
    else:
        cap += ("\nNo published reference is drawn: no verified OGBench number for "
                + (", ".join(sorted(ds)) if ds else "this dataset") + ".")
    if excluded:
        cap += "\nNot yet at that step, so omitted: " + ", ".join(excluded) + "."
    fig.text(0.5, -0.13, cap, color=MUTED, fontsize=8, ha="center")
    save(fig, out_dir, "fig1b_last3_vs_depth", rows, header)


# ---------------------------------------------------------------------------
# Figure 6 -- per-task breakdown. The aggregate hides a phase transition.
# ---------------------------------------------------------------------------

def fig_per_task(runs, out_dir, grid, n_tasks=5):
    """Success per evaluation task, averaged over seeds.

    The aggregate success rate is a mean over 5 tasks of very different difficulty. On
    antmaze-medium-stitch one task is unsolved by the shallow configurations at every
    seed, which caps the aggregate near 0.8 no matter how good the rest get. That is a
    different claim from "depth helps a bit everywhere", so it needs its own figure.
    """
    keys = [f"eval/task{i}/success" for i in range(1, n_tasks + 1)]
    labels, mat, note = [], [], []
    for agent in ("crl", "gcbc"):
        for d in sorted({r["depth"] for r in runs if r["agent"] == agent}):
            at_d = [r for r in runs if r["agent"] == agent and r["depth"] == d]
            per_task = []
            for k in keys:
                vals = []
                for r in at_d:
                    series = [float(t[k]) for t in r["eval"] if t.get(k) is not None]
                    if len(series) >= 3:
                        vals.append(float(np.mean(series[-3:])))
                per_task.append(float(np.mean(vals)) if vals else np.nan)
            if all(np.isnan(v) for v in per_task):
                continue
            partial = any(is_partial(r) for r in at_d)
            labels.append(f"{SERIES_LABEL[agent]}  d{d}" + ("  *" if partial else ""))
            mat.append(per_task)
            note.append(partial)
    if not mat:
        print("  WARNING fig6: no per-task columns found in eval.csv")
        return
    arr = np.array(mat)

    fig, ax = plt.subplots(figsize=(1.05 * n_tasks + 3.4, 0.42 * len(mat) + 2.0),
                           facecolor=SURFACE)
    # Sequential single-hue ramp: this is one quantity, 0..1, not categories.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "succ", ["#f4f3ee", "#cfe0f6", "#86b6ef", "#2a78d6", "#104281"])
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f"task {i}" for i in range(1, n_tasks + 1)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#f7f7f4" if v > 0.55 else INK_2)
    ax.set_title(f"{grid}: success per evaluation task (mean of last 3 evals)",
                 color=INK, fontsize=12, loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cap = "Averaged over seeds."
    if any(note):
        cap += "  * = contains runs still training."
    fig.text(0.0, -0.04 - 0.01 * len(mat), cap, color=MUTED, fontsize=8, ha="left")

    rows = [[labels[i].replace("  *", ""),
             "in-flight" if note[i] else "finished"]
            + [round(float(v), 4) for v in arr[i]]
            for i in range(len(labels))]
    save(fig, out_dir, "fig6_per_task",
         rows, ["config", "status"] + [f"task{i}" for i in range(1, n_tasks + 1)])


def main() -> int:
    ap = argparse.ArgumentParser()
    # No `choices`: see monitor.py.
    ap.add_argument("--grid", default="phase1_repro")
    ap.add_argument("--out-dir")
    ap.add_argument("--heatmap", action="store_true",
                    help="also draw the depth x data heatmap (needs >1 dataset size)")
    ap.add_argument("--no-publish", action="store_true",
                    help="write only to scratch; skip the copy into results/<grid>/")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(C.ANALYSIS_DIR, args.grid)
    runs = load_runs(args.grid)
    n_done = sum(1 for r in runs if r.get("done"))
    print(f"{args.grid}: {len(runs)} runs found, {n_done} finished")
    if not runs:
        print("nothing to plot yet")
        return 0

    fig_success_vs_depth(runs, out_dir, args.grid)
    fig_last3_vs_depth(runs, out_dir, args.grid)
    fig_learning_curves(runs, out_dir, args.grid)
    fig_per_task(runs, out_dir, args.grid)
    fig_critic_diagnostics(runs, out_dir, args.grid)
    fig_cost(runs, out_dir, args.grid)
    if args.heatmap or len({r["size"] for r in runs}) > 1:
        fig_heatmap(runs, out_dir, args.grid)
    print(f"figures in {out_dir}")
    if not (args.no_publish or args.out_dir):
        print(f"published {publish(args.grid, out_dir)} files to "
              f"{P.results_dir(args.grid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
