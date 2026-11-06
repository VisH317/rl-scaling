"""Fill a results markdown from finished eval tasks.

`RESULTS_TEMPLATE.md` holds the prose; this fills only the regions between
`<!-- AUTO:name -->` and `<!-- /AUTO:name -->`. Everything outside them survives, so
the interpretation written under a `> **Read:**` stub is never overwritten by a
re-run, and no number is ever copied by hand.

Figures are COPIED into the repo (`results/<suite>/`, the same folder as this
document) rather than linked into scratch, so the document renders on GitHub and in an
editor, and so a figure and the sentence reading it are versioned together.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import shutil

from offline import config as C
from offline.evals import common

from rl_scaling import paths

#: Where the template is READ from: beside this file, i.e. the code that is running.
_CODE_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_CODE_HERE, "RESULTS_TEMPLATE.md")
#: Where the results document and its figures are WRITTEN. Under a submit-time code
#: snapshot these differ, and writing beside `__file__` would bury the report in
#: scratch. See `rl_scaling.paths.OUTPUT_REPO`.
#: Where results documents and their figures are WRITTEN. Under a submit-time
#: code snapshot this must follow OUTPUT_REPO (the live checkout), never the
#: snapshot on scratch, or the document would be written where no one reads it.
HERE = paths.RESULTS


def _table(header, rows) -> str:
    if not rows:
        return "_(no finished runs yet)_"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join("—" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def _fmt(v, nd=3, sign=True):
    """A leading `+` only where the SIGN carries the meaning (topology_gain,
    oracle_gap, a difference). On a magnitude like `coverage` it is just noise."""
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return str(v)


def _load(suite_dir: str) -> list:
    recs = []
    for d in sorted(glob.glob(os.path.join(suite_dir, "critic_heatmap", "*"))):
        m, me = os.path.join(d, "metrics.json"), os.path.join(d, "meta.json")
        if os.path.exists(m) and os.path.exists(me):
            recs.append(dict(dir=d, metrics=json.load(open(m)),
                             meta=json.load(open(me))))
    recs.sort(key=lambda r: (r["meta"].get("grid", ""), r["meta"]["agent"],
                             r["meta"]["depth"]))
    return recs


def _load_probes(suite_dir: str) -> list:
    out = []
    for d in sorted(glob.glob(os.path.join(suite_dir, "state_probes", "*"))):
        f = os.path.join(d, "probes.json")
        if os.path.exists(f):
            out.append(json.load(open(f)))
    out.sort(key=lambda r: (r["meta"]["grid"], r["meta"]["depth"]))
    return out


def _per_grid(fn):
    """Render a table PER GRID when the suite spans more than one.

    Pooling antmaze and humanoidmaze rows into one depth-indexed table would read
    as a single trend over eight points when it is two trends over four, and the two
    go in opposite directions -- which is the whole finding.
    """
    def wrapped(cfg, recs, suite_dir):
        grids = sorted({r["meta"].get("grid", "?") for r in recs})
        if len(grids) <= 1:
            return fn(cfg, recs, suite_dir)
        out = []
        for g in grids:
            sub = [r for r in recs if r["meta"].get("grid", "?") == g]
            out.append(f"**`{g}`** — `{sub[0]['meta'].get('dataset', '?')}`\n\n"
                       + fn(cfg, sub, suite_dir))
        return "\n\n".join(out)
    return wrapped


# ---------------------------------------------------------------------------
# The AUTO blocks
# ---------------------------------------------------------------------------

def _meta(cfg, recs, suite_dir) -> str:
    if not recs:
        return "_(nothing has finished yet)_"
    m = recs[0]["meta"]
    secs = sum(r["meta"].get("secs", 0) for r in recs)
    return "\n".join([
        f"**Suite** `{cfg['name']}` · **generated** "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"**{len(recs)} checkpoints**",
        "",
        f"| | |", "|---|---|",
        f"| datasets | "
        + ", ".join(f"`{d}`" for d in sorted({r['meta']['dataset'] for r in recs}))
        + " |",
        f"| grids | {', '.join(cfg['grids'])} |",
        f"| states | {m['n_states_used']:,} from "
        f"`{', '.join(m['state_files'])}` ({m['state_source']}, held out) |",
        f"| binning | {m['subdiv']}x per cell = {m['bin_size']:.2f} world units; "
        f"{m['bins_usable']}/{m['free_bins']} free bins cleared the state floor |",
        f"| goals | {m['goal_samples']} draws per task "
        f"(`add_noise_to_goal={m['add_noise_to_goal']}`, so the goal is a "
        f"distribution) |",
        f"| actions | {', '.join('`%s`' % s for s in m['action_sources'])} |",
        f"| oracle | `{m['expert']['name']}` / `{m['expert']['checkpoint']}` — the "
        f"policy that generated this dataset |"
        if m.get("expert") else "| oracle | not used |",
        f"| compute | {secs / 60:.1f} min total across {len(recs)} checkpoints |",
    ])


def _checkpoints(cfg, recs, suite_dir) -> str:
    rows = []
    for r in recs:
        s = r["meta"].get("selection") or {}
        seeds = s.get("all_seeds") or {}
        rows.append([
            r["meta"]["depth"], r["meta"]["seed"],
            f"{s.get('score', float('nan')):.3f}" if s.get("score") is not None else "—",
            f"`{s.get('rule_used', '?')}`",
            ", ".join(f"s{k}={v:.3f}" for k, v in sorted(seeds.items())),
            f"{s.get('spread'):.3f}" if s.get("spread") is not None else "—",
            f"`{r['meta']['run_id']}`",
        ])
    return _table(["depth", "seed", "score", "rule", "all seeds", "spread", "run"], rows)


def _topology(cfg, recs, suite_dir) -> str:
    rows = []
    for r in recs:
        m = r["metrics"]["q_policy"]["mean"]
        rows.append([r["meta"]["depth"], _fmt(m.get("rho_geodesic"), sign=False),
                     _fmt(m.get("rho_euclid"), sign=False),
                     f"**{_fmt(m.get('topology_gain'))}**",
                     int(m["n_bins"]) if m.get("n_bins") is not None else None,
                     _fmt(m.get("coverage"), 2, sign=False)])
    return (_table(["depth", "ρ(Q, −geodesic)", "ρ(Q, −euclid)", "topology_gain",
                    "bins", "coverage"], rows)
            + "\n\nUnits: ρ is dimensionless in [−1, 1]; `topology_gain` is their "
              "difference. `coverage` is a control and must be identical across rows.")


def _field(cfg, recs, suite_dir) -> str:
    rows = []
    for r in recs:
        m = r["metrics"]["q_policy"]["mean"]
        g = r["metrics"].get("ens_gap", {}).get("mean", {})
        rows.append([r["meta"]["depth"], _fmt(m.get("peak_geodesic"), 2, sign=False),
                     _fmt(m.get("top5_geodesic"), 2, sign=False),
                     _fmt(m.get("path_monotone"), 3, sign=False),
                     _fmt(m.get("goal_percentile"), 3, sign=False),
                     _fmt(m.get("q_std"), 3, sign=False),
                     _fmt(g.get("q_mean"), 3, sign=False)])
    return (_table(["depth", "peak_geodesic", "top5_geodesic", "path_monotone",
                    "goal_percentile", "q_std", "ens_gap"], rows)
            + "\n\nUnits: the two `*_geodesic` columns are **world units** (0 is "
              "perfect); `path_monotone` and `goal_percentile` are fractions; `q_std` "
              "and `ens_gap` are **logits** and are not comparable across rows — read "
              "`ens_gap` relative to `q_std` in the same row.")


def _action(cfg, recs, suite_dir) -> str:
    rows = []
    for r in recs:
        c = r["metrics"].get("_cross", {})
        rows.append([
            r["meta"]["depth"],
            f"**{_fmt(c.get('action_topology_gain', {}).get('mean'))}**",
            _fmt(c.get("action_topology_gain_all_bins", {}).get("mean")),
            _fmt(c.get("oracle_gap", {}).get("mean")),
            _fmt(c.get("oracle_gap_split_bins", {}).get("mean")),
            _fmt(c.get("split_bin_fraction"), 2, sign=False)])
    return (_table(["depth", "action_topology_gain", "(all bins)", "oracle_gap",
                    "(split bins)", "split bin fraction"], rows)
            + "\n\nUnits: **logits**, so sign and trend only. "
              "`action_topology_gain` is restricted to bins where the geodesic and "
              "straight-line directions disagree by at least "
              f"{recs[0]['meta'].get('direction_split_deg', 45) if recs else 45:.0f}°; "
              "the `(all bins)` column is the same quantity without that restriction, "
              "as a check that the restriction concentrates the signal rather than "
              "creating it.")


def _pertask(cfg, recs, suite_dir) -> str:
    if not recs:
        return "_(nothing has finished yet)_"
    tasks = [t["task_id"] for t in recs[0]["meta"]["tasks"]]
    head = ["metric", "depth"] + [f"task {t}" for t in tasks]
    rows = []
    for r in recs:
        c = r["metrics"].get("_cross", {}).get("action_topology_gain", {})
        rows.append(["action_topology_gain", r["meta"]["depth"]]
                    + [_fmt(v) for v in c.get("per_task", [])])
    for r in recs:
        pt = r["metrics"]["q_policy"]["per_task"]
        rows.append(["topology_gain", r["meta"]["depth"]]
                    + [_fmt(m.get("topology_gain")) for m in pt])
    for r in recs:
        pt = r["metrics"]["q_policy"]["per_task"]
        rows.append(["path_monotone", r["meta"]["depth"]]
                    + [_fmt(m.get("path_monotone"), sign=False) for m in pt])
    geo = recs[0]["meta"]["tasks"]
    note = ("\n\nTask starts and goals: "
            + "; ".join(f"**{t['task_id']}** {tuple(t['init_ij'])}→{tuple(t['goal_ij'])}"
                        for t in geo))
    return _table(head, rows) + note


def _reproduce(cfg, recs, suite_dir) -> str:
    return "\n".join([
        "```bash",
        "src/eval.py                     # the plan: which checkpoints, which evals",
        "src/eval.py --submit            # queue the job",
        "src/eval.py --status            # per-task state",
        "src/eval.py --figures           # redraw from finished tasks",
        "src/eval.py --report            # regenerate THIS file (prose is preserved)",
        "```",
        "",
        f"Suite file: [`evals.yaml`](evals.yaml). Raw output: `{suite_dir}`",
        "(`field.npz`, `metrics.json`, `meta.json`, `maze_render.png` per checkpoint).",
    ])


def _provenance(cfg, recs, suite_dir) -> str:
    if not recs:
        return ""
    m = recs[0]["meta"]
    lines = ["---", "", "<sub>"]
    if m.get("render"):
        r = m["render"]
        lines.append(
            f"Figures are drawn on the environment's own top-down render. The "
            f"world→pixel map is measured, not derived: {r['n_calib']} known body "
            f"positions, affine fit RMS **{r['fit_rms_px']} px**, leave-one-out max "
            f"**{r['loo_max_px']} px** on a {r['width']} px image; floor-plane "
            f"correction `(d−z)/d = {r['floor_scale']}` with `torso_z = "
            f"{r['torso_z']}`. ")
    if m.get("expert"):
        lines.append(
            f"The oracle actions come from `{m['expert']['dir']}` "
            f"(`{m['expert']['checkpoint']}`, trained on "
            f"`{m['expert']['env_name']}`). ")
    lines.append(f"Metrics self-check: `python -m offline.evals.critic_heatmap.impl`. "
                 f"Generated by `src/eval.py --report`.")
    lines.append("</sub>")
    return "\n".join(lines)


def _probes(cfg, recs, suite_dir) -> str:
    """The headline probe table: can geodesic distance be read linearly off phi?"""
    probes = _load_probes(suite_dir)
    if not probes:
        return "_(state_probes has not finished yet)_"
    out = []
    for g in sorted({r["meta"]["grid"] for r in probes}):
        sub = [r for r in probes if r["meta"]["grid"] == g]
        rows = []
        for rec in sub:
            R = rec["records"]
            def pick(site, target, pred, sh=False):
                for r in R:
                    if (r["site"] == site and r["target"] == target
                            and r["shuffled"] == sh and pred(r["features"])):
                        return r["score"]
                return None
            prim = lambda f: (not f.startswith("random_init") and f != "raw_obs"
                              and f.endswith("[0]"))
            rnd = lambda f: f.startswith("random_init") and f.endswith("[0]")
            raw = lambda f: f == "raw_obs"
            rows.append([
                rec["meta"]["depth"],
                _fmt(pick("phi_state_action", "geodesic_to_task", prim), 3, sign=False),
                _fmt(pick("phi_state_action", "geodesic_to_task", raw), 3, sign=False),
                _fmt(pick("phi_state_action", "geodesic_to_task", rnd), 3, sign=False),
                _fmt(pick("psi_goal", "geodesic_to_task", prim), 3, sign=False),
                _fmt(pick("phi_state_action", "maze_cell", prim), 3, sign=False),
                _fmt(pick("phi_state_action", "joint_vel_norm", prim), 3, sign=False),
                _fmt(pick("phi_state_action", "geodesic_to_task", prim, True), 3),
            ])
        out.append(f"**`{g}`** — `{sub[0]['meta']['dataset']}`\n\n" + _table(
            ["depth", "geo R² φ", "geo R² raw", "geo R² rand", "geo R² ψ",
             "cell acc φ", "vel R² φ (ctrl)", "shuffled"], rows))
    return ("\n\n".join(out) + "\n\nHeld out by TRAJECTORY. `φ` must beat BOTH "
            "`raw` (the same probe on the raw input that tower sees) and `rand` (the "
            "same architecture untrained) to have added anything. `shuffled` is the "
            "leak check and must sit at or below 0.")


_topology = _per_grid(_topology)
_field = _per_grid(_field)
_action = _per_grid(_action)
_pertask = _per_grid(_pertask)
_checkpoints = _per_grid(_checkpoints)

def _img_block(found, stem) -> str:
    if not found:
        return f"_(no `{stem}` figure yet)_"
    out = []
    for f in found:
        base = os.path.basename(f)
        grid = base[len(stem) + 2:-4] if base.startswith(stem + "__") else ""
        if grid:
            out.append(f"**`{grid}`**")
        out.append(f"![{stem}]({base})")
    return "\n\n".join(out)


BLOCKS = {"meta": _meta, "checkpoints": _checkpoints, "topology": _topology,
          "field": _field, "action": _action, "pertask": _pertask,
          "probes": _probes,
          "reproduce": _reproduce, "provenance": _provenance}


# ---------------------------------------------------------------------------

def build(cfg: dict, suite_dir: str) -> str:
    recs = _load(suite_dir)
    # Document and figures share ONE folder: `results/<suite>/README.md` alongside
    # `results/<suite>/fig*.png`, so the image paths are bare filenames.
    fig_dir = paths.results_dir(cfg["name"])
    out_md = os.path.join(fig_dir, "README.md")

    copied = 0
    for src in glob.glob(os.path.join(suite_dir, "figures", "**", "*.png"),
                         recursive=True) + glob.glob(
            os.path.join(suite_dir, "figures", "**", "*.csv"), recursive=True):
        shutil.copy2(src, os.path.join(fig_dir, os.path.basename(src)))
        copied += 1

    if not os.path.exists(out_md):
        text = open(TEMPLATE).read()
        text = text[text.index("# {{SUITE}}"):]          # drop the template's own note
        text = text.replace("{{SUITE}}", cfg["name"])
        open(out_md, "w").write(text)
        print(f"  created {out_md} from the template")
    text = open(out_md).read()

    # Figure slots. A suite that spans two grids writes `figX__<grid>.png`, so a
    # static `![](figX.png)` in the template would be a broken link -- and a silently
    # broken one, since markdown renders a missing image as alt text. The slots are
    # generated from what is actually in the figure directory, and a grid that
    # produced no figure simply does not appear.
    blocks = dict(BLOCKS)
    for nm in re.findall(r"<!-- AUTO:(img_fig[0-9A-Za-z_]+) -->", text):
        stem = nm[4:]
        found = sorted(glob.glob(os.path.join(fig_dir, f"{stem}*.png")))
        # Both `found` AND `stem` are bound now: capturing only one leaves the other
        # late-bound, so every slot renders the LAST loop iteration's name.
        blocks[nm] = (lambda f, st: (lambda cfg, recs, sd: _img_block(f, st)))(found, stem)

    for name, fn in blocks.items():
        pat = re.compile(r"(<!-- AUTO:%s -->)(.*?)(<!-- /AUTO:%s -->)" % (name, name),
                         re.S)
        if not pat.search(text):
            continue
        body = fn(cfg, recs, suite_dir)
        text = pat.sub(lambda mt: mt.group(1) + "\n" + body + "\n" + mt.group(3), text)
    open(out_md, "w").write(text)
    print(f"  wrote {out_md}  ({len(recs)} checkpoints, {copied} files -> {fig_dir})")
    return out_md
