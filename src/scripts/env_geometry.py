#!/usr/bin/env python3
"""Maze geometry and data coverage for an OGBench maze environment.

Produces the numbers in section 1 of `src/offline/results_phase1.md`: the maze map, the
per-cell data coverage, how far one trajectory actually travels, and -- the part that
turned out to matter -- the angle between the straight line to each task's goal and the
first move that makes progress toward it.

That angle predicts GCBC's per-task success rate exactly on antmaze-medium-stitch, and
path length does not. Phase 2 probes should bin on it.

Usage:
    python src/scripts/env_geometry.py
    python src/scripts/env_geometry.py --env antmaze-large-navigate-v0
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline import config as C  # noqa: E402


def bfs(maze: np.ndarray, src: tuple[int, int]) -> dict:
    """Cell distances from `src` over 4-connected free cells."""
    dist, queue = {src: 0}, collections.deque([src])
    while queue:
        c = queue.popleft()
        for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                  (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if (0 <= n[0] < maze.shape[0] and 0 <= n[1] < maze.shape[1]
                    and not maze[n] and n not in dist):
                dist[n] = dist[c] + 1
                queue.append(n)
    return dist


def neighbors(maze: np.ndarray, c: tuple[int, int]) -> list:
    return [n for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                        (c[0], c[1] + 1), (c[0], c[1] - 1))
            if 0 <= n[0] < maze.shape[0] and 0 <= n[1] < maze.shape[1] and not maze[n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="antmaze-medium-stitch-v0")
    args = ap.parse_args()

    import ogbench
    env = ogbench.make_env_and_datasets(args.env, env_only=True)
    u = env.unwrapped
    maze = np.asarray(u.maze_map)
    unit = float(getattr(u, "maze_unit", 4.0))

    def to_xy(c):
        # OGBench maps cell (i, j) -> world (x, y) = (unit*(j-1), unit*(i-1)).
        return np.array([unit * (c[1] - 1), unit * (c[0] - 1)])

    print(f"{args.env}")
    print(f"  obs {env.observation_space.shape}  act {env.action_space.shape}  "
          f"maze {maze.shape}  unit {unit}  tasks {len(u.task_infos)}")

    # --- data coverage -----------------------------------------------------
    spec = C.DATASETS[args.env]
    train_p, _ = spec.rel_paths()
    path = os.path.join(C.DATA_DIR, train_p)
    occ = counts = None
    reach = np.array([])
    if os.path.exists(path):
        d = np.load(path)
        xy = d["observations"][:, :2]          # dims 0-1 are the agent's (x, y)
        ii = np.clip(np.round(xy[:, 1] / unit).astype(int) + 1, 0, maze.shape[0] - 1)
        jj = np.clip(np.round(xy[:, 0] / unit).astype(int) + 1, 0, maze.shape[1] - 1)
        occ = np.zeros(maze.shape, dtype=np.int64)
        np.add.at(occ, (ii, jj), 1)
        counts = len(xy)

        T = spec.traj_len
        n = len(xy) // T
        cells = np.stack([ii, jj], 1).reshape(n, T, 2)
        vals = []
        for k in range(n):
            a, b = tuple(cells[k, 0]), tuple(cells[k, -1])
            if maze[a] or maze[b]:
                continue
            v = bfs(maze, a).get(b, -1)
            if v >= 0:
                vals.append(v)
        reach = np.array(vals)

    print(f"\nmaze ('##' = wall"
          + (f", numbers = % of {counts:,} dataset states)" if occ is not None else ")"))
    for i in range(maze.shape[0]):
        row = []
        for j in range(maze.shape[1]):
            if maze[i, j]:
                row.append("  ## ")
            elif occ is not None:
                row.append("%5.2f" % (100 * occ[i, j] / counts))
            else:
                row.append("   . ")
        print("   " + " ".join(row))

    if reach.size:
        print(f"\n{len(reach)} trajectories. Cell-distance between a trajectory's own "
              f"first and last state:")
        print("   mean %.2f  median %.0f  p90 %.0f  max %d"
              % (reach.mean(), np.median(reach), np.percentile(reach, 90), reach.max()))
        u_, c_ = np.unique(reach, return_counts=True)
        print("   histogram:", dict(zip(u_.tolist(), c_.tolist())))

    # --- tasks -------------------------------------------------------------
    med = float(np.median(reach)) if reach.size else float("nan")
    print("\n%-5s %-9s %-9s %6s %7s %8s %7s %8s %10s" %
          ("task", "start", "goal", "cells", "units", "straight", "detour",
           "stitches", "1st-move deg"))
    for t, info in enumerate(u.task_infos, 1):
        s = tuple(int(v) for v in info["init_ij"])
        g = tuple(int(v) for v in info["goal_ij"])
        dist = bfs(maze, s).get(g)
        if dist is None:
            print(f"{t:<5} {s} -> {g}: unreachable")
            continue
        straight = float(np.linalg.norm(to_xy(g) - to_xy(s)))
        gdir = (to_xy(g) - to_xy(s)) / max(straight, 1e-9)
        # Only moves that actually shorten the remaining path count as progress.
        good = [n for n in neighbors(maze, s)
                if bfs(maze, n).get(g, 10**9) == dist - 1]
        angs = [float(np.degrees(np.arccos(np.clip(
            np.dot((to_xy(n) - to_xy(s)) / unit, gdir), -1, 1)))) for n in good]
        print("%-5d %-9s %-9s %6d %7.0f %8.0f %7.2f %8.1f %10.0f" %
              (t, str(s), str(g), dist, dist * unit, straight,
               dist * unit / max(straight, 1e-9), dist / med if med == med else 0,
               min(angs) if angs else float("nan")))
    print("\ndetour   = maze path / straight-line distance")
    print("stitches = path cells / median trajectory reach")
    print("1st-move = smallest angle between a progress-making first move and the "
          "straight line to the goal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
