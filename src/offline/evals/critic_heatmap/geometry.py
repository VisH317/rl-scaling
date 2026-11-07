"""Simple structures inside a maze, and the questions each one asks of a critic.

WHY THIS EXISTS
---------------
`critic_heatmap`'s aggregate metrics -- `rho_geodesic`, `topology_gain` -- score the
whole value field at once. A single correlation hides *which* geometry a critic gets
right. A critic can score a respectable rho by being roughly radial around the goal
while being wrong about every wall in the maze, because most cell pairs are not
separated by one.

These probes ask small questions with known answers instead. Each one is a named
structure found in the maze itself -- a straight run, an elbow, a pair of cells either
side of a wall -- paired with an assertion that any critic encoding maze distance must
satisfy. They are local, so a failure points at a geometry rather than at a number.

EVERYTHING HERE IS SCALE-FREE. Scores are accuracies in [0, 1], Spearman rho in
[-1, 1], or a Q difference already divided by the run's own `q_scale`. Q is an
unnormalised log-density-ratio, so a raw logit gap means nothing across runs.

DEFINED ON THE COARSE MAZE, scored on the fine bin grid. The structures are cell-level
facts about the layout; the critic is sampled per fine bin and averaged up to cells,
so a cell with too little data drops out rather than contributing a noisy mean.

The same 8x8 map backs antmaze-medium AND humanoidmaze-medium, so every structure
below is identical across the two environments and the scores are directly comparable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

#: 4-connectivity. Diagonal moves are not available to the agent, so a diagonal
#: neighbour is not "adjacent" for any purpose here.
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: A straight run must be at least this long to count as a corridor. Two cells are a
#: pair, not a line, and give a Spearman rho of +/-1 on no evidence.
MIN_LINE = 3

#: `wall_pair`: how much longer the real path must be than the obstacle-free one.
#: 2 is the smallest possible detour on a 4-connected grid (one step out, one back).
WALL_PAIR_MIN_DETOUR = 2
#: ...and how close the two cells must be in obstacle-free terms, so the pair is a
#: LOCAL confusion a radial value field would actually make.
WALL_PAIR_MAX_FREE = 3


def free_cells(maze) -> list:
    return [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1])
            if maze[i, j] == 0]


def neighbours(maze, ij) -> list:
    i, j = ij
    out = []
    for di, dj in STEPS:
        a, b = i + di, j + dj
        if 0 <= a < maze.shape[0] and 0 <= b < maze.shape[1] and maze[a, b] == 0:
            out.append((a, b))
    return out


def free_distance(a, b) -> int:
    """Obstacle-free distance under the SAME 4-connectivity: Manhattan.

    This, not Euclidean, is the foil for every pairwise test here. On a 4-connected
    grid the geodesic equals Manhattan whenever no wall intervenes, so comparing the
    two isolates the effect of WALLS exactly -- in an open room they agree and the
    tests correctly find nothing to measure. Euclidean would differ from the geodesic
    even with no walls present (L2 against L1), and those differences would be
    counted as maze structure when they are an artefact of the connectivity.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cell_geodesic(maze, src) -> dict:
    """BFS hop counts from `src` over free cells. Unreachable cells are absent."""
    dist = {src: 0}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nb in neighbours(maze, cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


# ---------------------------------------------------------------------------
# The structures
# ---------------------------------------------------------------------------

def find_lines(maze) -> list:
    """Maximal straight runs of free cells, length >= MIN_LINE.

    The simplest possible geometry: no turns, no walls in the way, so geodesic and
    straight-line distance agree exactly along the run. A critic that cannot order Q
    along a straight corridor has not learned distance in any form.
    """
    out = []
    H, W = maze.shape
    for axis in (0, 1):
        n_outer, n_inner = (H, W) if axis == 1 else (W, H)
        for a in range(n_outer):
            run = []
            for b in range(n_inner):
                ij = (a, b) if axis == 1 else (b, a)
                if maze[ij] == 0:
                    run.append(ij)
                else:
                    if len(run) >= MIN_LINE:
                        out.append(dict(kind="line",
                                        axis="row" if axis == 1 else "col",
                                        cells=run))
                    run = []
            if len(run) >= MIN_LINE:
                out.append(dict(kind="line", axis="row" if axis == 1 else "col",
                                cells=run))
    return out


def find_corners(maze) -> list:
    """Elbows: a free cell whose only two free neighbours are perpendicular.

    The cell is the turn. Its two arms are the question: which way is the goal? A
    critic using straight-line distance will often pick the wrong arm, because the
    arm pointing at the goal may be the one that dead-ends.
    """
    out = []
    for ij in free_cells(maze):
        nbs = neighbours(maze, ij)
        if len(nbs) != 2:
            continue
        (ai, aj), (bi, bj) = nbs
        # Perpendicular iff the two arms differ on both axes.
        if (ai - ij[0], aj - ij[1]) != (ij[0] - bi, ij[1] - bj):
            out.append(dict(kind="corner", cell=ij, arms=[nbs[0], nbs[1]]))
    return out


def find_wall_pairs(maze) -> list:
    """Cells that are close in a straight line and far by path: "two sections".

    This is the sharpest geometry in the maze. The two cells are a step or two apart
    on the floor but the agent must walk around; a critic that has learned the
    *environment* separates them, a critic that has learned a radial bump around the
    goal does not. Ranked by the ratio, which is the strength of the test.
    """
    cells = free_cells(maze)
    geos = {c: cell_geodesic(maze, c) for c in cells}
    out = []
    for n, a in enumerate(cells):
        for b in cells[n + 1:]:
            f = free_distance(a, b)
            g = geos[a].get(b)
            if g is None or f > WALL_PAIR_MAX_FREE or g - f < WALL_PAIR_MIN_DETOUR:
                continue
            out.append(dict(kind="wall_pair", a=a, b=b, free=int(f),
                            geodesic=int(g), detour=int(g - f),
                            ratio=round(g / max(f, 1), 3)))
    out.sort(key=lambda d: (-d["ratio"], -d["detour"]))
    return out


def find_deadends(maze) -> list:
    """Cul-de-sacs: a cell with exactly one free neighbour, plus the chain out of it.

    Walking into a dead end always increases distance to any goal outside it. The
    chain is followed until the corridor branches, so the test has more than one
    cell to be monotone over where the maze allows.
    """
    out = []
    for ij in free_cells(maze):
        if len(neighbours(maze, ij)) != 1:
            continue
        chain, cur, prev = [ij], neighbours(maze, ij)[0], ij
        while True:
            chain.append(cur)
            nxt = [n for n in neighbours(maze, cur) if n != prev]
            if len(nxt) != 1:
                break                      # a branch, or another dead end: stop
            prev, cur = cur, nxt[0]
        out.append(dict(kind="deadend", tip=ij, chain=chain, mouth=chain[-1]))
    return out


def find_bottlenecks(maze) -> list:
    """Articulation cells: removing one disconnects the maze into two sections.

    The strongest global structure a maze has. Every path between the two sides runs
    through this cell, so the value field should be ordered consistently on either
    side of it -- the far side uniformly worse than the near side, with no leakage.
    """
    cells = free_cells(maze)
    if not cells:
        return []
    out = []
    for ij in cells:
        rest = [c for c in cells if c != ij]
        if not rest:
            continue
        blocked = maze.copy()
        blocked[ij] = 1
        seen = cell_geodesic(blocked, rest[0])
        reached = {c for c in rest if c in seen}
        if len(reached) == len(rest):
            continue                        # still connected: not an articulation
        far = [c for c in rest if c not in reached]
        out.append(dict(kind="bottleneck", cell=ij,
                        side_a=sorted(reached), side_b=sorted(far),
                        smaller=min(len(reached), len(far))))
    out.sort(key=lambda d: -d["smaller"])
    return out


def find_detour_pairs(maze, goal) -> list:
    """Pairs the straight-line heuristic gets backwards, for one goal.

    `a` is nearer the goal in a straight line but FARTHER along the floor than `b`.
    Ranking these by Q is the single cleanest discriminator in the suite: a critic
    that has learned the maze ranks by geodesic, one that has learned a radial bump
    ranks by euclid, and chance is 0.5.
    """
    cells = free_cells(maze)
    gd = cell_geodesic(maze, goal)
    out = []
    for n, a in enumerate(cells):
        for b in cells[n + 1:]:
            ga, gb = gd.get(a), gd.get(b)
            if ga is None or gb is None or ga == gb:
                continue
            fa, fb = free_distance(a, goal), free_distance(b, goal)
            if fa == fb:
                continue
            # Disagreement: with the walls removed the pair orders one way, with the
            # walls present the other. Only a wall can cause this.
            if (fa < fb) == (ga < gb):
                continue
            nearer_geo, farther_geo = (a, b) if ga < gb else (b, a)
            out.append(dict(kind="detour_pair", nearer_geo=nearer_geo,
                            farther_geo=farther_geo,
                            d_geodesic=int(abs(ga - gb)),
                            d_free=int(abs(fa - fb))))
    out.sort(key=lambda d: -d["d_geodesic"])
    return out


def catalogue(maze) -> dict:
    """Every goal-independent structure. Goal-dependent ones take the goal separately."""
    return dict(lines=find_lines(maze), corners=find_corners(maze),
                wall_pairs=find_wall_pairs(maze), deadends=find_deadends(maze),
                bottlenecks=find_bottlenecks(maze))


def summarise(maze) -> str:
    c = catalogue(maze)
    return (f"{len(free_cells(maze))} free cells, {len(c['lines'])} lines, "
            f"{len(c['corners'])} corners, {len(c['wall_pairs'])} wall pairs, "
            f"{len(c['deadends'])} dead ends, {len(c['bottlenecks'])} bottlenecks")


# ---------------------------------------------------------------------------
# Self-check -- mazes whose structures can be counted by hand.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}: {got} (want {want})")

    # A 1x5 corridor: one line, no corners, no wall pairs, two dead ends, and every
    # interior cell is an articulation point.
    corridor = np.array([[1, 1, 1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1]])
    check("corridor: lines", len(find_lines(corridor)), 1)
    check("corridor: line length", len(find_lines(corridor)[0]["cells"]), 5)
    check("corridor: corners", len(find_corners(corridor)), 0)
    check("corridor: wall pairs", len(find_wall_pairs(corridor)), 0)
    check("corridor: dead ends", len(find_deadends(corridor)), 2)
    check("corridor: bottlenecks", len(find_bottlenecks(corridor)), 3)

    # An L: exactly one elbow, and the two arms are its only free neighbours.
    L = np.array([[1, 1, 1, 1, 1],
                  [1, 0, 1, 1, 1],
                  [1, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1]])
    check("L: corners", len(find_corners(L)), 1)
    check("L: elbow cell", find_corners(L)[0]["cell"], (3, 1))
    check("L: lines", len(find_lines(L)), 2)

    # Two 1x2 rooms either side of one wall: euclid 2, geodesic 6 the long way round.
    split = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 1, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1]])
    wp = find_wall_pairs(split)
    check("split: wall pairs", len(wp), 1)
    check("split: the separated pair", (wp[0]["a"], wp[0]["b"]), ((1, 1), (1, 3)))
    check("split: obstacle-free distance", wp[0]["free"], 2)
    check("split: geodesic", wp[0]["geodesic"], 4)
    check("split: detour forced by the wall", wp[0]["detour"], 2)

    # Detour pairs need a goal and must be antisymmetric in the right way.
    dp = find_detour_pairs(split, (1, 3))
    bad = [d for d in dp
           if cell_geodesic(split, (1, 3))[d["nearer_geo"]]
           >= cell_geodesic(split, (1, 3))[d["farther_geo"]]]
    check("split: detour pairs labelled correctly", len(bad), 0)

    # A fully open room has no walls to be wrong about: no wall pairs, no detours.
    openr = np.pad(np.zeros((4, 4), int), 1, constant_values=1)
    check("open room: wall pairs", len(find_wall_pairs(openr)), 0)
    # The point of using obstacle-free distance as the foil: with no walls there is
    # nothing to be wrong about, so BOTH pairwise tests must come back empty. Against
    # a Euclidean foil this returned 2 -- pure L1-vs-L2 artefact, no maze involved.
    check("open room: detour pairs", len(find_detour_pairs(openr, (1, 1))), 0)
    check("open room: dead ends", len(find_deadends(openr)), 0)

    print("\n" + ("maze geometry: ALL PASS" if ok else "SELF-CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
