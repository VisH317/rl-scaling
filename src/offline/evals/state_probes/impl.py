"""Linear probes on the critic's representations -- PLANNED, not implemented.

The design is in README.md beside this file. It is registered as an eval so that
`src/eval.py` lists it, prices it and refuses to submit it, rather than leaving the
plan in a document nothing executes.

Implement in this order; each step is useful on its own:

  1. `representations(agent, obs, acts, goals)` -> {"psi_goal": (N, d),
     "phi_state_action": (N, d)}, batched, ensemble axis kept separate.
  2. `targets(run_cfg, geo, obs, traj_ids)` -> {name: (N,) or (N, k)} plus a
     `kind` of "regression" or "classification" per target.
  3. `fit_probe(X, y, kind, groups, spec)` -> held-out R^2 / accuracy, with the
     lambda chosen by grouped CV INSIDE the training trajectories only.
  4. the three baselines, which are the result: a probe score means nothing without
     `raw_obs` and `random_init` beside it.
"""

from __future__ import annotations

import os

_README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")


def run(task: dict) -> dict:
    raise NotImplementedError(
        "state_probes is status: planned -- the design is in\n"
        f"  {_README}\n"
        "and the knobs it will take are already in eval.yaml beside it. Implement the "
        "four functions listed in impl.py's docstring, then set status: ready."
    )
