"""Interpretability / diagnostic evaluations of trained checkpoints.

An *eval* here is not a policy score. `scripts/reeval.py` already answers "how good is
this agent" by rolling it out. These evals answer "what did the network learn", and they
are read-only with respect to the runs: they load `params_*.pkl` and write only into
`<ANALYSIS_DIR>/evals/<suite>/`.

Each eval is one directory in this package and must contain:

    eval.yaml   the eval's own config: status, which agents it applies to, its params
    README.md   what the test is, what it measures, how to read the output
    impl.py     `run(task) -> dict`, executed once per selected checkpoint
    figures.py  `make_figures(suite_dir, out_dir) -> list[str]`  (optional)

`src/eval.py` is the only entry point: it reads the suite config (`evals.yaml`),
discovers the eval directories here, picks one checkpoint per (grid, agent, depth),
and compiles the cross product into a list of tasks it can hand to Slurm.
"""
