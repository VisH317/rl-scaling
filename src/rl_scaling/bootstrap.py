"""Interpreter guard and import paths. Import this first, from every entry point.

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rl_scaling.bootstrap  # noqa: F401

Three lines in place of the eighteen that used to be copied into every CLI.

WHY A GUARD AT ALL. Stellar's system `python3` is 3.6.8 and this project needs 3.9+.
Running `./src/run.py` directly picks up `/usr/bin/python3` via the shebang, which
used to die with "SyntaxError: future feature annotations is not defined" -- an error
naming neither the cause nor the fix. This re-execs into the project venv instead, so
`./src/<name>.py` and `<venv>/bin/python src/<name>.py` both work.

THIS FILE MUST STAY 3.6-PARSEABLE. It is read by whatever interpreter the shebang
picked, before any re-exec. So: no `from __future__ import annotations`, no variable
annotations, no walrus, no positional-only params. f-strings are fine (3.6).

It re-execs `sys.argv`, not `__file__`, so the caller is resumed -- which is what
makes one copy serve every entry point.
"""

import os
import sys

#: Overridable for a different interpreter; otherwise the project venv on scratch.
PYTHON = os.environ.get("RL_SCALING_PYTHON") or os.path.join(
    os.environ.get("RL_SCALING_SCRATCH", "/scratch/gpfs/vt7993"),
    "rl-scaling", "venv", "bin", "python")

MIN_VERSION = (3, 9)

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_SRC)
#: OGBench's reference implementations, vendored under lib/. Agents import each other
#: by bare name (`from agents.sac import SACAgent`), so this must be on sys.path.
_IMPLS = os.path.join(_REPO, "lib", "ogbench", "impls")
_SCRIPTS = os.path.join(_SRC, "scripts")


def _reexec():
    if sys.version_info >= MIN_VERSION:
        return
    if not os.path.exists(PYTHON):
        sys.stderr.write(
            "this needs Python %d.%d+ (found %d.%d) and the project venv is not at\n"
            "  %s\nSet RL_SCALING_PYTHON to your interpreter, or run it as\n"
            "  <venv>/bin/python %s\n"
            % (MIN_VERSION[0], MIN_VERSION[1], sys.version_info[0],
               sys.version_info[1], PYTHON, sys.argv[0]))
        raise SystemExit(1)
    argv = list(sys.argv)
    argv[0] = os.path.abspath(argv[0])
    os.execv(PYTHON, [PYTHON] + argv)


def _paths():
    for p in (_SRC, _IMPLS, _SCRIPTS):
        if p not in sys.path:
            sys.path.insert(0, p)


_reexec()
_paths()

SRC, REPO, IMPLS = _SRC, _REPO, _IMPLS
