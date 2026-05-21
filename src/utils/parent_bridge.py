"""Re-export persistence and stable-process utilities from the parent paper.

The parent paper code lives in a sibling project at
    /Users/chisondiwarioba/Documents/non_gaussian_stochastic/
On Sherlock it is expected at
    $SCRATCH/non_gaussian_stochastic/

We do not vendor or duplicate the parent paper's code: we load it via
``importlib`` so that both manuscripts share the same persistence and
stable-generator implementations and any improvements to the parent paper
propagate to this project automatically.

We deliberately avoid ``sys.path`` manipulation because the parent paper's
package is also named ``src`` (the standard ``src/`` layout); inserting the
parent paper's ``simulations/`` on ``sys.path`` would shadow this project's
own ``src`` package.

Set ``TOPO_DWMRI_PARENT_PATH`` in the environment to override the lookup
path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _find_parent_root() -> Path:
    env = os.environ.get("TOPO_DWMRI_PARENT_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path.home() / "Documents" / "non_gaussian_stochastic",
            Path("/Users/chisondiwarioba/Documents/non_gaussian_stochastic"),
            Path(os.environ.get("SCRATCH", "")) / "non_gaussian_stochastic",
        ]
    )
    for c in candidates:
        if (c / "simulations" / "src" / "utils" / "persistence.py").is_file():
            return c
    raise ImportError(
        "Could not locate parent paper project. Set TOPO_DWMRI_PARENT_PATH "
        "or clone csw-research/topological-levy-phase-transition next to "
        "this repository."
    )


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PARENT_ROOT = _find_parent_root()
_PARENT_SIMS = PARENT_ROOT / "simulations"

_persistence = _load_module(
    "_parent_paper_persistence",
    _PARENT_SIMS / "src" / "utils" / "persistence.py",
)
_stable = _load_module(
    "_parent_paper_stable",
    _PARENT_SIMS / "src" / "generators" / "stable.py",
)

sublevel_persistence_1d = _persistence.sublevel_persistence_1d
persistence_lifetimes = _persistence.persistence_lifetimes
total_persistence = _persistence.total_persistence
persistence_entropy = _persistence.persistence_entropy
stable_rvs = _stable.stable_rvs
stable_levy_process = _stable.stable_levy_process

__all__ = [
    "sublevel_persistence_1d",
    "persistence_lifetimes",
    "total_persistence",
    "persistence_entropy",
    "stable_rvs",
    "stable_levy_process",
    "PARENT_ROOT",
]
