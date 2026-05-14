"""sit.paths — canonical project-root, data and plots paths.

Resolves paths *relative to the repository root* so that
``python -m sit.solvers.admm`` (or any other module) works no matter where
the user is when they invoke it.

Repo layout assumption (Phase 0):

    <repo_root>/
        src/sit/...     ← this file lives here at sit/paths.py
        data/           ← raw + cached artefacts
        plots/          ← generated PNG figures
        pyproject.toml  ← marker used to detect the root
"""

from __future__ import annotations

from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def _find_root(start: Path) -> Path:
    """Walk upward from *start* looking for a directory containing pyproject.toml."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: 3 levels up from src/sit/paths.py (= src/sit, src, repo)
    return start.parents[2]


REPO_ROOT: Path = _find_root(_THIS_FILE)
DATA_DIR: Path = REPO_ROOT / "data"
PLOTS_DIR: Path = REPO_ROOT / "plots"
BENCHMARK_DIR: Path = REPO_ROOT / "benchmarks" / "_results"


def ensure_dirs() -> None:
    """Create the standard output directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "BENCHMARK_DIR",
    "DATA_DIR",
    "PLOTS_DIR",
    "REPO_ROOT",
    "ensure_dirs",
]
