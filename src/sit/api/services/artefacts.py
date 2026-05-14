"""Cached loaders for Phase-3/4 JSON artefacts.

The cache is keyed by ``(path, mtime, size)`` so editing a results file on
disk invalidates automatically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sit.api.settings import get_settings

logger = logging.getLogger(__name__)


_artefact_cache: dict[tuple[str, float, int], Any] = {}


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artefact missing: {path}")
    stat = path.stat()
    key = (str(path), stat.st_mtime, stat.st_size)
    cached = _artefact_cache.get(key)
    if cached is not None:
        return cached
    with path.open("r") as f:
        data = json.load(f)
    _artefact_cache[key] = data
    return data


def load_walkforward(window: str = "wf_20180101_20251231") -> dict[str, Any]:
    settings = get_settings()
    return _load_json(settings.data_dir / "backtest" / window / "results.json")


def load_multi_index() -> dict[str, Any]:
    settings = get_settings()
    return _load_json(settings.benchmarks_dir / "multi_index.json")


def load_method_comparison() -> dict[str, Any]:
    settings = get_settings()
    return _load_json(settings.benchmarks_dir / "method_comparison.json")


def load_cvxpy_speedup() -> dict[str, Any]:
    settings = get_settings()
    return _load_json(settings.benchmarks_dir / "cvxpy_speedup.json")


def load_regime_results() -> dict[str, Any]:
    settings = get_settings()
    return _load_json(settings.data_dir / "regime_results.json")


def downsample_curve(curve: dict[str, float], max_points: int = 1500) -> list[dict[str, Any]]:
    """Down-sample a date→value mapping to ≤ ``max_points`` points uniformly."""
    items = sorted(curve.items())
    if len(items) <= max_points:
        return [{"date": d, "value": float(v)} for d, v in items]
    step = max(1, -(-len(items) // max_points))  # ceil division
    sampled = items[::step]
    if sampled and sampled[-1] != items[-1]:
        sampled.append(items[-1])
    return [{"date": d, "value": float(v)} for d, v in sampled]


def filter_by_window(
    curve: dict[str, float],
    start: str | None,
    end: str | None,
) -> dict[str, float]:
    """Optional date-window filter for walkforward results."""
    if not start and not end:
        return curve
    out: dict[str, float] = {}
    for d, v in curve.items():
        if start and d < start:
            continue
        if end and d > end:
            continue
        out[d] = v
    return out


def parse_window(start: str | None, end: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, val in [("start", start), ("end", end)]:
        if val is None:
            continue
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid {label} date '{val}'; expected YYYY-MM-DD.") from exc
        out[label] = val
    return out


def clear_cache() -> None:
    """Test helper."""
    _artefact_cache.clear()
