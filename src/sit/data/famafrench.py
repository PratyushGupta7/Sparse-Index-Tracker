"""sit/data/famafrench.py — Fama-French 3-factor daily returns loader + cache.

Downloads Ken French's *F-F Research Data Factors (Daily)* CSV — the canonical
source of MKT-RF, SMB, HML, RF — once, parses it to a tidy
``pd.DataFrame[date → factor pct]``, and caches the parsed CSV under
``data/famafrench_daily.csv`` so subsequent runs are network-free.

The raw archive at the URL below is a zip containing a single CSV with two
sections (daily then annual) separated by a blank line; both have the same
column header and require some idiomatic clean-up. Robustness notes:

* The header line is not the first line — we strip the leading
  research-description block by detecting the first row that begins with a
  date-like ``YYYYMMDD`` token.
* Only the *daily* section is kept (we stop reading at the second
  ``"Annual"`` section).
* All percentages in the file are quoted as numbers, e.g. ``0.45`` ⇒ 0.45 %
  ⇒ 0.0045. We divide by 100 on parse so downstream code can multiply
  factor exposures by *decimal* portfolio returns directly.

The result has columns ``["mkt_rf", "smb", "hml", "rf"]`` and a
``pd.DatetimeIndex`` named ``date``.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sit.paths import DATA_DIR

FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
DEFAULT_CACHE_PATH = DATA_DIR / "famafrench_daily.csv"


@dataclass(frozen=True)
class FamaFrenchFactors:
    """Container around a parsed FF3 daily DataFrame.

    Attributes
    ----------
    df
        Frame with ``DatetimeIndex`` and columns
        ``["mkt_rf", "smb", "hml", "rf"]`` in *decimal* daily returns.
    source
        Free-form provenance string (URL or "cache").
    """

    df: pd.DataFrame
    source: str

    def aligned_to(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return the FF3 frame reindexed (forward-filled) to a target date index.

        For daily portfolio analytics where weekends / market holidays may
        differ from CRSP, we forward-fill across at most 3 days and drop any
        residual NaNs.
        """
        idx = pd.DatetimeIndex(index).tz_localize(None)
        out = self.df.reindex(idx).ffill(limit=3)
        return out.dropna()


# ---------------------------------------------------------------------------
# Download / parse / cache
# ---------------------------------------------------------------------------


def _parse_ff3_csv(text: str) -> pd.DataFrame:
    """Parse Ken French's daily CSV text into a tidy frame.

    Robust to header preambles and the trailing annual-frequency section.
    """
    lines = text.splitlines()
    # Find header (first line whose first cell is `Mkt-RF` etc.) — Ken French
    # always names columns "Mkt-RF, SMB, HML, RF". Some files prefix the header
    # with whitespace.
    header_idx = -1
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        if {"Mkt-RF", "SMB", "HML", "RF"}.issubset(set(cells)):
            header_idx = i
            break
    if header_idx < 0:
        raise RuntimeError("Could not locate FF3 header row in CSV.")

    # The body starts the line after the header. Stop at first non-numeric
    # date or the appearance of a second "Mkt-RF" header (annual section).
    body: list[str] = []
    for line in lines[header_idx + 1 :]:
        cells = [c.strip() for c in line.split(",")]
        if not cells or not cells[0]:
            # Blank line ⇒ end of daily section.
            if body:
                break
            continue
        if cells[0].isdigit() and len(cells[0]) == 8:
            body.append(line)
        elif "Mkt-RF" in cells:
            break
        else:
            # Defensive: any other non-date row ⇒ stop.
            if body:
                break

    if not body:
        raise RuntimeError("Empty FF3 daily section after parsing.")

    raw = pd.read_csv(
        io.StringIO("\n".join([lines[header_idx], *body])),
        skipinitialspace=True,
    )
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    raw = raw.set_index("date").sort_index()
    raw = raw.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"})

    # Convert percent → decimal
    for col in ("mkt_rf", "smb", "hml", "rf"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce") / 100.0
    return raw[["mkt_rf", "smb", "hml", "rf"]].dropna()


def _download_ff3() -> str:
    req = urllib.request.Request(
        FF3_URL,
        headers={"User-Agent": "Mozilla/5.0 sparse-index-tracker"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("FF3 zip contained no CSV files.")
        with zf.open(names[0]) as f:
            return f.read().decode("latin-1")


def load_famafrench_daily(
    *,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> FamaFrenchFactors:
    """Return the FF3 daily factor frame, hitting the cache first.

    Parameters
    ----------
    cache_path
        Override the cache file location (defaults to
        ``data/famafrench_daily.csv``).
    force_refresh
        If ``True``, re-download even when a cache file exists.

    Notes
    -----
    Network is contacted only when the cache is absent or
    ``force_refresh=True``. Tests should monkeypatch ``_download_ff3`` or pass
    ``cache_path`` to a tmp dir holding a pre-baked CSV.
    """
    cache_path = cache_path or DEFAULT_CACHE_PATH
    if cache_path.is_file() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
        return FamaFrenchFactors(df=df, source=f"cache:{cache_path}")

    csv_text = _download_ff3()
    df = _parse_ff3_csv(csv_text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index_label="date")
    return FamaFrenchFactors(df=df, source=FF3_URL)


# ---------------------------------------------------------------------------
# Synthetic factory — used in unit tests when network is not available.
# ---------------------------------------------------------------------------


def make_synthetic_ff3(
    dates: pd.DatetimeIndex,
    *,
    seed: int = 20260511,
    mkt_vol: float = 0.012,
    smb_vol: float = 0.005,
    hml_vol: float = 0.005,
    rf_daily: float = 1e-5,
) -> FamaFrenchFactors:
    """Generate a fully reproducible synthetic FF3 frame for tests.

    Mean-zero, Gaussian, uncorrelated factors with a constant tiny risk-free
    rate. Calibrated to roughly match the empirical daily volatilities of the
    real factor returns over 2010–2025.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)
    df = pd.DataFrame(
        {
            "mkt_rf": rng.standard_normal(n) * mkt_vol,
            "smb": rng.standard_normal(n) * smb_vol,
            "hml": rng.standard_normal(n) * hml_vol,
            "rf": np.full(n, rf_daily),
        },
        index=pd.DatetimeIndex(dates).tz_localize(None),
    )
    df.index.name = "date"
    return FamaFrenchFactors(df=df, source="synthetic")


__all__ = [
    "DEFAULT_CACHE_PATH",
    "FF3_URL",
    "FamaFrenchFactors",
    "load_famafrench_daily",
    "make_synthetic_ff3",
]
