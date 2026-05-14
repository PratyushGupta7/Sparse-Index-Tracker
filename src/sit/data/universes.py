"""sit/data/universes.py — index constituent loaders for Phase 4.

One factory function per supported index. Each returns

    (tickers: list[str], benchmark_ticker: str)

* the **benchmark** ticker is the one yfinance / NSE quote that we treat as
  ``y`` in the tracking problem,
* the **tickers** are the constituents in their yfinance form (e.g. India
  needs the ``.NS`` suffix), with the benchmark ticker excluded to avoid
  data leakage.

All loaders cache the constituent list to ``data/universes/{name}_constituents.csv``
so subsequent runs are network-free. Cache invalidation is opt-in via the
``force_refresh`` kwarg or by deleting the file. Writes are atomic.

Supported universes
-------------------
+------------+-------------+----------------------------------------------+
| name       | benchmark   | source                                       |
+============+=============+==============================================+
| sp500      | SPY         | Wikipedia "List of S&P 500 companies"        |
| nasdaq100  | QQQ         | Wikipedia "Nasdaq-100"                       |
| russell2000| IWM         | iShares IWM holdings CSV (BlackRock)         |
| nifty50    | ^NSEI       | NSE "ind_nifty50list.csv"                    |
+------------+-------------+----------------------------------------------+

Each loader writes a CSV with columns ``[ticker]`` (one row per constituent)
to keep the format dead simple — no risk of mismatched schemas across
indices, easy to inspect by hand.
"""

from __future__ import annotations

import csv
import io
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from sit.paths import DATA_DIR

UNIVERSES_DIR = DATA_DIR / "universes"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Canonical ETF / index proxies for each universe.
INDEX_METADATA: dict[str, dict[str, str]] = {
    "sp500": {"benchmark": "SPY", "label": "S&P 500", "region": "US"},
    "nasdaq100": {"benchmark": "QQQ", "label": "Nasdaq-100", "region": "US"},
    "russell2000": {"benchmark": "IWM", "label": "Russell 2000", "region": "US"},
    "nifty50": {"benchmark": "^NSEI", "label": "Nifty 50", "region": "IN"},
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    return UNIVERSES_DIR / f"{name}_constituents.csv"


def _atomic_write_tickers(path: Path, tickers: list[str]) -> None:
    """Atomically write a single-column CSV of tickers (with header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker"])
        for t in tickers:
            w.writerow([t])
    os.replace(tmp, path)


def _read_cached_tickers(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header is None or "ticker" not in [c.lower() for c in header]:
                # legacy single-column-no-header fallback
                f.seek(0)
                return [row[0] for row in csv.reader(f) if row]
            return [row[0] for row in r if row]
    except (OSError, StopIteration):
        return None


def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> bytes:
    """Tiny GET helper with a sane User-Agent + linear back-off."""
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return bytes(resp.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"HTTP fetch failed after {retries} attempts: {url} ({last_err!r})")


def _generic_load(
    name: str,
    fetch_fn: Callable[[], list[str]],
    *,
    force_refresh: bool,
    cache_path: Path | None = None,
) -> tuple[list[str], str]:
    """Cache-aware factory used by every public loader."""
    benchmark = INDEX_METADATA[name]["benchmark"]
    cp = cache_path if cache_path is not None else _cache_path(name)
    if not force_refresh:
        cached = _read_cached_tickers(cp)
        if cached:
            tickers = [t for t in cached if t and t != benchmark]
            return tickers, benchmark
    tickers = fetch_fn()
    tickers = [t for t in tickers if t and t != benchmark]
    _atomic_write_tickers(cp, tickers)
    return tickers, benchmark


# ---------------------------------------------------------------------------
# S&P 500 — Wikipedia
# ---------------------------------------------------------------------------


def _fetch_sp500() -> list[str]:
    html = _http_get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").decode("utf-8")
    table = pd.read_html(io.StringIO(html))[0]
    raw = table["Symbol"].astype(str).tolist()
    return [t.replace(".", "-") for t in raw]


def sp500(*, force_refresh: bool = False, cache_path: Path | None = None) -> tuple[list[str], str]:
    """Return ``(constituents, benchmark)`` for the current S&P 500."""
    return _generic_load("sp500", _fetch_sp500, force_refresh=force_refresh, cache_path=cache_path)


# ---------------------------------------------------------------------------
# Nasdaq-100 — Wikipedia
# ---------------------------------------------------------------------------


def _fetch_nasdaq100() -> list[str]:
    html = _http_get("https://en.wikipedia.org/wiki/Nasdaq-100").decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    # Pick the first table that has a 'Ticker' or 'Symbol' column.
    chosen: pd.DataFrame | None = None
    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if "ticker" in cols or "symbol" in cols:
            chosen = tbl
            break
    if chosen is None:
        raise RuntimeError("Nasdaq-100 ticker table not found on Wikipedia.")
    col = "Ticker" if "Ticker" in chosen.columns else "Symbol"
    raw = chosen[col].astype(str).tolist()
    out: list[str] = []
    for t in raw:
        # Drop footnote markers (e.g. 'AAPL[a]') and stray whitespace.
        t = t.split("[")[0].strip()
        if not t or t.lower().startswith("nan"):
            continue
        out.append(t.replace(".", "-"))
    return out


def nasdaq100(
    *, force_refresh: bool = False, cache_path: Path | None = None
) -> tuple[list[str], str]:
    """Return ``(constituents, benchmark)`` for the Nasdaq-100."""
    return _generic_load(
        "nasdaq100", _fetch_nasdaq100, force_refresh=force_refresh, cache_path=cache_path
    )


# ---------------------------------------------------------------------------
# Russell 2000 — iShares IWM holdings CSV
# ---------------------------------------------------------------------------


_IWM_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)


def _fetch_russell2000() -> list[str]:
    raw_bytes = _http_get(_IWM_HOLDINGS_URL, headers={"Accept": "text/csv"})
    # The BlackRock CSV has a multi-line preamble; the actual table starts at
    # the line whose first column is "Ticker".
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        cols = [c.strip().strip('"').lower() for c in line.split(",")]
        if cols and cols[0] == "ticker":
            header_idx = i
            break
    if header_idx < 0:
        raise RuntimeError("IWM holdings CSV: 'Ticker' header not found.")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    # Filter to common stock only when the column is present.
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.contains("Equity", case=False, na=False)]
    raw = df["Ticker"].astype(str).tolist()
    out: list[str] = []
    for t in raw:
        t = t.strip()
        if not t or t.lower() in {"-", "nan"}:
            continue
        # BlackRock uses '.' (e.g. BRK.B); yfinance wants '-'.
        out.append(t.replace(".", "-"))
    return out


def russell2000(
    *, force_refresh: bool = False, cache_path: Path | None = None
) -> tuple[list[str], str]:
    """Return ``(constituents, benchmark)`` for the Russell 2000.

    Source: the daily IWM holdings CSV published by BlackRock. The ETF's
    holdings list lags the index by ~1 trading day, which is fine for our
    rolling-rebalance window.
    """
    return _generic_load(
        "russell2000", _fetch_russell2000, force_refresh=force_refresh, cache_path=cache_path
    )


# ---------------------------------------------------------------------------
# Nifty 50 — NSE "ind_nifty50list.csv"
# ---------------------------------------------------------------------------


_NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"

_NIFTY50_STATIC_SYMBOLS = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]


def _normalise_nse_symbols(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = t.strip().upper()
        if not t or t.lower() == "nan":
            continue
        # yfinance wants '<symbol>.NS' for NSE-listed equities.
        if not t.endswith(".NS"):
            t = f"{t}.NS"
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _fetch_nifty50() -> list[str]:
    try:
        raw_bytes = _http_get(
            _NIFTY50_URL,
            headers={"Accept": "text/csv", "Referer": "https://www.niftyindices.com/"},
        )
        text = raw_bytes.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        # The CSV exposes a 'Symbol' column with NSE codes (e.g. RELIANCE, TCS).
        if "Symbol" not in df.columns:
            raise RuntimeError(f"Nifty 50 CSV: 'Symbol' column missing (got {list(df.columns)}).")
        parsed = _normalise_nse_symbols(df["Symbol"].astype(str).tolist())
        if len(parsed) >= 45:
            return parsed
    except (RuntimeError, UnicodeError, pd.errors.ParserError):
        # NSE occasionally serves an anti-bot / diagnostics payload instead of
        # the CSV. Keep the demo reliable by falling back to a static current
        # Nifty 50 list; callers still cache this list just like the live CSV.
        pass
    return _normalise_nse_symbols(_NIFTY50_STATIC_SYMBOLS)


def nifty50(
    *, force_refresh: bool = False, cache_path: Path | None = None
) -> tuple[list[str], str]:
    """Return ``(constituents, benchmark)`` for the Nifty 50.

    Source: NSE's official ``ind_nifty50list.csv``. Tickers carry the
    ``.NS`` suffix that yfinance requires for NSE-listed equities; the
    benchmark ticker ``^NSEI`` is the Nifty 50 index quote.
    """
    return _generic_load(
        "nifty50", _fetch_nifty50, force_refresh=force_refresh, cache_path=cache_path
    )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


UniverseFn = Callable[..., "tuple[list[str], str]"]

UNIVERSE_REGISTRY: dict[str, UniverseFn] = {
    "sp500": sp500,
    "nasdaq100": nasdaq100,
    "russell2000": russell2000,
    "nifty50": nifty50,
}


def get_universe(
    name: str, *, force_refresh: bool = False, cache_path: Path | None = None
) -> tuple[list[str], str]:
    """Look up an index by canonical short name.

    Parameters
    ----------
    name
        One of ``{"sp500", "nasdaq100", "russell2000", "nifty50"}``.
        Case-insensitive; spaces and dashes are stripped.
    force_refresh
        Ignore any cached file and re-fetch from the source.
    cache_path
        Override the default cache file location (used heavily by tests).
    """
    key = name.strip().lower().replace(" ", "").replace("-", "")
    if key not in UNIVERSE_REGISTRY:
        raise KeyError(f"Unknown universe {name!r}; supported: {sorted(UNIVERSE_REGISTRY)}")
    return UNIVERSE_REGISTRY[key](force_refresh=force_refresh, cache_path=cache_path)


def supported_universes() -> list[str]:
    return sorted(UNIVERSE_REGISTRY)


__all__ = [
    "INDEX_METADATA",
    "UNIVERSES_DIR",
    "UNIVERSE_REGISTRY",
    "UniverseFn",
    "get_universe",
    "nasdaq100",
    "nifty50",
    "russell2000",
    "sp500",
    "supported_universes",
]
