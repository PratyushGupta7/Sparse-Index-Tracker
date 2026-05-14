"""Unit tests for ``sit.data.universes``.

Network is heavily monkey-patched here — every public loader is exercised
end-to-end via fake HTML / CSV fixtures so the test runs offline. A separate
``@pytest.mark.network`` test exercises the live S&P 500 fetch as a smoke
check; CI opts into it via ``pytest -m network``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sit.data import universes

# ---------------------------------------------------------------------------
# Fake source payloads (stable; written to mimic real HTML / CSV shapes)
# ---------------------------------------------------------------------------


_FAKE_SP500_HTML = """
<html><body>
<table>
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td></tr>
<tr><td>MSFT</td><td>Microsoft Corp.</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway B</td></tr>
<tr><td>SPY</td><td>SPDR S&amp;P 500 ETF</td></tr>
</table>
</body></html>
""".strip()


_FAKE_NDX_HTML = """
<html><body>
<table>
<tr><th>Company</th><th>Ticker</th></tr>
<tr><td>Apple</td><td>AAPL</td></tr>
<tr><td>Microsoft</td><td>MSFT[a]</td></tr>
<tr><td>NVIDIA</td><td>NVDA</td></tr>
<tr><td>Meta</td><td>META</td></tr>
</table>
</body></html>
""".strip()


_FAKE_IWM_CSV = """
"iShares Russell 2000 ETF (IWM) - Daily Holdings","",""
"As of 2026-05-09","",""
"","",""
Ticker,Name,Asset Class
ABC,Alpha Corp,Equity
DEF,Delta Industries,Equity
GHI,Gamma Holdings,Equity
JKL,Junk Bond Fund,Bond
MNO,Mu Group Plc,Equity
""".strip()


_FAKE_NIFTY_CSV = (
    "Company Name,Industry,Symbol,Series,ISIN Code\n"
    "Reliance Industries Ltd.,Refineries,RELIANCE,EQ,INE002A01018\n"
    "Tata Consultancy Services Ltd.,Computers - Software,TCS,EQ,INE467B01029\n"
    "HDFC Bank Ltd.,Banks,HDFCBANK,EQ,INE040A01034\n"
)


# ---------------------------------------------------------------------------
# Monkeypatched _http_get
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Replace ``_http_get`` with a route table keyed by URL substring."""
    routes: dict[str, bytes] = {
        "List_of_S%26P_500_companies": _FAKE_SP500_HTML.encode("utf-8"),
        "Nasdaq-100": _FAKE_NDX_HTML.encode("utf-8"),
        "IWM_holdings": _FAKE_IWM_CSV.encode("utf-8"),
        "ind_nifty50list.csv": _FAKE_NIFTY_CSV.encode("utf-8"),
    }

    def _fake_get(url: str, **kwargs: object) -> bytes:
        for needle, payload in routes.items():
            if needle in url:
                return payload
        raise RuntimeError(f"No fake registered for URL: {url}")

    monkeypatch.setattr(universes, "_http_get", _fake_get)
    return routes


@pytest.fixture
def tmp_universes_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the package-level cache dir to a tmp path."""
    target = tmp_path / "universes"
    target.mkdir()
    monkeypatch.setattr(universes, "UNIVERSES_DIR", target)
    return target


# ---------------------------------------------------------------------------
# Per-universe shape tests
# ---------------------------------------------------------------------------


def test_sp500_basic_shape(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    tickers, bench = universes.sp500(cache_path=cp)
    assert isinstance(tickers, list)
    assert len(tickers) >= 3
    assert bench == "SPY"
    assert bench not in tickers, "Benchmark must not appear in constituents."
    # Wikipedia uses BRK.B; we map dots to dashes for yfinance compatibility.
    assert "BRK-B" in tickers
    assert "BRK.B" not in tickers


def test_nasdaq100_basic_shape(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "ndx.csv"
    tickers, bench = universes.nasdaq100(cache_path=cp)
    assert bench == "QQQ"
    assert "AAPL" in tickers
    # Footnote markers are stripped.
    assert "MSFT" in tickers
    assert all("[" not in t for t in tickers)


def test_russell2000_filters_non_equity(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "iwm.csv"
    tickers, bench = universes.russell2000(cache_path=cp)
    assert bench == "IWM"
    assert "ABC" in tickers and "DEF" in tickers and "GHI" in tickers
    assert "JKL" not in tickers, "Bond rows must be filtered out."
    assert "MNO" in tickers
    assert bench not in tickers


def test_nifty50_appends_ns_suffix(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "nifty.csv"
    tickers, bench = universes.nifty50(cache_path=cp)
    assert bench == "^NSEI"
    assert all(t.endswith(".NS") for t in tickers)
    assert "RELIANCE.NS" in tickers
    assert "TCS.NS" in tickers
    assert "HDFCBANK.NS" in tickers


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cache_writes_csv_with_header(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    universes.sp500(cache_path=cp)
    assert cp.is_file()
    rows = list(csv.reader(cp.open()))
    assert rows[0] == ["ticker"]
    assert len(rows) >= 4  # header + 3 tickers


def test_cache_is_used_on_second_call(monkeypatch, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    # Pre-populate cache by hand.
    cp.write_text("ticker\nFOO\nBAR\nBAZ\n")

    # If the loader hits the network, _http_get should fail loudly.
    def _fail(*_a, **_kw):  # pragma: no cover (must not be called)
        raise AssertionError("Network was called even though cache exists.")

    monkeypatch.setattr(universes, "_http_get", _fail)
    tickers, bench = universes.sp500(cache_path=cp)
    assert tickers == ["FOO", "BAR", "BAZ"]
    assert bench == "SPY"


def test_force_refresh_overwrites_cache(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    cp.write_text("ticker\nSTALE\n")
    tickers, _ = universes.sp500(cache_path=cp, force_refresh=True)
    assert "STALE" not in tickers
    assert "AAPL" in tickers


def test_cache_excludes_benchmark_even_if_present(monkeypatch, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    cp.write_text("ticker\nAAPL\nMSFT\nSPY\n")  # benchmark sneaked in
    # No network needed because cache exists.
    monkeypatch.setattr(
        universes, "_http_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    tickers, bench = universes.sp500(cache_path=cp)
    assert bench == "SPY"
    assert "SPY" not in tickers
    assert "AAPL" in tickers and "MSFT" in tickers


def test_cache_is_atomic_write(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    universes.sp500(cache_path=cp)
    # No leftover .tmp file should remain.
    assert not cp.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_bench"),
    [
        ("sp500", "SPY"),
        ("nasdaq100", "QQQ"),
        ("russell2000", "IWM"),
        ("nifty50", "^NSEI"),
    ],
)
def test_get_universe_dispatches(name: str, expected_bench: str, mock_http, tmp_path: Path) -> None:
    cp = tmp_path / f"{name}.csv"
    tickers, bench = universes.get_universe(name, cache_path=cp)
    assert bench == expected_bench
    assert tickers, f"{name} returned no tickers"
    assert bench not in tickers


def test_get_universe_normalises_name(mock_http, tmp_path: Path) -> None:
    cp = tmp_path / "sp500.csv"
    a = universes.get_universe("sp500", cache_path=cp)
    b = universes.get_universe("SP500", cache_path=cp)
    c = universes.get_universe("  sp-500 ", cache_path=cp)
    assert a == b == c


def test_get_universe_unknown_raises() -> None:
    with pytest.raises(KeyError):
        universes.get_universe("dax40")


def test_supported_universes_lists_all_four() -> None:
    assert set(universes.supported_universes()) == {
        "sp500",
        "nasdaq100",
        "russell2000",
        "nifty50",
    }


def test_index_metadata_consistency() -> None:
    for key, meta in universes.INDEX_METADATA.items():
        assert "benchmark" in meta and "label" in meta and "region" in meta
        assert isinstance(meta["benchmark"], str)


# ---------------------------------------------------------------------------
# Smoke: parse function eats real-shape DataFrame schemas
# ---------------------------------------------------------------------------


def test_nasdaq100_parser_handles_alternate_column_name(monkeypatch, tmp_path: Path) -> None:
    """Some historical revisions of the Wikipedia table use 'Symbol' instead."""
    html = """
    <table>
    <tr><th>Symbol</th><th>Company</th></tr>
    <tr><td>AAPL</td><td>Apple</td></tr>
    <tr><td>MSFT</td><td>Microsoft</td></tr>
    </table>
    """
    monkeypatch.setattr(universes, "_http_get", lambda *a, **k: html.encode("utf-8"))
    cp = tmp_path / "ndx.csv"
    tickers, bench = universes.nasdaq100(cache_path=cp)
    assert bench == "QQQ"
    assert "AAPL" in tickers and "MSFT" in tickers


def test_russell2000_parser_skips_preamble(monkeypatch, tmp_path: Path) -> None:
    raw = (
        "row1,foo,bar\n"
        "row2,baz,qux\n"
        "Ticker,Name,Asset Class\n"
        "ZZZ,Zeta Inc,Equity\n"
        "WWW,Omega Corp,Equity\n"
    )
    monkeypatch.setattr(universes, "_http_get", lambda *a, **k: raw.encode("utf-8"))
    cp = tmp_path / "iwm.csv"
    tickers, _ = universes.russell2000(cache_path=cp)
    assert tickers == ["ZZZ", "WWW"]


def test_nifty50_does_not_double_suffix(monkeypatch, tmp_path: Path) -> None:
    raw = "Symbol\nRELIANCE\nTCS.NS\n"
    monkeypatch.setattr(universes, "_http_get", lambda *a, **k: raw.encode("utf-8"))
    cp = tmp_path / "nifty.csv"
    tickers, _ = universes.nifty50(cache_path=cp)
    assert "RELIANCE.NS" in tickers
    assert "TCS.NS" in tickers
    # Make sure we didn't get TCS.NS.NS.
    assert all(t.count(".NS") == 1 for t in tickers)


def test_nifty50_falls_back_when_nse_serves_non_csv(monkeypatch, tmp_path: Path) -> None:
    raw = "Access Denied\nDiagnostics,not,a,constituent,csv\n"
    monkeypatch.setattr(universes, "_http_get", lambda *a, **k: raw.encode("utf-8"))
    cp = tmp_path / "nifty.csv"
    tickers, bench = universes.nifty50(cache_path=cp)
    assert bench == "^NSEI"
    assert len(tickers) >= 45
    assert "RELIANCE.NS" in tickers
    assert "TCS.NS" in tickers
    assert cp.exists()


# ---------------------------------------------------------------------------
# Real-network smoke (opt-in via -m network)
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_sp500_real_network_smoke(tmp_path: Path) -> None:  # pragma: no cover (network)
    cp = tmp_path / "sp500_live.csv"
    tickers, bench = universes.sp500(cache_path=cp, force_refresh=True)
    assert bench == "SPY"
    assert 480 <= len(tickers) <= 520
