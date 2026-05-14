"""Portfolio endpoint — raw weight breakdown of the pre-baked S&P 500 sparse portfolio."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from sit.api.deps import DEFAULT_INDEX, validate_index
from sit.api.schemas import PortfolioResponse, PortfolioWeight
from sit.data.universes import INDEX_METADATA, supported_universes

router = APIRouter(tags=["portfolio"])


@router.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Sparse portfolio weights",
)
def portfolio(
    request: Request,
    index: str = Query(
        DEFAULT_INDEX,
        description="Index to track (sp500/nasdaq100/russell2000/nifty50).",
        examples=["sp500"],
    ),
) -> PortfolioResponse:
    name = validate_index(index)
    bundle = getattr(request.app.state, "weights_bundle", {}) or {}
    if name != DEFAULT_INDEX:
        meta = INDEX_METADATA[name]
        return PortfolioResponse(
            index=name,
            label=meta["label"],
            benchmark=meta["benchmark"],
            note=(
                f"Pre-baked weights are not yet available for {meta['label']}. "
                f"Use /api/v1/invest_live?capital=...&index={name} for a live solve."
            ),
            supported_indices=sorted(supported_universes()),
        )
    active = bundle.get("active_stocks", {}) or {}
    n_active = int(bundle.get("n_active", 0))
    n_total = int(bundle.get("n_total", 0))
    sorted_stocks = sorted(active.items(), key=lambda x: -x[1])
    return PortfolioResponse(
        index=DEFAULT_INDEX,
        label=INDEX_METADATA[DEFAULT_INDEX]["label"],
        benchmark=INDEX_METADATA[DEFAULT_INDEX]["benchmark"],
        portfolio_name=f"Sparse S&P 500 Tracker ({n_active} stocks)",
        active_stocks=n_active,
        total_universe=n_total,
        reduction_pct=round((1 - n_active / n_total) * 100, 1) if n_total else 0.0,
        weights=[
            PortfolioWeight(
                rank=i + 1,
                ticker=ticker,
                weight=round(weight, 6),
                pct=f"{weight * 100:.2f}%",
            )
            for i, (ticker, weight) in enumerate(sorted_stocks)
        ],
    )
