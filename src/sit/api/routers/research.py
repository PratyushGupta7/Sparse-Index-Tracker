"""Research endpoints — λ-path slider data + 8-regime stress test results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from sit.api.deps import DEFAULT_INDEX, validate_index
from sit.api.schemas import LambdaPathPoint, LambdaPathResponse, RegimeResult, RegimesResponse
from sit.api.services import artefacts, lambda_path

router = APIRouter(tags=["research"])


@router.get(
    "/lambda-path",
    response_model=LambdaPathResponse,
    summary="Regularization path for the interactive λ slider",
)
def lambda_path_endpoint(
    index: str = Query(DEFAULT_INDEX, examples=["sp500"]),
) -> LambdaPathResponse:
    name = validate_index(index)
    payload = lambda_path.compute_path(name, prefer_real=False)
    return LambdaPathResponse(
        index=payload["index"],
        n_train=payload["n_train"],
        n_test=payload["n_test"],
        universe_size=payload["universe_size"],
        points=[LambdaPathPoint(**p) for p in payload["points"]],
        cached=bool(payload.get("cached", False)),
    )


@router.get(
    "/regimes",
    response_model=RegimesResponse,
    summary="8-regime stress test summary",
)
def regimes() -> RegimesResponse:
    try:
        raw = artefacts.load_regime_results()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Regime test results not generated yet.") from exc
    parsed: dict[str, RegimeResult] = {}
    for k, v in raw.items():
        try:
            parsed[k] = RegimeResult(**v)
        except Exception:
            continue
    return RegimesResponse(regimes=parsed, n_regimes=len(parsed), cached=True)
