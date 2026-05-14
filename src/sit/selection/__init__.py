"""λ selection via information criteria (BIC / EBIC) and cross-validation."""

from __future__ import annotations

from sit.selection.bic import (
    LambdaSelectionResult,
    ebic_lambda_selection,
    select_lambda_bic,
)

__all__ = [
    "LambdaSelectionResult",
    "ebic_lambda_selection",
    "select_lambda_bic",
]
