"""sit.api — FastAPI application for the Sparse Index Tracker.

Phase 5 refactor of the legacy ``app.py`` monolith into a modular package
with typed Pydantic v2 schemas, Redis-backed pricing cache, slowapi rate
limiting, and a versioned ``/api/v1/...`` surface.
"""

from sit.api.main import app

__all__ = ["app"]
