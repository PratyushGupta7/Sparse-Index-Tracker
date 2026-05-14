"""Legacy entry-point — re-exports the Phase 5 FastAPI ``app``.

Kept so that the existing ``Dockerfile`` (which calls
``uvicorn app:app --host 0.0.0.0 --port 8000``) and any external smoke
scripts continue to work unchanged. All real logic lives in
``sit.api.main``.
"""

from sit.api.main import app

__all__ = ["app"]
