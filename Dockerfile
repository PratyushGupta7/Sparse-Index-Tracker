# =============================================================================
# Sparse Index Tracker — Dockerfile
# Phase 4: MLOps & API Engineering
# =============================================================================
# Builds a lightweight container running the FastAPI investment API.
# Usage:
#   docker build -t sparse-tracker .
#   docker run -p 8000:8000 sparse-tracker
#   curl http://localhost:8000/invest?capital=100000
# =============================================================================

# =============================================================================
# Sparse Index Tracker — multi-stage Dockerfile
# Phase 0: Python 3.11 base, runtime-only image, ~350 MB target.
# Phase 5+ will add Redis side-car via docker-compose.
# =============================================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a dedicated prefix so we can copy them cleanly.
COPY requirements.txt pyproject.toml ./
COPY src/ src/
RUN pip install --prefix=/install -r requirements.txt
RUN pip install --prefix=/install --no-deps .

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# Copy installed site-packages from the builder.
COPY --from=builder /install /usr/local

# Copy app code and bundled data artefacts.
COPY app.py ./
COPY src/ src/
COPY data/ data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
