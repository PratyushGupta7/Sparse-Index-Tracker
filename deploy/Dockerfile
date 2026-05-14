# syntax=docker/dockerfile:1.7

# =============================================================================
# Sparse Index Tracker — Azure Container Apps Dockerfile
# =============================================================================
# Builds the Phase 6 production API image for FastAPI.
#
# Local smoke:
#   docker build -f deploy/Dockerfile -t sparse-tracker-api:local .
#   docker run --rm -p 8000:8000 sparse-tracker-api:local
#   curl http://localhost:8000/api/v1/health
# =============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --prefix=/install -r requirements.txt \
    && pip install --prefix=/install --no-deps .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    SIT_ENV=prod \
    SIT_DATA_DIR=/app/data \
    SIT_BENCHMARKS_DIR=/app/benchmarks/_results \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

COPY --from=builder /install /usr/local

# Runtime code + artefacts needed by the API endpoints.
COPY app.py ./
COPY src/ src/
COPY data/ data/
COPY benchmarks/_results/ benchmarks/_results/

RUN mkdir -p /app/data/lambda_paths \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=5)" || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
