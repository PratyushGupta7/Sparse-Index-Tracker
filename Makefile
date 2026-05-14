# =============================================================================
# Sparse Index Tracker — developer convenience targets
# =============================================================================
# Always run inside the project venv. Most targets are idempotent.
# =============================================================================

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest

# MOSEK academic licence path (used by Phase 2 MIQP baseline and CVXPY benchmark)
export MOSEKLM_LICENSE_FILE := $(HOME)/mosek/mosek.lic

.PHONY: help venv install install-dev lint format type test test-fast \
        test-network benchmark clean clean-data run-api ingest backtest \
        regimes solver docker-build docker-up docker-down docker-smoke verify

help:
	@echo "Sparse Index Tracker — make targets"
	@echo "  make venv         Create .venv on Python 3.11"
	@echo "  make install      Install runtime dependencies"
	@echo "  make install-dev  Install dev tooling (ruff, mypy, pytest, ...)"
	@echo "  make lint         Run ruff + black --check"
	@echo "  make format       Auto-format with ruff format + black"
	@echo "  make type         Run mypy"
	@echo "  make test         Run full pytest suite with coverage"
	@echo "  make test-fast    Run tests excluding 'slow' and 'network'"
	@echo "  make benchmark    Run ADMM vs CVXPY benchmark script"
	@echo "  make solver       Smoke-test ADMM on the pickled S&P 500 data"
	@echo "  make ingest       Run the data pipeline"
	@echo "  make backtest     Run the legacy phase-3 train/test backtest"
	@echo "  make regimes      Run the 8-regime stress test"
	@echo "  make run-api      Launch FastAPI on :8000"
	@echo "  make docker-build Build the Phase 6 API Docker image"
	@echo "  make docker-up    Run API + Redis via docker compose"
	@echo "  make docker-down  Stop docker compose stack"
	@echo "  make docker-smoke Smoke-test docker API health endpoint"
	@echo "  make verify       End-to-end sanity check"

venv:
	/opt/homebrew/bin/python3.11 -m venv .venv

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	$(PYTHON) -m pre_commit install || true

lint:
	.venv/bin/ruff check src tests benchmarks
	.venv/bin/black --check src tests benchmarks

format:
	.venv/bin/ruff check --fix src tests benchmarks
	.venv/bin/ruff format src tests benchmarks
	.venv/bin/black src tests benchmarks

type:
	.venv/bin/mypy src

test:
	$(PYTEST)

test-fast:
	$(PYTEST) -m "not slow and not network and not mosek"

test-network:
	$(PYTEST) -m "network" --no-cov

benchmark:
	$(PYTHON) -m benchmarks.cvxpy_benchmark

solver:
	$(PYTHON) -m sit.solvers.admm

ingest:
	$(PYTHON) -m sit.data.loader

backtest:
	$(PYTHON) -m sit.backtest.phase3_validator

regimes:
	$(PYTHON) -m sit.regimes.tester

run-api:
	.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f deploy/Dockerfile -t sparse-tracker-api:local .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-smoke:
	python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=10).read().decode()[:500])"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data:
	rm -rf data/cache data/_tmp

verify: lint type test
	@echo "✅ verify OK"
