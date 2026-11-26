.PHONY: install test lint typecheck audit reproduce demo docker-up all clean

install:
	python -m pip install -e ".[dev,experiments,api,dashboard]"

lint:
	python -m ruff check src tests dashboard experiments scripts

typecheck:
	python -m mypy src dashboard

test:
	python -m pytest tests/ --tb=short -q --cov=probcount --cov-report=term-missing -m "not slow"

audit:
	python -m bandit -r src dashboard -ll
	python -m pip_audit

reproduce:
	python -m experiments.run_all --config experiments/config.yaml --quick
	python scripts/verify_report_numbers.py

demo:
	python scripts/run_demo.py

docker-up:
	docker compose up --build

all: install lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
