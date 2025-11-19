.PHONY: help install install-dev sync test lint format clean docs

help:
	@echo "NeMo Fusion - Makefile Commands"
	@echo "================================"
	@echo "install          - Install package with uv"
	@echo "install-dev      - Install package with dev dependencies"
	@echo "sync             - Sync dependencies with uv"
	@echo "test             - Run tests with pytest"
	@echo "lint             - Run linters (ruff, black check, isort check)"
	@echo "format           - Format code with black and isort"
	@echo "clean            - Remove build artifacts and cache"
	@echo "docs             - Build documentation"
	@echo "benchmark        - Run benchmarks"

install:
	uv pip install -e .

install-dev:
	uv sync --all-extras

sync:
	uv sync

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=nemo_fusion --cov-report=html --cov-report=term-missing

lint:
	uv run ruff check nemo_fusion/
	uv run black --check nemo_fusion/
	uv run isort --check-only nemo_fusion/

format:
	uv run black nemo_fusion/ tests/ examples/
	uv run isort nemo_fusion/ tests/ examples/
	uv run ruff check --fix nemo_fusion/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && uv run make html

benchmark:
	uv run python benchmarks/parallelism_comparison.py

