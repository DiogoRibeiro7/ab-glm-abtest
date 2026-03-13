# Makefile for ab-glm-abtest development

.PHONY: help install test test-fast test-cov test-verbose lint format type security clean docs benchmark demo all

help:  ## Show this help message
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install dependencies with Poetry
	poetry install --with dev
	@echo "Installing pre-commit hooks..."
	@which pre-commit > /dev/null 2>&1 && pre-commit install || echo "pre-commit not installed, skipping hooks"

test:  ## Run tests with coverage (quiet)
	poetry run pytest -q --cov=src/ab_glm --cov-report=term --cov-fail-under=80

test-fast:  ## Run tests without coverage (faster)
	poetry run pytest tests/ -v

test-cov:  ## Run tests with detailed coverage report
	poetry run pytest --cov=src/ab_glm --cov-report=term-missing --cov-report=html --cov-fail-under=80
	@echo "Coverage report generated in htmlcov/index.html"

test-verbose:  ## Run tests in verbose mode
	poetry run pytest -v --cov=src/ab_glm --cov-report=term-missing --cov-fail-under=80

lint:  ## Run linters (ruff)
	poetry run ruff check src/ tests/ scripts/
	poetry run ruff format --check src/ tests/

format:  ## Format code with ruff
	poetry run ruff check --fix src/ tests/ scripts/
	poetry run ruff format src/ tests/ scripts/

type:  ## Run type checking with mypy
	poetry run mypy src/ab_glm --ignore-missing-imports

security:  ## Run security checks
	@echo "Running bandit security check..."
	@which bandit > /dev/null 2>&1 && bandit -r src/ -ll || pip install bandit && bandit -r src/ -ll
	@echo "Checking for known vulnerabilities..."
	@poetry export -f requirements.txt --without-hashes | safety check --stdin || echo "Safety check completed"

clean:  ## Clean build artifacts and cache
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ *.egg-info 2>/dev/null || true
	rm -rf docs/_build/ 2>/dev/null || true
	@echo "Cleaned all build artifacts and cache"

docs:  ## Build documentation
	@if [ -d "docs" ]; then \
		cd docs && make clean && make html; \
		echo "Documentation built in docs/_build/html/"; \
	else \
		echo "No docs directory found"; \
	fi

benchmark:  ## Run performance benchmarks
	poetry run python -m ab_glm.performance

demo:  ## Run the demo script
	poetry run python scripts/run_demo.py

notebook:  ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook server..."
	@which jupyter > /dev/null 2>&1 && poetry run jupyter notebook notebooks/ || echo "Jupyter not installed"

pre-commit:  ## Run pre-commit hooks on all files
	@which pre-commit > /dev/null 2>&1 && pre-commit run --all-files || echo "pre-commit not installed"

update:  ## Update dependencies
	poetry update
	@which pre-commit > /dev/null 2>&1 && pre-commit autoupdate || echo "pre-commit not installed"

ci-local:  ## Run full CI pipeline locally
	@echo "===================="
	@echo "Running CI pipeline locally..."
	@echo "===================="
	@make clean
	@make install
	@echo "\n--- Running formatters ---"
	@make format
	@echo "\n--- Running linters ---"
	@make lint
	@echo "\n--- Running type checks ---"
	@make type
	@echo "\n--- Running security checks ---"
	@make security || true
	@echo "\n--- Running tests ---"
	@make test-cov
	@echo "\n===================="
	@echo "CI pipeline completed!"
	@echo "===================="

# Development shortcuts
dev:  ## Quick development check (format + fast test)
	@make format
	@make test-fast

watch:  ## Watch for changes and run tests (requires pytest-watch)
	@which ptw > /dev/null 2>&1 && poetry run ptw tests/ -- -v || echo "pytest-watch not installed"

all: clean install format lint type security test-cov  ## Run all checks

.DEFAULT_GOAL := help
