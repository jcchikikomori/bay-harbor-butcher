.PHONY: help lint format test coverage clean

help:
	@echo "Available commands:"
	@echo "  make lint      - Run ruff linter"
	@echo "  make format    - Auto-format code with ruff"
	@echo "  make test      - Run pytest"
	@echo "  make coverage  - Run tests with coverage report"
	@echo "  make clean     - Remove build artifacts and cache"
	@echo "  make dev-setup - Install dev dependencies"

dev-setup:
	pip install -r requirements-dev.txt

lint:
	ruff check app.py utils/ pipelines/ tests/

format:
	ruff format app.py utils/ pipelines/ tests/
	ruff check app.py utils/ pipelines/ tests/ --fix

test:
	pytest

coverage:
	pytest --cov=. --cov-report=html --cov-report=term-missing
	@echo "HTML coverage report generated in htmlcov/index.html"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf *.egg-info dist build
