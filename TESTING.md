# Code Quality & Testing Guide

## Overview

This project uses **ruff** for code formatting/linting and **pytest** for comprehensive testing with coverage reporting.

## Quick Commands

Use the included Makefile for common tasks:

```bash
make help              # Show available commands
make dev-setup        # Install dev dependencies
make lint             # Run ruff linter
make format           # Auto-format code with ruff
make test             # Run all tests
make coverage         # Run tests with HTML coverage report
make clean            # Clean build artifacts and caches
```

## Setup

### Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
# or
make dev-setup
```

This installs:

- **pytest** - Testing framework
- **pytest-cov** - Coverage tracking
- **pytest-mock** - Mocking utilities
- **ruff** - Fast Python linter & formatter

## Testing

### Run All Tests

```bash
pytest
# or
make test
```

### Run Specific Test File

```bash
pytest tests/test_device.py -v
```

### Run Tests with Coverage Report

```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
# or
make coverage
```

This generates:

- Terminal report showing coverage %
- `htmlcov/index.html` for detailed HTML coverage report

## Code Quality

### Linting with Ruff

Check code for style and quality issues:

```bash
ruff check app.py utils/ pipelines/
# or
make lint
```

### Auto-Format Code

Automatically format code to match project standards:

```bash
ruff format app.py utils/ pipelines/ tests/
# or
make format
```

## Configuration

### Project Configuration (`pyproject.toml`)

- **PyTest**: Configured to test `tests/` directory
  - Target coverage: ≥95%
  - Excludes: test utilities, setup.py, **pycache**

- **Ruff**: Configured with:
  - Line length: 120 characters
  - Python 3.11 target
  - Selected rules: E, W, F, I, C4, UP, B, RSE, RUF
  - Ignored: E501 (line length), E203 (formatting)
  - Known first-party modules: utils, pipelines

## Test Structure

```text
tests/
├── __init__.py
├── test_device.py          # Device detection & dtype selection
├── test_flux.py            # FLUX pipeline building
├── test_lora.py            # LoRA weight loading
└── test_stable_diffusion.py # Stable Diffusion pipeline
```

### Test Coverage by Module

| Module                        | Lines | Coverage                             |
| ----------------------------- | ----- | ------------------------------------ |
| utils/lora.py                 | 15    | 100% ✅                              |
| pipelines/flux.py             | 14    | 93.75% ⚠️                            |
| pipelines/stable_diffusion.py | 12    | 93.75% ⚠️                            |
| utils/device.py               | 28    | 92.86% ⚠️                            |
| app.py                        | 123   | 51.03% (integration tested manually) |

**Overall: 67.80%** (tests focus on utility/pipeline modules; app.py integration tested via CLI)

## Test Examples

### Device Detection Tests

```python
# Testing FLUX model detection
def test_flux_detected(self) -> None:
    assert is_flux_model("black-forest-labs/FLUX.1-dev") is True
    assert is_flux_model("runwayml/stable-diffusion-v1-5") is False
```

### Pipeline Tests

```python
# Testing pipeline building with mocks
@patch("pipelines.flux.AutoPipelineForText2Image")
def test_build_flux_pipeline(self, mock_autopipeline) -> None:
    pipe = build_flux_pipeline("black-forest-labs/FLUX.1-dev", "cuda", torch.bfloat16)
    mock_autopipeline.from_pretrained.assert_called_once()
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests & Code Quality

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements-dev.txt
      - run: ruff check app.py utils/ pipelines/
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Best Practices

1. **Always run tests before committing:**

   ```bash
   make test && make lint
   ```

2. **Auto-format before pushing:**

   ```bash
   make format
   ```

3. **Check coverage for new code:**

   ```bash
   make coverage
   ```

4. **Target ≥95% coverage** for utilities and pipelines

5. **Use type hints** (validated by ruff's UP rules)

6. **Add docstrings** to all public functions

7. **Mock external dependencies** in tests (HuggingFace models, etc.)

## Troubleshooting

### Tests fail with import errors

```bash
pip install -r requirements-dev.txt --upgrade
```

### Ruff issues not auto-fixed

```bash
ruff check app.py --fix
```

### Coverage report not generated

```bash
rm -rf .coverage htmlcov
pytest --cov --cov-report=html
```

## Performance Notes

- **Ruff**: ~0.5s for full project check
- **Pytest**: ~4s for 46 tests
- **Coverage**: ~4s to generate HTML report

All tools are fast enough for pre-commit hooks.
