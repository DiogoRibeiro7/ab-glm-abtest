# CI/CD Guide for ab-glm-abtest

This guide explains the continuous integration and deployment setup for the ab-glm-abtest project.

## Table of Contents
- [Overview](#overview)
- [GitHub Actions Workflows](#github-actions-workflows)
- [Local Development](#local-development)
- [Code Quality Tools](#code-quality-tools)
- [Release Process](#release-process)
- [Docker Deployment](#docker-deployment)
- [Monitoring and Badges](#monitoring-and-badges)

## Overview

The project uses a comprehensive CI/CD pipeline to ensure code quality, security, and reliable deployments:

- **Continuous Integration**: Automated testing on every push and PR
- **Code Quality**: Linting, formatting, type checking, and security scanning
- **Coverage Reporting**: Automated coverage tracking with Codecov
- **Dependency Management**: Automated updates with Renovate
- **Release Automation**: Tag-based releases to PyPI and Docker Hub

## GitHub Actions Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

Runs on every push to main/develop and on all PRs.

**Jobs:**
- **test**: Runs pytest on multiple OS and Python versions
- **lint**: Checks code style with ruff
- **type-check**: Static type checking with mypy
- **security**: Scans for vulnerabilities with bandit and safety
- **docs**: Verifies documentation builds
- **benchmark**: Performance testing on PRs

**Matrix Testing:**
```yaml
os: [ubuntu-latest, windows-latest, macos-latest]
python-version: ['3.10', '3.11', '3.12']
```

### 2. Release Workflow (`.github/workflows/release.yml`)

Triggered by version tags (e.g., `v0.2.0`).

**Steps:**
1. Run full test suite
2. Build Python package
3. Create GitHub release
4. Publish to PyPI
5. Build and push Docker image

### 3. Monthly Reminder Workflow

Sends monthly maintenance reminders for:
- Dependency updates
- Security patches
- Documentation reviews

## Local Development

### Quick Start
```bash
# Install dependencies and pre-commit hooks
make install

# Run all checks locally
make ci-local

# Quick development cycle
make dev
```

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install dependencies and pre-commit hooks |
| `make test` | Run tests with coverage |
| `make test-fast` | Quick test run without coverage |
| `make lint` | Check code style |
| `make format` | Auto-format code |
| `make type` | Run type checking |
| `make security` | Run security scans |
| `make clean` | Remove build artifacts |
| `make docs` | Build documentation |
| `make benchmark` | Run performance tests |
| `make ci-local` | Run full CI pipeline locally |

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

**Configured hooks:**
- Trailing whitespace removal
- End-of-file fixing
- YAML/JSON/TOML validation
- Python formatting (ruff)
- Type checking (mypy)
- Security scanning (bandit)
- Import sorting (isort)
- Python upgrade syntax (pyupgrade)

## Code Quality Tools

### 1. Ruff
Fast Python linter and formatter.

```bash
# Check code
poetry run ruff check src/ tests/

# Format code
poetry run ruff format src/ tests/

# Fix issues automatically
poetry run ruff check --fix src/
```

Configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I"]
```

### 2. MyPy
Static type checker for Python.

```bash
# Type check
poetry run mypy src/ab_glm

# Ignore missing imports
poetry run mypy src/ --ignore-missing-imports
```

Configuration in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.10"
strict_optional = true
check_untyped_defs = true
```

### 3. Pytest
Testing framework with coverage.

```bash
# Run with coverage
poetry run pytest --cov=src/ab_glm --cov-report=html

# Run specific tests
poetry run pytest tests/test_pipeline.py -v

# Run with markers
poetry run pytest -m "not slow"
```

### 4. Bandit
Security vulnerability scanner.

```bash
# Scan for security issues
bandit -r src/ -ll

# Generate JSON report
bandit -r src/ -f json -o security-report.json
```

### 5. Safety
Checks dependencies for known vulnerabilities.

```bash
# Check current dependencies
poetry export -f requirements.txt | safety check --stdin

# Check with detailed output
safety check --json
```

## Release Process

### 1. Version Bump
Update version in `pyproject.toml`:
```toml
[tool.poetry]
version = "0.2.0"
```

Update `__version__` in `src/ab_glm/__init__.py`:
```python
__version__ = "0.2.0"
```

### 2. Update CHANGELOG
Add release notes to `CHANGELOG.md`:
```markdown
## [0.2.0] - 2024-01-30
### Added
- New feature X
### Fixed
- Bug fix Y
```

### 3. Create Release
```bash
# Commit changes
git add .
git commit -m "chore: release v0.2.0"

# Create and push tag
git tag v0.2.0
git push origin main --tags
```

The release workflow will automatically:
1. Run tests
2. Build package
3. Create GitHub release
4. Publish to PyPI
5. Build Docker image

### 4. Manual Release (if needed)
```bash
# Build package
poetry build

# Test upload to TestPyPI
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi

# Upload to PyPI
poetry publish
```

## Docker Deployment

### Building Images

```bash
# Build image locally
docker build -t ab-glm-abtest:latest .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t ab-glm-abtest:latest .
```

### Running Containers

```bash
# Run interactive Python
docker run -it ab-glm-abtest python

# Run Jupyter notebook
docker run -p 8888:8888 ab-glm-abtest \
  jupyter notebook --ip=0.0.0.0 --no-browser

# Run analysis script
docker run -v $(pwd)/data:/app/data ab-glm-abtest \
  python scripts/analyze_experiment.py

# Run with custom command
docker run ab-glm-abtest python -c "from ab_glm import simulate_ab_data; print(simulate_ab_data(100))"
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  ab-glm:
    image: ab-glm-abtest:latest
    volumes:
      - ./data:/app/data
      - ./notebooks:/app/notebooks
      - ./output:/app/output
    ports:
      - "8888:8888"
    command: jupyter notebook --ip=0.0.0.0 --no-browser
    environment:
      - PYTHONUNBUFFERED=1
```

## Monitoring and Badges

### Status Badges

Add these badges to your README:

```markdown
[![CI](https://github.com/DiogoRibeiro7/ab-glm-abtest/workflows/CI/badge.svg)](https://github.com/DiogoRibeiro7/ab-glm-abtest/actions)
[![codecov](https://codecov.io/gh/DiogoRibeiro7/ab-glm-abtest/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/ab-glm-abtest)
[![PyPI version](https://badge.fury.io/py/ab-glm-abtest.svg)](https://badge.fury.io/py/ab-glm-abtest)
[![Python versions](https://img.shields.io/pypi/pyversions/ab-glm-abtest.svg)](https://pypi.org/project/ab-glm-abtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

### Coverage Reports

Coverage is automatically reported to Codecov on successful CI runs.

View reports at: `https://codecov.io/gh/DiogoRibeiro7/ab-glm-abtest`

### Dependency Updates

Renovate bot automatically creates PRs for dependency updates:
- Major updates: Manual review required
- Minor/patch updates: Auto-merged after tests pass
- Security updates: High priority, assigned for immediate review

### Performance Monitoring

Performance benchmarks run on PRs and post results as comments:
- Data generation speed
- Model fitting time
- Memory usage
- Scalability metrics

## Troubleshooting

### Common Issues

**1. Pre-commit hooks failing**
```bash
# Skip hooks temporarily
git commit --no-verify -m "message"

# Fix issues and amend
make format
git add .
git commit --amend
```

**2. Coverage below threshold**
```bash
# Find uncovered lines
poetry run pytest --cov=src/ab_glm --cov-report=term-missing

# Generate HTML report
poetry run pytest --cov=src/ab_glm --cov-report=html
# Open htmlcov/index.html in browser
```

**3. Type checking errors**
```bash
# Check specific file
poetry run mypy src/ab_glm/pipeline.py

# Reveal type of variable
# Add in code: reveal_type(variable)
poetry run mypy src/
```

**4. Docker build failures**
```bash
# Build with no cache
docker build --no-cache -t ab-glm-abtest .

# Debug build
docker build --progress=plain -t ab-glm-abtest .

# Check intermediate stages
docker build --target builder -t ab-glm-builder .
```

### Getting Help

- Check [GitHub Issues](https://github.com/DiogoRibeiro7/ab-glm-abtest/issues)
- Review [CI logs](https://github.com/DiogoRibeiro7/ab-glm-abtest/actions)
- Consult [documentation](https://ab-glm-abtest.readthedocs.io/)
- Contact maintainers