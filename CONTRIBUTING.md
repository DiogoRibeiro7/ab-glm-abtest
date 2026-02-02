# Contributing to ab-glm-abtest

Thank you for your interest in contributing to ab-glm-abtest! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept responsibility and apologize for mistakes
- Prioritize the community's best interests

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Any conduct that could be considered inappropriate in a professional setting

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ab-glm-abtest.git
   cd ab-glm-abtest
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/DiogoRibeiro7/ab-glm-abtest.git
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**Bug reports should include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, package versions)
- Minimal code example that reproduces the issue

**Example bug report:**
```markdown
### Description
fit_binomial_glm raises LinAlgError when all treatment users convert

### Steps to Reproduce
```python
import pandas as pd
from ab_glm import fit_binomial_glm

df = pd.DataFrame({
    'user_id': [1, 2, 3, 4],
    'T': [0, 0, 1, 1],
    'y': [0, 0, 1, 1],
    # ... other required columns
})
fit_binomial_glm(df)  # Raises LinAlgError
```

### Expected
Should handle perfect separation gracefully

### System
- OS: Windows 10
- Python: 3.10.2
- ab-glm-abtest: 0.1.0
- numpy: 2.0.1
```

### Suggesting Enhancements

**Enhancement proposals should include:**
- Use case and motivation
- Detailed description of the proposed solution
- Alternative solutions considered
- Potential drawbacks or breaking changes

### Contributing Code

1. **Find an issue** labeled "good first issue" or "help wanted"
2. **Comment on the issue** to claim it
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following our coding standards
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request**

## Development Setup

### Prerequisites
- Python 3.10+
- Poetry 1.8.3+
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/ab-glm-abtest.git
cd ab-glm-abtest

# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Running Tests
```bash
# Run all tests with coverage
poetry run pytest --cov=ab_glm --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_pipeline.py

# Run with verbose output
poetry run pytest -v

# Run only fast tests (if marked)
poetry run pytest -m "not slow"
```

### Code Quality Checks
```bash
# Run linting
poetry run ruff check src tests

# Fix auto-fixable issues
poetry run ruff check src tests --fix

# Run type checking
poetry run mypy src

# Check all
make lint type test
```

## Coding Standards

### Python Style
- Follow PEP 8 with 100-character line limit
- Use Black for formatting (if added to project)
- Use meaningful variable names
- Add type hints for all public functions

### Type Hints
```python
# Good
def calculate_ate(
    treatment_prob: float,
    control_prob: float
) -> float:
    """Calculate Average Treatment Effect."""
    return treatment_prob - control_prob

# Bad
def calculate_ate(treatment_prob, control_prob):
    return treatment_prob - control_prob
```

### Docstrings
Use NumPy-style docstrings:
```python
def fit_model(df: pd.DataFrame, link: str = "logit") -> GLMResults:
    """
    Fit a binomial GLM to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing outcome and predictors
    link : str, default "logit"
        Link function ("logit" or "probit")

    Returns
    -------
    GLMResults
        Fitted model results

    Raises
    ------
    ValueError
        If required columns are missing
    LinAlgError
        If matrix is singular

    Examples
    --------
    >>> df = pd.read_csv("data.csv")
    >>> results = fit_model(df, link="probit")
    """
```

### Error Handling
```python
# Good: Specific error messages
if not set(df["T"].unique()) <= {0, 1}:
    raise ValueError(
        f"Treatment column 'T' must be binary (0/1). "
        f"Found values: {sorted(df['T'].unique())}"
    )

# Bad: Generic messages
if not set(df["T"].unique()) <= {0, 1}:
    raise ValueError("Invalid data")
```

## Testing Guidelines

### Test Structure
```python
import pytest
import numpy as np
from ab_glm import function_to_test

class TestFunctionName:
    """Test suite for function_name."""

    def test_normal_case(self):
        """Test normal expected behavior."""
        result = function_to_test(valid_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test boundary conditions."""
        result = function_to_test(edge_input)
        assert result == edge_output

    def test_error_handling(self):
        """Test that appropriate errors are raised."""
        with pytest.raises(ValueError, match="specific message"):
            function_to_test(invalid_input)

    @pytest.mark.parametrize("input,expected", [
        (input1, expected1),
        (input2, expected2),
    ])
    def test_multiple_cases(self, input, expected):
        """Test multiple similar cases."""
        assert function_to_test(input) == expected
```

### Test Coverage
- Maintain >90% test coverage
- Test all public functions
- Include edge cases and error conditions
- Add integration tests for complex workflows

### Performance Tests
```python
@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset (mark as slow)."""
    df = simulate_large_data(n=1000000)
    # ... test performance
```

## Documentation

### When to Update Documentation
- Adding new functions or classes
- Changing function signatures
- Modifying behavior
- Fixing documentation errors

### Documentation Checklist
- [ ] Docstrings for all public functions
- [ ] Type hints for parameters and returns
- [ ] Examples in docstrings
- [ ] Update API reference if needed
- [ ] Update README if adding major features
- [ ] Add to CHANGELOG.md

### Building Documentation
```bash
# If using Sphinx (future enhancement)
cd docs
make html
```

## Pull Request Process

### Before Submitting
1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. **Run all checks**:
   ```bash
   make lint type test
   ```

3. **Update documentation**

4. **Add tests** for new features

5. **Update CHANGELOG.md** with your changes

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Coverage remains >90%

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process
1. Submit PR against `develop` branch
2. Ensure CI passes
3. Wait for code review
4. Address feedback
5. Squash commits if requested
6. PR will be merged by maintainers

## Release Process

Maintainers handle releases:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Push to PyPI
5. Create GitHub release

## Questions?

- Open a [discussion](https://github.com/DiogoRibeiro7/ab-glm-abtest/discussions)
- Ask in [issues](https://github.com/DiogoRibeiro7/ab-glm-abtest/issues)
- Email maintainers (see AUTHORS file)

## Recognition

Contributors will be:
- Added to AUTHORS file
- Mentioned in release notes
- Listed on GitHub contributors page

Thank you for contributing to ab-glm-abtest!