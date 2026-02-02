# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with 97% coverage (55+ tests)
- Three detailed Jupyter notebooks for learning and analysis
- Complete API reference documentation
- Interpretation guide for understanding GLM results
- Troubleshooting guide for common issues
- Assumptions and limitations documentation
- Production-ready analysis script (`analyze_experiment.py`)
- Sample experiment data for testing
- Power analysis and sample size calculator
- Model comparison tools (logit vs probit)

### Changed
- Updated README with clearer structure and examples
- Improved type hints throughout codebase
- Fixed statsmodels API compatibility for newer versions
- Enhanced error messages for better debugging

### Fixed
- Deprecated statsmodels link function calls
- Cluster-robust standard error computation
- Import ordering and linting issues
- Windows encoding issues in example scripts

## [0.1.0] - 2025-01-27

### Added
- Initial release of ab-glm-abtest package
- Core GLM fitting with cluster-robust standard errors
- Marginal effects calculation (ATE and Risk Ratio)
- Brier score for model calibration
- Data simulation for testing
- Basic demo script
- Poetry-based dependency management
- GitHub Actions CI/CD pipeline
- Dependabot configuration
- MIT License

### Features
- Binomial GLM with logit and probit link functions
- Automatic handling of clustered data
- Covariate adjustment for improved power
- Business-friendly metrics output
- Type hints and strict type checking
- Comprehensive docstrings

### Documentation
- Basic README with quick start guide
- Installation instructions
- Example output demonstration

[Unreleased]: https://github.com/DiogoRibeiro7/ab-glm-abtest/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/ab-glm-abtest/releases/tag/v0.1.0