# ab-glm-abtest

A production-ready Python package for A/B testing analysis using Binomial Generalized Linear Models (GLMs) with **logit** and **probit** link functions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Why Use This Package?

Most A/B testing tools make critical statistical errors:
- ❌ Ignore clustering when users have multiple sessions
- ❌ Report coefficients instead of business metrics
- ❌ Use naive standard errors that underestimate uncertainty
- ❌ Waste power by not adjusting for covariates

**This package does it right:**
- ✅ **Cluster-robust standard errors** for valid inference with repeated measures
- ✅ **Marginal effects** to convert coefficients into business metrics (ATE, Risk Ratio)
- ✅ **Covariate adjustment** to increase statistical power
- ✅ **Model diagnostics** including Brier scores for calibration
- ✅ **Production-ready** with 97% test coverage and comprehensive documentation

## 📊 Key Features

- **Binomial GLM** with logit and probit link functions
- **Automatic handling** of clustered data (multiple sessions per user)
- **Business metrics**: Absolute Treatment Effect (ATE) and Risk Ratio (RR)
- **Covariate adjustment** for improved precision
- **Model diagnostics** including calibration metrics
- **Simulation tools** for power analysis and testing
- **Comprehensive examples** with Jupyter notebooks

## 🚀 Quick Start

### Installation

```bash
# Using Poetry (recommended)
poetry add ab-glm-abtest

# Or using pip
pip install ab-glm-abtest
```

### Basic Usage

```python
import pandas as pd
from ab_glm import fit_binomial_glm, marginal_effects_ate_and_rr

# Load your A/B test data
df = pd.read_csv('your_experiment_data.csv')
# Required columns: user_id, T, country_EU, device_mobile, prior_views, y

# Fit GLM with cluster-robust standard errors
glm, _, df_model, results = fit_binomial_glm(
    df,
    link="logit",
    cluster_col="user_id"
)

# Get business metrics
ate, risk_ratio, p_treatment, p_control = marginal_effects_ate_and_rr(
    results, df_model
)

print(f"Control conversion: {p_control:.1%}")
print(f"Treatment conversion: {p_treatment:.1%}")
print(f"Absolute lift: {ate*100:.2f} percentage points")
print(f"Relative lift: {(risk_ratio-1)*100:.1f}%")
```

### Command-Line Analysis

```bash
# Analyze your experiment data
python examples/analyze_experiment.py --data your_data.csv

# Use probit link function
python examples/analyze_experiment.py --data your_data.csv --link probit

# Save results to file
python examples/analyze_experiment.py --data your_data.csv --output results.txt
```

## 📈 Example Output

```
============================================================
RUNNING LOGIT GLM ANALYSIS
============================================================

Sample Size:
  Users: 5,000
  Observations: 15,234
  Avg Sessions/User: 3.05

Covariate-Adjusted Results:
  Control Rate: 0.129 (12.9%)
  Treatment Rate: 0.160 (16.0%)
  ATE (Risk Diff): 0.031 (3.1 pp)
  Risk Ratio: 1.240
  Relative Lift: 24.0%
  P-value: 0.002
  Significant: Yes

Model Diagnostics:
  Brier Score: 0.105 (Good calibration)
  Link Function: logit
```

## 📚 Documentation

### Comprehensive Guides
- [**Interpretation Guide**](docs/interpretation_guide.md) - Understanding GLM coefficients and business metrics
- [**Troubleshooting**](docs/troubleshooting.md) - Solutions for common issues
- [**API Reference**](docs/api_reference.md) - Complete function documentation

### Interactive Notebooks
- [**Real-World Example**](notebooks/01_real_world_example.ipynb) - Complete A/B test analysis workflow
- [**Logit vs Probit**](notebooks/02_logit_vs_probit_comparison.ipynb) - Comparing link functions
- [**Power Analysis**](notebooks/03_power_analysis_sample_size.ipynb) - Sample size planning

### Example Scripts
- [**analyze_experiment.py**](examples/analyze_experiment.py) - Production-ready analysis script
- [**sample_experiment_data.csv**](examples/sample_experiment_data.csv) - Example data format

## 🔧 Data Requirements

Your data must include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int/str | Unique user identifier |
| `T` | binary | Treatment assignment (0=control, 1=treatment) |
| `country_EU` | binary | User location (0=non-EU, 1=EU) |
| `device_mobile` | binary | Device type (0=desktop, 1=mobile) |
| `prior_views` | int | Prior engagement metric |
| `y` | binary | Outcome (0=no conversion, 1=conversion) |

**Important:** Treatment must be assigned at the user level, not session level.

## 🧮 Statistical Methods

### Model Specification
The package fits the following model:
```
y ~ Binomial(p)
g(p) = β₀ + β₁T + β₂country_EU + β₃device_mobile + β₄prior_views
```
where g() is the link function (logit or probit).

### Marginal Effects
Business metrics are calculated using the G-computation formula:
- **ATE** = E[Y|T=1, X] - E[Y|T=0, X]
- **Risk Ratio** = E[Y|T=1, X] / E[Y|T=0, X]

### Cluster-Robust Standard Errors
When users have multiple sessions, observations are correlated. The package automatically computes cluster-robust standard errors to account for this.

## 🎯 When to Use This Package

**Perfect for:**
- E-commerce conversion optimization
- SaaS feature experiments
- Marketing campaign testing
- UX/UI improvements
- Any binary outcome with user-level randomization

**Not suitable for:**
- Continuous outcomes (use linear models)
- Count data (use Poisson/Negative Binomial)
- Time-to-event data (use survival analysis)
- Cluster-randomized trials (use mixed effects models)

## 📊 Performance & Scalability

- **Memory**: ~100 bytes per observation
- **Speed**: Handles 1M observations in <10 seconds
- **Limits**: Tested up to 10M observations
- **Coverage**: 97% test coverage with 55+ tests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ab-glm-abtest.git
cd ab-glm-abtest

# Install with development dependencies
poetry install

# Run tests with coverage
poetry run pytest --cov=ab_glm --cov-report=term-missing

# Run linting
poetry run ruff check src tests

# Run type checking
poetry run mypy src
```

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{ab_glm_abtest,
  title = {ab-glm-abtest: Production-ready A/B testing with Binomial GLMs},
  author = {Diogo Ribeiro},
  year = {2025},
  url = {https://github.com/DiogoRibeiro7/ab-glm-abtest}
}
```

## 🔗 See Also

- [Statsmodels Documentation](https://www.statsmodels.org/) - Underlying statistical library
- [Causal Inference Mixtape](https://mixtape.scunning.com/) - Causal inference methods
- [Trustworthy Online Experiments](https://experimentguide.com/) - A/B testing best practices

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [Statsmodels](https://www.statsmodels.org/) for GLM implementation
- [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) for data handling
- [Poetry](https://python-poetry.org/) for dependency management

---

**Questions?** Open an [issue](https://github.com/DiogoRibeiro7/ab-glm-abtest/issues) or check the [documentation](docs/).