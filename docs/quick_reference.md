# Quick Reference Guide

## Installation
```bash
poetry add ab-glm-abtest
# or
pip install ab-glm-abtest
```

## Essential Imports
```python
from ab_glm import (
    simulate_ab_data,        # Generate test data
    fit_binomial_glm,        # Fit GLM model
    marginal_effects_ate_and_rr,  # Calculate treatment effects
    brier_score,             # Model calibration
    run_pipeline,            # Run complete analysis
    ABResults                # Results dataclass
)
```

## Data Format
```python
# Required columns
df = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, ...],      # User identifier
    'T': [0, 0, 1, 1, 0, ...],            # Treatment (0/1)
    'country_EU': [1, 1, 0, 0, 1, ...],   # Binary covariate
    'device_mobile': [0, 0, 1, 1, 0, ...], # Binary covariate
    'prior_views': [3, 3, 5, 5, 2, ...],  # Numeric covariate
    'y': [0, 1, 1, 0, 0, ...]             # Outcome (0/1)
})
```

## Basic Workflow
```python
# 1. Load and validate data
df = pd.read_csv('experiment_data.csv')
assert df.groupby('user_id')['T'].nunique().max() == 1

# 2. Fit model
glm, _, df_model, results = fit_binomial_glm(df, link="logit")

# 3. Get treatment effects
ate, rr, p_treat, p_ctrl = marginal_effects_ate_and_rr(results, df_model)

# 4. Check model quality
predictions = results.predict(df_model)
brier = brier_score(df_model['y'].values, predictions)
```

## Key Metrics Formulas

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ATE** | P(Y=1\|T=1) - P(Y=1\|T=0) | Absolute difference in probability |
| **Risk Ratio** | P(Y=1\|T=1) / P(Y=1\|T=0) | Relative probability ratio |
| **NNT** | 1 / ATE | Number needed to treat for one conversion |
| **Relative Lift** | (RR - 1) × 100% | Percentage increase |
| **Brier Score** | mean((y - p̂)²) | Calibration (lower is better) |

## Statistical Tests
```python
from scipy import stats

# P-value for treatment effect
treat_coef = results.params['T']
treat_se = results.bse['T']
z_stat = treat_coef / treat_se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Confidence interval for ATE
ate_se = treat_se * p_ctrl * (1 - p_ctrl)  # Approximation
ci_lower = ate - 1.96 * ate_se
ci_upper = ate + 1.96 * ate_se
```

## Common Patterns

### Check Randomization Balance
```python
user_df = df.groupby('user_id').first()
balance = user_df.groupby('T')[['country_EU', 'device_mobile', 'prior_views']].mean()
print(balance)
```

### Subgroup Analysis
```python
# Analyze by device type
for device in [0, 1]:
    subset = df[df['device_mobile'] == device]
    _, _, df_sub, res_sub = fit_binomial_glm(subset)
    ate_sub, _, _, _ = marginal_effects_ate_and_rr(res_sub, df_sub)
    print(f"Device={device}: ATE={ate_sub:.3f}")
```

### Compare Link Functions
```python
# Logit
_, _, df_l, res_l = fit_binomial_glm(df, link="logit")
ate_l, rr_l, _, _ = marginal_effects_ate_and_rr(res_l, df_l)

# Probit
_, _, df_p, res_p = fit_binomial_glm(df, link="probit")
ate_p, rr_p, _, _ = marginal_effects_ate_and_rr(res_p, df_p)

print(f"Logit ATE: {ate_l:.3f}, Probit ATE: {ate_p:.3f}")
```

### Sample Size Calculation
```python
from scipy.stats import norm

def sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """Calculate required sample size per group."""
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_bar = (p1 + p2) / 2

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    n = (z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2)) / mde**2
    return int(np.ceil(n))

# Example: 10% baseline, 2pp MDE
n_required = sample_size(0.10, 0.02)
print(f"Need {n_required} users per group")
```

## Error Handling

### Common Errors and Fixes

| Error | Cause | Solution |
|-------|-------|----------|
| `LinAlgError: Singular matrix` | No variation or perfect collinearity | Remove constant columns |
| `ValueError: Missing required columns` | Wrong column names | Rename to match expected |
| `ValueError: T must be binary` | Non-binary treatment | Convert to 0/1 |
| `PerfectSeparationWarning` | Perfect prediction | Remove/group predictors |
| `ValueError: Treatment varies within users` | Randomization issue | Check assignment logic |

### Debug Checklist
```python
# 1. Check data shape
print(f"Shape: {df.shape}")
print(f"Users: {df['user_id'].nunique()}")

# 2. Check for missing values
print(df.isnull().sum())

# 3. Check value distributions
print(df['T'].value_counts())
print(df['y'].value_counts())

# 4. Check treatment consistency
inconsistent = df.groupby('user_id')['T'].nunique()
print(f"Inconsistent users: {(inconsistent > 1).sum()}")

# 5. Check for perfect separation
for col in ['T', 'country_EU', 'device_mobile']:
    rates = df.groupby(col)['y'].mean()
    if (rates == 0).any() or (rates == 1).any():
        print(f"Warning: Perfect separation in {col}")
```

## Decision Framework

```python
def make_decision(ate, p_value, minimum_effect=0.01):
    """Simple decision framework."""
    if p_value >= 0.05:
        return "Inconclusive - gather more data"
    elif ate > minimum_effect:
        return "Ship it - positive significant effect"
    elif ate < -minimum_effect:
        return "Don't ship - negative significant effect"
    else:
        return "Significant but too small - consider cost/benefit"

decision = make_decision(ate, p_value, minimum_effect=0.01)
print(f"Recommendation: {decision}")
```

## Performance Tips

### For Large Datasets
```python
# Sample for exploration
sample_users = df['user_id'].sample(n=1000)
df_sample = df[df['user_id'].isin(sample_users)]

# Optimize data types
df['user_id'] = df['user_id'].astype('category')
df[['T', 'y', 'country_EU', 'device_mobile']] = df[['T', 'y', 'country_EU', 'device_mobile']].astype('int8')
df['prior_views'] = df['prior_views'].astype('int16')
```

### For Multiple Tests
```python
# Bonferroni correction
n_tests = 5
alpha_corrected = 0.05 / n_tests

# Check if significant after correction
significant = p_value < alpha_corrected
```

## Reporting Template

```python
def generate_report(results, ate, rr, p_value, brier):
    """Generate standardized report."""
    report = f"""
    A/B Test Results Summary
    ========================
    Sample Size: {results.n_users:,} users ({results.n_obs:,} observations)

    Treatment Effects:
    - Absolute Effect: {ate:.3f} ({ate*100:.1f} pp)
    - Relative Effect: {(rr-1)*100:.1f}%
    - Risk Ratio: {rr:.3f}
    - P-value: {p_value:.4f}
    - Significant: {'Yes' if p_value < 0.05 else 'No'}

    Model Quality:
    - Brier Score: {brier:.3f}
    - Link Function: {results.link}

    Recommendation: {'Ship' if p_value < 0.05 and ate > 0 else 'Do not ship'}
    """
    return report
```

## Useful Snippets

### Load and Validate
```python
df = pd.read_csv('data.csv')
assert set(df.columns) >= {'user_id', 'T', 'y', 'country_EU', 'device_mobile', 'prior_views'}
assert df.groupby('user_id')['T'].nunique().max() == 1
```

### Full Analysis
```python
glm, _, df_model, results = fit_binomial_glm(df)
ate, rr, p_t, p_c = marginal_effects_ate_and_rr(results, df_model)
brier = brier_score(df_model['y'].values, results.predict(df_model))
print(f"ATE: {ate:.3f}, RR: {rr:.3f}, Brier: {brier:.3f}")
```

### Export Results
```python
results_dict = {
    'ate': ate, 'rr': rr, 'p_treatment': p_t, 'p_control': p_c,
    'brier': brier, 'n_users': df['user_id'].nunique()
}
pd.DataFrame([results_dict]).to_csv('results.csv', index=False)
```