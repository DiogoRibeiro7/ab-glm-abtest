# API Reference

Complete API documentation for the ab-glm-abtest package.

## Core Functions

### `simulate_ab_data`

Generate simulated A/B test data with clustering and covariates.

```python
simulate_ab_data(
    n_users: int = 4000,
    sessions_per_user: Tuple[int, int] = (1, 5),
    seed: int = 42
) -> pd.DataFrame
```

**Parameters:**
- `n_users` (int): Number of unique users to simulate. Must be positive.
- `sessions_per_user` (tuple): Min and max sessions per user (inclusive).
- `seed` (int): Random seed for reproducibility.

**Returns:**
- `pd.DataFrame`: DataFrame with columns:
  - `user_id`: Unique user identifier
  - `T`: Treatment assignment (0=control, 1=treatment)
  - `country_EU`: Binary covariate (0=non-EU, 1=EU)
  - `device_mobile`: Binary covariate (0=desktop, 1=mobile)
  - `prior_views`: Count covariate (Poisson distributed)
  - `y`: Binary outcome (0=no conversion, 1=conversion)

**Example:**
```python
from ab_glm import simulate_ab_data

# Generate data for 1000 users with 2-4 sessions each
df = simulate_ab_data(n_users=1000, sessions_per_user=(2, 4), seed=123)

print(f"Total observations: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Treatment rate: {df.groupby('user_id')['T'].first().mean():.2%}")
```

**Notes:**
- Treatment is assigned at the user level (constant within user)
- Data generating process includes user-level random effects
- Conversion probability follows logistic function with known coefficients

---

### `fit_binomial_glm`

Fit a Binomial Generalized Linear Model with cluster-robust standard errors.

```python
fit_binomial_glm(
    df: pd.DataFrame,
    link: LinkName = "logit",
    cluster_col: str = "user_id"
) -> Tuple[sm.GLM, sm.GLM, pd.DataFrame, GLMResultsWrapper]
```

**Parameters:**
- `df` (pd.DataFrame): Data with required columns
- `link` (str): Link function - either "logit" or "probit"
- `cluster_col` (str): Column name for clustering (typically user ID)

**Returns:**
- `tuple`: Four elements:
  1. Unfitted GLM object (for reference)
  2. Duplicate GLM object (for compatibility)
  3. DataFrame used for modeling (after dropping NAs)
  4. Fitted results with cluster-robust covariance

**Required columns in df:**
- `y`: Binary outcome (0/1)
- `T`: Binary treatment indicator (0/1)
- `country_EU`: Binary covariate (0/1)
- `device_mobile`: Binary covariate (0/1)
- `prior_views`: Numeric covariate (>=0)
- Column specified by `cluster_col`

**Model formula:**
```
y ~ T + country_EU + device_mobile + prior_views
```

**Example:**
```python
from ab_glm import fit_binomial_glm

# Fit logit model with cluster-robust SEs
glm, _, df_model, results = fit_binomial_glm(
    df,
    link="logit",
    cluster_col="user_id"
)

# Access coefficients
print("Treatment effect (link scale):", results.params['T'])
print("Cluster-robust SE:", results.bse['T'])

# Get p-value
from scipy import stats
z_stat = results.params['T'] / results.bse['T']
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"P-value: {p_value:.4f}")
```

**Raises:**
- `ValueError`: If required columns are missing
- `ValueError`: If T or y contains non-binary values
- `ValueError`: If no data remains after dropping NAs
- `LinAlgError`: If matrix is singular (perfect collinearity)

---

### `marginal_effects_ate_and_rr`

Calculate Average Treatment Effect and Risk Ratio via marginal predictions.

```python
marginal_effects_ate_and_rr(
    res_robust: GLMResultsWrapper,
    df_model: pd.DataFrame
) -> Tuple[float, float, float, float]
```

**Parameters:**
- `res_robust`: Fitted GLM results object
- `df_model`: DataFrame used for model fitting

**Returns:**
- `tuple`: Four floats:
  1. `ate_rd`: Average Treatment Effect (risk difference)
  2. `rr`: Risk Ratio (treated/control)
  3. `p_treated`: Mean predicted probability under treatment
  4. `p_control`: Mean predicted probability under control

**Method:**
1. Create two copies of data
2. Set T=1 for all observations, predict probabilities
3. Set T=0 for all observations, predict probabilities
4. ATE = mean(P|T=1) - mean(P|T=0)
5. RR = mean(P|T=1) / mean(P|T=0)

**Example:**
```python
from ab_glm import marginal_effects_ate_and_rr

# Calculate marginal effects
ate, rr, p_treat, p_ctrl = marginal_effects_ate_and_rr(results, df_model)

print(f"Control probability: {p_ctrl:.3%}")
print(f"Treatment probability: {p_treat:.3%}")
print(f"Absolute lift: {ate:.3f} ({ate*100:.1f} pp)")
print(f"Relative lift: {(rr-1)*100:.1f}%")

# Calculate confidence interval (approximation)
se_treat = results.bse['T']
ate_se = se_treat * p_ctrl * (1 - p_ctrl)  # Delta method approximation
ci_lower = ate - 1.96 * ate_se
ci_upper = ate + 1.96 * ate_se
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**Notes:**
- Predictions are clipped to [1e-12, 1-1e-12] for numerical stability
- This is the G-computation estimator
- Accounts for covariate distribution in the sample

---

### `brier_score`

Calculate Brier score for model calibration assessment.

```python
brier_score(
    y_true: np.ndarray,
    p_hat: np.ndarray
) -> float
```

**Parameters:**
- `y_true` (array): True binary outcomes (0/1)
- `p_hat` (array): Predicted probabilities [0,1]

**Returns:**
- `float`: Brier score (lower is better)

**Formula:**
```
BS = mean((y_true - p_hat)²)
```

**Interpretation:**
- 0.0: Perfect predictions
- 0.25: Random guessing (p=0.5 always)
- 1.0: Worst possible predictions

**Example:**
```python
from ab_glm import brier_score

# Calculate in-sample Brier score
predictions = results.predict(df_model)
bs = brier_score(df_model['y'].values, predictions)

print(f"Brier score: {bs:.4f}")
if bs < 0.1:
    print("Excellent calibration")
elif bs < 0.15:
    print("Good calibration")
elif bs < 0.2:
    print("Acceptable calibration")
else:
    print("Poor calibration - check model")
```

**Raises:**
- `ValueError`: If arrays have different shapes
- `ValueError`: If arrays contain non-finite values

---

### `run_pipeline`

Run complete analysis pipeline with simulated data.

```python
run_pipeline(
    link: LinkName = "logit"
) -> ABResults
```

**Parameters:**
- `link` (str): Link function - "logit" or "probit"

**Returns:**
- `ABResults`: Dataclass with complete results

**Example:**
```python
from ab_glm import run_pipeline

# Run full pipeline
results = run_pipeline(link="logit")

print(f"Sample size: {results.n_users} users, {results.n_obs} obs")
print(f"ATE: {results.ate_rd:.3f}")
print(f"Risk Ratio: {results.rr:.3f}")
print(f"P-value: {2*(1-stats.norm.cdf(abs(results.coef_treat/results.robust_se_treat))):.4f}")
print(f"Brier score: {results.brier:.3f}")
```

**Notes:**
- Uses fixed seed for reproducibility
- Primarily for testing and demonstration
- Replace with your data for real analysis

---

## Data Classes

### `ABResults`

Container for A/B test analysis results.

```python
@dataclass
class ABResults:
    link: LinkName
    ate_rd: float
    rr: float
    p_treated: float
    p_control: float
    brier: float
    n_obs: int
    n_users: int
    robust_se_treat: Optional[float]
    coef_treat: Optional[float]
```

**Attributes:**
- `link`: Link function used ("logit" or "probit")
- `ate_rd`: Average Treatment Effect (risk difference)
- `rr`: Risk Ratio
- `p_treated`: Mean probability under treatment
- `p_control`: Mean probability under control
- `brier`: Brier score for model calibration
- `n_obs`: Number of observations (sessions)
- `n_users`: Number of unique users (clusters)
- `robust_se_treat`: Cluster-robust SE for treatment coefficient
- `coef_treat`: Treatment coefficient on link scale

**Example:**
```python
from ab_glm import ABResults

# Manually create results object
results = ABResults(
    link="logit",
    ate_rd=0.025,
    rr=1.15,
    p_treated=0.195,
    p_control=0.170,
    brier=0.12,
    n_obs=10000,
    n_users=3000,
    robust_se_treat=0.008,
    coef_treat=0.15
)

# Calculate derived metrics
nnt = 1 / results.ate_rd  # Number needed to treat
relative_lift = (results.rr - 1) * 100
```

---

## Type Aliases

### `LinkName`

Valid link function names.

```python
LinkName = Literal["logit", "probit"]
```

**Usage:**
```python
from ab_glm.pipeline import LinkName

def my_analysis(link: LinkName):
    if link not in ["logit", "probit"]:
        raise ValueError(f"Invalid link: {link}")
    # ... rest of analysis
```

---

## Complete Example Workflow

```python
import pandas as pd
import numpy as np
from scipy import stats
from ab_glm import (
    simulate_ab_data,
    fit_binomial_glm,
    marginal_effects_ate_and_rr,
    brier_score
)

# 1. Prepare data (use your own or simulated)
df = simulate_ab_data(n_users=5000, sessions_per_user=(2, 5), seed=42)

# 2. Validate data
assert set(df.columns) >= {'user_id', 'T', 'y', 'country_EU', 'device_mobile', 'prior_views'}
assert df.groupby('user_id')['T'].nunique().max() == 1, "Treatment varies within users!"

# 3. Fit model
glm, _, df_model, results = fit_binomial_glm(df, link="logit")

# 4. Extract coefficients and significance
print("\nModel Coefficients:")
for var in results.params.index:
    coef = results.params[var]
    se = results.bse[var]
    z = coef / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"{var:20s}: {coef:7.3f} (SE: {se:6.3f}, p={p:.4f})")

# 5. Calculate business metrics
ate, rr, p_treat, p_ctrl = marginal_effects_ate_and_rr(results, df_model)

print("\nTreatment Effects:")
print(f"Control rate:    {p_ctrl:.3%}")
print(f"Treatment rate:  {p_treat:.3%}")
print(f"Absolute effect: {ate:.3f} ({ate*100:.1f} pp)")
print(f"Relative effect: {(rr-1)*100:.1f}%")

# 6. Assess model quality
predictions = results.predict(df_model)
bs = brier_score(df_model['y'].values, predictions)
print(f"\nModel Quality:")
print(f"Brier score: {bs:.4f}")

# 7. Make decision
treat_idx = list(results.params.index).index('T')
z_stat = results.params['T'] / results.bse['T']
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

if p_value < 0.05:
    if ate > 0:
        print("\n=> Implement treatment (positive significant effect)")
    else:
        print("\n=> Do not implement (negative significant effect)")
else:
    print(f"\n=> Inconclusive (p={p_value:.3f}), gather more data")
```

---

## Error Handling

All functions include input validation and raise appropriate errors:

```python
try:
    results = fit_binomial_glm(df, link="logit")
except ValueError as e:
    if "Missing required columns" in str(e):
        print("Check your DataFrame columns")
    elif "'T' must be binary" in str(e):
        print("Treatment must be 0/1")
    elif "No rows left" in str(e):
        print("All data was NA")
except np.linalg.LinAlgError:
    print("Singular matrix - check for perfect collinearity")
```

---

## Performance Considerations

### Memory Usage
- Each observation requires ~100 bytes
- 1M observations ≈ 100MB memory
- Cluster-robust SE calculation scales with number of clusters

### Speed
- Model fitting: O(n) for n observations
- Marginal effects: O(n) for predictions
- Cluster-robust SE: O(k²) for k clusters

### Recommendations
- Keep < 10M observations for interactive use
- Use sampling for exploration
- Consider chunking for very large datasets

---

## Version Compatibility

| ab-glm Version | Python | NumPy | Pandas | Statsmodels |
|----------------|--------|--------|--------|-------------|
| 0.1.0          | ≥3.10  | ≥2.0   | ≥2.2   | ≥0.14.2     |

---

## See Also

- [Interpretation Guide](interpretation_guide.md) - Understanding results
- [Troubleshooting](troubleshooting.md) - Solving common problems
- [Examples](../examples/) - Working code examples
- [Notebooks](../notebooks/) - Interactive tutorials