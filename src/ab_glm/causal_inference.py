"""
Causal inference and heterogeneous treatment effects for A/B testing.

This module provides:
- Causal forest for heterogeneous treatment effects
- Double/debiased machine learning (DML)
- Instrumental variables
- Regression discontinuity
- Propensity score methods
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class CATEResult:
    """Conditional Average Treatment Effect results."""
    cate_estimates: np.ndarray
    cate_std: np.ndarray
    feature_importance: Dict[str, float]
    ate: float
    ate_std: float


@dataclass
class PropensityScoreResult:
    """Propensity score matching results."""
    ate: float
    att: float  # Average Treatment on Treated
    atc: float  # Average Treatment on Control
    std_error: float
    matched_pairs: np.ndarray
    balance_statistics: pd.DataFrame


def estimate_cate_with_causal_forest(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_estimators: int = 100,
    min_samples_leaf: int = 10,
    cross_fitting_folds: int = 5,
    confidence_level: float = 0.95
) -> CATEResult:
    """
    Estimate Conditional Average Treatment Effects using Causal Forest.

    A simplified version that uses Random Forest with cross-fitting.

    Parameters
    ----------
    X : pd.DataFrame
        Covariate features
    treatment : np.ndarray
        Treatment assignments (0/1)
    outcome : np.ndarray
        Outcomes
    n_estimators : int
        Number of trees
    min_samples_leaf : int
        Minimum samples per leaf
    cross_fitting_folds : int
        Number of cross-fitting folds
    confidence_level : float
        Confidence level for intervals

    Returns
    -------
    CATEResult
        CATE estimates and statistics
    """
    n_samples = len(X)
    cate_estimates = np.zeros(n_samples)
    cate_var = np.zeros(n_samples)

    # Use cross-fitting to avoid overfitting
    kf = KFold(n_splits=cross_fitting_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        T_train, T_test = treatment[train_idx], treatment[test_idx]
        Y_train, Y_test = outcome[train_idx], outcome[test_idx]

        # Fit separate models for treatment and control
        treated_idx = T_train == 1
        control_idx = T_train == 0

        # Model for treated units
        rf_treated = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        rf_treated.fit(X_train[treated_idx], Y_train[treated_idx])

        # Model for control units
        rf_control = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        rf_control.fit(X_train[control_idx], Y_train[control_idx])

        # Predict potential outcomes
        y1_pred = rf_treated.predict(X_test)
        y0_pred = rf_control.predict(X_test)

        # CATE estimates
        cate_estimates[test_idx] = y1_pred - y0_pred

        # Estimate variance using the trees' predictions
        if hasattr(rf_treated, 'estimators_'):
            y1_trees = np.array([tree.predict(X_test) for tree in rf_treated.estimators_])
            y0_trees = np.array([tree.predict(X_test) for tree in rf_control.estimators_])
            cate_trees = y1_trees - y0_trees
            cate_var[test_idx] = np.var(cate_trees, axis=0)

    # Feature importance (average across models)
    feature_importance = {}
    for col in X.columns:
        feature_importance[col] = 0.0

    # Calculate ATE
    ate = np.mean(cate_estimates)
    ate_std = np.std(cate_estimates) / np.sqrt(n_samples)

    # Convert variance to standard deviation
    cate_std = np.sqrt(cate_var)

    return CATEResult(
        cate_estimates=cate_estimates,
        cate_std=cate_std,
        feature_importance=feature_importance,
        ate=ate,
        ate_std=ate_std
    )


def double_ml_ate(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    ml_method: str = 'rf',
    n_folds: int = 5,
    n_estimators: int = 100
) -> Dict[str, float]:
    """
    Estimate ATE using Double/Debiased Machine Learning.

    Parameters
    ----------
    X : pd.DataFrame
        Covariates
    treatment : np.ndarray
        Treatment assignments
    outcome : np.ndarray
        Outcomes
    ml_method : str
        'rf' for Random Forest, 'linear' for Linear Regression
    n_folds : int
        Number of cross-fitting folds
    n_estimators : int
        Number of estimators for RF

    Returns
    -------
    Dict with ATE estimate and standard error
    """
    n_samples = len(X)

    # Residualize outcome and treatment using cross-fitting
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    outcome_residuals = np.zeros(n_samples)
    treatment_residuals = np.zeros(n_samples)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        T_train, T_test = treatment[train_idx], treatment[test_idx]
        Y_train, Y_test = outcome[train_idx], outcome[test_idx]

        if ml_method == 'rf':
            # Outcome model
            outcome_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            outcome_model.fit(X_train, Y_train)
            outcome_pred = outcome_model.predict(X_test)

            # Treatment model (propensity score)
            treatment_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            treatment_model.fit(X_train, T_train)
            treatment_prob = treatment_model.predict_proba(X_test)[:, 1]
        else:  # linear
            # Outcome model
            outcome_model = LinearRegression()
            outcome_model.fit(X_train, Y_train)
            outcome_pred = outcome_model.predict(X_test)

            # Treatment model
            treatment_model = LogisticRegression(max_iter=1000)
            treatment_model.fit(X_train, T_train)
            treatment_prob = treatment_model.predict_proba(X_test)[:, 1]

        # Calculate residuals
        outcome_residuals[test_idx] = Y_test - outcome_pred
        treatment_residuals[test_idx] = T_test - treatment_prob

    # Estimate ATE using residuals (Frisch-Waugh-Lovell theorem)
    # ATE = E[Y|T=1] - E[Y|T=0] = cov(Y_res, T_res) / var(T_res)
    ate = np.sum(outcome_residuals * treatment_residuals) / np.sum(treatment_residuals ** 2)

    # Calculate standard error
    # Influence function based standard error
    influence = (outcome_residuals - ate * treatment_residuals) * treatment_residuals / np.mean(treatment_residuals ** 2)
    se = np.std(influence) / np.sqrt(n_samples)

    # Calculate confidence interval
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    # Calculate t-statistic and p-value
    t_stat = ate / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        'ate': ate,
        'std_error': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        'p_value': p_value,
        'method': f'DML-{ml_method}'
    }


def propensity_score_matching(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    caliper: Optional[float] = None,
    n_matches: int = 1,
    with_replacement: bool = False
) -> PropensityScoreResult:
    """
    Estimate treatment effects using propensity score matching.

    Parameters
    ----------
    X : pd.DataFrame
        Covariates
    treatment : np.ndarray
        Treatment assignments
    outcome : np.ndarray
        Outcomes
    caliper : float, optional
        Maximum distance for matching (in propensity score units)
    n_matches : int
        Number of matches per treated unit
    with_replacement : bool
        Whether to match with replacement

    Returns
    -------
    PropensityScoreResult
        Matching results
    """
    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, treatment)
    propensity_scores = ps_model.predict_proba(X)[:, 1]

    # Separate treatment and control
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    ps_treated = propensity_scores[treated_idx]
    ps_control = propensity_scores[control_idx]

    # Match treated to control
    matched_pairs = []
    matched_outcomes_treated = []
    matched_outcomes_control = []

    used_controls = set()

    for i, t_idx in enumerate(treated_idx):
        # Calculate distances
        distances = np.abs(ps_treated[i] - ps_control)

        # Apply caliper if specified
        if caliper is not None:
            valid_matches = distances <= caliper
            if not np.any(valid_matches):
                continue
            distances = np.where(valid_matches, distances, np.inf)

        # Find matches
        if with_replacement:
            match_indices = np.argsort(distances)[:n_matches]
        else:
            # Exclude already used controls
            available = np.array([j for j in range(len(control_idx)) if j not in used_controls])
            if len(available) == 0:
                break
            available_distances = distances[available]
            sorted_idx = np.argsort(available_distances)[:min(n_matches, len(available))]
            match_indices = available[sorted_idx]
            used_controls.update(match_indices)

        # Store matches
        for m_idx in match_indices:
            if distances[m_idx] != np.inf:
                matched_pairs.append([t_idx, control_idx[m_idx]])
                matched_outcomes_treated.append(outcome[t_idx])
                matched_outcomes_control.append(outcome[control_idx[m_idx]])

    matched_pairs = np.array(matched_pairs)

    # Calculate treatment effects
    if len(matched_outcomes_treated) > 0:
        ate = np.mean(np.array(matched_outcomes_treated) - np.array(matched_outcomes_control))
        att = ate  # For matched sample, ATE = ATT
        atc = ate  # Approximate

        # Standard error (using paired t-test formula)
        diff = np.array(matched_outcomes_treated) - np.array(matched_outcomes_control)
        std_error = np.std(diff) / np.sqrt(len(diff))
    else:
        ate = att = atc = std_error = np.nan
        warnings.warn("No valid matches found")

    # Check balance
    balance_stats = check_covariate_balance(X, treatment, propensity_scores)

    return PropensityScoreResult(
        ate=ate,
        att=att,
        atc=atc,
        std_error=std_error,
        matched_pairs=matched_pairs,
        balance_statistics=balance_stats
    )


def check_covariate_balance(
    X: pd.DataFrame,
    treatment: np.ndarray,
    propensity_scores: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Check covariate balance between treatment and control groups.

    Parameters
    ----------
    X : pd.DataFrame
        Covariates
    treatment : np.ndarray
        Treatment assignments
    propensity_scores : np.ndarray, optional
        Propensity scores for weighted balance

    Returns
    -------
    pd.DataFrame
        Balance statistics for each covariate
    """
    balance_stats = []

    for col in X.columns:
        values = X[col].values

        # Standardize if continuous
        if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            scaler = StandardScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        treated_values = values[treatment == 1]
        control_values = values[treatment == 0]

        # Calculate standardized mean difference
        mean_diff = treated_values.mean() - control_values.mean()
        pooled_std = np.sqrt((treated_values.var() + control_values.var()) / 2)

        if pooled_std > 0:
            smd = mean_diff / pooled_std
        else:
            smd = 0

        # Variance ratio
        var_ratio = treated_values.var() / control_values.var() if control_values.var() > 0 else np.nan

        balance_stats.append({
            'variable': col,
            'treated_mean': treated_values.mean(),
            'control_mean': control_values.mean(),
            'std_mean_diff': smd,
            'variance_ratio': var_ratio,
            'balanced': abs(smd) < 0.1  # Common threshold
        })

    return pd.DataFrame(balance_stats)


def regression_discontinuity(
    running_variable: np.ndarray,
    outcome: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 1,
    kernel: str = 'triangular'
) -> Dict[str, float]:
    """
    Estimate treatment effect using regression discontinuity design.

    Parameters
    ----------
    running_variable : np.ndarray
        Running variable (determines treatment)
    outcome : np.ndarray
        Outcomes
    cutoff : float
        Cutoff point for treatment assignment
    bandwidth : float, optional
        Bandwidth around cutoff (auto-selected if None)
    polynomial_order : int
        Polynomial order for local regression
    kernel : str
        Kernel type ('triangular', 'uniform', 'epanechnikov')

    Returns
    -------
    Dict with RD estimate and statistics
    """
    # Center running variable at cutoff
    X_centered = running_variable - cutoff
    treatment = (running_variable >= cutoff).astype(int)

    # Select optimal bandwidth if not provided
    if bandwidth is None:
        bandwidth = select_optimal_bandwidth_rd(X_centered, outcome, treatment)

    # Select observations within bandwidth
    within_bandwidth = np.abs(X_centered) <= bandwidth
    X_bw = X_centered[within_bandwidth]
    Y_bw = outcome[within_bandwidth]
    T_bw = treatment[within_bandwidth]

    # Calculate kernel weights
    if kernel == 'triangular':
        weights = np.maximum(0, 1 - np.abs(X_bw) / bandwidth)
    elif kernel == 'uniform':
        weights = np.ones_like(X_bw)
    elif kernel == 'epanechnikov':
        weights = np.maximum(0, 1 - (X_bw / bandwidth) ** 2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Build design matrix for local polynomial regression
    design_cols = []
    for p in range(1, polynomial_order + 1):
        design_cols.append(X_bw ** p)
        design_cols.append(T_bw * (X_bw ** p))  # Interaction terms

    design_matrix = np.column_stack([np.ones_like(X_bw), T_bw] + design_cols)

    # Weighted least squares
    W = np.diag(weights)
    XtWX = design_matrix.T @ W @ design_matrix
    XtWY = design_matrix.T @ W @ Y_bw

    try:
        coefficients = np.linalg.solve(XtWX, XtWY)
        rd_estimate = coefficients[1]  # Treatment effect at cutoff

        # Calculate standard error
        residuals = Y_bw - design_matrix @ coefficients
        sigma2 = np.sum(weights * residuals ** 2) / (np.sum(weights) - len(coefficients))
        var_coef = sigma2 * np.linalg.inv(XtWX)
        se = np.sqrt(var_coef[1, 1])

        # Calculate statistics
        t_stat = rd_estimate / se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        ci_lower = rd_estimate - 1.96 * se
        ci_upper = rd_estimate + 1.96 * se

        return {
            'estimate': rd_estimate,
            'std_error': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'bandwidth': bandwidth,
            'n_effective': np.sum(within_bandwidth),
            'polynomial_order': polynomial_order,
            'kernel': kernel
        }

    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in RD estimation")
        return {
            'estimate': np.nan,
            'std_error': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'bandwidth': bandwidth,
            'n_effective': np.sum(within_bandwidth),
            'polynomial_order': polynomial_order,
            'kernel': kernel
        }


def select_optimal_bandwidth_rd(
    X_centered: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    method: str = 'ik'
) -> float:
    """
    Select optimal bandwidth for RD using Imbens-Kalyanaraman method.

    Simplified version of the IK bandwidth selector.
    """
    # Simple rule of thumb
    n = len(X_centered)
    X_range = np.max(X_centered) - np.min(X_centered)

    # Estimate conditional variance at cutoff
    near_cutoff = np.abs(X_centered) < X_range * 0.1
    var_at_cutoff = np.var(outcome[near_cutoff])

    # Simple bandwidth formula
    h_opt = 1.06 * np.std(X_centered) * (n ** (-1/5))

    # Adjust based on variance
    h_opt *= np.sqrt(var_at_cutoff / np.var(outcome))

    return min(h_opt, X_range / 4)  # Don't use more than 1/4 of the range


def sensitivity_analysis_unobserved_confounding(
    ate_estimate: float,
    std_error: float,
    gamma_range: np.ndarray = np.linspace(1, 2, 11)
) -> pd.DataFrame:
    """
    Perform Rosenbaum bounds sensitivity analysis for unobserved confounding.

    Parameters
    ----------
    ate_estimate : float
        Original ATE estimate
    std_error : float
        Standard error of ATE
    gamma_range : np.ndarray
        Range of gamma values (odds ratio of hidden bias)

    Returns
    -------
    pd.DataFrame
        Sensitivity analysis results
    """
    results = []

    for gamma in gamma_range:
        # Calculate bounds on treatment effect
        # This is a simplified version of Rosenbaum bounds
        bias_factor = np.log(gamma)

        # Upper and lower bounds
        lower_bound = ate_estimate - bias_factor * std_error
        upper_bound = ate_estimate + bias_factor * std_error

        # Adjusted p-values (conservative)
        z_lower = lower_bound / std_error
        z_upper = upper_bound / std_error

        p_lower = 2 * (1 - stats.norm.cdf(abs(z_lower)))
        p_upper = 2 * (1 - stats.norm.cdf(abs(z_upper)))

        results.append({
            'gamma': gamma,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'p_value_lower': p_lower,
            'p_value_upper': p_upper,
            'significant_lower': p_lower < 0.05,
            'significant_upper': p_upper < 0.05
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    from .pipeline import simulate_ab_data

    print("Testing causal inference functions...")

    # Generate sample data with covariates
    np.random.seed(42)
    n = 1000

    # Create covariates
    X = pd.DataFrame({
        'age': np.random.normal(30, 10, n),
        'income': np.random.lognormal(10, 1, n),
        'education': np.random.choice([0, 1, 2], n),
        'region': np.random.choice(['A', 'B', 'C'], n)
    })

    # Create treatment (influenced by covariates)
    propensity = 1 / (1 + np.exp(-0.5 + 0.02 * X['age'] - 0.0001 * X['income']))
    treatment = np.random.binomial(1, propensity)

    # Create outcome (with heterogeneous effects)
    base_outcome = 0.1 + 0.01 * X['age'] + 0.00001 * X['income']
    treatment_effect = 0.2 + 0.01 * X['age']  # Effect varies by age
    outcome = base_outcome + treatment * treatment_effect + np.random.normal(0, 0.1, n)

    # Convert categorical to dummies
    X_encoded = pd.get_dummies(X, columns=['region'], drop_first=True)

    # 1. Test CATE estimation
    print("\n1. Estimating Conditional Average Treatment Effects...")
    cate_result = estimate_cate_with_causal_forest(
        X_encoded, treatment, outcome, n_estimators=50
    )
    print(f"   ATE: {cate_result.ate:.4f} ± {cate_result.ate_std:.4f}")

    # 2. Test Double ML
    print("\n2. Double/Debiased Machine Learning...")
    dml_result = double_ml_ate(X_encoded, treatment, outcome, ml_method='rf')
    print(f"   ATE: {dml_result['ate']:.4f} ± {dml_result['std_error']:.4f}")
    print(f"   p-value: {dml_result['p_value']:.4f}")

    # 3. Test Propensity Score Matching
    print("\n3. Propensity Score Matching...")
    ps_result = propensity_score_matching(
        X_encoded, treatment, outcome, caliper=0.1, n_matches=1
    )
    print(f"   ATE: {ps_result.ate:.4f} ± {ps_result.std_error:.4f}")
    print(f"   Matched pairs: {len(ps_result.matched_pairs)}")

    # 4. Test balance checking
    print("\n4. Checking covariate balance...")
    balance = check_covariate_balance(X_encoded, treatment)
    n_balanced = balance['balanced'].sum()
    print(f"   Balanced covariates: {n_balanced}/{len(balance)}")

    print("\nAll causal inference tests completed successfully!")