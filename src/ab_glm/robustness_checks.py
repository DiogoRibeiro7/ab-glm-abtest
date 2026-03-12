"""
Robustness checks and sensitivity analyses for A/B testing.

This module provides:
- Outlier detection and robust estimation
- Model specification tests
- Assumption validation
- Sensitivity to different specifications
- Meta-analysis across experiments
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import jarque_bera, kstest, normaltest
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .pipeline import fit_binomial_glm, marginal_effects_ate_and_rr


@dataclass
class RobustnessResult:
    """Container for robustness check results."""
    baseline_estimate: float
    robust_estimates: Dict[str, float]
    sensitivity_range: Tuple[float, float]
    robust: bool
    warnings: List[str]
    diagnostics: Dict[str, any]


def check_model_assumptions(
    model_results: sm.regression.linear_model.RegressionResults,
    X: pd.DataFrame,
    y: np.ndarray
) -> Dict[str, any]:
    """
    Check standard regression model assumptions.

    Parameters
    ----------
    model_results : RegressionResults
        Fitted model results
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Outcome variable

    Returns
    -------
    Dict with assumption test results
    """
    diagnostics = {}
    warnings_list = []

    # 1. Normality of residuals
    residuals = model_results.resid
    jb_stat, jb_p = jarque_bera(residuals)
    diagnostics['jarque_bera'] = {
        'statistic': jb_stat,
        'p_value': jb_p,
        'normal': jb_p > 0.05
    }

    if jb_p < 0.05:
        warnings_list.append("Residuals may not be normally distributed")

    # 2. Heteroskedasticity tests
    try:
        # Breusch-Pagan test
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
        diagnostics['breusch_pagan'] = {
            'statistic': bp_stat,
            'p_value': bp_p,
            'homoskedastic': bp_p > 0.05
        }

        if bp_p < 0.05:
            warnings_list.append("Heteroskedasticity detected (Breusch-Pagan test)")
    except:
        diagnostics['breusch_pagan'] = {'error': 'Could not compute'}

    # 3. Multicollinearity (VIF)
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        high_vif = vif_data[vif_data['VIF'] > 10]
        diagnostics['vif'] = vif_data.to_dict('records')

        if len(high_vif) > 0:
            warnings_list.append(f"High multicollinearity detected for: {high_vif['Variable'].tolist()}")
    except:
        diagnostics['vif'] = {'error': 'Could not compute VIF'}

    # 4. Influential observations
    try:
        influence = model_results.get_influence()
        cooks_d = influence.cooks_distance[0]

        # Flag observations with Cook's D > 4/n
        n = len(y)
        influential_threshold = 4 / n
        n_influential = np.sum(cooks_d > influential_threshold)

        diagnostics['influential_obs'] = {
            'n_influential': n_influential,
            'threshold': influential_threshold,
            'max_cooks_d': np.max(cooks_d)
        }

        if n_influential > n * 0.05:  # More than 5% influential
            warnings_list.append(f"{n_influential} influential observations detected")
    except:
        diagnostics['influential_obs'] = {'error': 'Could not compute'}

    # 5. Autocorrelation (Durbin-Watson)
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        diagnostics['durbin_watson'] = {
            'statistic': dw_stat,
            'autocorrelated': dw_stat < 1.5 or dw_stat > 2.5
        }

        if dw_stat < 1.5 or dw_stat > 2.5:
            warnings_list.append("Autocorrelation detected in residuals")
    except:
        diagnostics['durbin_watson'] = {'error': 'Could not compute'}

    diagnostics['warnings'] = warnings_list
    diagnostics['all_assumptions_met'] = len(warnings_list) == 0

    return diagnostics


def robust_estimation_with_outliers(
    df: pd.DataFrame,
    link_name: str = 'logit',
    outlier_methods: List[str] = ['iqr', 'zscore', 'isolation'],
    contamination: float = 0.05
) -> RobustnessResult:
    """
    Perform robust estimation with different outlier handling methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    link_name : str
        Link function ('logit' or 'probit')
    outlier_methods : List[str]
        Methods for outlier detection
    contamination : float
        Expected proportion of outliers

    Returns
    -------
    RobustnessResult
        Robust estimation results
    """
    # Baseline estimate
    baseline_family, baseline_link, df_model, results_baseline = fit_binomial_glm(df, link_name)
    ate_baseline, _, _, _ = marginal_effects_ate_and_rr(results_baseline, df_model)

    robust_estimates = {}
    diagnostics = {}

    for method in outlier_methods:
        try:
            # Detect outliers
            outliers = detect_outliers(df, method=method, contamination=contamination)
            df_clean = df[~outliers]

            # Refit model
            _, _, df_model_clean, results_clean = fit_binomial_glm(df_clean, link_name)
            ate_clean, _, _, _ = marginal_effects_ate_and_rr(results_clean, df_model_clean)

            robust_estimates[f'ate_{method}'] = ate_clean
            diagnostics[f'n_outliers_{method}'] = np.sum(outliers)

        except Exception as e:
            robust_estimates[f'ate_{method}'] = np.nan
            diagnostics[f'error_{method}'] = str(e)

    # Calculate sensitivity range
    valid_estimates = [v for v in robust_estimates.values() if not np.isnan(v)]
    if valid_estimates:
        sensitivity_range = (min(valid_estimates), max(valid_estimates))
        robust = (max(valid_estimates) - min(valid_estimates)) / abs(ate_baseline) < 0.2
    else:
        sensitivity_range = (ate_baseline, ate_baseline)
        robust = False

    # Generate warnings
    warnings_list = []
    if not robust:
        warnings_list.append("Estimates sensitive to outlier handling method")

    for method in outlier_methods:
        n_outliers = diagnostics.get(f'n_outliers_{method}', 0)
        if n_outliers > len(df) * 0.1:
            warnings_list.append(f"High proportion of outliers detected with {method}: {n_outliers}")

    return RobustnessResult(
        baseline_estimate=ate_baseline,
        robust_estimates=robust_estimates,
        sensitivity_range=sensitivity_range,
        robust=robust,
        warnings=warnings_list,
        diagnostics=diagnostics
    )


def detect_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    contamination: float = 0.05,
    features: Optional[List[str]] = None
) -> np.ndarray:
    """
    Detect outliers using various methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    method : str
        Detection method ('iqr', 'zscore', 'isolation', 'lof')
    contamination : float
        Expected proportion of outliers
    features : List[str], optional
        Features to use for detection

    Returns
    -------
    np.ndarray
        Boolean array indicating outliers
    """
    if features is None:
        # Use numeric features
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'y' in features:
            features.remove('y')  # Don't use outcome for outlier detection
        if 'T' in features:
            features.remove('T')  # Don't use treatment

    X = df[features].values

    if method == 'iqr':
        # IQR method
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = np.any((X < lower_bound) | (X > upper_bound), axis=1)

    elif method == 'zscore':
        # Z-score method
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        outliers = np.any(np.abs(X_scaled) > 3, axis=1)

    elif method == 'isolation':
        # Isolation Forest
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(X) == -1
        except ImportError:
            warnings.warn("IsolationForest not available, using IQR instead")
            return detect_outliers(df, method='iqr', contamination=contamination, features=features)

    elif method == 'lof':
        # Local Outlier Factor
        try:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(contamination=contamination)
            outliers = lof.fit_predict(X) == -1
        except ImportError:
            warnings.warn("LocalOutlierFactor not available, using IQR instead")
            return detect_outliers(df, method='iqr', contamination=contamination, features=features)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return outliers


def sensitivity_to_specification(
    df: pd.DataFrame,
    specifications: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Test sensitivity to model specification choices.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    specifications : Dict[str, List[str]], optional
        Different model specifications to test

    Returns
    -------
    pd.DataFrame
        Results across specifications
    """
    if specifications is None:
        # Default specifications
        base_covariates = ['country_EU', 'device_mobile', 'prior_views']
        specifications = {
            'base': base_covariates,
            'no_controls': [],
            'interactions': base_covariates + ['T:country_EU', 'T:device_mobile'],
            'quadratic': base_covariates + ['prior_views_sq'],
            'all': df.columns.drop(['user_id', 'T', 'y']).tolist()
        }

    results = []
    link_map = {
        'logit': sm.families.links.Logit(),
        'probit': sm.families.links.Probit(),
    }

    for spec_name, covariates in specifications.items():
        try:
            # Prepare data for this specification
            df_spec = df.copy()

            # Add quadratic terms if specified
            for cov in covariates:
                if '_sq' in cov:
                    base_var = cov.replace('_sq', '')
                    if base_var in df.columns:
                        df_spec[cov] = df_spec[base_var] ** 2

            # Fit models with different link functions
            for link in ['logit', 'probit']:
                try:
                    formula_terms = []
                    for cov in covariates:
                        if ":" in cov:
                            var1, var2 = cov.split(":")
                            if var1 in df_spec.columns and var2 in df_spec.columns:
                                formula_terms.append(cov)
                        elif cov in df_spec.columns:
                            formula_terms.append(cov)

                    formula = "y ~ T" if not formula_terms else f"y ~ T + {' + '.join(formula_terms)}"
                    glm = smf.glm(
                        formula=formula,
                        data=df_spec,
                        family=sm.families.Binomial(link=link_map[link]),
                    )

                    if "user_id" in df_spec.columns:
                        results_glm = glm.fit(
                            cov_type="cluster",
                            cov_kwds={"groups": df_spec["user_id"].to_numpy()},
                        )
                    else:
                        results_glm = glm.fit()

                    df_model = results_glm.model.data.frame.copy()
                    ate, rr, _, _ = marginal_effects_ate_and_rr(results_glm, df_model)

                    results.append({
                        'specification': spec_name,
                        'link': link,
                        'n_covariates': len(covariates),
                        'ate': ate,
                        'risk_ratio': rr,
                        'aic': results_glm.aic,
                        'bic': results_glm.bic,
                        'converged': True
                    })
                except Exception as e:
                    results.append({
                        'specification': spec_name,
                        'link': link,
                        'n_covariates': len(covariates),
                        'ate': np.nan,
                        'risk_ratio': np.nan,
                        'aic': np.nan,
                        'bic': np.nan,
                        'converged': False,
                        'error': str(e)
                    })

        except Exception as e:
            warnings.warn(f"Failed to test specification {spec_name}: {e}")

    return pd.DataFrame(results)


def meta_analysis(
    experiments: List[Dict[str, float]],
    method: str = 'fixed_effects'
) -> Dict[str, float]:
    """
    Perform meta-analysis across multiple experiments.

    Parameters
    ----------
    experiments : List[Dict]
        List of experiment results with 'effect', 'se', and 'n' keys
    method : str
        'fixed_effects' or 'random_effects'

    Returns
    -------
    Dict with pooled estimates
    """
    effects = np.array([exp['effect'] for exp in experiments])
    std_errors = np.array([exp['se'] for exp in experiments])
    sample_sizes = np.array([exp['n'] for exp in experiments])

    # Calculate weights
    if method == 'fixed_effects':
        # Inverse variance weighting
        weights = 1 / (std_errors ** 2)
    else:  # random_effects
        # DerSimonian-Laird method
        # First, calculate heterogeneity
        weights_fixed = 1 / (std_errors ** 2)
        pooled_fixed = np.sum(effects * weights_fixed) / np.sum(weights_fixed)

        Q = np.sum(weights_fixed * (effects - pooled_fixed) ** 2)
        df = len(effects) - 1

        # Calculate tau-squared (between-study variance)
        C = np.sum(weights_fixed) - np.sum(weights_fixed ** 2) / np.sum(weights_fixed)
        tau2 = max(0, (Q - df) / C)

        # Random effects weights
        weights = 1 / (std_errors ** 2 + tau2)

    # Calculate pooled estimate
    pooled_effect = np.sum(effects * weights) / np.sum(weights)
    pooled_se = np.sqrt(1 / np.sum(weights))

    # Confidence interval
    ci_lower = pooled_effect - 1.96 * pooled_se
    ci_upper = pooled_effect + 1.96 * pooled_se

    # Test for heterogeneity
    Q = np.sum(weights * (effects - pooled_effect) ** 2)
    df = len(effects) - 1
    p_heterogeneity = 1 - stats.chi2.cdf(Q, df)

    # I-squared statistic
    I2 = max(0, (Q - df) / Q * 100)

    return {
        'pooled_effect': pooled_effect,
        'pooled_se': pooled_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z_statistic': pooled_effect / pooled_se,
        'p_value': 2 * (1 - stats.norm.cdf(abs(pooled_effect / pooled_se))),
        'n_experiments': len(experiments),
        'total_n': np.sum(sample_sizes),
        'Q_statistic': Q,
        'p_heterogeneity': p_heterogeneity,
        'I_squared': I2,
        'method': method
    }


def placebo_test(
    df: pd.DataFrame,
    placebo_outcome_col: str,
    treatment_col: str = 'T'
) -> Dict[str, float]:
    """
    Perform placebo test using pre-treatment outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    placebo_outcome_col : str
        Column with pre-treatment outcome
    treatment_col : str
        Treatment column

    Returns
    -------
    Dict with placebo test results
    """
    if placebo_outcome_col not in df.columns:
        raise ValueError(f"Placebo outcome column {placebo_outcome_col} not found")

    # Run standard analysis on placebo outcome
    df_placebo = df.copy()
    df_placebo['y_original'] = df_placebo['y']
    df_placebo['y'] = df_placebo[placebo_outcome_col]

    try:
        _, _, df_model, results_glm = fit_binomial_glm(df_placebo)
        ate_placebo, _, _, _ = marginal_effects_ate_and_rr(results_glm, df_model)

        # Get standard error
        se_placebo = float(results_glm.bse["T"]) if "T" in results_glm.bse.index else np.nan

        # Calculate p-value
        z_stat = ate_placebo / se_placebo if se_placebo > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Test passed if effect is not significant
        test_passed = p_value > 0.05

        return {
            'placebo_effect': ate_placebo,
            'placebo_se': se_placebo,
            'placebo_p_value': p_value,
            'test_passed': test_passed,
            'interpretation': 'No pre-existing differences' if test_passed else 'Pre-existing differences detected'
        }

    except Exception as e:
        return {
            'placebo_effect': np.nan,
            'placebo_se': np.nan,
            'placebo_p_value': np.nan,
            'test_passed': False,
            'error': str(e)
        }


def complete_robustness_analysis(
    df: pd.DataFrame,
    link_name: str = 'logit'
) -> Dict[str, any]:
    """
    Perform complete robustness analysis suite.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    link_name : str
        Link function

    Returns
    -------
    Dict with all robustness check results
    """
    print("Performing comprehensive robustness analysis...")

    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_observations': len(df),
        'link_function': link_name
    }

    # 1. Baseline model
    print("1. Fitting baseline model...")
    try:
        family, link, df_model, results_baseline = fit_binomial_glm(df, link_name)
        ate_baseline, rr_baseline, _, _ = marginal_effects_ate_and_rr(results_baseline, df_model)

        results['baseline'] = {
            'ate': ate_baseline,
            'risk_ratio': rr_baseline,
            'converged': results_baseline.converged
        }
    except Exception as e:
        results['baseline'] = {'error': str(e)}
        return results

    # 2. Model assumptions
    print("2. Checking model assumptions...")
    try:
        X = df_model.drop('y', axis=1) if 'y' in df_model.columns else df_model
        y = df_model['y'] if 'y' in df_model.columns else df['y']
        results['assumptions'] = check_model_assumptions(results_baseline, X, y)
    except Exception as e:
        results['assumptions'] = {'error': str(e)}

    # 3. Outlier robustness
    print("3. Testing outlier robustness...")
    try:
        outlier_results = robust_estimation_with_outliers(df, link_name)
        results['outlier_robustness'] = {
            'robust': outlier_results.robust,
            'sensitivity_range': outlier_results.sensitivity_range,
            'estimates': outlier_results.robust_estimates
        }
    except Exception as e:
        results['outlier_robustness'] = {'error': str(e)}

    # 4. Specification sensitivity
    print("4. Testing specification sensitivity...")
    try:
        spec_results = sensitivity_to_specification(df)
        results['specification_sensitivity'] = {
            'n_specifications': len(spec_results),
            'ate_range': (spec_results['ate'].min(), spec_results['ate'].max()),
            'best_aic': spec_results.loc[spec_results['aic'].idxmin()].to_dict() if len(spec_results) > 0 else None
        }
    except Exception as e:
        results['specification_sensitivity'] = {'error': str(e)}

    # 5. Summary
    all_robust = True
    warnings = []

    if results.get('assumptions', {}).get('warnings'):
        warnings.extend(results['assumptions']['warnings'])
        all_robust = False

    if not results.get('outlier_robustness', {}).get('robust', True):
        warnings.append("Estimates sensitive to outliers")
        all_robust = False

    results['summary'] = {
        'all_checks_passed': all_robust,
        'warnings': warnings,
        'recommendation': 'Results are robust' if all_robust else 'Interpret with caution'
    }

    return results


if __name__ == "__main__":
    # Example usage
    from .pipeline import simulate_ab_data

    print("Testing robustness check functions...")

    # Generate sample data
    df = simulate_ab_data(n_users=500)

    # Run complete robustness analysis
    robustness_results = complete_robustness_analysis(df)

    print("\nRobustness Analysis Summary:")
    print(f"  All checks passed: {robustness_results['summary']['all_checks_passed']}")
    if robustness_results['summary']['warnings']:
        print(f"  Warnings: {robustness_results['summary']['warnings']}")
    print(f"  Recommendation: {robustness_results['summary']['recommendation']}")

    # Test meta-analysis
    print("\nTesting meta-analysis...")
    experiments = [
        {'effect': 0.05, 'se': 0.02, 'n': 1000},
        {'effect': 0.03, 'se': 0.015, 'n': 1500},
        {'effect': 0.06, 'se': 0.025, 'n': 800}
    ]
    meta_results = meta_analysis(experiments)
    print(f"  Pooled effect: {meta_results['pooled_effect']:.4f} ± {meta_results['pooled_se']:.4f}")
    print(f"  I-squared: {meta_results['I_squared']:.1f}%")

    print("\nAll robustness tests completed!")
