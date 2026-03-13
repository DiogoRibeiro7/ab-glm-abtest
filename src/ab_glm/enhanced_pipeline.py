"""
Enhanced pipeline for A/B testing with Binomial GLMs.

This module provides improved versions of the core functions with:
- Comprehensive logging
- Better error handling
- Progress tracking
- Additional statistical metrics
- Input validation
- Performance optimizations
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

LinkName = Literal["logit", "probit", "cloglog"]


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ModelConvergenceError(Exception):
    """Raised when model fails to converge."""
    pass


@dataclass
class EnhancedABResults:
    """Enhanced container for A/B test results with additional metrics.

    Includes confidence intervals, additional effect measures, and diagnostics.
    """
    # Core metrics
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

    # Additional metrics
    ate_ci_lower: float = 0.0
    ate_ci_upper: float = 0.0
    rr_ci_lower: float = 0.0
    rr_ci_upper: float = 0.0
    odds_ratio: float = 0.0
    nnt: float = 0.0  # Number needed to treat
    p_value: float = 1.0

    # Model diagnostics
    aic: float = 0.0
    bic: float = 0.0
    log_likelihood: float = 0.0
    deviance: float = 0.0
    pearson_chi2: float = 0.0

    # Data quality metrics
    covariate_balance: Dict[str, float] = field(default_factory=dict)
    missing_data_pct: float = 0.0
    sessions_per_user_mean: float = 0.0
    sessions_per_user_std: float = 0.0

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a formatted summary of results."""
        summary_lines = [
            "=" * 60,
            "ENHANCED A/B TEST RESULTS",
            "=" * 60,
            f"Link Function: {self.link}",
            f"Sample Size: {self.n_users:,} users, {self.n_obs:,} observations",
            f"Sessions per user: {self.sessions_per_user_mean:.2f} ± {self.sessions_per_user_std:.2f}",
            "",
            "TREATMENT EFFECTS:",
            f"  ATE: {self.ate_rd:.4f} [{self.ate_ci_lower:.4f}, {self.ate_ci_upper:.4f}]",
            f"  Risk Ratio: {self.rr:.4f} [{self.rr_ci_lower:.4f}, {self.rr_ci_upper:.4f}]",
            f"  Odds Ratio: {self.odds_ratio:.4f}",
            f"  NNT: {self.nnt:.1f}" if self.nnt < 1000 else f"  NNT: >{1000:.0f}",
            f"  P-value: {self.p_value:.4f}",
            "",
            "MODEL QUALITY:",
            f"  Brier Score: {self.brier:.4f}",
            f"  AIC: {self.aic:.1f}",
            f"  BIC: {self.bic:.1f}",
            f"  Log-Likelihood: {self.log_likelihood:.1f}",
        ]

        if self.warnings:
            summary_lines.extend(["", "WARNINGS:"] + [f"  - {w}" for w in self.warnings])

        return "\n".join(summary_lines)


def validate_data(
    df: pd.DataFrame,
    treatment_col: str = "T",
    outcome_col: str = "y",
    cluster_col: str = "user_id",
    covariates: Optional[List[str]] = None,
    strict: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Comprehensive data validation with detailed error messages.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column
    cluster_col : str
        Name of clustering column
    covariates : List[str], optional
        Names of covariate columns
    strict : bool
        If True, raise errors; if False, return warnings

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned data
    warnings : List[str]
        List of warnings encountered
    """
    warnings_list = []

    # Check for required columns
    if covariates is None:
        covariates = ["country_EU", "device_mobile", "prior_views"]

    required_cols = [cluster_col, treatment_col, outcome_col] + covariates
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        msg = f"Missing required columns: {sorted(missing_cols)}"
        if strict:
            raise DataValidationError(msg)
        warnings_list.append(msg)
        logger.warning(msg)
        # Continue with available columns when strict=False
        required_cols = [c for c in required_cols if c in df.columns]

    n_clusters = df[cluster_col].nunique() if cluster_col in df.columns else 0
    logger.info(f"Validating data with {len(df)} rows and {n_clusters} clusters")

    # Check for missing values
    na_counts = df[required_cols].isnull().sum()
    if na_counts.any():
        na_pct = (na_counts.sum() / len(df)) * 100
        msg = f"Found {na_counts.sum()} missing values ({na_pct:.1f}% of data)"
        warnings_list.append(msg)
        logger.warning(msg)

        # Drop missing values
        df_clean = df[required_cols].dropna()
        logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
    else:
        df_clean = df[required_cols].copy()

    # Check binary columns
    for col in [treatment_col, outcome_col]:
        unique_vals = df_clean[col].unique()
        if not set(unique_vals) <= {0, 1}:
            msg = f"Column '{col}' must be binary (0/1). Found: {sorted(unique_vals)}"
            if strict:
                raise DataValidationError(msg)
            warnings_list.append(msg)
            logger.error(msg)

    # Check treatment consistency within clusters
    treatment_consistency = df_clean.groupby(cluster_col)[treatment_col].nunique()
    inconsistent_clusters = treatment_consistency[treatment_consistency > 1]

    if len(inconsistent_clusters) > 0:
        msg = f"Treatment varies within {len(inconsistent_clusters)} clusters"
        warnings_list.append(msg)
        logger.warning(msg)

        # Fix by taking first treatment value per cluster
        first_treatment = df_clean.groupby(cluster_col)[treatment_col].first()
        df_clean[treatment_col] = df_clean[cluster_col].map(first_treatment)
        logger.info("Fixed treatment inconsistency by using first value per cluster")

    # Check for rare events
    outcome_rate = df_clean[outcome_col].mean()
    if outcome_rate < 0.01 or outcome_rate > 0.99:
        msg = f"Extreme outcome rate: {outcome_rate:.1%}"
        warnings_list.append(msg)
        logger.warning(msg)

    # Check for small sample size
    n_clusters = df_clean[cluster_col].nunique()
    n_treated = df_clean.groupby(cluster_col)[treatment_col].first().sum()
    n_control = n_clusters - n_treated

    if min(n_treated, n_control) < 30:
        msg = f"Small sample size: {n_control} control, {n_treated} treated clusters"
        warnings_list.append(msg)
        logger.warning(msg)

    # Check covariate balance
    user_df = df_clean.groupby(cluster_col).first()
    balance_stats = {}

    for cov in covariates:
        if cov in df_clean.columns:
            control_mean = user_df[user_df[treatment_col] == 0][cov].mean()
            treated_mean = user_df[user_df[treatment_col] == 1][cov].mean()
            pooled_std = np.sqrt(
                (user_df[user_df[treatment_col] == 0][cov].var() +
                 user_df[user_df[treatment_col] == 1][cov].var()) / 2
            )
            if pooled_std > 0:
                std_diff = abs(treated_mean - control_mean) / pooled_std
                balance_stats[cov] = std_diff

                if std_diff > 0.25:
                    msg = f"Large imbalance in {cov}: standardized diff = {std_diff:.3f}"
                    warnings_list.append(msg)
                    logger.warning(msg)

    logger.info(f"Data validation complete: {len(df_clean)} rows, {len(warnings_list)} warnings")
    return df_clean, warnings_list


def fit_binomial_glm_enhanced(
    df: pd.DataFrame,
    link: LinkName = "logit",
    cluster_col: str = "user_id",
    formula: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    validate: bool = True,
    progress_bar: bool = False,
    max_iter: int = 100,
    tolerance: float = 1e-8
) -> Tuple[sm.GLM, pd.DataFrame, sm.genmod.generalized_linear_model.GLMResultsWrapper, Dict[str, Any]]:
    """
    Enhanced GLM fitting with logging, validation, and additional diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    link : LinkName
        Link function ("logit", "probit", or "cloglog")
    cluster_col : str
        Column for clustering
    formula : str, optional
        Custom model formula
    covariates : List[str], optional
        List of covariate names
    validate : bool
        Whether to validate data first
    progress_bar : bool
        Show progress bar during fitting
    max_iter : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance

    Returns
    -------
    glm : sm.GLM
        Unfitted GLM object
    df_model : pd.DataFrame
        Data used for modeling
    results : GLMResultsWrapper
        Fitted results with cluster-robust SE
    diagnostics : Dict[str, Any]
        Additional diagnostics and metadata
    """
    logger.info(f"Starting GLM fitting with {link} link function")

    # Data validation
    if validate:
        df_clean, warnings_list = validate_data(df, cluster_col=cluster_col, covariates=covariates)
    else:
        df_clean = df.copy()
        warnings_list = []

    # Set default covariates
    if covariates is None:
        covariates = ["country_EU", "device_mobile", "prior_views"]

    # Set formula
    if formula is None:
        formula = f"y ~ T + {' + '.join(covariates)}"

    logger.info(f"Model formula: {formula}")

    # Get link function
    link_functions = {
        "logit": sm.families.links.Logit(),
        "probit": sm.families.links.Probit(),
        "cloglog": sm.families.links.CLogLog()
    }

    if link not in link_functions:
        raise ValueError(f"Unsupported link: {link}. Choose from {list(link_functions.keys())}")

    # Fit model
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

            if progress_bar:
                logger.info("Fitting model (progress bar not shown in logs)...")

            glm = smf.glm(
                formula=formula,
                data=df_clean,
                family=sm.families.Binomial(link=link_functions[link])
            )

            # Fit with cluster-robust SEs
            results = glm.fit(
                maxiter=max_iter,
                tol=tolerance,
                cov_type="cluster",
                cov_kwds={"groups": df_clean[cluster_col].values}
            )

            # Check for convergence
            if not results.converged:
                msg = f"Model did not converge after {max_iter} iterations"
                logger.warning(msg)
                warnings_list.append(msg)

    except np.linalg.LinAlgError as e:
        logger.error(f"Singular matrix error: {e}")
        raise ModelConvergenceError(f"Model fitting failed due to singular matrix. Check for perfect collinearity or separation.")

    except Exception as e:
        logger.error(f"Unexpected error during model fitting: {e}")
        raise

    # Calculate diagnostics
    diagnostics = {
        "warnings": warnings_list,
        "formula": formula,
        "n_iterations": results.fit_history.get("iteration", 0) if hasattr(results, "fit_history") else None,
        "converged": results.converged if hasattr(results, "converged") else True,
        "condition_number": np.linalg.cond(results.model.exog) if hasattr(results.model, "exog") else None,
        "effective_n": len(df_clean),
        "n_clusters": df_clean[cluster_col].nunique(),
        "cluster_sizes": df_clean.groupby(cluster_col).size().describe().to_dict()
    }

    # Check for separation
    predictions = results.predict(df_clean)
    if (predictions < 0.001).any() or (predictions > 0.999).any():
        msg = "Warning: Some predictions near 0 or 1, possible separation"
        logger.warning(msg)
        diagnostics["warnings"].append(msg)

    logger.info(f"Model fitting complete. Converged: {diagnostics['converged']}")
    return glm, df_clean, results, diagnostics


def calculate_confidence_intervals(
    results: sm.genmod.generalized_linear_model.GLMResultsWrapper,
    df_model: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "delta",
    n_bootstrap: int = 1000,
    cluster_col: str = "user_id",
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for treatment effects.

    Parameters
    ----------
    results : GLMResultsWrapper
        Fitted model results
    df_model : pd.DataFrame
        Model data
    alpha : float
        Significance level
    method : str
        Method for CI calculation ("delta" or "bootstrap")
    n_bootstrap : int
        Number of bootstrap iterations if method="bootstrap"
    cluster_col : str
        Cluster identifier column for cluster bootstrap.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Confidence intervals for various metrics
    """
    logger.info(f"Calculating {(1-alpha)*100}% confidence intervals using {method} method")

    if method == "delta":
        # Delta method approximation
        treat_idx = list(results.params.index).index("T")
        treat_coef = results.params["T"]
        treat_se = np.sqrt(np.diag(results.cov_params()))[treat_idx]

        # For ATE (simplified)
        df1 = df_model.copy()
        df0 = df_model.copy()
        df1["T"] = 1
        df0["T"] = 0

        p1 = results.predict(df1).mean()
        p0 = results.predict(df0).mean()
        ate = p1 - p0

        # Approximate SE for ATE using delta method
        ate_se = treat_se * p0 * (1 - p0)
        z_crit = stats.norm.ppf(1 - alpha/2)

        ate_ci = (ate - z_crit * ate_se, ate + z_crit * ate_se)

        # For Risk Ratio (log scale for better approximation)
        log_rr = np.log(p1 / p0)
        log_rr_se = treat_se / p0  # Simplified
        log_rr_ci = (log_rr - z_crit * log_rr_se, log_rr + z_crit * log_rr_se)
        rr_ci = (np.exp(log_rr_ci[0]), np.exp(log_rr_ci[1]))

    elif method == "bootstrap":
        # Bootstrap confidence intervals
        logger.info(f"Running {n_bootstrap} bootstrap iterations...")
        if cluster_col not in df_model.columns:
            raise ValueError(
                f"Cluster column '{cluster_col}' not found in df_model for bootstrap CI."
            )

        ate_samples = []
        rr_samples = []
        clusters = df_model[cluster_col].unique()

        for _ in tqdm(range(n_bootstrap), disable=not logger.isEnabledFor(logging.DEBUG)):
            # Resample clusters
            sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)

            # Create bootstrap sample
            df_boot = pd.concat([
                df_model[df_model[cluster_col] == c]
                for c in sampled_clusters
            ])

            try:
                # Refit model
                glm_boot = smf.glm(
                    formula=results.model.formula,
                    data=df_boot,
                    family=results.model.family
                )
                res_boot = glm_boot.fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df_boot[cluster_col].to_numpy()},
                )

                # Calculate metrics
                df1 = df_boot.copy()
                df0 = df_boot.copy()
                df1["T"] = 1
                df0["T"] = 0

                p1_boot = res_boot.predict(df1).mean()
                p0_boot = res_boot.predict(df0).mean()

                ate_samples.append(p1_boot - p0_boot)
                rr_samples.append(p1_boot / p0_boot)

            except Exception:
                continue

        if not ate_samples or not rr_samples:
            raise ModelConvergenceError(
                "Bootstrap CI failed: no successful bootstrap fits."
            )

        # Calculate percentile CIs
        ate_ci = (np.percentile(ate_samples, alpha/2 * 100),
                  np.percentile(ate_samples, (1 - alpha/2) * 100))
        rr_ci = (np.percentile(rr_samples, alpha/2 * 100),
                 np.percentile(rr_samples, (1 - alpha/2) * 100))

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'delta' or 'bootstrap'")

    confidence_intervals = {
        "ate": ate_ci,
        "risk_ratio": rr_ci
    }

    logger.info(f"Confidence intervals calculated: ATE {ate_ci}, RR {rr_ci}")
    return confidence_intervals


def calculate_additional_metrics(
    results: sm.genmod.generalized_linear_model.GLMResultsWrapper,
    df_model: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate additional statistical metrics beyond ATE and RR.

    Parameters
    ----------
    results : GLMResultsWrapper
        Fitted model results
    df_model : pd.DataFrame
        Model data

    Returns
    -------
    Dict[str, float]
        Additional metrics including NNT, OR, etc.
    """
    logger.info("Calculating additional statistical metrics")

    # Get predictions for treatment and control
    df1 = df_model.copy()
    df0 = df_model.copy()
    df1["T"] = 1
    df0["T"] = 0

    p1 = results.predict(df1).mean()
    p0 = results.predict(df0).mean()

    # Calculate metrics
    metrics = {}

    # Number Needed to Treat
    ate = p1 - p0
    metrics["nnt"] = 1 / abs(ate) if abs(ate) > 0.001 else float('inf')

    # Odds Ratio
    odds1 = p1 / (1 - p1) if p1 < 0.999 else float('inf')
    odds0 = p0 / (1 - p0) if p0 < 0.999 else float('inf')
    metrics["odds_ratio"] = odds1 / odds0 if odds0 > 0 else float('inf')

    # Relative Risk Reduction
    metrics["rrr"] = (p0 - p1) / p0 if p0 > 0 else 0

    # Attributable Risk
    metrics["attributable_risk"] = ate

    # Population Attributable Risk
    treatment_prevalence = df_model["T"].mean()
    metrics["par"] = treatment_prevalence * ate

    # Log odds ratio (for forest plots)
    metrics["log_or"] = np.log(metrics["odds_ratio"]) if metrics["odds_ratio"] > 0 else 0

    logger.info(f"Additional metrics calculated: NNT={metrics['nnt']:.1f}, OR={metrics['odds_ratio']:.3f}")
    return metrics


def run_enhanced_pipeline(
    df: Optional[pd.DataFrame] = None,
    link: LinkName = "logit",
    cluster_col: str = "user_id",
    n_users: int = 4000,
    sessions_per_user: Tuple[int, int] = (1, 5),
    seed: int = 42,
    validate: bool = True,
    calculate_cis: bool = True,
    ci_method: str = "delta",
    verbose: bool = True
) -> EnhancedABResults:
    """
    Run complete enhanced analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input data. If None, simulated data is generated
    link : LinkName
        Link function to use
    cluster_col : str
        Cluster identifier column for robust covariance and summaries.
    n_users : int
        Number of users for simulation (if df is None)
    sessions_per_user : Tuple[int, int]
        Session range for simulation (if df is None)
    seed : int
        Random seed for simulation
    validate : bool
        Whether to validate input data
    calculate_cis : bool
        Whether to calculate confidence intervals
    ci_method : str
        Method for CI calculation
    verbose : bool
        Whether to print progress

    Returns
    -------
    EnhancedABResults
        Complete analysis results
    """
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.info("Starting enhanced analysis pipeline")

    # Get or simulate data
    if df is None:
        from .pipeline import simulate_ab_data
        logger.info(f"Simulating data for {n_users} users")
        df = simulate_ab_data(n_users=n_users, sessions_per_user=sessions_per_user, seed=seed)

    # Fit model
    glm, df_model, results, diagnostics = fit_binomial_glm_enhanced(
        df,
        link=link,
        cluster_col=cluster_col,
        validate=validate,
        progress_bar=verbose
    )

    # Calculate main effects
    from .pipeline import marginal_effects_ate_and_rr, brier_score
    ate, rr, p_treat, p_ctrl = marginal_effects_ate_and_rr(results, df_model)

    # Calculate Brier score
    predictions = results.predict(df_model)
    brier = brier_score(df_model["y"].values, predictions)

    # Calculate confidence intervals
    if calculate_cis:
        cis = calculate_confidence_intervals(
            results, df_model, method=ci_method, cluster_col=cluster_col
        )
        ate_ci = cis["ate"]
        rr_ci = cis["risk_ratio"]
    else:
        ate_ci = (ate, ate)
        rr_ci = (rr, rr)

    # Calculate additional metrics
    additional = calculate_additional_metrics(results, df_model)

    # Calculate p-value
    if "T" in results.params.index:
        treat_idx = list(results.params.index).index("T")
        treat_coef = results.params["T"]
        treat_se = np.sqrt(np.diag(results.cov_params()))[treat_idx]
        z_stat = treat_coef / treat_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        treat_coef = None
        treat_se = None
        p_value = 1.0

    # Calculate covariate balance
    user_df = df_model.groupby(cluster_col).first()
    balance_stats = {}

    for cov in ["country_EU", "device_mobile", "prior_views"]:
        if cov in df_model.columns:
            control_mean = user_df[user_df["T"] == 0][cov].mean()
            treated_mean = user_df[user_df["T"] == 1][cov].mean()
            balance_stats[cov] = abs(treated_mean - control_mean)

    # Calculate sessions per user stats
    sessions_stats = df_model.groupby(cluster_col).size()

    # Create results object
    results_obj = EnhancedABResults(
        link=link,
        ate_rd=ate,
        rr=rr,
        p_treated=p_treat,
        p_control=p_ctrl,
        brier=brier,
        n_obs=len(df_model),
        n_users=df_model[cluster_col].nunique(),
        robust_se_treat=treat_se,
        coef_treat=treat_coef,
        ate_ci_lower=ate_ci[0],
        ate_ci_upper=ate_ci[1],
        rr_ci_lower=rr_ci[0],
        rr_ci_upper=rr_ci[1],
        odds_ratio=additional["odds_ratio"],
        nnt=additional["nnt"],
        p_value=p_value,
        aic=results.aic,
        bic=results.bic,
        log_likelihood=results.llf,
        deviance=results.deviance,
        pearson_chi2=results.pearson_chi2,
        covariate_balance=balance_stats,
        missing_data_pct=(1 - len(df_model) / len(df)) * 100 if df is not None else 0,
        sessions_per_user_mean=sessions_stats.mean(),
        sessions_per_user_std=sessions_stats.std(),
        warnings=diagnostics.get("warnings", [])
    )

    logger.info("Enhanced pipeline complete")

    if verbose:
        print(results_obj.summary())

    return results_obj
