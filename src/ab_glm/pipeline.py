"""
Core pipeline for A/B testing with Binomial GLMs (logit & probit).

Features:
- Simulated clustered A/B data (multiple sessions/user)
- Covariate-adjusted GLM (Binomial with chosen link)
- ATE (risk difference) & Risk Ratio via marginal predictions
- Cluster-robust SEs at the user level
- Brier score for probabilistic calibration

All public functions include type hints, basic type checks, and concise docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

LinkName = Literal["logit", "probit"]


@dataclass
class ABResults:
    """Container for key A/B metrics and model descriptors.

    Attributes
    ----------
    link : LinkName
        Link used by the Binomial GLM ("logit" or "probit").
    ate_rd : float
        Average treatment effect on the probability scale (absolute lift).
    rr : float
        Risk ratio: treated probability / control probability.
    p_treated : float
        Average predicted probability under treatment (covariate-adjusted).
    p_control : float
        Average predicted probability under control (covariate-adjusted).
    brier : float
        Brier score of in-sample predictions.
    n_obs : int
        Number of rows used for fitting and evaluation.
    n_users : int
        Number of unique users (clusters).
    robust_se_treat : Optional[float]
        Cluster-robust standard error for the treatment coefficient (link scale).
    coef_treat : Optional[float]
        Estimated treatment coefficient (link scale).
    """

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


def simulate_ab_data(
    n_users: int = 4000,
    sessions_per_user: Tuple[int, int] = (1, 5),
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate clustered A/B data with pre-treatment covariates.

    Parameters
    ----------
    n_users : int
        Number of unique users (clusters). Must be positive.
    sessions_per_user : (int, int)
        Inclusive range for sessions per user. Lower bound must be >=1 and <= upper.
    seed : int
        Random generator seed.

    Returns
    -------
    pd.DataFrame
        Columns: user_id, T, country_EU, device_mobile, prior_views, y
    """
    if n_users <= 0:
        raise ValueError("n_users must be positive.")
    lo, hi = sessions_per_user
    if lo < 1 or hi < lo:
        raise ValueError("sessions_per_user must be like (lo>=1, hi>=lo).")

    rng = np.random.default_rng(seed)

    T_user = rng.integers(0, 2, size=n_users)  # treatment at user level (50/50)

    # Pre-treatment covariates
    country_EU = rng.binomial(1, 0.6, size=n_users)
    device_mobile = rng.binomial(1, 0.55, size=n_users)
    prior_views = rng.poisson(3.0, size=n_users)

    # Random intercept (user-level latent propensity)
    u = rng.normal(0.0, 0.5, size=n_users)

    rows = []
    for i in range(n_users):
        n_sessions = rng.integers(lo, hi + 1)
        eta_base = (
            -2.0
            + 0.20 * country_EU[i]
            + 0.25 * device_mobile[i]
            + 0.08 * prior_views[i]
            + 0.35 * T_user[i]
            + u[i]
        )
        p = 1.0 / (1.0 + np.exp(-eta_base))  # logistic to generate outcomes
        for _ in range(n_sessions):
            y = rng.binomial(1, p)
            rows.append(
                (i, T_user[i], country_EU[i], device_mobile[i], prior_views[i], y)
            )

    df = pd.DataFrame(
        rows,
        columns=["user_id", "T", "country_EU", "device_mobile", "prior_views", "y"],
    )
    return df


def _get_link(link: LinkName) -> sm.families.links.Link:
    """Return statsmodels link object for the given name."""
    if link == "logit":
        return sm.families.links.Logit()
    if link == "probit":
        return sm.families.links.Probit()
    raise ValueError(f"Unsupported link: {link!r}.")


def fit_binomial_glm(
    df: pd.DataFrame,
    link: LinkName = "logit",
    cluster_col: str = "user_id",
) -> Tuple[sm.GLM, sm.GLM, pd.DataFrame, sm.genmod.generalized_linear_model.GLMResultsWrapper]:
    """Fit a Binomial GLM with treatment and covariates, cluster-robust SEs.

    Model: y ~ T + country_EU + device_mobile + prior_views

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'y', 'T', 'country_EU', 'device_mobile', 'prior_views',
        and the cluster column.
    link : LinkName
        Either 'logit' or 'probit'.
    cluster_col : str
        Column used for clustering (user id).

    Returns
    -------
    (glm, glm, df_model, res_robust)
        Unfitted GLM (for reference), duplicate (for symmetry), the data used,
        and the fitted results with cluster-robust covariance.
    """
    required = {"y", "T", "country_EU", "device_mobile", "prior_views", cluster_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not set(pd.unique(df["T"])) <= {0, 1}:
        raise ValueError("'T' must be binary (0/1).")
    if not set(pd.unique(df["y"])) <= {0, 1}:
        raise ValueError("'y' must be binary (0/1).")

    used_cols = list(required)
    df_model = df[used_cols].dropna(axis=0).copy()
    if df_model.empty:
        raise ValueError("No rows left after dropping NA in required columns.")

    formula = "y ~ T + country_EU + device_mobile + prior_views"

    glm = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(link=_get_link(link)),
    )

    # Fit with cluster-robust covariance
    res_robust = glm.fit(
        cov_type="cluster",
        cov_kwds={"groups": df_model[cluster_col].astype(int).to_numpy()}
    )

    return glm, glm, df_model, res_robust


def marginal_effects_ate_and_rr(
    res_robust: sm.genmod.generalized_linear_model.GLMResultsWrapper,
    df_model: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """Compute covariate-adjusted ATE (RD) and risk ratio via marginal predictions."""
    df1 = df_model.copy()
    df0 = df_model.copy()
    df1["T"] = 1
    df0["T"] = 0

    p1 = res_robust.predict(df1)
    p0 = res_robust.predict(df0)

    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    p0 = np.clip(p0, 1e-12, 1 - 1e-12)

    p_treated = float(np.mean(p1))
    p_control = float(np.mean(p0))
    ate_rd = p_treated - p_control
    rr = p_treated / p_control

    return ate_rd, rr, p_treated, p_control


def brier_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Return Brier score (mean squared error for probabilistic predictions)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_hat, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and p_hat must have the same shape.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("y_true and p_hat must be finite.")
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((y - p) ** 2))


def run_pipeline(link: LinkName = "logit") -> ABResults:
    """Simulate data, fit GLM, compute ATE/RR, and summarize into ABResults."""
    df = simulate_ab_data()
    _, _, df_model, res_robust = fit_binomial_glm(df, link=link, cluster_col="user_id")

    ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(res_robust, df_model)

    p_in_sample = res_robust.predict(df_model)
    bs = brier_score(df_model["y"].to_numpy(), p_in_sample)

    if "T" in res_robust.params.index:
        coef_treat = float(res_robust.params["T"])
        try:
            ix = list(res_robust.params.index).index("T")
            se_treat = float(np.sqrt(np.diag(res_robust.cov_params()))[ix])
        except Exception:
            se_treat = None
    else:
        coef_treat, se_treat = None, None

    return ABResults(
        link=link,
        ate_rd=ate_rd,
        rr=rr,
        p_treated=p_treated,
        p_control=p_control,
        brier=bs,
        n_obs=int(df_model.shape[0]),
        n_users=int(df_model["user_id"].nunique()),
        robust_se_treat=se_treat,
        coef_treat=coef_treat,
    )
