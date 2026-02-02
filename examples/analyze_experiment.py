#!/usr/bin/env python
"""
Example script showing how to analyze a real A/B test using ab-glm.

This script demonstrates:
1. Loading experimental data from CSV
2. Running data quality checks
3. Fitting GLM with cluster-robust SEs
4. Computing business metrics (ATE, Risk Ratio)
5. Generating a results report

Usage:
    python analyze_experiment.py [--data PATH] [--link LINK] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path if running as script
sys.path.append(str(Path(__file__).parent.parent))

from ab_glm import (
    fit_binomial_glm,
    marginal_effects_ate_and_rr,
    brier_score,
)


def load_and_validate_data(filepath):
    """Load and validate experimental data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Check required columns
    required = {"user_id", "T", "country_EU", "device_mobile", "prior_views", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Validate data types and values
    assert df["T"].isin([0, 1]).all(), "Treatment must be binary (0/1)"
    assert df["y"].isin([0, 1]).all(), "Outcome must be binary (0/1)"
    assert df["country_EU"].isin([0, 1]).all(), "country_EU must be binary"
    assert df["device_mobile"].isin([0, 1]).all(), "device_mobile must be binary"
    assert (df["prior_views"] >= 0).all(), "prior_views must be non-negative"

    # Check treatment assignment consistency
    treatment_consistency = df.groupby("user_id")["T"].nunique()
    if not (treatment_consistency == 1).all():
        raise ValueError("Treatment varies within users! Check randomization.")

    print(f"[OK] Data validated: {len(df):,} observations from {df['user_id'].nunique():,} users")
    return df


def check_balance(df):
    """Check covariate balance between treatment groups."""
    print("\nChecking randomization balance...")

    # Aggregate to user level
    user_df = df.groupby("user_id").first()

    # Calculate standardized differences
    covariates = ["country_EU", "device_mobile", "prior_views"]

    print("\nCovariate Balance (user-level means):")
    print("-" * 50)
    print(f"{'Covariate':<20} {'Control':<10} {'Treatment':<10} {'Std Diff':<10}")
    print("-" * 50)

    for col in covariates:
        control_mean = user_df[user_df["T"] == 0][col].mean()
        treat_mean = user_df[user_df["T"] == 1][col].mean()

        # Standardized difference
        pooled_std = np.sqrt(
            (user_df[user_df["T"] == 0][col].var() +
             user_df[user_df["T"] == 1][col].var()) / 2
        )
        std_diff = (treat_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        print(f"{col:<20} {control_mean:<10.3f} {treat_mean:<10.3f} {std_diff:<10.3f}")

    print("\n[OK] Standardized differences < 0.1 indicate good balance")


def analyze_experiment(df, link="logit"):
    """Run main analysis."""
    print(f"\n{'='*60}")
    print(f"RUNNING {link.upper()} GLM ANALYSIS")
    print(f"{'='*60}")

    # Fit model
    print("\nFitting Binomial GLM with cluster-robust SEs...")
    glm, _, df_model, results = fit_binomial_glm(df, link=link, cluster_col="user_id")

    # Extract coefficients
    print("\nModel Coefficients:")
    print("-" * 50)
    coef_df = pd.DataFrame({
        "Coefficient": results.params,
        "Std Error": np.sqrt(np.diag(results.cov_params())),
    })
    coef_df["Z-score"] = coef_df["Coefficient"] / coef_df["Std Error"]
    from scipy import stats
    coef_df["P-value"] = 2 * (1 - stats.norm.cdf(np.abs(coef_df["Z-score"])))
    coef_df["Significant"] = coef_df["P-value"] < 0.05
    print(coef_df.round(4))

    # Calculate marginal effects
    print("\nCalculating marginal effects...")
    ate_rd, rr, p_treated, p_control = marginal_effects_ate_and_rr(results, df_model)

    # Calculate Brier score
    predictions = results.predict(df_model)
    brier = brier_score(df_model["y"].values, predictions)

    # Calculate confidence intervals (approximate)
    treat_idx = list(results.params.index).index("T")
    treat_se = np.sqrt(np.diag(results.cov_params()))[treat_idx]
    ate_se_approx = treat_se * p_control * (1 - p_control)
    ate_ci_lower = ate_rd - 1.96 * ate_se_approx
    ate_ci_upper = ate_rd + 1.96 * ate_se_approx

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    results_dict = {
        "Sample Size": {
            "Users": df_model["user_id"].nunique(),
            "Observations": len(df_model),
            "Avg Sessions/User": len(df_model) / df_model["user_id"].nunique(),
        },
        "Raw Conversion Rates": {
            "Control": df[df["T"] == 0]["y"].mean(),
            "Treatment": df[df["T"] == 1]["y"].mean(),
            "Raw Difference": df[df["T"] == 1]["y"].mean() - df[df["T"] == 0]["y"].mean(),
        },
        "Covariate-Adjusted Results": {
            "Control Rate": p_control,
            "Treatment Rate": p_treated,
            "ATE (Risk Diff)": ate_rd,
            "ATE 95% CI Lower": ate_ci_lower,
            "ATE 95% CI Upper": ate_ci_upper,
            "Risk Ratio": rr,
            "Relative Lift": (rr - 1) * 100,
            "P-value": coef_df.loc["T", "P-value"],
            "Significant": coef_df.loc["T", "Significant"],
        },
        "Model Diagnostics": {
            "Brier Score": brier,
            "Link Function": link,
        }
    }

    for section, metrics in results_dict.items():
        print(f"\n{section}:")
        for key, value in metrics.items():
            if isinstance(value, bool):
                print(f"  {key}: {'Yes [OK]' if value else 'No [NO]'}")
            elif isinstance(value, (int, np.integer)):
                print(f"  {key}: {value:,}")
            elif isinstance(value, (float, np.floating)):
                if abs(value) < 0.01:
                    print(f"  {key}: {value:.4f}")
                elif abs(value) < 1:
                    print(f"  {key}: {value:.3f}")
                elif abs(value) < 100:
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")

    return results_dict


def generate_report(results, output_path=None):
    """Generate a summary report."""
    report = []
    report.append("=" * 60)
    report.append("A/B TEST ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")

    # Executive summary
    adj_results = results["Covariate-Adjusted Results"]
    if adj_results["Significant"]:
        if adj_results["ATE (Risk Diff)"] > 0:
            report.append("[POSITIVE] RESULT: Statistically significant positive effect detected")
            report.append(f"The treatment increases conversion by {adj_results['ATE (Risk Diff)']*100:.2f} percentage points")
            report.append(f"This represents a {adj_results['Relative Lift']:.1f}% relative improvement")
        else:
            report.append("[NEGATIVE] RESULT: Statistically significant negative effect detected")
            report.append(f"The treatment decreases conversion by {abs(adj_results['ATE (Risk Diff)'])*100:.2f} percentage points")
            report.append(f"This represents a {abs(adj_results['Relative Lift']):.1f}% relative decline")
    else:
        report.append("[INCONCLUSIVE] RESULT: No statistically significant effect detected")
        report.append(f"Observed difference: {adj_results['ATE (Risk Diff)']*100:.2f} percentage points")
        report.append(f"P-value: {adj_results['P-value']:.4f} (not significant at α=0.05)")

    report.append("")
    report.append("DETAILS:")
    report.append("-" * 40)

    # Sample size
    sample = results["Sample Size"]
    report.append(f"Sample size: {sample['Users']:,} users ({sample['Observations']:,} observations)")
    report.append(f"Average sessions per user: {sample['Avg Sessions/User']:.1f}")

    # Effect sizes
    report.append("")
    report.append("Effect Estimates (covariate-adjusted):")
    report.append(f"  Control rate: {adj_results['Control Rate']*100:.2f}%")
    report.append(f"  Treatment rate: {adj_results['Treatment Rate']*100:.2f}%")
    report.append(f"  Absolute effect: {adj_results['ATE (Risk Diff)']*100:.2f} pp")
    report.append(f"  95% CI: [{adj_results['ATE 95% CI Lower']*100:.2f}, {adj_results['ATE 95% CI Upper']*100:.2f}] pp")
    report.append(f"  Risk ratio: {adj_results['Risk Ratio']:.3f}")

    # Model quality
    report.append("")
    report.append("Model Diagnostics:")
    diag = results["Model Diagnostics"]
    report.append(f"  Brier score: {diag['Brier Score']:.4f}")
    report.append(f"  Link function: {diag['Link Function']}")

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"\n[OK] Report saved to {output_path}")
    else:
        print("\n" + report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze A/B test data using Binomial GLM"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="sample_experiment_data.csv",
        help="Path to CSV file with experiment data",
    )
    parser.add_argument(
        "--link",
        type=str,
        default="logit",
        choices=["logit", "probit"],
        help="Link function to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save report (optional)",
    )

    args = parser.parse_args()

    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("\nExpected CSV format:")
        print("  user_id,T,country_EU,device_mobile,prior_views,y")
        print("  1001,0,1,1,3,0")
        print("  1001,0,1,1,3,1")
        print("  ...")
        sys.exit(1)

    try:
        # Load data
        df = load_and_validate_data(data_path)

        # Check balance
        check_balance(df)

        # Run analysis
        results = analyze_experiment(df, link=args.link)

        # Generate report
        generate_report(results, args.output)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()