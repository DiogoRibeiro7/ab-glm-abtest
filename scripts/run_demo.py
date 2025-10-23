from __future__ import annotations

from ab_glm.pipeline import run_pipeline


def pretty_print(res) -> None:
    print(f"\n=== Binomial GLM ({res.link}) for A/B test ===")
    print(f"Observations: {res.n_obs:,}    Users (clusters): {res.n_users:,}")
    print(f"Baseline (control) p̂: {res.p_control:.4f}")
    print(f"Treated p̂:          {res.p_treated:.4f}")
    print(f"ATE (RD, abs. lift): {res.ate_rd:+.4f}")
    print(f"Risk ratio (RR):     {res.rr:.4f}")
    print(f"Brier score:         {res.brier:.4f}")
    if res.coef_treat is not None:
        line = f"Treat coef (link):   {res.coef_treat:+.3f}"
        if res.robust_se_treat is not None:
            line += f"  (cluster-robust SE: {res.robust_se_treat:.3f})"
        print(line)
    print("Note: ATE/RR computed via marginal predictions (covariate-adjusted).")


if __name__ == "__main__":
    res_logit = run_pipeline(link="logit")
    pretty_print(res_logit)

    res_probit = run_pipeline(link="probit")
    pretty_print(res_probit)
