import numpy as np
import pandas as pd


def check_loss(u, tau):
    """Quantile check loss."""
    return u * (tau - (u < 0))


def check_loss_grid(alpha, test_blocks, taus, M):
    """Average check loss at each composite quantile level."""
    beta = alpha[M:]
    out = []

    for mi, tau in enumerate(taus):
        num = 0.0
        den = 0

        for _, Xte, Yte in test_blocks:
            u = Yte - alpha[mi] - Xte @ beta
            num += check_loss(u, tau).sum()
            den += len(Yte)

        out.append(num / den)

    return np.asarray(out)


def build_intercept_table(method_alpha, taus, M, response_label):
    rows = []
    for method, alpha in method_alpha.items():
        for i, tau in enumerate(taus):
            intercept = float(alpha[i])
            rows.append({
                "method": method,
                "tau": float(tau),
                "intercept": intercept,
                f"{response_label}_back_transformed": float(np.exp(intercept) - 1.0),
            })
    return pd.DataFrame(rows)


def build_check_loss_table(method_alpha, test_blocks, taus, M):
    rows = []
    for method, alpha in method_alpha.items():
        losses = check_loss_grid(alpha, test_blocks, taus, M)
        for tau, loss in zip(taus, losses):
            rows.append({
                "method": method,
                "tau": float(tau),
                "check_loss": float(loss),
            })
    return pd.DataFrame(rows)


def build_slope_table_for_beijing(covariate_names, M, results):
    """
    Slope coefficient table for Beijing only.

    The ACS slope-coefficient table is intentionally not generated in this repository.
    """
    rows = []
    beta_g, lo_g, hi_g = results["beta_g"], results["lo_g"], results["hi_g"]
    beta_c, lo_c, hi_c = results["beta_c"], results["lo_c"], results["hi_c"]
    beta_csl = results["beta_csl"]
    beta_dp = results["beta_dp"]

    for j, cov in enumerate(covariate_names):
        rows.append({
            "covariate": cov,
            "gscqr": float(beta_g[j]),
            "gscqr_ci_lower": float(lo_g[j]),
            "gscqr_ci_upper": float(hi_g[j]),
            "csl_csqr": float(beta_csl[j]),
            "cst_dcqr": float(beta_c[j]),
            "cst_dcqr_ci_lower": float(lo_c[j]),
            "cst_dcqr_ci_upper": float(hi_c[j]),
            "dp_cst_eps_0.5": float(beta_dp[0.5][j]),
            "dp_cst_eps_1.0": float(beta_dp[1.0][j]),
            "dp_cst_eps_5.0": float(beta_dp[5.0][j]),
        })

    return pd.DataFrame(rows)


def build_distance_table(results):
    return pd.DataFrame([{
        "distance_cst_to_gscqr": float(results["dist_cst"]),
        "distance_csl_to_gscqr": float(results["dist_csl"]),
        "ratio_csl_over_cst": float(results["dist_csl"] / results["dist_cst"]),
    }])
