import numpy as np

from .kernels import local_bw, central_bw
from .estimators import fit_local_scqr, cst_aggregate, global_grad_at, central_hessian_at
from .covariance import plugin_cov, slope_ci
from .privacy import estimate_privacy_scale_parameters, dp_cst_average


def run_cst_dcqr_analysis(train_blocks, test_blocks, M, eps_list, seed):
    """
    Run GSCQR, CSL-CSQR, CST-DCQR, and DP-CST-DCQR on prepared train/test blocks.

    Each train block is a tuple:
        (name, X_train, y_train).
    """
    rng = np.random.default_rng(seed)

    n_sizes = np.asarray([b[1].shape[0] for b in train_blocks], dtype=int)
    N_train = int(n_sizes.sum())
    N_test = int(sum(b[1].shape[0] for b in test_blocks))

    p = train_blocks[0][1].shape[1]
    d = M + p
    taus = np.arange(1, M + 1) / (M + 1.0)

    Xc = train_blocks[0][1]
    Yc = train_blocks[0][2]
    n1 = Xc.shape[0]
    h1 = central_bw(n1, d, 1.0)

    print(f"K={len(train_blocks)} p={p} M={M} d={d}")
    print(f"N_train={N_train:,} N_test={N_test:,} central={train_blocks[0][0]} n1={n1}")

    # Local estimators
    alpha_locals = []
    for name, Xk, Yk in train_blocks:
        hk = local_bw(Xk.shape[0], d, 1.0)
        alpha_locals.append(fit_local_scqr(Xk, Yk, taus, hk))
    alpha_locals = np.asarray(alpha_locals)

    # GSCQR pooled benchmark
    Xall = np.vstack([b[1] for b in train_blocks])
    Yall = np.concatenate([b[2] for b in train_blocks])
    h_global = local_bw(N_train, d, 1.0)
    alpha_gscqr = fit_local_scqr(
        Xall, Yall, taus, h_global, alpha0=alpha_locals.mean(axis=0)
    )

    # CST-DCQR
    alpha_cst, _ = cst_aggregate(alpha_locals, n_sizes, Xc, Yc, taus, h1)

    # CSL-CSQR
    alpha_expand = alpha_locals[0]
    g_global = global_grad_at(alpha_expand, train_blocks, taus)
    H_central = central_hessian_at(Xc, Yc, alpha_expand, taus, h1) + 1e-8 * np.eye(d)
    alpha_csl = alpha_expand - np.linalg.solve(H_central, g_global)

    # DP-CST-DCQR
    C_W, mu = estimate_privacy_scale_parameters(train_blocks, alpha_cst, taus, h1, M, p)
    alpha_dp = {}
    for eps in eps_list:
        alpha_dp[float(eps)] = dp_cst_average(
            alpha_locals=alpha_locals,
            sizes=n_sizes,
            Xc=Xc,
            Yc=Yc,
            taus=taus,
            h1=h1,
            eps=float(eps),
            C_W=C_W,
            mu=mu,
            M=M,
            p=p,
            rng=rng,
            n_rep=40,
        )

    # Covariance and slope CI for GSCQR / CST-DCQR
    cov_g = plugin_cov(alpha_gscqr, Xall, Yall, taus, h_global, N_train)
    cov_c = plugin_cov(alpha_cst, Xc, Yc, taus, h1, N_train)

    beta_g, lo_g, hi_g = slope_ci(alpha_gscqr, cov_g, M)
    beta_c, lo_c, hi_c = slope_ci(alpha_cst, cov_c, M)
    beta_csl = alpha_csl[M:]
    beta_dp = {float(eps): alpha_dp[float(eps)][M:] for eps in eps_list}

    dist_cst = float(np.linalg.norm(beta_c - beta_g))
    dist_csl = float(np.linalg.norm(beta_csl - beta_g))

    method_alpha = {
        "GSCQR": alpha_gscqr,
        "CSL-CSQR": alpha_csl,
        "CST-DCQR": alpha_cst,
    }
    for eps in eps_list:
        method_alpha[f"DP-CST-DCQR_eps_{float(eps)}"] = alpha_dp[float(eps)]

    results = {
        "taus": taus,
        "M": M,
        "p": p,
        "d": d,
        "n_sizes": n_sizes,
        "N_train": N_train,
        "N_test": N_test,
        "n1": n1,
        "alpha_gscqr": alpha_gscqr,
        "alpha_csl": alpha_csl,
        "alpha_cst": alpha_cst,
        "alpha_dp": alpha_dp,
        "method_alpha": method_alpha,
        "beta_g": beta_g,
        "lo_g": lo_g,
        "hi_g": hi_g,
        "beta_c": beta_c,
        "lo_c": lo_c,
        "hi_c": hi_c,
        "beta_csl": beta_csl,
        "beta_dp": beta_dp,
        "dist_cst": dist_cst,
        "dist_csl": dist_csl,
        "C_W": C_W,
        "mu": mu,
    }

    return results
