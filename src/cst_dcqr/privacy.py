import numpy as np

from .estimators import cst_aggregate, central_hessian_at


def estimate_privacy_scale_parameters(train_blocks, alpha_cst, taus, h1, M, p):
    """Estimate C_W and mu from the data and the central Hessian."""
    X_blocks = [b[1] for b in train_blocks]
    C_X = float(np.max([np.abs(X).max() for X in X_blocks]))
    C_W = max(1.0, C_X)

    Xc, Yc = train_blocks[0][1], train_blocks[0][2]
    H_at_cst = central_hessian_at(Xc, Yc, alpha_cst, taus, h1)
    eig = np.linalg.eigvalsh(H_at_cst)
    mu = 1.0 / max(np.median(eig), 1e-3)

    return C_W, mu


def dp_cst_average(
    alpha_locals,
    sizes,
    Xc,
    Yc,
    taus,
    h1,
    eps,
    C_W,
    mu,
    M,
    p,
    rng,
    n_rep=40,
):
    """
    DP-CST-DCQR point estimate averaged across repeated Laplace perturbations.

    Noise scale: 2 C_W (M+p) mu / (n_k eps).
    """
    d = M + p
    estimates = []

    for _ in range(n_rep):
        noisy = alpha_locals.copy()
        for k in range(alpha_locals.shape[0]):
            scale = 2.0 * C_W * d * mu / (sizes[k] * eps)
            noisy[k] = alpha_locals[k] + rng.laplace(0.0, scale, size=d)

        alpha_dp, _ = cst_aggregate(noisy, sizes, Xc, Yc, taus, h1)
        estimates.append(alpha_dp)

    return np.mean(estimates, axis=0)
