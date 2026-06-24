import numpy as np

from .estimators import central_hessian_at


def omega_hat(Xc, taus):
    """
    Plug-in long-run variance component for the composite score.

    The block structure uses W_im=(e_m, X_i^T)^T.
    """
    n, p = Xc.shape
    M = len(taus)
    d = M + p

    C = np.minimum.outer(taus, taus) - np.outer(taus, taus)

    Om = np.zeros((d, d))

    Om[:M, :M] = n * C

    rowsum = C.sum(axis=1)
    Xsum = Xc.sum(axis=0)

    Om[:M, M:] = np.outer(rowsum, Xsum)
    Om[M:, :M] = Om[:M, M:].T

    Om[M:, M:] = C.sum() * (Xc.T @ Xc)

    return Om / (n * M * M)


def plugin_cov(alpha, Xc, Yc, taus, h1, N_total):
    """Plug-in covariance for the estimator."""
    d = len(alpha)
    H = central_hessian_at(Xc, Yc, alpha, taus, h1) + 1e-8 * np.eye(d)
    H_inv = np.linalg.inv(H)
    Sig = H_inv @ omega_hat(Xc, taus) @ H_inv
    return Sig / N_total


def slope_ci(alpha, cov, M, z=1.96):
    """Return beta estimates and Wald confidence intervals."""
    beta = alpha[M:]
    se = np.sqrt(np.maximum(np.diag(cov)[M:], 0.0))
    return beta, beta - z * se, beta + z * se
