import numpy as np

from .kernels import gaussian_kernel, gaussian_cdf, local_bw


def smoothed_grad_hess(X, Y, b, beta, taus, h, need_hess=True):
    """
    Smoothed composite quantile gradient and Hessian.

    Composite parameter:
        alpha = (b_tau1, ..., b_tauM, beta_1, ..., beta_p).

    X has no intercept column because quantile-specific intercepts are included in b.
    """
    n, p = X.shape
    M = len(taus)

    Xb = X @ beta
    S = b[None, :] + (Xb - Y)[:, None]

    Kt = gaussian_cdf(S / h)
    score = Kt - taus[None, :]

    g_b = score.mean(axis=0) / M
    g_beta = (X.T @ score.sum(axis=1)) / (n * M)
    grad = np.concatenate([g_b, g_beta])

    if not need_hess:
        return grad, None

    Kh = gaussian_kernel(S / h) / h

    H_bb = np.diag(Kh.mean(axis=0) / M)
    H_beta_b = (X.T @ Kh) / (n * M)

    w = Kh.sum(axis=1)
    H_beta_beta = (X * w[:, None]).T @ X / (n * M)

    d = M + p
    H = np.zeros((d, d))
    H[:M, :M] = H_bb
    H[:M, M:] = H_beta_b.T
    H[M:, :M] = H_beta_b
    H[M:, M:] = H_beta_beta

    return grad, H


def fit_local_scqr(X, Y, taus, h, alpha0=None, maxit=100, tol=1e-8):
    """Fit a local smoothed composite quantile regression estimator."""
    n, p = X.shape
    M = len(taus)
    d = M + p

    if alpha0 is None:
        alpha = np.concatenate([np.quantile(Y, taus), np.zeros(p)])
    else:
        alpha = np.asarray(alpha0, dtype=float).copy()

    for _ in range(maxit):
        grad, H = smoothed_grad_hess(X, Y, alpha[:M], alpha[M:], taus, h, True)
        H = H + 1e-8 * np.eye(d)

        step = np.linalg.solve(H, grad)
        step_norm = np.linalg.norm(step)
        grad_norm = np.linalg.norm(grad)

        t = 1.0
        for _ in range(30):
            new_alpha = alpha - t * step
            new_grad, _ = smoothed_grad_hess(
                X, Y, new_alpha[:M], new_alpha[M:], taus, h, False
            )
            if np.linalg.norm(new_grad) <= (1 - 0.1 * t) * grad_norm or np.linalg.norm(new_grad) < tol:
                break
            t *= 0.5

        alpha = new_alpha

        if step_norm < tol:
            break

    return alpha


def central_hessian_at(Xc, Yc, alpha_anchor, taus, h1):
    M = len(taus)
    _, H = smoothed_grad_hess(
        Xc, Yc, alpha_anchor[:M], alpha_anchor[M:], taus, h1, True
    )
    return H


def cst_aggregate(anchors, sizes, Xc, Yc, taus, h1):
    """CST-DCQR aggregation."""
    d = anchors.shape[1]
    A = np.zeros((d, d))
    rhs = np.zeros(d)

    for k in range(anchors.shape[0]):
        Hk = central_hessian_at(Xc, Yc, anchors[k], taus, h1)
        A += sizes[k] * Hk
        rhs += sizes[k] * (Hk @ anchors[k])

    A += 1e-8 * np.eye(d)
    return np.linalg.solve(A, rhs), A


def global_grad_at(alpha, train_blocks, taus, l=1.0):
    """Weighted average of local gradients at a common expansion point."""
    d = len(alpha)
    M = len(taus)
    grad = np.zeros(d)
    N = 0

    for _, Xk, Yk in train_blocks:
        hk = local_bw(Xk.shape[0], d, l)
        gk, _ = smoothed_grad_hess(Xk, Yk, alpha[:M], alpha[M:], taus, hk, False)
        grad += Xk.shape[0] * gk
        N += Xk.shape[0]

    return grad / N
