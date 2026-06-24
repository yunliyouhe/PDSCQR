import numpy as np
from math import sqrt, pi
from scipy.special import ndtr

SQRT2PI = sqrt(2.0 * pi)


def gaussian_kernel(u):
    """Gaussian kernel K(u)=phi(u)."""
    return np.exp(-0.5 * u * u) / SQRT2PI


def gaussian_cdf(u):
    """Gaussian kernel distribution function."""
    return ndtr(u)


def local_bw(nk: int, d: int, l: float = 1.0) -> float:
    """Local bandwidth h_k = l (d/n_k)^(2/5)."""
    return l * (d / nk) ** (2.0 / 5.0)


def central_bw(n1: int, d: int, l: float = 1.0) -> float:
    """Central bandwidth h_1 = l (d log n_1/n_1)^(1/3)."""
    return l * (d * np.log(n1) / n1) ** (1.0 / 3.0)
