"""
Tools for signed distance
"""

import functools
import torch


__all__ = [
    "norm",
    "dot_product",
    "check_dist",
]


def norm(X, p=2, weights=None):
    """
    Vector lp norm

    Also works on generator to allow some optmization
    (avoid stacking components before calculating the norm).

    Parameters
    ----------
    X: iterable
        Iterable over vector components
    p: int, float or inf
        p of the l-p norm
    weights: iterable of flaot
        scaling divider for each coordinate
    """
    if weights is not None:
        X = [x / w for x, w in zip(X, weights)]

    if p == float("inf"):
        return functools.reduce(
            lambda x, y: torch.max(x, y), # prefer torch.maximum in PyTorch 1.7
            (x.abs() for x in X))

    else:
        # TODO: if needed, could be optimized for 1 and even power
        return sum(x.abs().pow(p) for x in X).pow(1 / p)


def dot_product(A, B):
    """
    Returns the dot product along first axis.
    """
    return sum(a * b for a, b in zip(A, B))


def check_dist(shape, domain, p=2):
    """
    Returns the norm of the gradient of the signed distance using the dual norm of lp (ie l^{p/(p-1)}).

    Should be equal to one everywhere except on the shape skeleton.

    TODO: take into account the domain periodicity
    """
    dist = shape(*domain.X)
    ddist = []
    for i in range(domain.dim):
        down_slice = [slice(1, -1)] * domain.dim
        down_slice[i] = slice(0, -2)
        up_slice = [slice(1, -1)] * domain.dim
        up_slice[i] = slice(2, None)
        ddist.append(
            (dist[tuple(up_slice)] - dist[tuple(down_slice)]) / (2 * domain.dX[i])
        )

    if p == float("inf"):
        dp = 1
    elif p == 1:
        dp = float("inf")
    else:
        dp = p / (p - 1)

    return norm(ddist, dp)

