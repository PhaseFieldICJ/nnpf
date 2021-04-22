"""
Tools for signed distance
"""

import functools
import torch

__all__ = [
    "norm",
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

