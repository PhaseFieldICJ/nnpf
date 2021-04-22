"""
Operators from signed distances
"""

import torch
import functools


__all__ = [
    "reduce",
    "reverse",
    "unsign",
    "union",
    "intersection",
    "subtraction",
    "symmetric_difference",
    "translation",
    "scaling",
    "rounding",
    "periodic",
    "replicate",
    "onion",
    "elongate",
    "displace",
    "transform",
]


def reduce(op, *shapes):
    """ Reduce an operator over multiple shapes """
    def dist(*X):
        return functools.reduce(op, (s(*X) for s in shapes))

    return dist

def reverse(shape):
    """ Reverse interior/exterior of given shape """
    def dist(*X):
        return -shape(*X)
    return dist

def unsign(shape):
    """ Return absolute value of distance to the given shape """
    def dist(*X):
        return shape(*X).abs()
    return dist

def union(*shapes):
    """ Union of shapes (not exact in the interior) """
    return reduce(torch.min, *shapes)

def intersection(*shapes):
    """ Intersection of shapes (not exact) """
    return reduce(torch.max, *shapes)

def subtraction(*shapes):
    """ First shape substracted by the other shapes (not exact) """
    def op(a, b):
        return torch.max(a, -b)

    return reduce(op, *shapes)
    #return intersection(reverse(shapes[0]), union(*shapes[1:))

def symmetric_difference(*shapes):
    """ Symmetric difference of given shapes (not exact) """
    def op(a, b):
        return torch.max(torch.min(a, b), -torch.max(a, b))

    return reduce(op, *shapes)
    """
    if len(shapes) == 1:
        return shapes[0]
    else:
        return symmetric_difference(
            subtraction(
                union(shapes[0], shapes[1]),
                intersection(shapes[0], shapes[1])),
            *shapes[2:])
    """

def translation(shape, shift):
    """ Translation of a shape (exact) """
    def dist(*X):
        return shape(*(X[i] - shift[i] for i in range(len(X))))

    return dist

def scaling(shape, s):
    """ Scale shape by given factor """
    def dist(*X):
        return shape(*(x / s for x in X)) * s

    return dist

def rounding(shape, radius):
    """ Rounds a shape by given radius (shift distance outward) """
    def dist(*X):
        return shape(*X) - radius

    return dist

def periodic(shape, bounds):
    """ Periodicize a shape using given bounds

    None bound means no periodicity.
    """

    def shift_gen(shift=[]):
        """ Generates all combinations of shift of the current window """
        i = len(shift)
        if i == len(bounds):
            yield shift
        elif bounds[i] is None:
            yield from shift_gen(shift + [0.])
        else:
            for sign in (-1., 0., 1.):
                yield from shift_gen(shift + [sign * (bounds[i][1] - bounds[i][0])])

    return union(*(translation(shape, shift) for shift in shift_gen()))

def replicate(shape, periods, limits=None):
    """ Replicate a shape with given periods (None for no replication) and limits """
    limits = limits or [None] * len(periods)

    def X_transform(*X):
        for x, p, l in zip(X, periods, limits):
            if l is None:
                yield torch.remainder(x + 0.5 * p, p) - 0.5 * p if p is not None else x
            else:
                ll = (l - 1) // 2
                rl = l - 1 - ll
                yield x - p * torch.clamp(torch.round(x / p), -ll, rl)

    def dist(*X):
        return shape(*X_transform(*X))

    return dist

def onion(shape, thickness):
    """ Makes a shape annular with given thickness """
    def dist(*X):
        return shape(*X).abs() - thickness

    return dist

def elongate(shape, sizes):
    """ Elonge a shape in each direction with given sizes """
    def dist(*X):
        q = torch.stack([x.abs() - s/2 for x, s in zip(X, sizes)])
        z = q.new_zeros(1)
        return shape(*torch.max(q, z)) + torch.min(q.max(dim=0).values, z)

    return dist

def displace(shape, displacement):
    """ Displacement of a shape """
    def dist(*X):
        return shape(*X) + displacement(*X)

    return dist

def transform(shape, t):
    """
    Affine transformation of a shape.

    Transformation matrix t should be of size d*d
    or d*(d+1) where the last column is a translation.
    """
    dim = t.shape[0]
    if t.shape[-1] == dim + 1:
        translation = t[:, -1]
    else:
        translation = t.new_zeros(dim)

    t = torch.inverse(t[:dim, :dim])

    def dist(*X):
        return shape(*(sum(X[j] * t[i, j] for j in range(dim)) - translation[i] for i in range(dim)))

    return dist

