"""
Operators from signed distances
"""

import torch
import functools
import math

from .utils import *


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
    "rotate",
    "linear_extrusion",
    "rotational_extrusion",
    "rotational_twist",
    "expand_dim",
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
    """ Makes a shape annular with given thickness

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1]] * 2, [256] * 2)
    >>> s = shapes.onion(shapes.box([1., 0.75], p=3.5), 0.1)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX, p=3.5).item() < 0.05
    True
    """
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

def rotate(shape, theta, axis1=0, axis2=1):
    """
    Rotates a shape in the plane defined by the two given axis.

    Not exact for lp-norm with p != 2 and theta not a multiple of pi/2.

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1], [1, 1]], (256, 256))
    >>> s = shapes.rotate(shapes.box([0.7, 0.3]), 1.24, 0, 1)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX).item() < 0.05
    True
    """

    theta = torch.as_tensor(theta)

    def dist(*X):
        X = list(X)
        X[axis1], X[axis2] = (
            torch.cos(theta) * X[axis1] + torch.sin(theta) * X[axis2],
            torch.cos(theta) * X[axis2] - torch.sin(theta) * X[axis1],
        )
        return shape(*X)

    return dist

def linear_extrusion(shape, axis=0):
    """ Extrude a N-1 dimensional shape along the given axis

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1], [1, 1]], (256, 256))
    >>> s = shapes.linear_extrusion(shapes.sphere(0.1, [0.7]), 1)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX).item() < 0.05
    True
    """

    def dist(*X):
        idx = [slice(None) if i != axis else 0 for i in range(X[0].ndim)]
        d = shape(*(x[idx] for i, x in enumerate(X) if i != axis))
        return d.unsqueeze(axis).expand_as(X[0])

    return dist

def rotational_extrusion(shape, axis1=0, axis2=1, p=2):
    """
    Rotational extrusion of a N-1 dimensional shape in the given plane and around the origin.

    Norm of the projection on the given plane will be passed as the first argument to the shape.

    Parameters
    ----------
    shape: function
        Shape to be extruded
    axis1, axis2: int
        Dimension of the plane in which the rotation is made
    p: int or float
        The p in the lp norm

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1], [-1, 1]], (256, 256))
    >>> s = shapes.rotational_extrusion(shapes.sphere(0.1, [0.7]), 0, 1)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX).item() < 0.05
    True

    >>> domain = Domain([[-1, 1]] * 3, (64,) * 3)
    >>> s = shapes.rotational_extrusion(shapes.box([0.6, 0.3], p=3.5), 0, 2, p=3.5) # Cylinder
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX, p=3.5).item() < 0.05
    True
    """

    def dist(*X):
        X = [norm((X[axis1], X[axis2]), p=p)] + [x for i, x in enumerate(X) if i != axis1 and i != axis2]
        return shape(*X)

    return dist

def rotational_twist(shape, center, k, axis1=0, axis2=1):
    """
    Rotational extrusion of a twisted 2 dimensional shape

    Norm of the projection on the given plane will be passed as the first argument to the shape.
    Twist is applied as a rotation of k times the argument of (axis1, axis2)
    and around given center.

    Note1: only works in 3D and for 2D shape.
    Note2: the rotation of given shape of angle 2k\pi must return the same shape.
    Note3: this operation doesn't preserve the distance. Error increases with k.

    Parameters
    ----------
    shape: function
        A 2D shape to be extruded
    center: iteratable of float
        Rotation center of the 2D shape
    k: int or float
        Number of full rotation applied along the circle path
    axis1, axis2: int
        Dimension of the plane in which the rotation is made

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1]] * 3, (64,) * 3)
    >>> s = shapes.translation(
    ...     shapes.cross(radius=0.2, arms=4),
    ...     [0.7, 0.]
    ... )
    >>> s = shapes.rotational_twist(s, center=[0.7, 0], k=0.25)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX).item() < 0.05
    True
    """
    center = torch.as_tensor(center)
    assert len(center) == 2, "Center must be specified for the 2D shape"
    axis3 = 3 - axis1 - axis2

    def dist(*X):
        assert len(X) == 3, "Rotational twist works only in 3D"
        theta = torch.atan2(X[axis2], X[axis1])
        tcos = torch.cos(k * theta)
        tsin = torch.sin(k * theta)
        X = [norm((X[axis1], X[axis2])) - center[0], X[axis3] - center[1]]

        return shape(
            center[0] + (tcos * X[0] + tsin * X[1]),
            center[1] + (tcos * X[1] - tsin * X[0]),
        )

    return dist

def expand_dim(shape, axes, p=2):
    """
    Immerses a shape in a higher dimensional space

    Parameters
    ----------
    shape: function
        The shape
    axes: iterable of int
        Dimensions where the shape is defined
    p: int or float
        The p in the lp norm

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1]] * 3, (64,) * 3)
    >>> s = shapes.expand_dim(shapes.box([0.6, 0.3], p=3.5), [0, 2], p=3.5)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX, p=3.5).item() < 0.05
    True
    """

    def dist(*X):
        return norm(
            [torch.clamp(shape(*(X[i] for i in axes)), min=0.)]
            + [x for i, x in enumerate(X) if i not in axes],
            p=p
        )

    return dist

