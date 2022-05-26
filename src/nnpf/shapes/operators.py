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
    "twist",
    "fit",
    "point_reflection",
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

def scaling(shape, s, eps=1e-30):
    """ Scale shape by given factor """
    s = torch.max(torch.as_tensor(s), torch.as_tensor(eps))
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
        return shape(*(x for i, x in enumerate(X) if i != axis))

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

def twist(shape, w, param_axis=2, rot_axis1=0, rot_axis2=1):
    """
    Twist a shape along a given axis.

    Rotate the shape in the plane (rot_axis1, rot_axis2) by a rotation linearly
    dependant on the projection on axe param_axis.

    Note: not exact!

    Parameters
    ----------
    shape: function
        Shape to be twisted
    w: real
        Rotation speed
    param_axis: int
        Coordinate used as rotation parameter
    rot_axis1, rot_axis2: int
        Rotation plane

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1]] * 3, (64,) * 3)
    >>> s = shapes.linear_extrusion(
    ...     shapes.sphere(0.2, [0.5, 0.]),
    ...     axis=2
    ... )
    >>> s = shapes.twist(s, 0.9, 2, 0, 1)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX).item() < 0.05
    True
    """

    def dist(*X):
        theta = X[param_axis] * w
        tcos = torch.cos(theta)
        tsin = torch.sin(theta)
        XX = [x for x in X]
        XX[rot_axis1] = tcos * X[rot_axis1] + tsin * X[rot_axis2]
        XX[rot_axis2] = tcos * X[rot_axis2] - tsin * X[rot_axis1]
        return shape(*XX)

    return dist

def fit(shape, from_bounds, to_bounds):
    """
    Scale and translate a shape accordingly to a bounds mapping.

    Scale is calculated in order to keep aspect ratio of the given shape
    and so that transformed input domain is included in output domain.

    Parameters
    ----------
    shape: function
        Shape to be fitted
    from_domain, to_domain: iterable of pair of float
        Input and output bounds (iterable of bounds along each dimension)

    Example
    -------
    >>> from nnpf import shapes
    >>> s = shapes.fit(
    ...     shapes.sphere(0.2, [0.5, 0.]),
    ...     [[-1, 1]] * 2,
    ...     [[0, 4], [0, 6]],
    ... )
    >>> torch.allclose(
    ...     s(torch.tensor(3.), torch.tensor(3.)),
    ...     torch.tensor(-0.4),
    ... )
    True

    >>> from nnpf.domain import Domain
    >>> domain = Domain([[0, 4], [0, 6]] , (256,) * 2)
    >>> dist = s(*domain.X)
    >>> check_dist(dist, domain.dX).item() < 0.05
    True
    """

    from_extent = [b - a for a, b in from_bounds]
    to_extent = [b - a for a, b in to_bounds]
    from_center = [(a + b) / 2 for a, b in from_bounds]
    to_center = [(a + b) / 2 for a, b in to_bounds]
    assert len(from_extent) == len(to_extent), "Input and output bounds must be of same dimension"

    scale = min(t / f for f, t in zip(from_extent, to_extent))
    shift = [t - f * scale for f, t in zip(from_center, to_center)]

    return translation(scaling(shape, scale), shift)

def point_reflection(shape, dims=None, center=None):
    """
    Apply a point reflection to a given shape.

    Parameters
    ----------
    shape: function
        Shape to be modified
    dims: None or iterable of int
        Dimensions on which reflection will be applied (all dimensions if None)
    center: None or iterable of float
        Center of reflection

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf import shapes
    >>> domain = Domain([[-1, 1], [0, 2]], [256, 256])
    >>> s = shapes.point_reflection(
    ...     shapes.translation(
    ...         shapes.regular_polygon(3, outer_radius=0.7),
    ...         [0, 1],
    ...     ), center=[0, 1])
    >>> ref_s = shapes.translation(
    ...     shapes.regular_polygon(3, outer_radius=0.7, phase=0.5),
    ...     [0, 1],
    ... )
    >>> torch.allclose(s(*domain.X), ref_s(*domain.X), atol=1e-7)
    True
    """

    def dist(*X):
        curr_dims = dims or list(range(len(X)))
        curr_center = center or [0.] * len(X)
        assert len(curr_dims) <= len(X), "Cannot have more dims than space dimension!"
        assert len(curr_center) == len(X), "Center point should have same dimension than the space!"
        return shape(*(2 * c - x if d in curr_dims else x for d, (x, c) in enumerate(zip(X, curr_center))))

    return dist

