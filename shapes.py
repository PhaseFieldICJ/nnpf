"""
Signed distances functions and tools

Example
-------

Centered sphere:
>>> X, Y = torch.meshgrid(torch.linspace(0, 1, 5), torch.linspace(0, 1, 5))
>>> s = sphere(0.25, [0.5, 0.5])
>>> s(X, Y)
tensor([[ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.2500,  0.0000, -0.2500,  0.0000,  0.2500],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571]])

Shifted sphere with periodicity:
>>> sp = periodic(sphere(0.25, [0.75, 0.25]), [(0, 1), (0, 1)])
>>> sp(X, Y)
tensor([[ 0.1036,  0.0000,  0.1036,  0.3090,  0.1036],
        [ 0.3090,  0.2500,  0.3090,  0.4571,  0.3090],
        [ 0.1036,  0.0000,  0.1036,  0.3090,  0.1036],
        [ 0.0000, -0.2500,  0.0000,  0.2500,  0.0000],
        [ 0.1036,  0.0000,  0.1036,  0.3090,  0.1036]])

Distance calculation for multiple spheres at once:
>>> radius = torch.rand(10, 1, 1)
>>> center = torch.rand(2, 10, 1, 1)
>>> sm = sphere(radius, center)
>>> sm(X, Y).shape
torch.Size([10, 5, 5])

Distance to two spheres:
>>> X, Y = torch.meshgrid(torch.linspace(0, 2, 9), torch.linspace(0, 1, 5))
>>> s2 = union(s, translation(s, (1, 0)))
>>> s2(X, Y)
tensor([[ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.2500,  0.0000, -0.2500,  0.0000,  0.2500],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.2500,  0.0000, -0.2500,  0.0000,  0.2500],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571]])
"""

import torch
from torch.distributions.utils import broadcast_all

###############################################################################
# Shapes

def dot(p=2):
    """ Signed lp distance to a dot """

    if p == float("inf"):
        def dist(*X):
            result = X[0].abs()
            for x in X[1:]:
                result = torch.max(result, x.abs())
            return result

    else:
        # TODO: if needed, could be optimized for 1 and even power
        def dist(*X):
            return sum(x.abs().pow(p) for x in X).pow(1 / p)

    return dist

def sphere(radius, center=None, p=2):
    """ Signed distance to a sphere """
    if center is None:
        return rounding(dot(p), radius)
    else:
        return translation(sphere(radius, p=p), center)

def box(sizes):
    """ Signed distance to a box """
    def dist(*X):
        assert len(X) == len(sizes), "Box & coords dimensions do not match!"
        # FIXME: better way than using broadcast_all!!!
        q = torch.stack(broadcast_all(*[x.abs() - s/2 for x, s in zip(X, sizes)]))
        z = q.new_zeros(1)
        return torch.max(q, z).norm(dim=0) + torch.min(q.max(dim=0).values, z)

    return dist

def half_plane(dim, shift=0.):
    """ Signed distance to the half plane with normal along dim """
    def dist(*X):
        return sum((dim == i) * (x - shift) for i, x in enumerate(X))

    return dist

def beam(dim, thickness):
    """ Signed distance to the infinite beam with normal along dim """
    return subtraction(half_plane(dim, thickness / 2), half_plane(dim, -thickness / 2))

###############################################################################
# Operations

def reduce(op, *shapes):
    """ Reduce an operator over multiple shapes """
    def dist(*X):
        result = shapes[0](*X)
        for s in shapes[1:]:
            result = op(result, s(*X)) # TODO: using out parameters instead of returning
        return result

    return dist

def reverse(shape):
    """ Reverse interior/exterior of given shape """
    def dist(*X):
        return -shape(*X)

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

def display(shape_or_dist, X=None, scale=1., extent=None, return_image=False):
    """ Display a 2D shape or distance function.

    Parameters
    ----------
    shape_or_dist: shape or torch.tensor
        Shape definition (X needed) or directly the signed distance field
    X: tuple or None
        If shape_or_dist is a shape, X are the point coordinates
    scale: real
        Scale of the visualization
    extent: tuple/list or None
        Domain extent. If None, calculated from X (if given)
    return_image: bool
        If True, don't display the distance function and returns the image instead

    Example
    -------
    >>> from domain import Domain
    >>> d = Domain([[-1, 1], [-1, 1]], [256, 256])
    >>> s = periodic(union(sphere(0.5, [0, 0]), sphere(0.3, [0.4, 0.3])), d.bounds)
    >>> display(s, d.X)
    """

    import warnings
    import visu

    if return_image:
        warnings.warn("You should use visu.distance_to_img instead", FutureWarning)
        return visu.distance_to_img(shape_or_dist, X, scale, extent)
    else:
        warnings.warn("You should use visu.DistanceShow instead", FutureWarning)
        import matplotlib.pyplot as plt
        visu.DistanceShow(shape_or_dist, X, scale, extent)
        plt.show()

