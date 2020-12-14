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
import functools

###############################################################################
# Tools

def norm(X, p=2):
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
    """
    if p == float("inf"):
        return functools.reduce(
            lambda x, y: torch.max(x, y), # prefer torch.maximum in PyTorch 1.7
            (x.abs() for x in X))

    else:
        # TODO: if needed, could be optimized for 1 and even power
        return sum(x.abs().pow(p) for x in X).pow(1 / p)


###############################################################################
# Shapes

def dot(p=2):
    """ Signed lp distance to a dot """
    def dist(*X):
        return norm(X, p)

    return dist

def sphere(radius, center=None, p=2):
    """ Signed distance to a sphere """
    if center is None:
        return rounding(dot(p), radius)
    else:
        return translation(sphere(radius, p=p), center)

def box(sizes, p=2):
    """ Signed distance to a box """
    return elongate(dot(p), sizes)

def half_plane(dim_or_normal, pt_or_shift=0., normalize=False):
    """
    Signed distance to the half plane with given normal

    Parameters
    ----------
    dim_or_normal: int or iterable
        If integer, dimension orthogonal to the plane.
        If iterable, normal vector to the plane.
    pt_or_shift: real or iterable
        If real, distance of the line to the origin (factor of the normal's norm)
        If iterable, a point that lies on the line
    normalize: bool
        True to normalize the given normal before applying the shift
    """
    pt_or_shift = torch.as_tensor(pt_or_shift)
    dim_or_normal = torch.as_tensor(dim_or_normal)

    # Integer dimension
    if dim_or_normal.ndim == 0:
        if pt_or_shift.ndim > 0:
            pt_or_shift = pt_or_shift[dim_or_normal]

        def dist(*X):
            return sum((dim_or_normal == i) * x for i, x in enumerate(X)) - pt_or_shift

    # Normal
    else:
        if pt_or_shift.ndim == 0 and not normalize:
            pt_or_shift = pt_or_shift * dim_or_normal.norm()

        normal = torch.nn.functional.normalize(dim_or_normal, p=2, dim=0)

        if pt_or_shift.ndim > 0:
            pt_or_shift = pt_or_shift.dot(normal)

        def dist(*X):
            return sum(n * x for n, x in zip(normal, X)) - pt_or_shift

    return dist

def beam(dim_or_normal, thickness, pt_or_shift=0., normalize=False):
    """
    Signed distance to the infinite beam with normal along dim

    Parameters
    ----------
    dim_or_normal: int or iterable
        If integer, dimension orthogonal to the plane.
        If iterable, normal vector to the plane.
    thickness: real
        thickness of the beam
    pt_or_shift: real or iterable
        If real, distance of the center line to the origin (factor of the normal's norm)
        If iterable, a point that lies on the center line
    normalize: bool
        True to normalize the given normal before applying the shift
    """
    return rounding(line(dim_or_normal, pt_or_shift, normalize), thickness / 2)

def line(dim_or_normal, pt_or_shift=0., normalize=False):
    """
    Distance to a line

    Parameters
    ----------
    dim_or_normal: int or iterable
        If integer, dimension orthogonal to the plane.
        If iterable, normal vector to the plane.
    pt_or_shift: real or iterable
        If real, distance of the line to the origin (factor of the normal's norm)
        If iterable, a point that lies on the line
    normalize: bool
        True to normalize the given normal before applying the shift
    """
    return unsign(half_plane(dim_or_normal, pt_or_shift, normalize))

def segment(a, b):
    """ Segment between two points """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    ab = b - a
    dot_ab = torch.dot(ab, ab)

    def dist(*X):
        h = torch.clamp(sum((X[i] - a[i]) * ab[i] for i in range(len(X))) / dot_ab, 0., 1.)
        return norm((X[i] - a[i] - ab[i] * h for i in range(len(X))), p=2)

    return dist

def capsule(a, b, thickness):
    """ Capsule between two points """
    return rounding(segment(a, b), thickness / 2)


###############################################################################
# Operations

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


