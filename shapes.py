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
import math

###############################################################################
# Tools

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


###############################################################################
# Shapes

def dot(p=2, weights=None):
    """ Signed lp distance to a dot """
    def dist(*X):
        return norm(X, p, weights)

    return dist

def sphere(radius, center=None, p=2, weights=None):
    """ Signed distance to a sphere """
    if center is None:
        return rounding(dot(p, weights), radius)
    else:
        return translation(sphere(radius, p=p, weights=weights), center)

def ellipse(radius):
    """ Signed distance to an ellipse (2D only)

    Source: https://iquilezles.org/www/articles/ellipsedist/ellipsedist.htm

    Doesn't work well due to loss of precision during computation aroung the axe of the great radius.
    """

    def msign(v):
        s = torch.empty_like(v)
        mask = v < 0.
        s[mask] = -1.
        s[~mask] = 1.
        return s

    def dist(*X):
        assert len(radius) == 2 and len(X) == 2, "Signed distance to an ellipse is defined in 2D only!"

        a, b = radius
        # Symmetries
        X = [torch.abs(x) for x in X]
        if a > b:
            X = X[::-1]
            a, b = b, a

        l = b**2 - a**2
        m = a * X[0] / l
        n = b * X[1] / l
        n2 = n**2
        m2 = m**2

        c = (m2 + n2 - 1.) / 3.
        c3 = c**3

        d = c3 + m2 * n2
        q = d + m2 * n2
        g = m + m * n2

        co = torch.empty_like(X[0])

        # if d < 0
        mask = d < 0.
        h = torch.acos(torch.clamp(q[mask] / c3[mask], max=1.)) / 3.
        s = torch.cos(h) + 2.
        t = torch.sin(h) * math.sqrt(3.)
        rx = torch.sqrt(m2[mask] - c[mask] * (s + t))
        ry = torch.sqrt(m2[mask] - c[mask] * (s - t))
        co[mask] = ry + math.copysign(1, l) * rx + torch.abs(g[mask]) / (rx * ry)

        # if d >= 0
        mask = ~mask
        h = 2. * m[mask] * n[mask] * torch.sqrt(d[mask])
        s = msign(q[mask] + h) * (q[mask] + h).abs().pow(1/3)
        t = msign(q[mask] - h) * (q[mask] - h).abs().pow(1/3)
        rx = - (s + t) - c[mask] * 4. + 2. * m2[mask]
        #ry = (s - t) * math.sqrt(3.)
        ry = torch.clamp(2*h - 3*s*s*t + 3*s*t*t, min=0.)**(1/3) * math.sqrt(3.)
        rm = torch.sqrt(rx**2 + ry**2)
        co[mask] = ry / torch.sqrt(rm - rx) + 2. * g[mask] / rm
        #print("s = ", s)
        #print("t = ", t)
        #print("rx = ", rx)
        #print("ry = ", ry)
        #print("h = ", h)
        #print("n = ", n[mask])
        #print("ryy = ", torch.clamp(2*h - 3*s*s*t + 3*s*t*t, min=0.)**(1/3) * math.sqrt(3.))

        co = (co - m) / 2.
        si = torch.sqrt(torch.clamp(1. - co**2, min=0.))
        r = [a * co, b * si]
        return torch.sqrt((r[0] - X[0])**2 + (r[1] - X[1])**2) * msign(X[1] - r[1])

    return dist


def box(sizes, p=2, weights=None):
    """ Signed distance to a box """
    return elongate(dot(p, weights), sizes)

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


