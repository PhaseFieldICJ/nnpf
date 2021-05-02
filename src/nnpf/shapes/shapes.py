"""
Signed distances functions
"""

import torch
import math

from .operators import *
from .utils import *


__all__ = [
    "dot",
    "sphere",
    "ellipse",
    "box",
    "half_plane",
    "beam",
    "line",
    "segment",
    "capsule",
    "arc",
]


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

def arc(radius, theta_start, theta_stop=None, p=2, weights=None):
    """
    Sphere arc

    Not really an arc since it returns a portion of a sphere in nD for n > 2 (the distance will be wrong!)
    Probably not exact for p != 2 or custom weights (to be checked)!

    Parameters
    ----------
    radius: float
        Arc radius
    theta_start: float
        The angle from the first axis where the arc begins.
    thetastop: float
        The angle rom the first axis where the arc ends.
        If None, the arc ends at 2pi - theta_start (clockwise).
    p: int or float
        The p in the lp norm.
    """
    if theta_stop is None:
        theta_stop = 2 * math.pi - theta_stop

    theta_axe = math.pi + (theta_start + theta_stop) / 2
    axe = torch.tensor([math.cos(theta_axe), math.sin(theta_axe)])

    start_normal = torch.tensor([math.cos(theta_start), math.sin(theta_start)])
    start_normal = start_normal / norm(start_normal, p=p, weights=weights)
    stop_normal = torch.tensor([math.cos(theta_stop), math.sin(theta_stop)])
    stop_normal = stop_normal / norm(stop_normal, p=p, weights=weights)

    sphere_dist = unsign(sphere(radius, p=p, weights=weights))
    start_dist = sphere(0., center=radius * start_normal, p=p, weights=weights)
    stop_dist = sphere(0., center=radius * stop_normal, p=p, weights=weights)

    def dist(*X):
        P1 = dot_product(X, axe)
        P = P1, norm((x - P1 * a for x, a in zip(X, axe))) # Will not work in nD for n > 2 and p != 2 ...
        side = P[1] * math.cos(min(theta_start, theta_stop) - theta_axe) - P[0] * math.sin(min(theta_start, theta_stop) - theta_axe)
        return torch.where(side >= 0, sphere_dist(*X), torch.min(start_dist(*X), stop_dist(*X)))

    return dist

