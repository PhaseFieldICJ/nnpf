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

###############################################################################
# Shapes

def dot():
    """ Signed distance to a dot """
    def dist(*X):
        return sum(x**2 for x in X).sqrt()

    return dist

def sphere(radius, center=None):
    """ Signed distance to a sphere """
    if center is None:
        return rounding(dot(), radius)
    else:
        return translation(sphere(radius), center)

def box(sizes):
    """ Signed distance to a box """
    def dist(*X):
        q = torch.stack([x.abs() - s for x, s in zip(X, sizes)])
        z = q.new_zeros(1)
        return torch.max(q, z).norm(dim=0) + torch.min(q.max(dim=0).values, z)

    return dist

###############################################################################
# Operations

def reduce(op, *shapes):
    """ Reduce an operator over multiple shapes """
    def dist(*X):
        result = shapes[0](*X)
        for s in shapes[1:]:
            result = op(result, s(*X))
        return result

    return dist

def union(*shapes):
    """ Union of shapes (not exact in the interior) """
    return reduce(torch.min, *shapes)


def intersection(*shapes):
    """ Intersection of shapes (not exact) """
    return reduce(torch.max, *shapes)


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

    def smoothstep(a, b, x):
        x = torch.clamp((x - a) / (b - a), 0., 1.)
        return x.square() * (3 - 2. * x)

    def mix(a, b, r):
        return a + (b - a) * r

    # Color from Inigo Quilez
    # See e.g. https://www.shadertoy.com/view/3t33WH
    def color(dist):
        adist = dist[..., None].abs()
        col = torch.where(dist[..., None] < 0., dist.new([0.6, 0.8, 1.0]), dist.new([0.9, 0.6, 0.3]))
        col *= 1.0 - (-9.0 / scale * adist).exp()
        col *= 1.0 + 0.2 * torch.cos(128.0 / scale * adist)
        return mix(col, dist.new_ones(3), 1.0 - smoothstep(0., scale * 0.015, adist))

    # Calculating distance
    if not torch.is_tensor(shape_or_dist):
        shape_or_dist = shape_or_dist(*X)
    shape_or_dist = shape_or_dist.squeeze()
    assert shape_or_dist.dim() == 2, "Can only display 2D distance fields"

    # Image
    image = color(shape_or_dist).clamp(0., 1.)
    if return_image:
        return image

    # Extent
    if X is not None and extent is None:
        extent = [X[0].min(), X[0].max(), X[1].min(), X[1].max()]

    # Display
    import matplotlib.pyplot as plt
    plt.imshow(image, extent=extent)
    plt.show()

