"""
Signed distances functions and tools

Example
-------
>>> X, Y = torch.meshgrid(torch.linspace(0, 1, 5), torch.linspace(0, 1, 5))
>>> s = sphere([0.5, 0.5], 0.25)
>>> s(X, Y)
tensor([[ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.2500,  0.0000, -0.2500,  0.0000,  0.2500],
        [ 0.3090,  0.1036,  0.0000,  0.1036,  0.3090],
        [ 0.4571,  0.3090,  0.2500,  0.3090,  0.4571]])
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

def sphere(center, radius):
    """ Signed distance to a sphere """
    def dist(*X):
        return torch.sqrt(sum((X[i] - center[i])**2 for i in range(len(X)))) - radius

    return dist


###############################################################################
# Operations

def union(*shapes):
    """ Union of shapes (not exact) """
    def dist(*X):
        return torch.min(*(shape(*X) for shape in shapes))

    return dist


def intersection(*shapes):
    """ Intersection of shapes (not exact) """
    def dist(*X):
        return torch.max(*(shape(*X) for shape in shapes))

    return dist


def translation(shape, shift):
    """ Translation of a shape (exact) """
    def dist(*X):
        return shape(*(X[i] - shift[i] for i in range(len(X))))

    return dist



