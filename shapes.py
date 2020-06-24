"""
Signed distances functions and tools

Example
-------
>>> import numpy as np
>>> import shapes
>>> X, Y = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
>>> s = shapes.sphere([0.3, 0.5], 0.2)
>>> dist = s(X, Y)
"""

import numpy as np

###############################################################################
# Shapes

def sphere(center, radius):
    """ Signed distance to a sphere """
    def dist(*X):
        return np.sqrt(sum((X[i] - center[i])**2 for i in range(len(X)))) - radius

    return dist


###############################################################################
# Operations

def union(*shapes):
    """ Union of shapes (not exact) """
    def dist(*X):
        return np.minimum(*(shape(*X) for shape in shapes))

    return dist


def intersection(*shapes):
    """ Intersection of shapes (not exact) """
    def dist(*X):
        return np.maximum(*(shape(*X) for shape in shapes))

    return dist


def translation(shape, shift):
    """ Translation of a shape (exact) """
    def dist(*X):
        return shape(*(X[i] - shift[i] for i in range(len(X))))

    return dist



