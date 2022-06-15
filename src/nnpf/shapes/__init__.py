"""
Signed distances functions

Example
-------

Centered sphere:
>>> import torch
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

Some visualization and norm check:
>>> from nnpf.domain import Domain
>>> import nnpf.visu as visu
>>> import matplotlib.pyplot as plt
>>> d = Domain([(-1,1),(-1,1)], [1024, 1024])
>>> p_list = [1.1, 2.]
>>> fig = plt.figure(figsize=[2 * 4, len(p_list) * 4])
>>> for i, p in enumerate(p_list):
...     s = periodic(union(translation(box([0.5, 0.5], p=p), [-0.5, -0.25]), rounding(translation(arc(0.5, 2.28, 6.2, p=p), [0.5, 0.25]), 0.1)), d.bounds)
...     ax = plt.subplot(len(p_list), 2, 2*i + 1)
...     im = visu.DistanceShow(s(*d.X), X=d.X)
...     ax = plt.subplot(len(p_list), 2, 2*i + 2)
...     im = visu.ImShow(gradient_norm(s(*d.X), d.dX, p=p), vmin=0, vmax=2, cmap='seismic')
...     cb = plt.colorbar(im.mappable)
>>> plt.savefig("doctest_shapes.png")
>>> plt.pause(0.5)

"""

from .shapes import *
from .utils import *
from .operators import *
from .slices import *
