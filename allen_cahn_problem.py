"""
Base module and utils for the Allen-Cahn equation learning problem
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

import shapes


def sphere_dist_MC(X, radius=1., t=0., center=None):
    """
    Signed distance field to a sphere evolving by mean curvature field

    Parameters
    ----------
    X: tuple of Tensor
        Coordinates of each discretization point
    radius: float
        Sphere radius
    t: float
        Evaluation time
    center: list of float
        Sphere center

    Returns
    -------
    dist: Tensor
        The signed distance field
    """
    center = center or [0.] * len(X)
    radius = torch.as_tensor(radius**2 - 2 * t).sqrt()
    return sum((X[i] - center[i]).square() for i in range(len(X))).sqrt() - radius

    
