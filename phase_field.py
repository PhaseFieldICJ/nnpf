"""
Phase Field base functions
"""

import torch

def profil(dist, epsilon):
    """
    Phase-field profil from the given distance field and interface sharpness

    Parameters
    ----------
    dist: Tensor
        Signed distance field to the interface
    epsilon: float
        Interface sharpness

    Returns
    -------
    u_eps: Tensor
        Corresponding phase-field solution.
    """
    return 0.5 * (1 - torch.tanh(dist / (2 * epsilon)))

def iprofil(u_eps, epsilon):
    """
    Inverse of profil: returns signed distance from phase field function

    Parameters
    ----------
    u_exp: Tensor
        Phase field function
    epsilon: float
        Interface sharpness

    Returns:
    dist: Tensor
        Signed distance field to the interface
    """
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    return 2 * epsilon * atanh(1 - 2 * u_eps)
