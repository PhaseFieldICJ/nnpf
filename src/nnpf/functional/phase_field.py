"""
Phase Field base functions
"""

import torch


__all__ = [
    "profil",
    "iprofil",
    "dprofil",
    "idprofil",
]


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

    Examples
    --------
    >>> d = torch.linspace(-1., 1., 11, dtype=torch.float64)
    >>> profil(d, 0.1)
    tensor([9.9995e-01, 9.9966e-01, 9.9753e-01, 9.8201e-01, 8.8080e-01, 5.0000e-01,
            1.1920e-01, 1.7986e-02, 2.4726e-03, 3.3535e-04, 4.5398e-05],
           dtype=torch.float64)
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

    Returns
    -------
    dist: Tensor
        Signed distance field to the interface

    Examples
    --------
    >>> d = torch.linspace(-1., 1., 11, dtype=torch.float64)
    >>> u_eps = profil(d, 0.1)
    >>> torch.allclose(d, iprofil(u_eps, 0.1))
    True
    """
    return 2 * epsilon * torch.atanh(1 - 2 * u_eps)

def dprofil(dist, epsilon):
    """
    Derivative of the phase-field profil from the given distance field and interface sharpness

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

    Examples
    --------
    >>> d = torch.linspace(-1., 1., 11, dtype=torch.float64)
    >>> dprofil(d, 0.1)
    tensor([-4.5396e-05, -3.3524e-04, -2.4665e-03, -1.7663e-02, -1.0499e-01,
            -2.5000e-01, -1.0499e-01, -1.7663e-02, -2.4665e-03, -3.3524e-04,
            -4.5396e-05], dtype=torch.float64)
    """
    return -0.25 * (1 - torch.tanh(dist / (2 * epsilon))**2)

def idprofil(u_eps, epsilon):
    """
    Inverse of the profil's derivative: returns signed distance from phase field function

    Parameters
    ----------
    u_exp: Tensor
        Phase field function
    epsilon: float
        Interface sharpness

    Returns
    -------
    dist: Tensor
        Signed distance field to the interface

    Examples
    --------
    >>> d = torch.linspace(-1., 1., 11, dtype=torch.float64)
    >>> u_eps = dprofil(d, 0.1)
    >>> torch.allclose(d.abs(), idprofil(u_eps, 0.1))
    True
    """
    return 2 * epsilon * torch.atanh((1 + 4 * u_eps).sqrt())

