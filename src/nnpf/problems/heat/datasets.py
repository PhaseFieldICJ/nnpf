import torch
from torch.utils.data import TensorDataset

from .solution import HeatSolution
from nnpf.domain import Domain
from nnpf.datasets import generate_sphere_phase, generate_phase_field_union, evolve_phase_field


__all__ = [
    "HeatDataset",
]


class HeatDataset(TensorDataset):
    """
    Base dataset for the heat equation

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset
    bounds: iterable of pairs of float
        Bounds of the domain
    N: int or iterable of int
        Number of discretization points
    dt: float
        Time step.
    radius: float or (float, float)
        Radius or bounds on the radius
    epsilon: float or (float, float)
        Value or bounds on the sharpness of the interface
    num_shapes: int or (int, int)
        Bounds on the number of superimposed phase fields
    steps: int
        Number of evolving steps

    Examples
    --------
    >>> dataset = HeatDataset(100, [[0., 1.], [0., 1.]], 256, 1e-2, [0, 0.25], [0, 0.1], [1, 3], 10)
    >>> len(dataset)
    100
    >>> dataset[0][0].shape
    torch.Size([1, 256, 256])
    >>> len(dataset[0])
    11
    """

    def __init__(self, num_samples, bounds, N, dt, radius, epsilon, num_shapes, steps):
        domain = Domain(bounds, N)
        exact_sol = HeatSolution(domain, dt)
        base_shape_gen = lambda num_samples: generate_sphere_phase(num_samples, domain, radius, epsilon)
        data = generate_phase_field_union(num_samples, base_shape_gen, num_shapes)
        data = evolve_phase_field(exact_sol, data, steps)
        super().__init__(*(data[i, ...] for i in range(data.shape[0])))


