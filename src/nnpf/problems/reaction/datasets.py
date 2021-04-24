import torch
from torch.utils.data import TensorDataset

from .solution import ReactionSolution


__all__ = [
    "ReactionDataset",
]


class ReactionDataset(TensorDataset):
    """
    Base dataset for the Allen-Cahn reaction term.

    It is the exact solution for linear spaced samples in the [0, 1] interval
    with a given external margin.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step.
    margin: float
        Expanding length of the sampled [0, 1] interval

    Examples
    --------
    >>> dataset = ReactionDataset(13, 1e-2, 1e-4, 0.1)
    >>> len(dataset)
    13
    >>> dataset[1]
    (tensor([0.]), tensor([0.]))
    >>> dataset[2]
    (tensor([0.1000]), tensor([0.0449]))
    >>> dataset[11]
    (tensor([1.]), tensor([1.]))
    """

    def __init__(self, num_samples, epsilon, dt, margin):
        lower_bound = 0. - margin
        upper_bound = 1. + margin
        exact_sol = ReactionSolution(epsilon, dt)
        train_x = torch.linspace(lower_bound, upper_bound, num_samples)[:, None]
        train_y = exact_sol(train_x)
        super().__init__(train_x, train_y)


