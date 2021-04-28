from torch.utils.data import TensorDataset
import nnpf.problems.mean_curvature as mcp


__all__ = [
    "SteinerSphereLazyDataset",
    "SteinerSphereDataset",
]


class SteinerSphereLazyDataset(mcp.MCSphereLazyDataset):
    """
    Dataset of spheres for Steiner problem, with samples generated at loading.

    See documentation of mean_curvature_problem.MCSphereLazyDataset
    """
    def __init__(self, X, radius, center, epsilon, dt, lp=2, steps=1, reverse=False):
        from nnpf.problems import SteinerProblem
        super().__init__(
            SteinerProblem.sphere_dist,
            SteinerProblem.profil,
            X, radius, center, epsilon, dt, lp, steps, reverse)


class SteinerSphereDataset(TensorDataset):
    """
    Dataset of spheres for Steiner problem (non-lazy version).

    See documentation of SteinerSphereLazyDataset
    """
    def __init__(self, *args, **kwargs):
        ds = SteinerSphereLazyDataset(*args, **kwargs)
        super().__init__(*ds[:])


