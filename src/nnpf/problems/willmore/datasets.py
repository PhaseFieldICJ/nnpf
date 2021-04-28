from torch.utils.data import TensorDataset
import nnpf.problems.mean_curvature as mcp


__all__ = [
    "WillmoreSphereLazyDataset",
    "WillmoreSphereDataset",
]


class WillmoreSphereLazyDataset(mcp.MCSphereLazyDataset):
    """
    Dataset of spheres for Willmore problem, with samples generated at loading.

    See documentation of mean_curvature_problem.MCSphereLazyDataset
    """
    def __init__(self, X, radius, center, epsilon, dt, lp=2, steps=1, reverse=False):
        from nnpf.problems import WillmoreProblem
        super().__init__(
            WillmoreProblem.sphere_dist,
            WillmoreProblem.profil,
            X, radius, center, epsilon, dt, lp, steps, reverse)


class WillmoreSphereDataset(TensorDataset):
    """
    Dataset of spheres for Willmore problem (non-lazy version).

    See documentation of WillmoreSphereLazyDataset
    """
    def __init__(self, *args, **kwargs):
        ds = WillmoreSphereLazyDataset(*args, **kwargs)
        super().__init__(*ds[:])

