from torch.utils.data import TensorDataset
import nnpf.problems.mean_curvature as mcp


__all__ = [
    "ACSphereLazyDataset",
    "ACSphereDataset",
]


class ACSphereLazyDataset(mcp.MCSphereLazyDataset):
    """
    Dataset of spheres for Allen-Cahn problem, with samples generated at loading.

    See documentation of mean_curvature_problem.MCSphereLazyDataset
    """
    def __init__(self, *args, **kwargs):
        from nnpf.problems import AllenCahnProblem
        super().__init__(
            AllenCahnProblem.sphere_dist,
            AllenCahnProblem.profil,
            *args,
            **kwargs,
        )


class ACSphereDataset(TensorDataset):
    """
    Dataset of spheres for Allen-Cahn problem (non-lazy version).

    See documentation of ACSphereLazyDataset
    """
    def __init__(self, *args, **kwargs):
        ds = ACSphereLazyDataset(*args, **kwargs)
        super().__init__(*ds[:])

