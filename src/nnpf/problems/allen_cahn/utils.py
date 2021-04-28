import nnpf.problems.mean_curvature as mcp


__all__ = [
    "check_sphere_mass",
]


def check_sphere_mass(*args, **kwargs):
    """
    Check an Allen-Cahn model by measuring the decreasing of the mass of the profil associated to a sphere.

    See documentation of mean_curvature_problem.check_sphere_volume
    """
    from nnpf.problems import AllenCahnProblem
    return mcp.check_sphere_volume(AllenCahnProblem.sphere_radius, AllenCahnProblem.profil, *args, **kwargs)


