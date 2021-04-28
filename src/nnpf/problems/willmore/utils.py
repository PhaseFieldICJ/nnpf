import nnpf.problems.mean_curvature as mcp


__all__ = [
    "check_sphere_mass",
]


def check_sphere_mass(*args, **kwargs):
    """
    Check a Willmore model by measuring sphere volume decreasing

    See documentation of mean_curvature_problem.check_sphere_volume
    """
    from nnpf.problems import WillmoreProblem
    return mcp.check_sphere_volume(WillmoreProblem.sphere_radius, WillmoreProblem.profil, *args, **kwargs)



