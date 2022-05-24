import torch
import nnpf.shapes as shapes


__all__ = [
    "check_sphere_mass",
]


def check_sphere_mass(sphere_radius, profil, model, domain, r0, epsilon, dt, num_steps=None, center=None, p=2, progress_bar=False, shape=shapes.sphere):
    """
    Check a mean curvature model by measuring the decreasing of the mass of the profil associated to a sphere (or other compatible shape).

    Parameters
    ----------
    sphere_radius: callable
        Time evolution of the radius of a sphere.
    profil: callable
        Transform a distance function into the appropriate solution profil
    model: callable
        Returns `u^{n+1}` from `u^{n}. Don't forget to disable grad calculation before!
    domain: domain.Domain
        Discretization domain
    r0: float
        Initial sphere radius
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step
    num_steps: int or None
        Number of time steps.
        If None, automatically calculated depending on domain size and epsilon.
    center: list of float
        Sphere center (domain center if None)
    p: int, float or float("inf")
        Power in the lp-norm
    progress_bar: bool
        True to display a progress bar
    shape: function(radius, center, p)
        Reference shape (sphere by design so that any other shape must have the save signature)

    Returns
    -------
    model_mass: torch.tensor
        Sphere mass evolution for the given model
    exact_mass: torch.tensor
        Sphere mass evolution of the solution
    """

    domain_diameter = min(b[1] - b[0] for b in domain.bounds)
    center = center or [0.5 * sum(b) for b in domain.bounds]

    def generate_solution(radius):
        return profil(shape(radius, center, p=p)(*domain.X), epsilon)[None, None, ...]

    def mass(u):
        return domain.dX.prod() * u.sum()

    # Calculating radius range
    radiuses = []
    step = 0
    while num_steps is None or step < num_steps:
        radius = sphere_radius(r0, dt * torch.arange(step, step + (num_steps or 100) + 1))
        inbound_mask = torch.logical_and(epsilon < radius, radius < domain_diameter / 2 - epsilon)
        radiuses.append(radius[inbound_mask])
        step += radiuses[-1].numel()

        if radiuses[-1].numel() < radius.numel():
            break

    radiuses = torch.cat(radiuses)

    # Calculating mass for each radius
    model_mass = domain.X[0].new_empty(radiuses.numel())
    exact_mass = model_mass.new_empty(radiuses.numel())

    u = generate_solution(radiuses[0])

    if progress_bar:
        from tqdm import tqdm
        radiuses = tqdm(radiuses)

    for idx, r in enumerate(radiuses):
        model_mass[idx] = mass(u)
        exact_mass[idx] = mass(generate_solution(r))
        u = model(u)

    return model_mass, exact_mass

