import torch
import nnpf.functional
import nnpf.shapes as shapes

__all__ = [
    "generate_sphere_phase",
    "generate_phase_field_union",
    "evolve_phase_field",
]


def generate_sphere_phase(num_samples, domain, radius, epsilon, profil=nnpf.functional.profil):
    """
    Generates phase field function for one sphere per sample

    With uniform random position, radius and interface's sharpness.

    Parameters
    ----------
    num_samples: int
        Number of samples
    domain: domain.Domain
        Discretization domain
    radius: float or (float, float)
        Radius or bounds on the radius
    epsilon: float or (float, float)
        Value or bounds on the sharpness of the interface
    profil: callable
        The phase field profil (e.g. function that map the signed distance to [0, 1])

    Returns
    -------
    data: Tensor
        Phase field function of a sphere foreach sample
    """

    if isinstance(radius, float):
        min_radius = max_radius = radius
    else:
        min_radius, max_radius = radius

    if isinstance(epsilon, float):
        min_epsilon = max_epsilon = epsilon
    else:
        min_epsilon, max_epsilon = epsilon

    # Additionnal dimensions for appropriate broadcasting
    sup_dims = (1,) + domain.dim * (1,)

    # Domain origin
    origin = torch.Tensor([a for a, b in domain.bounds]).resize_((domain.dim, 1) + sup_dims)

    # Domain width
    width = torch.Tensor([b - a for a, b in domain.bounds]).resize_((domain.dim, 1) + sup_dims)

    # Sphere centers
    # Resulting shape is (dimension, samples, 1, N, M, ...) because
    # shapes.sphere iterate through first dimension of center when calculating
    # distance.
    centers = origin + width * torch.rand((domain.dim, num_samples) + sup_dims)

    # Sphere radius
    radius = min_radius + (max_radius - min_radius) * torch.rand((num_samples,) + sup_dims)

    # Periodic sphere
    shape = shapes.periodic(shapes.sphere(radius, centers), domain.bounds)

    # Interface sharpness and corresponding phase field profil
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * torch.rand((num_samples,) + sup_dims)
    return profil(shape(*domain.X), epsilon)


def generate_phase_field_union(num_samples, phase_field_gen, num_shapes):
    """
    Union of random number of phase fields

    Parameters
    ----------
    num_samples: int
        Number of samples
    phase_field_gen: function
        Given a number of samples, returns one phase field per sample
    num_shapes: int or (int, int)
        Bounds on the number of superimposed phase fields

    Returns
    -------
    data: Tensor
        Phase field function foreach sample
    """


    if isinstance(num_shapes, int):
        min_shapes = max_shapes = num_shapes
    else:
        min_shapes, max_shapes = num_shapes

    assert min_shapes >= 1

    # Initial phase field
    data = phase_field_gen(num_samples)

    # Additional shapes
    shapes_count = torch.randint(min_shapes, max_shapes + 1, (num_samples,))
    for i in range(2, max_shapes + 1):
        mask = shapes_count >= i
        data[mask] = torch.max(data[mask], phase_field_gen(mask.sum().item()))

    return data


def evolve_phase_field(operator, phase_fields, steps):
    """
    Modify phase field samples under the given operator

    Parameters
    ----------
    operator: function like
        Evolving operator
    phase_fields: Tensor
        Samples of phase fields
    steps: int
        Number of evolving steps

    Returns
    -------
    data: Tensor
        initial phase fiels and concatenated evolutions.
    """
    assert steps >= 1
    data = phase_fields.new_empty(steps + 1, *phase_fields.shape)
    data[0, ...] = phase_fields
    for i in range(steps):
        data[i + 1, ...] = operator(data[i, ...])

    return data

