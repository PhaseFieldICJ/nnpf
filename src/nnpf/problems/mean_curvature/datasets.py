import torch
from torch.utils.data import Dataset, TensorDataset
import nnpf.shapes as shapes


__all__ = [
    "MCSphereLazyDataset",
    "MCSphereDataset",
]


class MCSphereLazyDataset(Dataset):
    """
    Dataset of spheres for mean curvature problem, with samples generated at loading.

    Dataset length depends on the broadcasted size of radius, t, center and epsilon.

    Phase fields are actually generated on-the-fly in `__getitem__`.

    Parameters
    ----------
    sphere_radius: callable
        Time evolution of the radius of a sphere
    profil: callable
        Transform a distance function into the appropriate solution profil
    X: tuple of Tensor
        Coordinates of each discretization point
    radius: float or Tensor
        Sphere radius
    dt: float or Tensor
        Time step
    center: list of float or Tensor
        Sphere center
    epsilon: float or Tensor
        Interface sharpness in phase field model
    p: int, float or float("inf")
        Power in the lp-norm
    steps: int
        Number of evolution steps applied to each input
    reverse: bool or Tensor of bool
        Reverse phase's inside and outside
    shape: function(radius, center, p)
        Reference shape (sphere by design so that any other shape must have the save signature)

    Examples
    --------

    # The domain
    >>> from nnpf.domain import Domain
    >>> d = Domain([[-1, 1], [-1, 1]], 21)

    # Radius evolution and solution profil
    >>> from nnpf.problems import AllenCahnProblem
    >>> sphere_radius = AllenCahnProblem.sphere_radius
    >>> profil = AllenCahnProblem.profil

    # Helper
    >>> import nnpf.shapes as shapes
    >>> def sphere_dist(X, radius, t, center, p):
    ...     return shapes.sphere(sphere_radius(radius, t), center, p=p)(*X)
    >>> def sol(X, radius, center, epsilon, t, lp=2, reverse=False):
    ...     return profil((1. - 2. * reverse) * sphere_dist(
    ...         torch.as_tensor(X),
    ...         radius,
    ...         t,
    ...         torch.as_tensor(center),
    ...         lp), epsilon)

    # With one data
    >>> ds = MCSphereLazyDataset(sphere_radius, profil, d.X, 0.5, [0., 0.], 0.1, 0.1)
    >>> len(ds)
    1
    >>> len(ds[0])
    2
    >>> ds[0][0].shape
    torch.Size([1, 21, 21])
    >>> torch.allclose(ds[0][0][0, 1, 11], sol([-0.9, 0.1], 0.5, [0., 0.], 0.1, 0.))
    True
    >>> torch.allclose(ds[0][1][0, 1, 11], sol([-0.9, 0.1], 0.5, [0., 0.], 0.1, 0.1))
    True

    # Reverse
    >>> ds = MCSphereLazyDataset(sphere_radius, profil, d.X, 0.5, [0., 0.], 0.1, 0.1, reverse=True)
    >>> torch.allclose(ds[0][0][0, 1, 11], sol([-0.9, 0.1], 0.5, [0., 0.], 0.1, 0., reverse=True))
    True

    # With multiple radius
    >>> ds = MCSphereLazyDataset(sphere_radius, profil, d.X, [0.1, 0.2, 0.3, 0.4, 0.5], [0., 0.], 0.1, 0.1)
    >>> len(ds)
    5
    >>> ds[1:4][0].shape
    torch.Size([3, 1, 21, 21])
    >>> torch.allclose(ds[2][0][0, 1, 11], sol([-0.9, 0.1], 0.3, [0., 0.], 0.1, 0.))
    True

    # Everything
    >>> ds = MCSphereLazyDataset(
    ...     sphere_radius,
    ...     profil,
    ...     d.X,
    ...     torch.linspace(0., 1., 11), # radius
    ...     torch.stack((torch.linspace(-1., 0, 11), torch.linspace(0, 1., 11))), # center
    ...     torch.linspace(0.1, 1.1, 11), # epsilon
    ...     torch.linspace(0.1, 1.1, 11), # dt
    ...     lp=1,
    ...     steps=5,
    ...     reverse=torch.linspace(0., 1., 11) <= 0.5,
    ... )
    >>> len(ds)
    11
    >>> len(ds[2:6])
    6
    >>> ds[2:6][3].shape
    torch.Size([4, 1, 21, 21])
    >>> torch.allclose(ds[4][2][0, 1, 11], sol([-0.9, 0.1], 0.4, [-0.6, 0.4], 0.5, 2*0.5, lp=1, reverse=True))
    True
    """

    def __init__(self, sphere_radius, profil, X, radius, center, epsilon, dt, lp=2, steps=1, reverse=False, shape=shapes.sphere):
        self.sphere_radius = sphere_radius
        self.profil = profil
        self.shape = shape

        # Additionnal dimensions for appropriate broadcasting
        dim = X[0].ndim
        sup_dims = (1,) + dim * (1,)

        radius = torch.as_tensor(radius).reshape(-1, *sup_dims)
        center = torch.as_tensor(center).reshape(dim, -1, *sup_dims)
        epsilon = torch.as_tensor(epsilon).reshape(-1, *sup_dims)
        dt = torch.as_tensor(dt).reshape(-1, *sup_dims)
        reverse = torch.as_tensor(reverse).reshape(-1, *sup_dims)

        self.num_samples = max(
            radius.shape[0],
            center.shape[1],
            epsilon.shape[0],
            dt.shape[0],
            reverse.shape[0],
        )

        self.X = X
        self.lp = lp
        self.steps = steps
        self.radius = radius.expand(self.num_samples, *sup_dims)
        self.center = center.expand(dim, self.num_samples, *sup_dims)
        self.epsilon = epsilon.expand(self.num_samples, *sup_dims)
        self.dt = dt.expand(self.num_samples, *sup_dims)
        self.reverse = reverse.expand(self.num_samples, *sup_dims)
        # Should be a better way...

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Flip coordinates and inside/outside
        sign = 1. - 2. * self.reverse[idx, ...]
        X = [c + sign * (x - c) for x, c in zip(self.X, self.center[:, idx, ...])]

        return tuple(
            self.profil(
                sign * self.shape(
                    self.sphere_radius(
                        self.radius[idx, ...],
                        i * self.dt[idx, ...],
                    ),
                    self.center[:, idx, ...],
                    self.lp,
                )(*X),
                self.epsilon[idx, ...],
            )
            for i in range(self.steps + 1))


class MCSphereDataset(TensorDataset):
    """
    Dataset of spheres for mean curvature problem (non-lazy version).

    See documentation of MCSphereLazyDataset

    Examples
    --------

    # The domain
    >>> from nnpf.domain import Domain
    >>> d = Domain([[-1, 1], [-1, 1]], 21)

    # Radius evolution and solution profil
    >>> from nnpf.problems import AllenCahnProblem
    >>> sphere_radius = AllenCahnProblem.sphere_radius
    >>> profil = AllenCahnProblem.profil

    # Helper
    >>> import nnpf.shapes as shapes
    >>> def sphere_dist(X, radius, t, center, p):
    ...     return shapes.sphere(sphere_radius(radius, t), center, p=p)(*X)
    >>> def sol(X, radius, center, epsilon, t, lp=2, reverse=False):
    ...     return profil((1. - 2. * reverse) * sphere_dist(
    ...         torch.as_tensor(X),
    ...         radius,
    ...         t,
    ...         torch.as_tensor(center),
    ...         lp), epsilon)

    # Everything
    >>> ds = MCSphereDataset(
    ...     sphere_radius,
    ...     profil,
    ...     d.X,
    ...     torch.linspace(0., 1., 11), # radius
    ...     torch.stack((torch.linspace(-1., 0, 11), torch.linspace(0, 1., 11))), # center
    ...     torch.linspace(0.1, 1.1, 11), # epsilon
    ...     torch.linspace(0.1, 1.1, 11), # dt
    ...     lp=1,
    ...     steps=5,
    ...     reverse=torch.linspace(0., 1., 11) <= 0.5,
    ... )
    >>> len(ds)
    11
    >>> len(ds[2:6])
    6
    >>> ds[2:6][3].shape
    torch.Size([4, 1, 21, 21])
    >>> torch.allclose(ds[4][2][0, 1, 11], sol([-0.9, 0.1], 0.4, [-0.6, 0.4], 0.5, 2*0.5, lp=1, reverse=True))
    True
    """

    def __init__(self, *args, **kwargs):
        ds = MCSphereLazyDataset(*args, **kwargs)
        super().__init__(*ds[:])

