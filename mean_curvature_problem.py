#!/usr/bin/env python3

"""
Base module and utils for the mean curvature based-equation learning problem
"""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import math

from problem import Problem, get_default_args
from domain import Domain
import nn_toolbox
import shapes


def check_sphere_mass(sphere_radius, profil, model, domain, r0, epsilon, dt, num_steps=None, center=None, p=2, progress_bar=False):
    """
    Check a mean curvature model by measuring the decreasing of the mass of the profil associated to a sphere.

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
        return profil(shapes.sphere(radius, center, p=p)(*domain.X), epsilon)[None, None, ...]

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

    Examples
    --------

    # The domain
    >>> from domain import Domain
    >>> d = Domain([[-1, 1], [-1, 1]], 21)

    # Radius evolution and solution profil
    >>> from allen_cahn_problem import AllenCahnProblem
    >>> sphere_radius = AllenCahnProblem.sphere_radius
    >>> profil = AllenCahnProblem.profil

    # Helper
    >>> import shapes
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

    def __init__(self, sphere_radius, profil, X, radius, center, epsilon, dt, lp=2, steps=1, reverse=False):
        self.sphere_radius = sphere_radius
        self.profil = profil

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
        sign = 1. - 2. * self.reverse[idx, ...]
        return tuple(
            self.profil(
                sign * shapes.sphere(
                    self.sphere_radius(
                        self.radius[idx, ...],
                        i * self.dt[idx, ...],
                    ),
                    self.center[:, idx, ...],
                    self.lp,
                )(*self.X),
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
    >>> from domain import Domain
    >>> d = Domain([[-1, 1], [-1, 1]], 21)

    # Radius evolution and solution profil
    >>> from allen_cahn_problem import AllenCahnProblem
    >>> sphere_radius = AllenCahnProblem.sphere_radius
    >>> profil = AllenCahnProblem.profil

    # Helper
    >>> import shapes
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


class MeanCurvatureProblem(Problem):
    """
    Base class for the mean curvature problem

    Features the train and validation data (MCSphereDataset), and the metric.

    Parameters
    ----------
    bounds: iterable of pairs of float
        Bounds of the domain
    N: int or iterable of int
        Number of discretization points
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step. If None, set to the maximum time step that guarantee the scheme stability.
    batch_size: int
        Size of the batch during training and validation steps. Full data if None.
    batch_shuffle: bool
        Shuffle batch content.
    lr: float
        Learning rate of the optimizer
    loss_norms: list of pair (p, weight)
        Compose loss as sum of weight * (output - target).norm(p).pow(e).
        Default to l2 norm.
        Exponent e is defined with loss_power parameter.
    loss_power: float
        Power applied to each loss term (for regularization purpose).
    radius: list of 2 floats
        Bounds on sphere radius (ratio of domain bounds) used for training and validation datasets.
    lp: int or float
        Power of the lp-norm used to defined the spheres in training and validation datasets.
    train_N, val_N: int
        Size of the training and validation datasets
    train_steps, val_steps: int
        Number of evolution steps applied to each input
    train_reverse, val_reverse: float
        Probability of having a sample with reversed inside and outside
    kwargs: dict
        Parameters passed to Problem (see doc)
    """

    def __init__(self, bounds=[[0., 1.], [0., 1.]], N=256, epsilon=2/256, dt=None,
                 batch_size=10, batch_shuffle=True, lr=1e-4,
                 loss_norms=None, loss_power=2.,
                 radius=[0.05, 0.45], lp=2,
                 train_N=100, train_steps=1, train_reverse=0.,
                 val_N=200, val_steps=5, val_reverse=0.,
                 **kwargs):

        super().__init__(**kwargs)

        # Default values
        dt = dt or self.stability_dt(epsilon)
        loss_norms = loss_norms or [[2, 1.]]

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
            'bounds', 'N', 'dt', 'epsilon',
            'batch_size', 'batch_shuffle', 'lr', 'loss_norms', 'loss_power',
            'radius', 'lp', 'train_N', 'val_N', 'train_steps', 'val_steps',
            'train_reverse', 'val_reverse',
        )


    @staticmethod
    def stability_dt(epsilon):
        """ Maximum allowed time step that guarantee stability of the scheme """
        ...

    @staticmethod
    def profil(dist, epsilon):
        """ Solution profil from distance field """
        ...

    @staticmethod
    def iprofil(dist, epsilon):
        """ Distance field from solution profil """
        ...

    @staticmethod
    def sphere_radius(r0, t):
        """ Time evolution of the radius of the sphere """
        ...

    @property
    def domain(self):
        return Domain(self.hparams.bounds, self.hparams.N, device=self.device)

    def loss(self, output, target):
        """ Default loss function """
        dim = tuple(range(2, 2 + self.domain.dim))
        error = target - output
        return sum(
            w * nn_toolbox.norm(error, p, dim).pow(self.hparams.loss_power)
            for p, w in self.hparams.loss_norms).mean() / torch.tensor(self.domain.N).prod()

        # Below, loss with rescale depending on the solution norm (i.e. the circle size)
        # The idea was to compensate the less importance of small circle in the final loss
        # Need a lit more work on the scaling to work good
        """
        return sum(
            w * (nn_toolbox.norm(error, p, dim) / nn_toolbox.norm(target, p, dim)).pow(self.hparams.loss_power)
            for p, w in self.hparams.loss_norms).mean()
        """

    def check_sphere_mass(self, radius=0.45, num_steps=100, center=None, progress_bar=False):
        """
        Check a mean curvature model by measuring the decreasing of the mass of the profil associated to a sphere.

        Parameters
        ----------
        radius: float or list of float
            List of sphere radius (ratio of domain diameter) used as initial condition.
        num_steps: int or list of int
            Number of time steps (possibly per radius)
        center: list of float
            Sphere center (domain center if None)
        progress_bar: bool
            True to display a progress bar

        Returns
        -------
        model_mass: torch.tensor
            Sphere mass evolution for the given model
        exact_mass: torch.tensor
            Sphere mass evolution of the solution
        """

        def as_list(v):
            try: iter(v)
            except TypeError: return [v]
            else: return list(v)

        radius = as_list(radius)
        num_steps = as_list(num_steps)
        num_steps += [num_steps[-1]] * (len(num_steps) - len(radius))

        domain_diameter = min(b[1] - b[0] for b in self.domain.bounds)
        model_mass, exact_mass = [], []

        for r, n in zip(radius, num_steps):
            r = r * domain_diameter

            with torch.no_grad():
                mm, em = check_sphere_mass(
                    self.sphere_radius,
                    self.profil,
                    self, self.domain, r, self.hparams.epsilon, self.hparams.dt,
                    n, center, p=self.hparams.lp, progress_bar=progress_bar)
                model_mass.append(mm)
                exact_mass.append(em)

        return torch.cat(model_mass), torch.cat(exact_mass)

    def training_step(self, batch, batch_idx):
        """ Default training step with custom loss function """
        data, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self.forward(data)
            loss += self.loss(data, target)
        loss /= len(targets)

        return self.dispatch_metrics({'loss': loss})

    def validation_step(self, batch, batch_idx):
        """ Called at each batch of the validation data """
        data, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """ Called at epoch end of the validation step (after all batches) """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Metric calculation
        model_mass, exact_mass = self.check_sphere_mass()
        mass_error = ((model_mass - exact_mass) / exact_mass).norm() / model_mass.numel()

        self.dispatch_metrics({'val_loss': avg_loss, 'metric': mass_error})

    def configure_optimizers(self):
        """ Default optimizer """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def prepare_data(self):
        """ Prepare training and validation data """

        # Additionnal dimensions for appropriate broadcasting
        sup_dims = (1,) + self.domain.dim * (1,)

        # Shape properties
        domain_diameter = min(b[1] - b[0] for b in self.domain.bounds)
        radius = [domain_diameter * r for r in self.hparams.radius]
        center = [0.5 * sum(b) for b in self.domain.bounds]

        # Datasets
        def generate_data(num_samples, steps, reverse):
            return MCSphereDataset(
                self.sphere_radius,
                self.profil,
                self.domain.X,
                radius=torch.linspace(radius[0], radius[1], num_samples),
                center=center,
                epsilon=self.hparams.epsilon,
                dt=self.hparams.dt,
                lp=self.hparams.lp,
                steps=steps,
                reverse=torch.rand(num_samples) < reverse,
            )

        self.train_dataset = generate_data(self.hparams.train_N, self.hparams.train_steps, self.hparams.train_reverse)
        self.val_dataset = generate_data(self.hparams.val_N, self.hparams.val_steps, self.hparams.val_reverse)

    def train_dataloader(self):
        """ Returns the training data loader """
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size or len(self.train_dataset), shuffle=self.hparams.batch_shuffle)

    def val_dataloader(self):
        """ Returns the validation data loader """
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size or len(self.val_dataset))

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):

        # Parser for the domain bounds
        import re
        def bounds_parser(s):
            bounds = []
            per_dim = s.split('x')
            for dim_spec in per_dim:
                match = re.fullmatch(r'\s*\[([^\]]*)\]\s*', dim_spec)
                if not match:
                    raise ValueError(f"Invalid bound specification {dim_spec}")
                bounds.append([float(b) for b in match.group(1).split(',')])
            return bounds

        # Parser for loss definition
        def float_or_str(v):
            try:
                return float(v)
            except ValueError:
                return v

        # Parser for lp norm
        def int_or_float(v):
            try:
                return int(v)
            except ValueError:
                return float(v)

        from distutils.util import strtobool

        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Mean curvature problem", "Options common to all models of mean curvature based equation.")
        group.add_argument('--bounds', type=bounds_parser, help="Domain bounds in format like '[0, 1]x[1, 2.5]'")
        group.add_argument('--N', type=int, nargs='+', help="Domain discretization")
        group.add_argument('--epsilon', type=float, help="Interface sharpness")
        group.add_argument('--dt', type=float, help="Time step (epsilon**2 if None)")
        group.add_argument('--train_N', type=int, help="Number of initial conditions in the training dataset")
        group.add_argument('--val_N', type=int, help="Number of initial conditions in the validation dataset")
        group.add_argument('--train_steps', type=int, help="Number of evolution steps in the training dataset")
        group.add_argument('--val_steps', type=int, help="Number of evolution steps in the validation dataset")
        group.add_argument('--radius', type=float, nargs=2, help="Bounds on sphere radius (ratio of domain bounds) used for training and validation dataset.")
        group.add_argument('--lp', type=int_or_float, help="Power of the lp-norm used to define the spheres for training and validation dataset.")
        group.add_argument('--batch_size', type=int, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Shuffle batch")
        group.add_argument('--lr', type=float, help="Learning rate")
        group.add_argument('--loss_norms', type=float_or_str, nargs=2, action='append', help="List of (p, weight). Compose loss as sum of weight * (output - target).norm(p).pow(e). Default to l2 norm. Exponent e is defined with loss_power parameter.")
        group.add_argument('--loss_power', type=float, help="Power applied to each loss term (for regularization purpose)")
        group.set_defaults(**{**get_default_args(MeanCurvatureProblem), **defaults})

        return parser


###############################################################################
# Command-line interface
###############################################################################

if __name__ == "__main__":

    from mean_curvature_problem import MeanCurvatureProblem
    import shapes
    import visu
    import argparse
    import imageio
    import torch
    import tqdm
    import math
    from functools import reduce
    from distutils.util import strtobool

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Spheres evolution compared to the solution of the mean curvature flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Path to the model's checkpoint")
    parser.add_argument("--no_save", action="store_true", help="Don't save the animation")
    parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance used as a stop criteron")
    parser.add_argument("--max_it", type=int, default=-1, help="Maximum number of calculated iterations (-1 for illimited")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of rendered frames (-1 for illimited)")
    parser.add_argument("--max_duration", type=float, default=-1, help="Maximum duration of the animation (-1 for illimited")
    parser.add_argument("--no_dist", action="store_true", help="Display phase field instead of distance")
    parser.add_argument("--scale", type=float, default=1., help="Initial shape scale")
    parser.add_argument("--shape", type=str, choices=["one", "two", "three"], default="one", help="Initial shape")
    parser.add_argument("--offscreen", action="store_true", help="Don't display the animation (but still saving")
    parser.add_argument("--gpu", action="store_true", help="Evaluation model on your GPU")
    parser.add_argument("--display_step", type=int, default=1, help="Render frame every given number")
    parser.add_argument("--fps", type=int, default=25, help="Frame per second in the saved animation")
    parser.add_argument("--figsize", type=int, default=[6, 6], nargs=2, help="Figure size in inches")
    parser.add_argument("--revert", type=lambda s:bool(strtobool(s)), nargs='?', const=True, default=False, help="Revert inside and outside of the phase")

    args = parser.parse_args()

    if args.max_duration < 0.:
        args.max_duration = float('inf')

    # Matplotlib rendering backend
    if args.offscreen:
        import matplotlib as mpl
        mpl.use('Agg')
    import matplotlib.pyplot as plt

    # Loading model
    model = MeanCurvatureProblem.load_from_checkpoint(args.checkpoint, map_location=torch.device("cpu"))
    model.freeze()

    if args.gpu:
        model.cuda()

    domain = model.domain

    # Defining initial shape
    bounds = domain.bounds
    domain_extent = [b[1] - b[0] for b in bounds]
    domain_diameter = min(domain_extent)

    def radius(r, scale):
        return scale * r * domain_diameter

    def pos(X, scale):
        #return [b[0] + (0.5 + scale * (x - 0.5)) * (b[1] - b[0]) for x, b in zip(X, bounds)]
        return [b[0] + x * (b[1] - b[0]) for x, b in zip(X, bounds)]

    # Shape
    if args.shape == "one":
        spheres = [(0.3, 0.5, 0.5)]

    elif args.shape == "two":
        spheres = [(0.1, 0.2, 0.2), (0.2, 0.7, 0.7)]

    elif args.shape == "three":
        spheres = [(0.1, 0.2, 0.2), (0.2, 0.3, 0.7), (0.05, 0.7, 0.3)]

    s = shapes.union(*(shapes.sphere(radius(p[0], args.scale), pos(p[1:], args.scale), model.hparams.lp) for p in spheres))
    dist_sol = lambda t: reduce(torch.min, [shapes.sphere(model.sphere_radius(radius(p[0], args.scale), t), pos(p[1:], args.scale), model.hparams.lp)(*domain.X) for p in spheres])

    # Periodizing
    s = shapes.periodic(s, bounds)

    # Phase field
    u = model.profil(s(*domain.X), model.hparams.epsilon)
    if args.revert:
        u = 1. - u

    # Graph
    scale = 0.25 * max(b[1] - b[0] for b, n in zip(domain.bounds, domain.N))
    extent = [*domain.bounds[0], *domain.bounds[1]]
    interpolation = "kaiser"

    plt.figure(figsize=args.figsize)

    if args.no_dist:
        def data_from(u):
            return u.cpu()
        graph = visu.PhaseFieldShow(data_from(u), extent=extent, interpolation=interpolation)
    else:
        def data_from(u):
            return model.iprofil(u, model.hparams.epsilon).cpu()
        graph = visu.DistanceShow(data_from(u), scale=scale, extent=extent, interpolation=interpolation)

    contour = visu.ContourShow(dist_sol(0.).cpu(), [0.], X=[x.cpu() for x in domain.X], colors='red')

    title = plt.title(f"t = 0 ; it = 0")
    plt.tight_layout()
    plt.pause(1)

    with visu.AnimWriter('anim.avi', fps=args.fps, do_nothing=args.no_save) as anim:

        for i in range(25):
            anim.add_frame()

        last_diff = [args.tol + 1] * 25

        with tqdm.tqdm() as pbar:
            while max(last_diff) > args.tol and pbar.n != args.max_it and pbar.n != args.max_frames * args.display_step and pbar.n / args.display_step / args.fps < args.max_duration:
                last_u = u.clone()
                u = model(u[None, None, ...])[0, 0, ...]

                if pbar.n % args.display_step == 0:
                    graph.update(data_from(u))
                    contour.update(dist_sol(pbar.n * model.hparams.dt).cpu())
                    title.set_text(f"t = {pbar.n*model.hparams.dt:.5} ; it = {pbar.n}")
                    plt.pause(0.01)

                    anim.add_frame()

                vol = model.domain.dX.prod() * u.sum()
                last_diff[1:] = last_diff[:-1]
                last_diff[0] = (u - last_u).norm().item()

                pbar.update(1)
                pbar.set_postfix({
                    'volume': vol.item(),
                    'diff': last_diff[0],
                    'max diff': max(last_diff),
                    't': pbar.n * model.hparams.dt,
                    'frames': pbar.n // args.display_step,
                    'duration': pbar.n / args.display_step / args.fps,
                })

