"""
Base module and utils for the Allen-Cahn equation learning problem
"""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import math

from problem import Problem, get_default_args
from domain import Domain
from phase_field import profil
import nn_toolbox
import shapes


def sphere_dist_MC(X, radius=1., t=0., center=None, p=2):
    """
    Signed distance (lp norm) field to a sphere evolving by mean curvature field

    Parameters
    ----------
    X: tuple of Tensor
        Coordinates of each discretization point
    radius: float or Tensor
        Sphere radius
    t: float or Tensor
        Evaluation time
    center: list of (float or Tensor)
        Sphere center
    p: int, float or float("inf")
        Power in the lp-norm

    Returns
    -------
    dist: Tensor
        The signed distance field
    """
    radius = torch.as_tensor(radius, device=X[0].device, dtype=X[0].dtype)
    t = torch.as_tensor(t, device=X[0].device, dtype=X[0].dtype)
    radius = (radius**2 - 2 * t).max(X[0].new_zeros(1)).sqrt()
    return shapes.sphere(radius, center, p=p)(*X)


def check_sphere_volume(model, domain, radius, epsilon, dt, num_steps, center=None, p=2, progress_bar=False):
    """
    Check an Allen-Cahn model by measuring sphere volume decreasing

    Parameters
    ----------
    model: callable
        Returns `u^{n+1}` from `u^{n}. Don't forget to disable grad calculation before!
    domain: domain.Domain
        Discretization domain
    radius: float
        Sphere radius
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step
    num_steps: int
        Number of time steps
    center: list of float
        Sphere center (domain center if None)
    p: int, float or float("inf")
        Power in the lp-norm
    progress_bar: bool
        True to display a progress bar

    Returns
    -------
    model_volume: torch.tensor
        Sphere volume evolution for the given model
    exact_volume: torch.tensor
        Sphere volume evolution of the solution
    """

    center = center or [0.5 * sum(b) for b in domain.bounds]

    def generate_solution(i):
        return profil(sphere_dist_MC(domain.X, radius, i * dt, center, p=p), epsilon)[None, None, ...]

    def vol(u):
        return domain.dX.prod() * u.sum()

    model_volume = domain.X[0].new_empty(num_steps + 1)
    exact_volume = model_volume.new_empty(num_steps + 1)

    u = generate_solution(0)

    rg = range(num_steps + 1)
    if progress_bar:
        from tqdm import tqdm
        rg = tqdm(rg)

    for i in rg:
        model_volume[i] = vol(u)
        exact_volume[i] = vol(generate_solution(i))
        u = model(u)

    return model_volume, exact_volume


class AllenCahnLazyDataset(Dataset):
    """
    Base dataset for Allen-Cahn problem, with samples generated at loading.

    Dataset length depends on the broadcasted size of radius, t, center and epsilon.

    Phase fields are actually generated on-the-fly in `__getitem__`.

    Parameters
    ----------
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

    # Helper
    >>> def sol(X, radius, center, epsilon, t, lp=2, reverse=False):
    ...     return profil((1. - 2. * reverse) * sphere_dist_MC(
    ...         torch.as_tensor(X),
    ...         radius,
    ...         t,
    ...         torch.as_tensor(center),
    ...         lp), epsilon)

    # With one data
    >>> ds = AllenCahnLazyDataset(d.X, 0.5, [0., 0.], 0.1, 0.1)
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
    >>> ds = AllenCahnLazyDataset(d.X, 0.5, [0., 0.], 0.1, 0.1, reverse=True)
    >>> torch.allclose(ds[0][0][0, 1, 11], sol([-0.9, 0.1], 0.5, [0., 0.], 0.1, 0., reverse=True))
    True

    # With multiple radius
    >>> ds = AllenCahnLazyDataset(d.X, [0.1, 0.2, 0.3, 0.4, 0.5], [0., 0.], 0.1, 0.1)
    >>> len(ds)
    5
    >>> ds[1:4][0].shape
    torch.Size([3, 1, 21, 21])
    >>> torch.allclose(ds[2][0][0, 1, 11], sol([-0.9, 0.1], 0.3, [0., 0.], 0.1, 0))
    True

    # Everything
    >>> ds = AllenCahnLazyDataset(
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

    def __init__(self, X, radius, center, epsilon, dt, lp=2, steps=1, reverse=False):
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
            profil( sign *
                sphere_dist_MC(
                    self.X,
                    self.radius[idx, ...],
                    i * self.dt[idx, ...],
                    self.center[:, idx, ...],
                    self.lp),
                self.epsilon[idx, ...])
            for i in range(self.steps + 1))


class AllenCahnDataset(TensorDataset):
    """
    Base dataset for Allen-Cahn problem (non-lazy version).

    See documentatin of AllenCahnLazyDataset

    Examples
    --------

    # The domain
    >>> from domain import Domain
    >>> d = Domain([[-1, 1], [-1, 1]], 21)

    # Helper
    >>> def sol(X, radius, center, epsilon, t, lp=2, reverse=False):
    ...     return profil((1. - 2. * reverse) * sphere_dist_MC(
    ...         torch.as_tensor(X),
    ...         radius,
    ...         t,
    ...         torch.as_tensor(center),
    ...         lp), epsilon)

    # Everything
    >>> ds = AllenCahnDataset(
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
        ds = AllenCahnLazyDataset(*args, **kwargs)
        super().__init__(*ds[:])


class AllenCahnProblem(Problem):
    """
    Base class for the Allen-Cahn equation problem

    Features the train and validation data, and the metric.

    Parameters
    ----------
    bounds: iterable of pairs of float
        Bounds of the domain
    N: int or iterable of int
        Number of discretization points
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step. epsilon**2 if None.
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
                 batch_size=10, batch_shuffle=True, lr=1e-3,
                 loss_norms=None, loss_power=2.,
                 radius=[0.05, 0.45], lp=2,
                 train_N=100, train_steps=1, train_reverse=0.,
                 val_N=200, val_steps=5, val_reverse=0.,
                 **kwargs):

        super().__init__(**kwargs)

        # Default values
        dt = dt or epsilon**2
        loss_norms = loss_norms or [[2, 1.]]

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
            'bounds', 'N', 'dt', 'epsilon',
            'batch_size', 'batch_shuffle', 'lr', 'loss_norms', 'loss_power',
            'radius', 'lp', 'train_N', 'val_N', 'train_steps', 'val_steps',
            'train_reverse', 'val_reverse',
        )

    @property
    def domain(self):
        return Domain(self.hparams.bounds, self.hparams.N, device=self.device)

    def loss(self, output, target):
        """ Default loss function """
        dim = tuple(range(2, 2 + self.domain.dim))
        error = target - output
        return self.domain.dX.prod() * sum(
            w * nn_toolbox.norm(error, p, dim).pow(self.hparams.loss_power)
            for p, w in self.hparams.loss_norms).mean()

    def check_sphere_volume(self, radius=0.45, num_steps=None, center=None, progress_bar=False):
        """
        Check an Allen-Cahn model by measuring sphere volume decreasing

        Note: Remember to freeze the model if gradient calculation is not needed!!!

        Parameters
        ----------
        radius: float
            Sphere radius (ratio of domain diameter)
        num_steps: int
            Number of time steps. If None, calculate it to reach radius 0.01 * domain diameter
        center: list of float
            Sphere center (domain center if None)
        progress_bar: bool
            True to display a progress bar

        Returns
        -------
        model_volume: torch.tensor
            Sphere volume evolution for the given model
        exact_volume: torch.tensor
            Sphere volume evolution of the solution
        """
        domain_diameter = min(b[1] - b[0] for b in self.domain.bounds)
        radius = radius * domain_diameter
        num_steps = num_steps or math.floor((radius**2 - (0.01 * domain_diameter)**2) / (2 * self.hparams.dt))

        with torch.no_grad():
            return check_sphere_volume(self, self.domain, radius, self.hparams.epsilon, self.hparams.dt, num_steps, center, p=self.hparams.lp, progress_bar=progress_bar)

    def training_step(self, batch, batch_idx):
        """ Default training step with custom loss function """
        data, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self.forward(data)
            loss += self.hparams.dt * self.loss(data, target)

        return self.dispatch_metrics({'loss': loss})

    def validation_step(self, batch, batch_idx):
        """ Called at each batch of the validation data """
        data, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data)
            loss += self.hparams.dt * self.loss(data, target)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """ Called at epoch end of the validation step (after all batches) """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Metric calculation
        model_volume, exact_volume = self.check_sphere_volume()
        volume_error = self.hparams.dt * (model_volume - exact_volume).norm()

        self.dispatch_metrics({'val_loss': avg_loss, 'metric': volume_error})

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
            return AllenCahnDataset(
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
        group = parser.add_argument_group("Allen-Cahn problem", "Options common to all models of Allen-Cahn equation.")
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
        group.set_defaults(**{**get_default_args(AllenCahnProblem), **defaults})

        return parser


