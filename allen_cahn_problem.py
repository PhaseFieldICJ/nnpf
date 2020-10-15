"""
Base module and utils for the Allen-Cahn equation learning problem
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import math

from problem import Problem
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
    radius: float
        Sphere radius
    t: float
        Evaluation time
    center: list of float
        Sphere center
    p: int, float or float("inf")
        Power in the lp-norm

    Returns
    -------
    dist: Tensor
        The signed distance field
    """
    radius = torch.as_tensor(radius**2 - 2 * t, device=X[0].device, dtype=X[0].dtype) \
                  .max(X[0].new_zeros(1)).sqrt()
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


class AllenCahnProblem(Problem):
    """
    Base class for the Allen-Cahn equation problem

    Features the train and validation data, and the metric.
    """

    def __init__(self, bounds, N, epsilon, dt=None,
                 batch_size=None, batch_shuffle=None, lr=1e-3,
                 loss_norms=[[2, 1.]], loss_power=2.,
                 radius=[0.05, 0.45], lp=2,
                 train_N=10, train_steps=1, val_N=20, val_steps=5,
                 **kwargs):
        """ Constructor

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
            Exponent e is defined with loss_power parameter.
        loss_power: float
            Power applied to each loss term (for regularization purpose).
        radius: tuple or list of 2 floats
            Bounds on sphere radius (ratio of domain bounds) used for training and validation datasets.
        lp: int or float
            Power of the lp-norm used to defined the spheres in training and validation datasets.
        train_N, val_N: int
            Size of the training and validation datasets
        train_steps, val_steps: int
            Number of evolution step applied to each input
        kwargs: dict
            Parameters passed to Problem (see doc)
        """

        super().__init__(**kwargs)

        # Default values
        dt = dt or epsilon**2
        loss_norms = loss_norms or [[2, 1.]]

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
            'bounds', 'N', 'dt', 'epsilon',
            'batch_size', 'batch_shuffle', 'lr', 'loss_norms', 'loss_power',
            'radius', 'lp', 'train_N', 'val_N', 'train_steps', 'val_steps',
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

        return self.dispatch_metrics({'val_loss': avg_loss, 'metric': volume_error})

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

        def generate_data(num_samples, steps):
            return tuple(
                profil(
                    sphere_dist_MC(
                        self.domain.X,
                        radius=torch.linspace(radius[0], radius[1], num_samples).reshape(-1, *sup_dims),
                        center=center,
                        t=i*self.hparams.dt,
                        p=self.hparams.lp),
                    self.hparams.epsilon)
                for i in range(steps + 1))

        # Training dataset
        train_x, *train_y = generate_data(self.hparams.train_N, self.hparams.train_steps)
        self.train_dataset = TensorDataset(train_x, *train_y)

        # Validation dataset
        val_x, *val_y = generate_data(self.hparams.val_N, self.hparams.val_steps)
        self.val_dataset = TensorDataset(val_x, *val_y)

    def train_dataloader(self):
        """ Returns the training data loader """
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size or len(self.train_dataset), shuffle=self.hparams.batch_shuffle)

    def val_dataloader(self):
        """ Returns the validation data loader """
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size or len(self.val_dataset))

    @staticmethod
    def add_model_specific_args(parent_parser):
        import re

        # Parser for the domain bounds
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

        parser = Problem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Allen-Cahn problem", "Options common to all models of Allen-Cahn equation.")
        group.add_argument('--bounds', type=bounds_parser, default=[[0., 1.],[0., 1.]], help="Domain bounds in format like '[0, 1]x[1, 2.5]'")
        group.add_argument('--N', type=int, nargs='+', default=256, help="Domain discretization")
        group.add_argument('--epsilon', type=float, default=2/8**3, help="Interface sharpness")
        group.add_argument('--dt', type=float, default=None, help="Time step (epsilon**2 if None)")
        group.add_argument('--train_N', type=int, default=100, help="Number of initial conditions in the training dataset")
        group.add_argument('--val_N', type=int, default=200, help="Number of initial conditions in the validation dataset")
        group.add_argument('--train_steps', type=int, default=1, help="Number of evolution steps in the training dataset")
        group.add_argument('--val_steps', type=int, default=10, help="Number of evolution steps in the validation dataset")
        group.add_argument('--radius', type=float, nargs=2, default=[0.05, 0.45], help="Bounds on sphere radius (ratio of domain bounds) used for training and validation dataset.")
        group.add_argument('--lp', type=int_or_float, default=2, help="Power of the lp-norm used to define the spheres for training and validation dataset.")
        group.add_argument('--batch_size', type=int, default=None, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(int(v)), default=False, help="Shuffle batch (1 to activate)")
        group.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
        group.add_argument('--loss_norms', type=float_or_str, nargs=2, action='append', help="List of (p, weight). Compose loss as sum of weight * (output - target).norm(p).pow(e). Default to l2 norm. Exponent e is defined with loss_power parameter.")
        group.add_argument('--loss_power', type=float, default=2., help="Power applied to each loss term (for regularization purpose)")

        return parser


