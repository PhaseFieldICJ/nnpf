"""
Base module and utils for the mean curvature based-equation learning problem
"""

import torch
from torch.utils.data import DataLoader

from nnpf.utils import get_default_args
from nnpf.problems import Problem
from nnpf.domain import Domain
from nnpf.functional import norm
import nnpf.shapes as shapes
from .datasets import MCSphereDataset, MCSphereLazyDataset
from .utils import check_sphere_mass


__all__ = [
    "MeanCurvatureProblem",
]


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
                 use_lazy_datasets=False, num_workers=0,
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
            'train_reverse', 'val_reverse', 'use_lazy_datasets',
            'num_workers',
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

    def shape(self, radius, center, p=2):
        """ Reference shape (sphere by design or any shape with same signature) """
        return shapes.sphere(radius, center, p)

    @property
    def domain(self):
        return Domain(self.hparams.bounds, self.hparams.N, device=self.device)

    @property
    def example_input_array(self):
        """ Example of input (for graph generation) """
        return torch.rand(self.hparams.batch_size or self.hparams.train_N, 1, *self.domain.N)

    def loss(self, output, target):
        """ Default loss function """
        dim = tuple(range(2, 2 + self.domain.dim))
        error = target - output
        return sum(
            w * norm(error, p, dim).pow(self.hparams.loss_power)
            for p, w in self.hparams.loss_norms).mean() / torch.tensor(self.domain.N).prod()

        # Below, loss with rescale depending on the solution norm (i.e. the circle size)
        # The idea was to compensate the less importance of small circle in the final loss
        # Need a lit more work on the scaling to work good
        """
        return sum(
            w * (norm(error, p, dim) / norm(target, p, dim)).pow(self.hparams.loss_power)
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
                    n, center, p=self.hparams.lp, progress_bar=progress_bar,
                    shape=self.shape)
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
        Dataset = MCSphereLazyDataset if self.hparams.use_lazy_datasets else MCSphereDataset
        def generate_data(num_samples, steps, reverse):
            return Dataset(
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
                shape=self.shape,
            )

        self.train_dataset = generate_data(self.hparams.train_N, self.hparams.train_steps, self.hparams.train_reverse)
        self.val_dataset = generate_data(self.hparams.val_N, self.hparams.val_steps, self.hparams.val_reverse)

    def train_dataloader(self):
        """ Returns the training data loader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size or len(self.train_dataset),
            shuffle=self.hparams.batch_shuffle,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        """ Returns the validation data loader """
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size or len(self.val_dataset),
            num_workers=self.hparams.num_workers
        )

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
        group.add_argument('--train_reverse', type=float, help="Probability of having a sample with reversed inside and outside in training dataset")
        group.add_argument('--val_reverse', type=float, help="Probability of having a sample with reversed inside and outside in validation dataset")
        group.add_argument('--radius', type=float, nargs=2, help="Bounds on sphere radius (ratio of domain bounds) used for training and validation dataset.")
        group.add_argument('--lp', type=int_or_float, help="Power of the lp-norm used to define the spheres for training and validation dataset.")
        group.add_argument('--batch_size', type=int, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Shuffle batch")
        group.add_argument('--lr', type=float, help="Learning rate")
        group.add_argument('--loss_norms', type=float_or_str, nargs=2, action='append', help="List of (p, weight). Compose loss as sum of weight * (output - target).norm(p).pow(e). Default to l2 norm. Exponent e is defined with loss_power parameter.")
        group.add_argument('--loss_power', type=float, help="Power applied to each loss term (for regularization purpose)")
        group.add_argument('--use_lazy_datasets', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Use lazy-evaluation datasets")
        group.add_argument('--num_workers', type=int, help="Number of subprocesses used for data loading")
        group.set_defaults(**{**get_default_args(MeanCurvatureProblem), **defaults})

        return parser


