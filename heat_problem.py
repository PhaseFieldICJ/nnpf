"""
Base module and utils for the Heat equation problem

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import math
import argparse

from domain import complex_mul, Domain
import shapes

def heat_kernel_freq(domain, dt):
    """ Return the discretizes heat kernel in frequency domain.

    Parameters
    ----------
    domain: domain.Domain
        The discretized domain
    dt: float
        Time step

    Returns
    -------
    kernel: Tensor
        The discretized heat kernel (complex format)
    """
    kernel = torch.exp(-(2 * math.pi)**domain.dim * sum(k**2 for k in domain.K) * dt)
    return kernel[..., None] * torch.Tensor([1., 0.]) # To complex format


def heat_kernel_spatial(domain, dt, truncate=None):
    """ Return the discretizes heat kernel in spatial domain.

    Parameters
    ----------
    domain: domain.Domain
        The discretized domain
    dt: float
        Time step
    truncate: int or tuple of int
        Truncate kernel to have the given size.

    Returns
    -------
    kernel: numpy.array
        The discretized heat kernel.
    """
    kernel_freq = heat_kernel_freq(domain, dt)
    kernel_spatial = domain.spatial_shift(domain.ifft(kernel_freq))

    # Truncating kernel
    if truncate is not None:
        if type(truncate) == int:
            truncate = [truncate] * domain.dim

        indexing = tuple(slice(n//2 - t//2, n//2 + (t - t//2), None) for n, t in zip(domain.N, truncate))
        kernel_spatial = kernel_spatial[indexing]

    return kernel_spatial


def generate_sphere_phase(num_samples, domain, min_radius, max_radius, min_epsilon, max_epsilon):
    """
    Generates phase field function for one sphere per sample

    With uniform random position, radius and interface's sharpness.

    Parameters
    ----------
    num_samples: int
        Number of samples
    domain: domain.Domain
        Discretization domain
    min_radius, max_radius: double
        Bounds on the radius
    min_epsilon, max_epsilon: double
        Bounds on the sharpness of the interface

    Returns
    -------
    data: Tensor
        Phase field function of a sphere foreach sample
    """

    sup_dims = (1,) + domain.dim * (1,) # Additionnal dimensions for appropriate broadcasting
    origin = torch.Tensor([a for a, b in domain.bounds]).resize_((domain.dim, 1) + sup_dims)
    width = torch.Tensor([b - a for a, b in domain.bounds]).resize_((domain.dim, 1) + sup_dims)
    centers = origin + width * torch.rand((domain.dim, num_samples) + sup_dims)
    radius = min_radius + (max_radius - min_radius) * torch.rand((num_samples,) + sup_dims)
    shape = shapes.periodic(shapes.sphere(centers, radius), domain.bounds)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * torch.rand((num_samples,) + sup_dims)
    return 0.5 * (1 - torch.tanh(shape(*domain.X) / (2 * epsilon)))


def generate_phase_field_union(num_samples, phase_field_gen, min_shapes, max_shapes):
    """
    Union of random number of phase fields

    Parameters
    ----------
    num_samples: int
        Number of samples
    phase_field_gen: function
        Given a number of samples, returns one phase field per sample
    min_shapes, max_shapes: int
        Bounds on the number of superimposed phase fields

    Returns
    -------
    data: Tensor
        Phase field function foreach sample
    """

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
    num_samples = phase_fields.shapes[0]
    data = phase_fields.new_empty(num_samples * (steps + 1), *phase_fields.shape[1:])
    data[:num_samples] = phase_fields
    for i in range(steps):
        data[((i + 1) * num_samples):((i + 2) * num_samples)] = operator(data[(i * num_samples):((i + 1) * num_samples)])

    return data


class HeatSolution:
    """
    Exact solution to the heat equation
    """

    def __init__(self, domain, dt):
        """ Constructor

        Parameters
        ----------
        domain: domain.Domain
            Spatial domain
        dt: float
            Time step.
        """

        self.domain = domain
        self.dt = dt
        self.kernel = heat_kernel_freq(self.domain, self.dt)

    def __call__(self, u):
        """
        Returns u(t + dt) from u(t)

        Support batch and channels
        """
        return self.domain.ifft(complex_mul(self.kernel, self.domain.fft(u)))


class HeatProblem(pl.LightningModule):
    """
    Base class for the heat equation learning proble

    Features the train and validation data.
    """

    def __init__(self, bounds, N, dt, Ntrain=100, Nval=None, batch_size=None, lr=1e-3, seed=None, **kwargs):
        """ Constructor

        Parameters
        ----------
        bounds: iterable of pairs of float
            Bounds of the domain
        N: int or iterable of int
            Number of discretization points
        dt: float
            Time step.
        margin: float
            Expanding length of the sampled [0, 1] interval
        Ntrain: int
            Number of samples for the training step
        Nval: int
            Number of samples for the validation step. 10*Ntrain if None.
        batch_size: int
            Size of the batch during training and validation steps. Full data if None.
        lr: float
            Learning rate of the optimizer
        seed: int
            If set to an integer, use it as seed of all random generators.
        """

        super().__init__()

        # Default values
        Nval = Nval or 10 * Ntrain

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('bounds', 'N', 'dt', 'Ntrain', 'Nval', 'batch_size', 'lr', 'seed')

        # Seed random generators
        if self.hparams.seed is not None:
            pl.seed_everything(self.hparams.seed)
            # Should also enable deterministic behavior in the trainer parameters

        # Domain
        self.domain = Domain(self.hparams.bounds, self.hparams.N)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        """ Default training step with custom loss function """
        data, target = batch
        output = self.forward(data)

        try:
            loss = self.loss_fn(output, target)
        except AttributeError:
            loss = torch.nn.functional.mse_loss(output, target)

        return dispatch_metrics({'loss': loss})

    def configure_optimizers(self):
        """ Default optimizer """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def prepare_data(self):
        """ Prepare training and validation data """

        exact_sol = HeatSolution(self.domain, self.hparams.dt)

        def generate_data(num_samples, min_radius, max_radius, min_epsilon, max_epsilon, min_shapes, max_shapes, steps):
            base_shape_gen = lambda num_samples: generate_sphere_phase(num_samples, self.domain, min_radius, max_radius, min_epsilon, max_epsilon)
            data = generate_phase_field_union(num_samples, base_shape_gen, min_shapes, max_shapes)
            data = evolve_phase_field(exact_sol, data, steps + 1)
            return data[:(steps + 1) * num_samples], data[num_samples:(steps + 2) * num_samples]

        # Training dataset
        train_x, train_y = generate_data(
            self.hparams.train_N,
            self.hparams.train_min_radius,
            self.hparams.train_max_radius,
            self.hparams.train_min_epsilon,
            self.hparams.train_max_epsilon,
            self.hparams.train_min_shapes,
            self.hparams.train_max_shapes,
            self.hparams.train_steps,
        )
        self.train_dataset = TensorDataset(train_x, train_y)

        # Validation dataset
        val_x, val_y = generate_data(
            self.hparams.val_N,
            self.hparams.val_min_radius,
            self.hparams.val_max_radius,
            self.hparams.val_min_epsilon,
            self.hparams.val_max_epsilon,
            self.hparams.val_min_shapes,
            self.hparams.val_max_shapes,
            self.hparams.val_steps,
        )
        self.val_dataset = TensorDataset(val_x, val_y)

