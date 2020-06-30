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
    """ Return the discretizes heat kernel in frequency space.

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


def heat_kernel(domain, dt, truncate=None):
    """ Return the discretizes heat kernel.

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

    def _generate_peaks(self, num_samples, min_radius, max_radius, min_epsilon, max_epsilon, min_shapes, max_shapes, num_steps):

        # Generate one shapes
        sup_dims = (1,) + self.domain.dim * (1,)
        origin = torch.Tensor([a for a, b in self.domain.bounds]).resize_((self.domain.dim, 1) + sup_dims)
        width = torch.Tensor([b - a for a, b in self.domain.bounds]).resize_((self.domain.dim, 1) + sup_dims)

        def gen_shape_dist(Ns):
            centers = origin + width * torch.rand((self.domain.dim, Ns) + sup_dims) 
            radius = min_radius + (max_radius - min_radius) * torch.rand((Ns,) + sup_dims)
            shape = shapes.periodic(shapes.sphere(centers, radius), self.domain.bounds)
            return shape(*self.domain.X)

        # Number of peaks
        peak_counts = torch.randint(min_shapes, max_shapes + 1, (num_samples,))

        # Initial distance field
        data = torch.empty(num_samples * num_steps, 1, *self.domain.N)
        data[:num_samples] = gen_shape_dist(num_samples)

        # Additional shapes
        for i in range(1, max_peaks + 1):
            mask = peak_counts == i
            data[:num_samples][mask].min(gen_shape_dist(mask.sum().item()))

        # Phase field
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * torch.rand((num_shapes,) + sup_dims)
        data[:num_shapes] = 0.5 * (1 - torch.tanh(dist[:num_shapes] / (2 * epsilon)))

        return data
        


    def prepare_data(self):
        """ Prepare training and validation data """

        exact_sol = HeatSolution(self.domain, self.hparams.dt)


        # Training dataset
        train_x = torch.linspace(lower_bound, upper_bound, self.hparams.Ntrain)[:, None]
        train_y = exact_sol(train_x)
        self.train_dataset = TensorDataset(train_x, train_y)

        # Validation dataset
        val_x = torch.linspace(lower_bound, upper_bound, self.hparams.Nval)[:, None]
        val_y = exact_sol(val_x)
        self.val_dataset = TensorDataset(val_x, val_y)

