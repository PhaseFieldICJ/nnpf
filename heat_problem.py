#!/usr/bin/env python3
"""
Base module and utils for the Heat equation problem

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import math
import argparse

from domain import complex_mul, Domain
from problem import Problem
import shapes
import nn_toolbox
from phase_field import profil

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
    truncate: int or (int, int)
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


# TODO: move that in a dedicated file (maybe a part in shapes). It can be reused for other problems.
def generate_sphere_phase(num_samples, domain, radius, epsilon):
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


# TODO: move that in a dedicated file (maybe a part in shapes). It can be reused for other problems.
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


# TODO: move that in a dedicated file (maybe a part in shapes). It can be reused for other problems.
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


class HeatProblem(Problem):
    """
    Base class for the heat equation learning problem

    Features the train and validation data.
    """

    def __init__(self, bounds, N, dt, batch_size=None, batch_shuffle=False, lr=1e-3, loss_norms=[[2, 1.]], loss_power=2.,
                 train_N=100, train_radius=[0, 0.25], train_epsilon=[0, 0.1], train_num_shapes=1, train_steps=10,
                 val_N=100, val_radius=[0, 0.35], val_epsilon=[0, 0.2], val_num_shapes=[1, 3], val_steps=10,
                 **kwargs):
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
        train_N: int
            Number of samples for the training step
        val_N: int
            Number of samples for the validation step. 10*Ntrain if None.
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
        """

        super().__init__(**kwargs)

        loss_norms = loss_norms or [[2, 1.]]

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('bounds', 'N', 'dt', 'batch_size', 'batch_shuffle', 'lr', 'loss_norms', 'loss_power',
                                  'train_N', 'train_radius', 'train_epsilon', 'train_num_shapes', 'train_steps',
                                  'val_N', 'val_radius', 'val_epsilon', 'val_num_shapes', 'val_steps',)

        # Domain
        self.domain = Domain(self.hparams.bounds, self.hparams.N)

    def loss(self, output, target):
        """ Default loss function """
        dim = tuple(range(2, 2 + self.domain.dim))
        error = target - output
        return self.domain.dX.prod() * sum(
            w * nn_toolbox.norm(error, p, dim).pow(self.hparams.loss_power)
            for p, w in self.hparams.loss_norms).mean()

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
        metric_l2 = data.new_zeros([])
        for target in targets:
            data = self(data)
            metric_l2 += self.hparams.dt * self.domain.dX.prod() * (target - data).square().sum(dim=list(range(2, 2 + self.domain.dim))).sqrt().mean()
            loss += self.hparams.dt * self.loss(data, target)

        return {'val_loss': loss, 'metric_l2': metric_l2}

    def validation_epoch_end(self, outputs):
        """ Called at epoch end of the validation step (after all batches) """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_l2_metric = torch.stack([x['metric_l2'] for x in outputs]).mean()
        return self.dispatch_metrics({'val_loss': avg_loss, 'metric': avg_l2_metric})

    def configure_optimizers(self):
        """ Default optimizer """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # TODO: data generated by a dedicated script, stored in an archive
    def prepare_data(self):
        """ Prepare training and validation data """

        exact_sol = HeatSolution(self.domain, self.hparams.dt)

        def generate_data(num_samples, radius, epsilon, num_shapes, steps):
            base_shape_gen = lambda num_samples: generate_sphere_phase(num_samples, self.domain, radius, epsilon)
            data = generate_phase_field_union(num_samples, base_shape_gen, num_shapes)
            data = evolve_phase_field(exact_sol, data, steps + 1)
            return tuple(data[i, ...] for i in range(data.shape[0]))

        # Training dataset
        train_x, *train_y = generate_data(
            self.hparams.train_N,
            self.hparams.train_radius,
            self.hparams.train_epsilon,
            self.hparams.train_num_shapes,
            self.hparams.train_steps,
        )
        self.train_dataset = TensorDataset(train_x, *train_y)

        # Validation dataset
        val_x, *val_y = generate_data(
            self.hparams.val_N,
            self.hparams.val_radius,
            self.hparams.val_epsilon,
            self.hparams.val_num_shapes,
            self.hparams.val_steps,
        )
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

        parser = Problem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Heat equation problem", "Options common to all models of the heat equation.")
        group.add_argument('--bounds', type=bounds_parser, default=[[0., 1.],[0., 1.]], help="Domain bounds in format like '[0, 1]x[1, 2.5]'")
        group.add_argument('--N', type=int, nargs='+', default=256, help="Domain discretization")
        group.add_argument('--dt', type=float, default=(2 / 256)**2, help="Time step")
        group.add_argument('--train_N', type=int, default=100, help="Number of initial conditions in the training dataset")
        group.add_argument('--val_N', type=int, default=100, help="Number of initial conditions in the validation dataset")
        group.add_argument('--train_steps', type=int, default=1, help="Number of evolution steps in the training dataset")
        group.add_argument('--val_steps', type=int, default=10, help="Number of evolution steps in the validation dataset")
        group.add_argument('--batch_size', type=int, default=None, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(int(v)), default=False, help="Shuffle batch (1 to activate)")
        group.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
        group.add_argument('--loss_norms', type=float_or_str, nargs=2, action='append', help="List of (p, weight). Compose loss as sum of weight * (output - target).norm(p).pow(e). Default to l2 norm. Exponent e is defined with loss_power parameter.")
        group.add_argument('--loss_power', type=float, default=2., help="Power applied to each loss term (for regularization purpose)")

        return parser


###############################################################################
# Command-line interface
###############################################################################

if __name__ == "__main__":

    import problem
    import heat_problem

    # Command-line arguments
    parser = argparse.ArgumentParser(
            description="Evaluation of a model for the heat equation problem",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', type=str, help="Path to the model's checkpoint")
    parser.add_argument('--spheres', type=int, nargs='+', default=[1], help="Number (or interval) of spheres")
    parser.add_argument('--radius', type=float, nargs='+', default=[0.2], help="Radius (or interval) of the spheres")
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.01], help="Interface sharpness (or interval) of the phase field")
    parser.add_argument('--steps', type=int, default=1, help="Number of evolution steps")
    args = parser.parse_args()

    # Expanding scalar to bounds
    args.spheres += args.spheres * (2 - len(args.spheres))
    args.radius += args.radius * (2 - len(args.radius))
    args.epsilon += args.epsilon * (2 - len(args.epsilon))

    # Loading model
    checkpoint_path = problem.checkpoint_from_path(args.checkpoint)
    extra_data = torch.load(checkpoint_path)
    model = heat_problem.HeatProblem.load_from_checkpoint(args.checkpoint)
    model.freeze()

    # Some informations about the model
    print(f"""
Model summary:
    checkpoint path: {checkpoint_path}
    epochs: {extra_data['epoch']}
    steps: {extra_data['global_step']}
    best score: {extra_data['checkpoint_callback_best_model_score']}
    ndof: {nn_toolbox.ndof(model)}
""")

    print("Model hyper parameters:")
    for key, value in model.hparams.items():
        print(f"    {key}: {value}")

    # Exact solution
    exact_sol = heat_problem.HeatSolution(model.domain, model.hparams.dt)

    # Generate initial data
    exact_data = generate_phase_field_union(
            1,
            lambda num_samples: generate_sphere_phase(num_samples,
                                                      model.domain,
                                                      args.radius,
                                                      args.epsilon),
            args.spheres)
    model_data = exact_data.clone()

    # Display
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=args.steps + 1, ncols=3)

    for i in range(args.steps + 1):
        # Exact solution
        axes[i, 0].imshow(exact_data[0, 0, ...])
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Solution')

        # Model solution
        axes[i, 1].imshow(model_data[0, 0, ...])
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Model')

        # Difference
        im = axes[i, 2].imshow((exact_data - model_data)[0, 0, ...])
        fig.colorbar(im, ax=axes[i, 2])
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Difference')

        # Evolution
        exact_data = exact_sol(exact_data)
        model_data = model(model_data)

    fig.tight_layout(pad=0, w_pad=-10, h_pad=0)
    plt.show()

