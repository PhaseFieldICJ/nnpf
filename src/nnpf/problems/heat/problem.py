#!/usr/bin/env python3
"""
Base module and utils for the Heat equation problem

"""

import torch
from torch.utils.data import DataLoader
import math
import argparse

from nnpf.utils import get_default_args
from nnpf.problems import Problem
from nnpf.domain import Domain
from nnpf.functional import norm
from .datasets import HeatDataset


__all__ = [
    "HeatProblem",
]


class HeatProblem(Problem):
    """
    Base class for the heat equation learning problem

    Features the train and validation data.

    Parameters
    ----------
    bounds: iterable of pairs of float
        Bounds of the domain
    N: int or iterable of int
        Number of discretization points
    dt: float
        Time step.
    train_N: int
        Number of samples for the training step
    val_N: int
        Number of samples for the validation step. 10*train_N if None.
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
    """

    def __init__(self, bounds=[[0., 1.], [0., 1.]], N=256, dt=(2 / 256)**2,
                 batch_size=10, batch_shuffle=True, lr=1e-3, loss_norms=None, loss_power=2.,
                 train_N=100, train_radius=[0, 0.25], train_epsilon=[0, 0.1], train_num_shapes=1, train_steps=10,
                 val_N=100, val_radius=[0, 0.35], val_epsilon=[0, 0.2], val_num_shapes=[1, 3], val_steps=10,
                 **kwargs):

        super().__init__(**kwargs)

        loss_norms = loss_norms or [[2, 1.]]

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('bounds', 'N', 'dt', 'batch_size', 'batch_shuffle', 'lr', 'loss_norms', 'loss_power',
                                  'train_N', 'train_radius', 'train_epsilon', 'train_num_shapes', 'train_steps',
                                  'val_N', 'val_radius', 'val_epsilon', 'val_num_shapes', 'val_steps',)

    @property
    def domain(self):
        """
        Domain associated to the problem

        As a property so that to generate it on the save device as the model
        """
        return Domain(self.hparams.bounds, self.hparams.N, device=self.device)

    @property
    def example_input_array(self):
        """ Example of input (for graph generation) """
        return torch.rand(1, 1, *self.domain.N)

    def loss(self, output, target):
        """ Default loss function """
        dim = tuple(range(2, 2 + self.domain.dim))
        error = target - output
        return self.domain.dX.prod() * sum(
            w * norm(error, p, dim).pow(self.hparams.loss_power)
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

        self.dispatch_metrics({'val_loss': loss, 'metric': metric_l2})

    def configure_optimizers(self):
        """ Default optimizer """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # TODO: data generated by a dedicated script, stored in an archive
    def prepare_data(self):
        """ Prepare training and validation data """
        self.train_dataset = HeatDataset(
            self.hparams.train_N,
            self.hparams.bounds,
            self.hparams.N,
            self.hparams.dt,
            self.hparams.train_radius,
            self.hparams.train_epsilon,
            self.hparams.train_num_shapes,
            self.hparams.train_steps,
        )

        self.val_dataset = HeatDataset(
            self.hparams.train_N,
            self.hparams.bounds,
            self.hparams.N,
            self.hparams.dt,
            self.hparams.val_radius,
            self.hparams.val_epsilon,
            self.hparams.val_num_shapes,
            self.hparams.val_steps,
        )

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

        from distutils.util import strtobool

        # Parser for loss definition
        def float_or_str(v):
            try:
                return float(v)
            except ValueError:
                return v

        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Heat equation problem", "Options common to all models of the heat equation.")
        group.add_argument('--bounds', type=bounds_parser, help="Domain bounds in format like '[0, 1]x[1, 2.5]'")
        group.add_argument('--N', type=int, nargs='+', help="Domain discretization")
        group.add_argument('--dt', type=float, help="Time step")
        group.add_argument('--train_N', type=int, help="Number of initial conditions in the training dataset")
        group.add_argument('--val_N', type=int, help="Number of initial conditions in the validation dataset")
        group.add_argument('--train_steps', type=int, help="Number of evolution steps in the training dataset")
        group.add_argument('--val_steps', type=int, help="Number of evolution steps in the validation dataset")
        group.add_argument('--batch_size', type=int, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Shuffle batch")
        group.add_argument('--lr', type=float, help="Learning rate")
        group.add_argument('--loss_norms', type=float_or_str, nargs=2, action='append', help="List of (p, weight). Compose loss as sum of weight * (output - target).norm(p).pow(e). Default to l2 norm. Exponent e is defined with loss_power parameter.")
        group.add_argument('--loss_power', type=float, help="Power applied to each loss term (for regularization purpose)")
        group.set_defaults(**{**get_default_args(HeatProblem), **defaults})

        return parser

