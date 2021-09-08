#!/usr/bin/env python3

import torch
import torch.nn as nn
import argparse

from nnpf.problems import ReactionProblem
from nnpf.utils import get_default_args
from nnpf.trainer import Trainer
from nnpf.nn import Function, get_model_by_name


__all__ = [
    "Reaction",
]


class Reaction(ReactionProblem):
    """
    Reaction problem model using a Multiple Layer Perceptron

    Parameters
    ----------
    layer_dims: iterable of int
        Working dimensions of the hidden layers
    activation: str
        Name of the activation function class
    kwargs: dict
        Parameters passed to reaction_problem.ReactionProblem

    Examples
    --------

    Training:
    >>> from nnpf.trainer import Trainer
    >>> trainer = Trainer(default_root_dir="logs_doctest", name="Reaction", version="test0", max_epochs=1, log_every_n_steps=1)
    >>> model = Reaction(train_N=10, val_N=20, seed=0, num_workers=4)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)
    >>> trainer.global_step > 0
    True

    Loading from checkpoint:
    >>> import os
    >>> from nnpf.problems import Problem
    >>> model = Problem.load_from_checkpoint(os.path.join('logs_doctest', 'Reaction', 'test0'))
    >>> type(model).__name__
    'Reaction'
    """

    def __init__(self, layer_dims=[8, 3], activation='GaussActivation', **kwargs):
        super().__init__(**kwargs)

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('layer_dims', 'activation')

        # Model
        activation_class = get_model_by_name(self.hparams.activation)
        layers = ((activation_class(), d) for d in self.hparams.layer_dims)
        self.model = Function(1, 1, *layers)

    def forward(self, x):
        return self.model(x.reshape(-1, 1)).reshape(x.shape)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = ReactionProblem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Reaction model", "Options specific to this model")
        group.add_argument('--layer_dims', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
        group.set_defaults(**{**get_default_args(Reaction), **defaults})
        return parser

