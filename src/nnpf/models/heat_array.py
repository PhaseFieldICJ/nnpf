#!/usr/bin/env python3

import torch
import argparse

from nnpf.problems import HeatProblem
from nnpf.nn import FFTConvolutionArray
from nnpf.functional import norm, heat_kernel_spatial
from nnpf.utils import get_default_args
from nnpf.trainer import Trainer


__all__ = [
    "HeatArray",
]


class HeatArray(HeatProblem):
    """
    Heat problem model using a discretized kernel.

    Parameters
    ----------
    kernel_size: int or list
        Size of the convolution kernel (a list for 2D and 3D convolutions)
    padding_mode: string
        'zeros', 'reflect', 'replicate' or 'circular'
    bias: bool
        If True, adds a learnable bias to the output.
    init: ['zeros', 'random', 'solution']
        Initialization of the convolution kernel:
        - 'random' for the default from PyTorch.
        - 'zeros' for zero-initialization.
        - 'solution' to initialize with the truncated heat kernel.
    kernel_norms: list of pair (p, weight)
        Compose kernel penalization term as sum of weight * kernel.norm(p).exp(e). Exponent e is defined with kernel_power parameter.
    kernel_power: int
        Power applied to the kernel penalization term (for regularization purpose).

    Examples
    --------

    Training:
    >>> from nnpf.trainer import Trainer
    >>> trainer = Trainer(default_root_dir="logs_doctest", name="HeatArray", version="test0", max_epochs=1, log_every_n_steps=1)
    >>> model = HeatArray(N=64, train_N=10, val_N=20, num_workers=4)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)
    >>> trainer.global_step > 0
    True

    Loading from checkpoint:
    >>> import os
    >>> from nnpf.problems import Problem
    >>> model = Problem.load_from_checkpoint(os.path.join('logs_doctest', 'HeatArray', 'test0'))
    >>> type(model).__name__
    'HeatArray'
    """

    def __init__(self, kernel_size=17, padding_mode='circular', bias=False, init='zeros', kernel_norms=[], kernel_power=2, **kwargs):
        super().__init__(**kwargs)

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('kernel_size', 'padding_mode', 'bias', 'init', 'kernel_norms', 'kernel_power')

        # Model
        self.model = FFTConvolutionArray(
            self.hparams.kernel_size,
            padding='center',
            padding_mode=self.hparams.padding_mode,
            bias=self.hparams.bias,
        )

        # Kernel initialization (FIXME: ugly)
        if not self.is_loaded:
            if self.hparams.init != 'random':
                with torch.no_grad():
                    self.model.weight[:] *= 0
                    if self.hparams.init == 'solution':
                        self.model.weight[:] += heat_kernel_spatial(self.domain, self.hparams.dt, self.hparams.kernel_size)

    @property
    def weight(self):
        return self.model.weight

    # TODO: auto reshape when no batch nor channel are included
    def forward(self, x):
        return self.model(x)

    def loss(self, output, target):
        dim = tuple(range(2, 2 + self.domain.dim))
        return super().loss(output, target) + sum(
            w * norm(self.model.weight, p, dim).squeeze().pow(self.hparams.kernel_power)
            for p, w in self.hparams.kernel_norms)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):

        from distutils.util import strtobool
        # Parser for loss definition
        def float_or_str(v):
            try:
                return float(v)
            except ValueError:
                return v

        parser = HeatProblem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Heat equation by convolution array", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--padding_mode', choices=['zeros', 'reflect', 'replicate', 'circular'], help="Padding mode for the convolution")
        group.add_argument('--bias', type=lambda s:bool(strtobool(s)), nargs='?', const=True, help="Add a bias to the convolution")
        group.add_argument('--init', choices=['zeros', 'random', 'solution'], help="Initialization of the convolution kernel")
        group.add_argument('--kernel_norms', type=float_or_str, nargs=2, action='append', help="List of (p, weight). Compose the kernel penalization term as sum of weight * kernel.norm(p).pow(e). Exponent e is defined with --kernel_power option.")
        group.add_argument('--kernel_power', type=float, help="Power applied to each penalization term (for regularization purpose)")
        group.set_defaults(**{**get_default_args(HeatArray), **defaults})
        return parser

