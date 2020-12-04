#!/usr/bin/env python3

import torch
import argparse

from willmore_problem import WillmoreProblem
from allen_cahn_problem import AllenCahnProblem
from reaction_model import Reaction
from heat_array_model import HeatArray
from problem import get_default_args
from trainer import Trainer
from nn_models import Parallel, LinearChannels
from torch.nn import Sequential

#Problem = AllenCahnProblem
Problem = WillmoreProblem

class WillmoreParallel(Problem):
    """
    Experimental class for modeling the Willmore problem using parallel stack of base schemes

    Examples
    --------

    Training:
    >>> from trainer import Trainer
    >>> trainer = Trainer(default_root_dir="logs_doctest", name="WillmoreParallel", version="test0", max_epochs=1)
    >>> model = WillmoreParallel(N=64, train_N=10, val_N=20)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)
    >>> trainer.global_step > 0
    True

    Loading from checkpoint:
    >>> import os
    >>> from problem import Problem
    >>> model = Problem.load_from_checkpoint(os.path.join('logs_doctest', 'WillmoreParallel', 'test0'))
    >>> type(model).__name__
    'WillmoreParallel'
    """

    def __init__(self, scheme="DR", scheme_layers=[4], scheme_repeat=1, prefix="", suffix="",
                 kernel_size=17, kernel_init='zeros',
                 reaction_layers=[8, 3], reaction_activation='GaussActivation',
                 input_linear=False, input_bias=False, # LinearChannels before the Parallel
                 output_linear=True, output_bias=True, # LinearChannels after the Parallel (needed if depth > 1)
                 **kwargs):
        super().__init__(**kwargs)

        assert output_linear or depth == 1, "Output LinearChannels is mandatory if depth > 1"

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('scheme', 'scheme_layers', 'scheme_repeat', 'prefix', 'suffix',
                                  'kernel_size', 'kernel_init',
                                  'reaction_layers', 'reaction_activation',
                                  'input_linear', 'input_bias',
                                  'output_linear', 'output_bias',)


        # Helper function
        def str_to_module(scheme):
            module = Sequential()
            for i, t in enumerate(scheme):
                if t == 'R':
                    module.add_module(str(i), Reaction(reaction_layers, reaction_activation))
                elif t == 'D':
                    module.add_module(str(i), HeatArray(kernel_size=kernel_size, init=kernel_init,
                                                        bounds=self.hparams.bounds,
                                                        N=self.hparams.N,))
                else:
                    raise ValueError(f"Unknow module type {t} in scheme {scheme}")
            return module

        # Creating model
        self.model = Sequential()

        # Prefix
        if prefix:
            self.model.add_module("prefix", str_to_module(prefix))

        # Input layer
        if input_linear:
            self.model.add_module("input", LinearChannels(1, scheme_layers[0], bias=input_bias))

        # Parallel layers
        last_dim = None
        for i, dim in enumerate(scheme_layers):
            if last_dim is not None:
                self.model.add_module(f"linear{i-1}", LinearChannels(last_dim, dim, bias=True))
            parallel = Parallel(*[str_to_module(scheme * scheme_repeat) for _ in range(dim)])
            self.model.add_module(f"parallel{i}", parallel)
            last_dim = dim

        # Output layer
        if output_linear:
            self.model.add_module("output", LinearChannels(last_dim, 1, bias=output_bias))

        # Suffix
        if suffix:
            self.model.add_module("suffix", str_to_module(suffix))

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):

        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Willmore parallel", "Options specific to this model")
        group.add_argument('--scheme', type=str, help='Base scheme (sequence of R & D)')
        group.add_argument('--scheme_layers', type=int, nargs='+', help='Sizes of the hidden Parallel layers')
        group.add_argument('--scheme_repeat', type=int, help='Number of repetition of the scheme in each parallel layer')
        group.add_argument('--prefix', type=str, help='Prefix scheme (sequence of R & D)')
        group.add_argument('--suffix', type=str, help='Suffix scheme (sequence of R & D)')
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--reaction_layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--reaction_activation', type=str, help='Name of the activation function')
        group.add_argument('--input_linear', type=lambda s: bool(strtobool(s)), nargs='?', const=True, help="Add a LinearChannel before the Parallel")
        group.add_argument('--input_bias', type=lambda s: bool(strtobool(s)), nargs='?', const=True, help="Add a bias in the input LinearChannel")
        group.add_argument('--output_linear', type=lambda s: bool(strtobool(s)), nargs='?', const=True, help="Add a LinearChannel after the Parallel")
        group.add_argument('--output_bias', type=lambda s: bool(strtobool(s)), nargs='?', const=True, help="Add a bias in the output LinearChannel")
        group.set_defaults(**{**get_default_args(WillmoreParallel), **defaults})
        return parser


if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Model of the Willmore equation using parallel stack of repeated sequences of given scheme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser, dict(name="WillmoreParallel"))
    parser = WillmoreParallel.add_model_specific_args(parser, WillmoreParallel.defaults_from_config())
    args = parser.parse_args()

    # Model, training & fit
    model = WillmoreParallel(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

