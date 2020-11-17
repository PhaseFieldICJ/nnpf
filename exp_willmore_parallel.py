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

    def __init__(self, depth=4, width=1, scheme="DR",
                 kernel_size=17, init='zeros',
                 layer_dims=[8, 3], activation='GaussActivation',
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
        self.save_hyperparameters('depth', 'width', 'scheme',
                                  'kernel_size', 'init',
                                  'layer_dims', 'activation',
                                  'input_linear', 'input_bias',
                                  'output_linear', 'output_bias',)

        # Creating model
        self.model = Sequential()

        # Input layer
        if input_linear:
            self.model.add_module("input", LinearChannels(1, depth, bias=input_bias))

        # Parallel layer
        parallel = Parallel()
        for d in range(depth):
            module = Sequential()
            for i, t in enumerate(scheme):
                if t == 'R':
                    module.add_module(str(i), Reaction(layer_dims, activation))
                elif t == 'D':
                    module.add_module(str(i), HeatArray(kernel_size=kernel_size, init=init,
                                                        bounds=self.hparams.bounds,
                                                        N=self.hparams.N,))
                else:
                    raise ValueError(f"Unknow module type {t} in scheme {scheme}")
            parallel.add_module(str(d), module)
        self.model.add_module("parallel", parallel)

        # Output layer
        if output_linear:
            self.model.add_module("output", LinearChannels(depth, 1, bias=output_bias))

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):

        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Willmore parallel", "Options specific to this model")
        group.add_argument('--depth', type=int, help='Number of layers in the Parallel container')
        group.add_argument('--width', type=int, help='Number of repetition of the scheme in each parallel layer')
        group.add_argument('--scheme', type=str, help='Base scheme (sequence of R & D)')
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--layer_dims', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
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
    parser = Trainer.add_argparse_args(parser)
    parser = WillmoreParallel.add_model_specific_args(parser, WillmoreParallel.defaults_from_config())
    args = parser.parse_args()

    # Model, training & fit
    model = WillmoreParallel(**vars(args))
    trainer = Trainer.from_argparse_args(args, "WillmoreParallel")
    trainer.fit(model)

