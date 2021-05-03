#!/usr/bin/env python3

import torch
import argparse

from nnpf.problems import SteinerProblem
from nnpf.models import Reaction, HeatArray
from nnpf.utils import get_default_args
from nnpf.nn import Parallel, LinearChannels
from torch.nn import Sequential

class SteinerParallel(SteinerProblem):

    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 reaction_layers=[8, 3], reaction_activation='GaussActivation',
                 **kwargs):
        super().__init__(**kwargs)

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
          'kernel_size', 'kernel_init',
          'reaction_layers', 'reaction_activation',
        )

        # Helper function
        def create_diffusion():
            return HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            )

        def create_reaction():
            return Reaction(reaction_layers, reaction_activation)

        # Creating model
        self.model = Sequential(
            LinearChannels(1, 2, bias=False),
            Parallel(
                Sequential(create_diffusion(), create_reaction()),
                create_reaction(),
            ),
            LinearChannels(2, 1, bias=True),
            create_diffusion(),
            create_reaction(),
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):

        parser = SteinerProblem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Steiner parallel", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--reaction_layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--reaction_activation', type=str, help='Name of the activation function')
        group.set_defaults(**{**get_default_args(SteinerParallel), **defaults})
        return parser

