#!/usr/bin/env python3

import torch
import torch.nn as nn

import nn_models
import nn_toolbox
from domain import Domain
import heat_problem
from heat_problem import HeatProblem
from trainer import Trainer

import argparse

class HeatArray(HeatProblem):

    def __init__(self, kernel_size, padding_mode='circular', bias=False, init='zeros', **kwargs):
        """ Constructor

        Parameters
        ----------
        kernel_size: int or tuple
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
        """

        super().__init__(**kwargs)

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = tuple(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('kernel_size', 'padding_mode', 'bias', 'init')

        # Model
        self.model = nn_models.ConvolutionArray(
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
                        self.model.weight[:] += heat_problem.heat_kernel_spatial(self.domain, self.hparams.dt, self.hparams.kernel_size)

    # TODO: auto reshape when no batch nor channel are included
    def forward(self, x):
        return self.model(x)

    def loss(self, output, target):
        return super().loss(output, target) + 0 * 1e-2 * self.model.weight.square().mean()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HeatProblem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Heat equation by convolution array", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', required=True, help='Size of the kernel (nD)')
        group.add_argument('--padding_mode', choices=['zeros', 'reflect', 'replicate', 'circular'], default='circular', help="Padding mode for the convolution")
        group.add_argument('--bias', action='store_true', help="Add a bias to the convolution")
        group.add_argument('--init', choices=['zeros', 'random', 'solution'], default='zeros', help="Initialization of the convolution kernel")
        return parser


if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Model of the heat equation using an array as convolution kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser = HeatArray.add_model_specific_args(parser)
    args = parser.parse_args()

    # Model, training & fit
    model = HeatArray(**vars(args))
    trainer = Trainer.from_argparse_args(args, "HeatArray")
    trainer.fit(model)


    print(model.model.weight.detach().sum() - 1.)

    import matplotlib.pyplot as plt
    plt.imshow(model.model.weight.detach()[0, 0, ...])
    plt.colorbar()
    plt.show()

