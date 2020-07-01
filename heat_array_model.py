#!/usr/bin/env python3

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import nn_models
import nn_toolbox
from domain import Domain
from heat_problem import HeatProblem

import argparse

class HeatArray(HeatProblem):

    def __init__(self, kernel_size, padding_mode='circular', bias=False, **kwargs):
        """ Constructor

        Parameters
        ----------
        kernel_size: int or tuple
            Size of the convolution kernel (a list for 2D and 3D convolutions)
        padding_mode: string
            'zeros', 'reflect', 'replicate' or 'circular'
        bias: bool
             If True, adds a learnable bias to the output.
        """

        super().__init__(**kwargs)

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('kernel_size', 'padding_mode', 'bias')

        # Model
        self.model = nn_models.ConvolutionArray(
            self.hparams.kernel_size,
            padding='center',
            padding_mode=self.hparams.padding_mode,
            bias=self.hparams.bias,
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HeatProblem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Heat equation by convolution array", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', required=True, help='Size of the kernel (nD)')
        group.add_argument('--padding_mode', choices=['zeros', 'reflect', 'replicate', 'circular'], default='circular', help="Padding mode for the convolution")
        group.add_argument('--bias', action='store_true', help="Add a bias to the convolution")
        return parser


if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Model of the heat equation using an array as convolution kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser = HeatArray.add_model_specific_args(parser)
    parser.add_argument('--version', default=None, help="Experiment version (logger)")
    args = parser.parse_args()

    # Deterministic calculation
    try:
        deterministic = args.seed is not None
    except AttributeError:
        deterministic = False

    # Model, training & fit
    model = HeatArray(**vars(args))
    logger = TensorBoardLogger("logs", name="HeatArray", version=args.version)
    trainer = Trainer.from_argparse_args(args, logger=logger, early_stop_callback=True, deterministic=deterministic)
    trainer.fit(model)


    import matplotlib.pyplot as plt
    plt.imshow(model.model.weight.detach()[0, 0, ...])
    plt.colorbar()
    plt.show()

