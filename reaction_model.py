#!/usr/bin/env python3

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import nn_models
import nn_toolbox
from reaction_problem import ReactionProblem

import argparse

class Reaction(ReactionProblem):

    def __init__(self, layer_dims=[8, 3], activation='GaussActivation', **kwargs):
        """ Constructor

        Parameters
        ----------
        layer_dims: iterable of int
            Working dimensions of the hidden layers
        activation: str
            Name of the activation function class
        kwargs: dict
            Parameters passed to reaction_problem.ReactionProblem
        """

        super().__init__(**kwargs)

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('layer_dims', 'activation', 'lr')

        # Model
        activation_class = nn_toolbox.get_model_by_name(self.hparams.activation)
        layers = ((activation_class(), d) for d in self.hparams.layer_dims)
        self.model = nn_models.Function(1, 1, *layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ReactionProblem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Reaction model", "Options specific to this model")
        group.add_argument('--layer_dims', type=int, nargs='+', default=[8, 3], help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, default='GaussActivation', help='Name of the activation function')
        return parser


if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Model of the reaction operator of the Allen-Cahn equation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser = Reaction.add_model_specific_args(parser)
    parser.add_argument('--version', default=None, help="Experiment version (logger)")
    args = parser.parse_args()


    # Model, training & fit
    model = Reaction(**vars(args))
    logger = TensorBoardLogger("logs", name="Reaction", version=args.version)
    trainer = Trainer.from_argparse_args(args, logger=logger, early_stop_callback=True)
    trainer.fit(model)

