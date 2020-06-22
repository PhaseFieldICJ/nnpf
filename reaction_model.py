#!/usr/bin/env python3

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import nn_models
from reaction_problem import ReactionProblem

import argparse

class Reaction(ReactionProblem):

    def __init__(self, layers, lr=1e-3, **kwargs):
        """ Constructor

        Parameters
        ----------
        layers: iterable of pairs (fn, dim)
            Activation functions and working dimensions of the hidden layers
        lr: float
            Learning rate of the optimizer
        kwargs: dict
            Parameters passed to reaction_problem.ReactionProblem
        """

        super().__init__(**kwargs)

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('layers', 'lr')

        # Model
        self.model = nn_models.Function(1, 1, *self.hparams.layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = torch.nn.functional.mse_loss(output, target)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ReactionProblem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Reaction model", "Options specific to this model")
        group.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Model of the reaction operator of the Allen-Cahn equation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser = Reaction.add_model_specific_args(parser)
    args = parser.parse_args()

    from nn_models import GaussActivation

    trainer = Trainer.from_argparse_args(args, early_stop_callback=True)

    layers=((GaussActivation(), 8), (GaussActivation(), 4))
    model = Reaction(layers, **vars(args))


    trainer.fit(model)

