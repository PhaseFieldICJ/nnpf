#!/usr/bin/env python3

import torch
import argparse

from allen_cahn_problem import AllenCahnProblem
from heat_problem import HeatProblem
from reaction_problem import ReactionProblem
from trainer import Trainer

class AllenCahnDR(AllenCahnProblem):

    def __init__(self, heat_checkpoint, reaction_checkpoint, **kwargs):
        """
        Constructor

        Parameters
        ----------
        heat_checkpoint: str
            Path to the model's checkpoint of the heat operator
        reaction_checkpoint: str
            Path to the model's checkpoint of the reaction term
        """

        # Loading operators
        heat = HeatProblem.load_from_checkpoint(heat_checkpoint)
        reaction = ReactionProblem.load_from_checkpoint(reaction_checkpoint)

        # Checking consistency
        assert heat.hparams.dt == reaction.hparams.dt, "Inconsistent time step in heat and reaction operators"

        # Updating parameters
        # FIXME: remove command-line parameters for AllenCahnProblem?
        kwargs['bounds'] = heat.hparams.bounds
        kwargs['N'] = heat.hparams.N
        kwargs['dt'] = heat.hparams.dt
        kwargs['epsilon'] = reaction.hparams.epsilon

        super().__init__(**kwargs)

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('heat_checkpoint', 'reaction_checkpoint')

        self.heat = heat
        self.reaction = reaction

    def forward(self, x):
        x = self.heat(x)
        x = self.reaction(x.reshape(-1, 1)).reshape(x.shape)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = AllenCahnProblem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Allen-Cahn equation using diffusion-reaction splitting", "Options specific to this model")
        group.add_argument('heat_checkpoint', type=str, help="Path to the heat equation model's checkpoint")
        group.add_argument('reaction_checkpoint', type=str, help="Path to the reation term model's checkpoint")
        return parser


if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Model of the heat equation using an array as convolution kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser = AllenCahnDR.add_model_specific_args(parser)
    args = parser.parse_args()

    # Model, training & fit
    model = AllenCahnDR(**vars(args))
    trainer = Trainer.from_argparse_args(args, "AllenCahnDR")
    trainer.fit(model)




