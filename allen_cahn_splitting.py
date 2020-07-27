#!/usr/bin/env python3

import torch
from torch.nn import ModuleList
import argparse

from allen_cahn_problem import AllenCahnProblem
from problem import Problem
from heat_problem import HeatProblem
from reaction_problem import ReactionProblem
from trainer import Trainer

class AllenCahnSplitting(AllenCahnProblem):

    def __init__(self, checkpoints, **kwargs):
        """
        Constructor

        Parameters
        ----------
        checkpoints: list of str
            Paths to the model's checkpoints of the operators
        """

        # Loading operators and checking consistency
        operators = ModuleList()
        epsilon = None
        bounds = None
        N = None
        dt_heat, dt_reaction = 0., 0.

        def isclose(a, b):
            return torch.isclose(torch.tensor(a), torch.tensor(b))

        for path in checkpoints:
            # Loading model
            op = Problem.load_from_checkpoint(path)
            operators.append(op)

            # Checking consistency
            if isinstance(op, HeatProblem):
                dt_heat += op.hparams.dt

                if bounds is None:
                    bounds = op.hparams.bounds
                else:
                    assert len(bounds) == len(op.hparams.bounds), f"Dimension mismatch for heat operator {path}!"
                    assert all(isclose(a1, a2) and isclose(b1, b2) for (a1, b1), (a2, b2) in zip(bounds, op.hparams.bounds)), f"Bounds mismatch for heat operator {path}!"

                if N is None:
                    N = op.hparams.N
                else:
                    assert all(n1 == n2 for n1, n2 in zip(N, op.hparams.N)), f"Domain discretization mismatch for heat operator {path}!"

            elif isinstance(op, ReactionProblem):
                dt_reaction += op.hparams.dt

                if epsilon is None:
                    epsilon = op.hparams.epsilon
                else:
                    assert isclose(epsilon, op.hparams.epsilon), f"Inconsistent epsilon for model {path}!"

            else:
                raise ValueError(f"Model {path} is neither a Reaction or a Heat model!")


        assert isclose(dt_heat, dt_reaction), f"Heat time step {dt_heat} and Reaction time step {dt_reaction} are different!"

        # Updating parameters
        # FIXME: remove command-line parameters for AllenCahnProblem?
        kwargs['bounds'] = bounds
        kwargs['N'] = N
        kwargs['dt'] = dt_heat
        kwargs['epsilon'] = epsilon

        super().__init__(**kwargs)

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('checkpoints')

        self.operators = operators

    def forward(self, x):
        for op in self.operators:
            x = op(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = AllenCahnProblem.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Allen-Cahn equation using splitting", "Options specific to this model")
        group.add_argument('checkpoints', type=str, nargs='+', help="Path to the model's checkpoint")
        return parser


if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Model of the Allen-Cahn equation using splitting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser = AllenCahnSplitting.add_model_specific_args(parser)
    args = parser.parse_args()

    # Model, training & fit
    model = AllenCahnSplitting(**vars(args))
    trainer = Trainer.from_argparse_args(args, "AllenCahnSplitting")
    trainer.fit(model)




