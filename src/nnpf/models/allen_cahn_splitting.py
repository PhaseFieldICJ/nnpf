#!/usr/bin/env python3

import torch
from torch.nn import ModuleList
import argparse

from nnpf.problems import Problem, AllenCahnProblem, HeatProblem, ReactionProblem
from nnpf.utils import get_default_args
from nnpf.trainer import Trainer


__all__ = [
    "AllenCahnSplitting",
]


class AllenCahnSplitting(AllenCahnProblem):
    """
    Allen-Cahn problem model based on checkpoints of heat and reaction models.

    Parameters
    ----------
    checkpoints: list of str
        Paths to the model's checkpoints of the operators

    Examples
    --------

    Training reaction and heat model:
    >>> from nnpf.models import Reaction, HeatArray
    >>> from nnpf.trainer import Trainer

    >>> trainer = Trainer(default_root_dir="logs_doctest", name="Reaction", version="test_splitting", max_epochs=1, log_every_n_steps=1)
    >>> model = Reaction(train_N=10, val_N=20, seed=0, num_workers=4)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)

    >>> trainer = Trainer(default_root_dir="logs_doctest", name="HeatArray", version="test_splitting", max_epochs=1, log_every_n_steps=1)
    >>> model = HeatArray(N=64, train_N=10, val_N=20, num_workers=4)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)

    After launching examples from heat_array_model and reaction_model:
    >>> import os
    >>> from nnpf.trainer import Trainer
    >>> trainer = Trainer(default_root_dir="logs_doctest", name="AllenCahnSplitting", version="test_splitting", max_epochs=1, log_every_n_steps=1)
    >>> model = AllenCahnSplitting(
    ...     [os.path.join('logs_doctest', 'HeatArray', 'test_splitting'), os.path.join('logs_doctest', 'Reaction', 'test_splitting')],
    ...     test_N=10,
    ...     val_N=20,
    ...     val_reverse=0.5,
    ...     num_workers=4
    ... )
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)
    >>> trainer.global_step > 0
    True

    Loading from checkpoint:
    >>> from nnpf.problems import Problem
    >>> model = Problem.load_from_checkpoint(os.path.join('logs_doctest', 'AllenCahnSplitting', 'test_splitting'))
    >>> type(model).__name__
    'AllenCahnSplitting'
    """

    def __init__(self, checkpoints, **kwargs):
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
    def add_model_specific_args(parent_parser, defaults={}):
        parser = AllenCahnProblem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Allen-Cahn equation using splitting", "Options specific to this model")
        group.add_argument('checkpoints', type=str, nargs='*', help="Path to the model's checkpoint")
        group.set_defaults(**{**get_default_args(AllenCahnSplitting), **defaults})
        return parser



