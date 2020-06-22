"""
Base module and utils for the Allen-Cahn reaction term learning problem.

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import math

from argparse import ArgumentParser

def _exact_sol_c(s, dt, epsilon):
    return math.exp(-dt / epsilon**2) * s * (1 - s) / (1 - 2 * s)**2


def exact_sol(s, dt, epsilon):
    """ Exact solution for the reaction term

    EDO
    """

    result = torch.empty_like(s)
    sign = torch.sign(s - 0.5)

    result[s == 0.5] = 0.5

    a = torch.sqrt(1 + 4 * _exact_sol_c(s[s < 0.5], dt, epsilon))
    result[s < 0.5] = 1 - (a + 1) / (2 * a)

    a = torch.sqrt(1 + 4 * _exact_sol_c(s[s > 0.5], dt, epsilon))
    result[s > 0.5] = (a + 1) / (2 * a)

    return result



class ReactionProblem(pl.LightningModule):
    """
    Base class for the Allen-Cahn reation term learning problem

    Features the train and validation data.
    """

    def __init__(self, epsilon=2/2**8, dt=None, margin=0.1, Ntrain=100, Nval=None, batch_size=None, **kwargs):
        """ Constructor

        Parameters
        ----------
        epsilon: float
            Interface sharpness in phase field model
        dt: float
            Time step. epsilon**2 if None.
        margin: float
            Expanding length of the sampled [0, 1] interval
        Ntrain: int
            Number of samples for the training step
        Nval: int
            Number of samples for the validation step. 10*Ntrain if None.
        batch_size: int
            Size of the batch during training and validation steps. Full data if None.
        """

        super().__init__()

        # Default values
        if dt is None:
            dt = epsilon**2

        if Nval is None:
            Nval = 10 * Ntrain

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('dt', 'epsilon', 'margin', 'Ntrain', 'Nval', 'batch_size')


    def prepare_data(self):
        """ Prepare training and validation data """

        lower_bound = 0. - self.hparams.margin
        upper_bound = 1. + self.hparams.margin

        # Training dataset
        train_x = torch.linspace(lower_bound, upper_bound, self.hparams.Ntrain)[:, None]
        train_y = exact_sol(train_x, self.hparams.dt, self.hparams.epsilon)
        self.train_dataset = TensorDataset(train_x, train_y)

        # Validation dataset
        val_x = torch.linspace(lower_bound, upper_bound, self.hparams.Nval)[:, None]
        val_y = exact_sol(val_x, self.hparams.dt, self.hparams.epsilon)
        self.val_dataset = TensorDataset(val_x, val_y)

    def train_dataloader(self):
        """ Returns the training data loader """
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size or self.hparams.Ntrain)

    #def val_dataloader(self):
    #    """ Returns the validation data loader """
    #    return DataLoader(self.val_dataset, batch_size=self.batch_size if not None else self.Nval)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--epsilon', type=float, default=2/8**3, help="Interface sharpness")
        parser.add_argument('--dt', type=float, default=None, help="Time step")
        parser.add_argument('--margin', type=float, default=0.1, help="[0, 1] expanding length")
        parser.add_argument('--Ntrain', type=int, default=100, help="Size of the training dataset")
        parser.add_argument('--Nval', type=int, default=None, help="Size of the validatio dataset (10*Ntrain if None")
        parser.add_argument('--batch_size', type=int, default=None, help="Size of batch")
        return parser
