"""
Base module and utils for the Allen-Cahn reaction term learning problem.

"""

import torch
from torch.utils.data import DataLoader
import argparse

from nnpf.utils import get_default_args
from nnpf.problems import Problem

from .datasets import ReactionDataset


__all__ = [
    "ReactionProblem",
]


class ReactionProblem(Problem):
    """
    Base class for the Allen-Cahn reaction term learning problem

    Features the train and validation data from ReactionDataset.

    Parameters
    ----------
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step. epsilon**2 if None.
    margin: float
        Expanding length of the sampled [0, 1] interval
    train_N: int
        Number of samples for the training step
    val_N: int
        Number of samples for the validation step. 10*train_N if None.
    batch_size: int
        Size of the batch during training and validation steps. Full data if None.
    lr: float
        Learning rate of the optimizer
    """

    def __init__(self, epsilon=2/2**8, dt=None, margin=0.1, train_N=100, val_N=None, batch_size=10, batch_shuffle=True, lr=1e-3, num_workers=0, **kwargs):

        super().__init__(**kwargs)

        # Default values
        dt = dt or epsilon**2
        val_N = val_N or 10 * train_N

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('dt', 'epsilon', 'margin', 'train_N', 'val_N', 'batch_size', 'batch_shuffle', 'lr', 'num_workers')

    @property
    def example_input_array(self):
        """ Example of input (for graph generation) """
        return torch.rand(self.hparams.batch_size or self.hparams.train_N, 1, 256, 256)

    def loss(self, output, target):
        """ Default loss function """
        return torch.nn.functional.mse_loss(output, target)

    def training_step(self, batch, batch_idx):
        """ Default training step with custom loss function """
        data, target = batch
        output = self.forward(data)
        loss = self.loss(output, target)
        return self.dispatch_metrics({'loss': loss})

    def configure_optimizers(self):
        """ Default optimizer """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def prepare_data(self):
        """ Prepare training and validation data """
        self.train_dataset = ReactionDataset(self.hparams.train_N, self.hparams.epsilon, self.hparams.dt, self.hparams.margin)
        self.val_dataset = ReactionDataset(self.hparams.val_N, self.hparams.epsilon, self.hparams.dt, self.hparams.margin)

    def train_dataloader(self):
        """ Returns the training data loader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size or len(self.train_dataset),
            shuffle=self.hparams.batch_shuffle,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        """ Returns the validation data loader """
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size or len(self.val_dataset),
            num_workers=self.hparams.num_workers
        )

    def validation_step(self, batch, batch_idx):
        """ Called at each batch of the validation data """
        data, target = batch
        output = self(data)
        return {'val_loss': torch.nn.functional.mse_loss(output, target)}

    def validation_epoch_end(self, outputs):
        """ Called at epoch end of the validation step (after all batches) """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_loss = {'val_loss': avg_loss}
        self.dispatch_metrics({'val_loss': avg_loss})

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        from distutils.util import strtobool
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Reaction problem", "Options common to all models of the reaction term.")
        group.add_argument('--epsilon', type=float, help="Interface sharpness")
        group.add_argument('--dt', type=float, help="Time step (epsilon**2 if None)")
        group.add_argument('--margin', type=float, help="[0, 1] expansion length")
        group.add_argument('--train_N', type=int, help="Size of the training dataset")
        group.add_argument('--val_N', type=int, help="Size of the validation dataset (10*train_N if None)")
        group.add_argument('--batch_size', type=int, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Shuffle batch")
        group.add_argument('--lr', type=float, help="Learning rate")
        group.add_argument('--num_workers', type=int, help="Number of subprocesses used for data loading")
        group.set_defaults(**{**get_default_args(ReactionProblem), **defaults})
        return parser

