"""
Base module and utils for the Allen-Cahn reaction term learning problem.

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse

from problem import Problem, get_default_args

class ReactionSolution:
    """
    Exact solution for the reaction operator of the Allen-Cahn equation
    """

    def __init__(self, epsilon=2/2**8, dt=None):
        """ Constructor

        Parameters
        ----------
        epsilon: float
            Interface sharpness in phase field model
        dt: float
            Time step. epsilon**2 if None.
        """
        self.epsilon = epsilon
        self.dt = dt or self.epsilon**2

    def __call__(self, u):
        """ Returns u(t + dt) from u(t) """
        result = torch.empty_like(u)
        dt, epsilon = self.dt, self.epsilon

        def helper(u):
            return torch.as_tensor(-dt / epsilon**2).exp() * u * (1 - u) / (1 - 2 * u)**2

        result[u == 0.5] = 0.5

        a = torch.sqrt(1 + 4 * helper(u[u < 0.5]))
        result[u < 0.5] = 1 - (a + 1) / (2 * a)

        a = torch.sqrt(1 + 4 * helper(u[u > 0.5]))
        result[u > 0.5] = (a + 1) / (2 * a)

        return result


class ReactionDataset(TensorDataset):
    """
    Base dataset for the Allen-Cahn reaction term.

    It is the exact solution for linear spaced samples in the [0, 1] interval
    with a given external margin.

    Parameters
    ----------
    epsilon: float
        Interface sharpness in phase field model
    dt: float
        Time step.
    margin: float
        Expanding length of the sampled [0, 1] interval
    N: int
        Number of samples in the dataset

    Examples
    --------
    >>> dataset = ReactionDataset(13, 1e-2, 1e-4, 0.1)
    >>> len(dataset)
    13
    >>> dataset[1]
    (tensor([0.]), tensor([0.]))
    >>> dataset[2]
    (tensor([0.1000]), tensor([0.0449]))
    >>> dataset[11]
    (tensor([1.]), tensor([1.]))
    """

    def __init__(self, num_samples, epsilon, dt, margin):
        lower_bound = 0. - margin
        upper_bound = 1. + margin
        exact_sol = ReactionSolution(epsilon, dt)
        train_x = torch.linspace(lower_bound, upper_bound, num_samples)[:, None]
        train_y = exact_sol(train_x)
        super().__init__(train_x, train_y)


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
    Ntrain: int
        Number of samples for the training step
    Nval: int
        Number of samples for the validation step. 10*Ntrain if None.
    batch_size: int
        Size of the batch during training and validation steps. Full data if None.
    lr: float
        Learning rate of the optimizer
    """

    def __init__(self, epsilon=2/2**8, dt=None, margin=0.1, Ntrain=100, Nval=None, batch_size=10, batch_shuffle=True, lr=1e-3, **kwargs):

        super().__init__(**kwargs)

        # Default values
        dt = dt or epsilon**2
        Nval = Nval or 10 * Ntrain

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('dt', 'epsilon', 'margin', 'Ntrain', 'Nval', 'batch_size', 'batch_shuffle', 'lr')

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
        self.train_dataset = ReactionDataset(self.hparams.Ntrain, self.hparams.epsilon, self.hparams.dt, self.hparams.margin)
        self.val_dataset = ReactionDataset(self.hparams.Nval, self.hparams.epsilon, self.hparams.dt, self.hparams.margin)

    def train_dataloader(self):
        """ Returns the training data loader """
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size or len(self.train_dataset), shuffle=self.hparams.batch_shuffle)

    def val_dataloader(self):
        """ Returns the validation data loader """
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size or len(self.val_dataset))

    def validation_step(self, batch, batch_idx):
        """ Called at each batch of the validation data """
        data, target = batch
        output = self(data)
        return {'val_loss': torch.nn.functional.mse_loss(output, target)}

    def validation_epoch_end(self, outputs):
        """ Called at epoch end of the validation step (after all batches) """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_loss = {'val_loss': avg_loss}
        return self.dispatch_metrics({'val_loss': avg_loss})

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        from distutils.util import strtobool
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("Reaction problem", "Options common to all models of the reaction term.")
        group.add_argument('--epsilon', type=float, help="Interface sharpness")
        group.add_argument('--dt', type=float, help="Time step (epsilon**2 if None)")
        group.add_argument('--margin', type=float, help="[0, 1] expansion length")
        group.add_argument('--Ntrain', type=int, help="Size of the training dataset")
        group.add_argument('--Nval', type=int, help="Size of the validation dataset (10*Ntrain if None)")
        group.add_argument('--batch_size', type=int, help="Size of batch")
        group.add_argument('--batch_shuffle', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Shuffle batch")
        group.add_argument('--lr', type=float, help="Learning rate")
        group.set_defaults(**{**get_default_args(ReactionProblem), **defaults})
        return parser

