""" Lightning trainer with additional features """

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from nnpf.utils import get_default_args, merge_signature


__all__ = [
    "Trainer",
]


class Trainer(pl.Trainer):
    """
    Lightning trainer with additional features

    * problem name
    * experiment custom version name (also in command-line)

    Examples
    --------

    From command-line arguments:
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser = Trainer.add_argparse_args(parser)
    >>> args = parser.parse_args([
    ...     "--name", "DocTest",
    ...     "--version", "Trainer",
    ...     "--default_root_dir", "logs_doctest",
    ...     "--check_val_every_n_epoch", "100",
    ...     "--max_epochs", "1000",
    ... ])
    >>> trainer = Trainer.from_argparse_args(args)

    Checking:
    >>> model_checkpoint = trainer.checkpoint_callback
    >>> model_checkpoint.monitor
    'hp_metric'
    >>> model_checkpoint.save_top_k
    1
    >>> model_checkpoint.save_last
    True
    >>> trainer.logger.log_dir
    'logs_doctest/DocTest/Trainer'
    >>> trainer.check_val_every_n_epoch
    100
    >>> trainer.max_epochs
    1000

    Using the constructor:
    >>> trainer = Trainer(name="DocTest", version="Trainer", default_root_dir="logs_doctest", check_val_every_n_epoch=100, max_epochs=1000)

    Checking:
    >>> model_checkpoint = trainer.checkpoint_callback
    >>> model_checkpoint.monitor
    'hp_metric'
    >>> model_checkpoint.save_top_k
    1
    >>> model_checkpoint.save_last
    True
    >>> trainer.logger.log_dir
    'logs_doctest/DocTest/Trainer'
    >>> trainer.check_val_every_n_epoch
    100
    >>> trainer.max_epochs
    1000
    """

    @merge_signature(pl.Trainer.__init__)
    def __init__(self, version=None, name=None, log_graph=False, **kwargs):

        # Logger
        if kwargs.get('logger', True) is True:
            default_root_dir = kwargs.get('default_root_dir', None)
            kwargs['logger'] = TensorBoardLogger(
                default_root_dir if default_root_dir is not None else "logs",
                name=name,
                version=version,
                default_hp_metric=False, # hp_metric will be declared after the sanity check, see below
                log_graph=log_graph # need example_input_array attribute to be set in the model
            )

        # Checkpointer
        if kwargs.get('checkpoint_callback', True) is True:
            kwargs.setdefault('callbacks', [])
            if kwargs['callbacks'] is None:
                kwargs['callbacks'] = []

            kwargs['callbacks'].append(ModelCheckpoint(
                monitor='hp_metric',
                save_top_k=1,
                mode='min',
                every_n_epochs=1,
                save_last=True,
            ))

        # Create trainer
        super().__init__(
            **kwargs
        )

        # Creating log folder
        path = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)


    @classmethod
    def add_argparse_args(cls, parent_parser, defaults={}):
        """ Add trainer command-line options and experiment version """
        from distutils.util import strtobool

        parser = super().add_argparse_args(parent_parser)
        parser.add_argument('--version', help="Experiment version for the logger")
        parser.add_argument('--name', help="Experiment name for the logger")
        parser.add_argument('--log_graph', type=lambda v: bool(strtobool(v)), nargs='?', const=True, help="Calculates and log the computational graph of the model")
        parser.set_defaults(**{**get_default_args(Trainer), **defaults})
        return parser

    def fit(self, model, *args, **kwargs):
        # Disabing TensorBoardLogger.log_metrics since log_hyperparams
        # call log_metrics with step always set to 0, leading to a zigzag
        # in the graph
        def do_nothing(*args, **kwargs):
            pass
        log_metrics = self.logger.log_metrics
        self.logger.log_metrics = do_nothing

        self.logger.log_hyperparams(
            params=model.hparams,
            metrics={'hp_metric': 0}
        )

        # Restoring TensorBoard.log_metrics
        self.logger.log_metrics = log_metrics

        super().fit(model, *args, **kwargs)

    def train(self):
        # Saving initial state
        path = os.path.join(self.logger.log_dir, "checkpoints")
        self.save_checkpoint(os.path.join(path, f"epoch={self.current_epoch}.ckpt"))
        super().train()

