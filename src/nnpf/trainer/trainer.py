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
    def __init__(self, version=None, name=None, log_graph=False, force_gpu=False, **kwargs):

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

        # Force GPU
        if force_gpu:
            kwargs["gpus"] = max(1, kwargs.get("gpus") or 0)

        # Create trainer
        super().__init__(
            **kwargs
        )

        self.force_gpu = force_gpu

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
        parser.add_argument('--log_graph', type=lambda v: bool(strtobool(v)), nargs='?', default=False, const=True, help="Calculates and log the computational graph of the model")
        parser.add_argument('--force_gpu', type=lambda v: bool(strtobool(v)), nargs='?', default=False, const=True, help="Force model and dataset to be on the GPU (may fail if not enough RAM)")
        parser.set_defaults(**{**get_default_args(Trainer), **defaults})
        return parser

    def fit(self, model, train_dataloaders = None, val_dataloaders = None, *args, **kwargs):
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

        # Force GPU
        if self.force_gpu:
            if train_dataloaders is None or val_dataloaders is None:
                model.prepare_data()
                #model.setup_data() # FIXME: is that needed?

            if train_dataloaders is None:
                try:
                    train_dataloaders = model.train_dataloader()
                except AttributeError:
                    pass

            if val_dataloaders is None:
                try:
                    val_dataloaders = model.val_dataloader()
                except AttributeError:
                    pass

            from torch.utils.data import DataLoader, TensorDataset
            def move_to(dl, device='cuda'):
                if not isinstance(dl, DataLoader):
                    raise ValueError(f"Unsupported transfer to {device} for dataloader of type {type(dl).__name__}")
                if not isinstance(dl.dataset, TensorDataset):
                    raise ValueError(f"Unsupported transfer to {device} for dataset of type {type(dl.dataset).__name__}")
                from copy import deepcopy
                # FIXME: need a cleaner way to recreate the same DataLoader but with a customized dataset
                dl = deepcopy(dl)
                dl.dataset.tensors = tuple(ts.to(device) for ts in dl.dataset.tensors)
                return dl

            from warnings import warn
            if train_dataloaders is not None:
                try:
                    train_dataloaders = move_to(train_dataloaders, 'cuda')
                except ValueError as e:
                    warn(str(e))

            if val_dataloaders is not None:
                try:
                    val_dataloaders = move_to(val_dataloaders, 'cuda')
                except ValueError as e:
                    warn(str(e))

        super().fit(model, train_dataloaders, val_dataloaders, *args, **kwargs)

    def train(self):
        # Saving initial state
        path = os.path.join(self.logger.log_dir, "checkpoints")
        self.save_checkpoint(os.path.join(path, f"epoch={self.current_epoch}.ckpt"))
        super().train()

