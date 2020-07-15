""" Lightning trainer with additional features """

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class Trainer(pl.Trainer):
    """
    Lightning trainer with additional features

    * automatic deterministic behavior if seed is given in the command-line
    * problem name
    * experiment custom version name (also in command-line)
    """

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """ Add trainer command-line options and experiment version """
        parser = super().add_argparse_args(parent_parser)
        parser.add_argument('--version', default=None, help="Experiment version for the logger")
        return parser

    @classmethod
    def from_argparse_args(cls, args, name, **kwargs):
        """ Create an instance from CLI arguments

        Also activate deterministic behavior if a seed is specified.

        Parameters
        ----------
        args: Namespace or ArgumentParser
            The parser or namespace to take arguments from. Only known arguments will be
            parsed and passed to the :class:`Trainer`.
        name: str
            Name of the problem (used as subdir of logging directory)
        **kwargs: dict
            Additional keyword arguments that may override ones in the parser or namespace.
            These must be valid Trainer arguments.
        """

        default_root_dir = args.default_root_dir if args.default_root_dir else "logs"

        # Deterministic calculation
        try:
            deterministic = args.deterministic or (args.seed is not None)
        except AttributeError:
            deterministic = args.deterministic

        # Logger
        logger = TensorBoardLogger(default_root_dir, name=name, version=args.version)

        # Checkpointer
        checkpointer = ModelCheckpoint(
            filepath=None,
            monitor=kwargs.get('metric', 'val_loss'),
            save_top_k=1,
            mode='min',
            period=1,
        )

        # Create trainer
        return super().from_argparse_args(
            args,
            logger=logger,
            checkpoint_callback=checkpointer,
            default_root_dir=default_root_dir,
            deterministic=deterministic)

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        Overriding base class to add metric in logger, see
        https://github.com/PyTorchLightning/pytorch-lightning/issues/1778#issuecomment-653733246

        Args:
            model: The model to run sanity test on.
        """
        self.logger_backup = self.logger
        self.logger = None # Disabling logger to control hyperparameters
        super().run_pretrain_routine(model)

    def train(self):
        """
        Overriding base class to add metric in logger, see
        https://github.com/PyTorchLightning/pytorch-lightning/issues/1778#issuecomment-653733246
        """

        self.logger = self.logger_backup

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started
            self.logger.log_hyperparams(
                params=self.model.hparams,
                metrics={self.checkpoint_callback.monitor: self.checkpoint_callback.best_model_score}
            )
            self.logger.save()

        super().train()


