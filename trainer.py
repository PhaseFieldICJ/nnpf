""" Lightning trainer with additional features """

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

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

        # Deterministic calculation
        try:
            deterministic = args.deterministic or (args.seed is not None)
        except AttributeError:
            deterministic = args.deterministic

        # Logger
        logger = TensorBoardLogger("logs", name=name, version=args.version)
        return super().from_argparse_args(args, logger=logger, deterministic=deterministic)

