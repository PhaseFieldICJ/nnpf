"""
Base module and utils for every problem
"""

import pytorch_lightning as pl
import argparse

from trainer import get_default_args

def fix_path(path):
    """ Convert path from Posix to/from Windows filesystems """
    # Naive way to find out if given path if from Windows
    # No really reliable since \\ may come from a valid POSIX path...
    is_windows = path.find('\\') != -1

    # Returns given path if on right filesystem
    import os
    if (is_windows and os.sep == '\\') or (not is_windows and os.sep == '/'):
        return path

    import pathlib
    if is_windows:
        return pathlib.PureWindowsPath(path).as_posix() # Windows -> POSIX
    else:
        return str(pathlib.PureWindowsPath(path)) # POSIX -> Windows


def checkpoint_from_path(checkpoint_path):
    """
    Returns path if it points to an actual file,
    otherwise search for the last checkpoint of the form
    "path/checkpoints/epoch=*.ckpt"
    """
    import os
    checkpoint_path = fix_path(checkpoint_path)

    # If path if a folder, found last checkpoint from checkpoints subfolder
    if os.path.isdir(checkpoint_path):
        import glob
        import re
        glob_expr = os.path.join(os.path.expanduser(checkpoint_path), r"checkpoints", r"epoch=*.ckpt")
        checkpoint_list = glob.glob(glob_expr)
        if len(checkpoint_list) > 0:
            checkpoint_path = sorted(checkpoint_list, key=lambda s: int(re.search(r"epoch=([0-9]+)\.ckpt$", s).group(1)))[-1]

    return checkpoint_path


class Problem(pl.LightningModule):
    """
    Base class for every problem
    """

    def __init__(self, seed=None, is_loaded=False, **kwargs):
        """ Constructor

        Parameters
        ----------
        seed: int
            If set to an integer, use it as seed of all random generators.
        is_loaded: bool
            Internal variable that should be set to True when the model is loaded from a checkpoint.
        """

        super().__init__()

        # Is this model loaded from a checkpoint?
        self.is_loaded = is_loaded

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters('seed')

        # Seed random generators
        if self.hparams.seed is not None and not self.is_loaded:
            pl.seed_everything(self.hparams.seed)
            # Should also enable deterministic behavior in the trainer parameters

    def dispatch_metrics(self, metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, **kwargs):
        """ Dispatch metrics (e.g. loss) to log and progress bar """

        def transform_key(key):
            if key == 'loss':
                return 'train_loss'
            elif key == 'metric':
                return 'hp_metric'
            else:
                return key

        # Global metric (default to validation loss)
        if 'val_loss' in metrics and 'metric' not in metrics and 'hp_metric' not in metrics:
            metrics['hp_metric'] = metrics['val_loss']

        # Log metric
        for key, value in metrics.items():
            self.log(transform_key(key), value, prog_bar, logger, on_step, on_epoch, **kwargs)

        return metrics

    def on_save_checkpoint(self, checkpoint):
        """ Called just before checkpointing """
        import inspect

        # So that to detect add a initialization parameter to detect loading
        checkpoint[type(self).CHECKPOINT_HYPER_PARAMS_KEY]["is_loaded"] = True

        # Save module and class of the model
        cls = type(self)
        checkpoint['class_path'] = inspect.getfile(cls)
        checkpoint['class_name'] = cls.__name__

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, map_location=None, **kwargs):
        """ Load model from checkpoint with automatic model forward """
        import torch

        # If path if a folder, found last checkpoint from checkpoints subfolder
        checkpoint_path = checkpoint_from_path(checkpoint_path)

        # Forward to the right class if current class's name doesn't match
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if cls.__name__ != checkpoint["class_name"]:
            # From https://stackoverflow.com/a/67692
            import importlib.util
            spec = importlib.util.spec_from_file_location("model", fix_path(checkpoint["class_path"]))
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            model_cls = getattr(foo, checkpoint["class_name"])

            # Check that we load a model from the current problem
            if not issubclass(model_cls, cls):
                raise ImportError(f"{model_cls.__name__} is not a model of problem {cls.__name__}")

            # Load checkpoint from the right class
            return model_cls.load_from_checkpoint(checkpoint_path, *args, map_location=map_location, **kwargs)

        else:
            return super().load_from_checkpoint(checkpoint_path, *args, map_location=map_location, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = argparse.ArgumentParser(
            parents=[parent_parser],
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        group = parser.add_argument_group("Problem", "Options common to all problems.")
        group.add_argument('--seed', type=int, help="Seed the random generators and disable non-deterministic behavior")
        group.add_argument('--config', type=str, help="Load configuration from the given YAML file")
        group.set_defaults(**{**get_default_args(Problem), **defaults})
        return parser

    @staticmethod
    def defaults_from_config():
        """ Load defaults from a YAML file, if specified in the command line """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', type=str, help="Load configuration from the given YAML file")
        args, _ = parser.parse_known_args()

        if args.config:
            import yaml
            with open(args.config, 'r') as fh:
                return yaml.safe_load(fh)
        else:
            return {}

