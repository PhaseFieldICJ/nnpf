"""
Base module and utils for every problem
"""

import pytorch_lightning as pl
import argparse

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


def default_args(**defaults):
    """
    Function decorator that overwrites defaults using given dictionary

    Inspired from https://stackoverflow.com/a/58983447
    and https://stackoverflow.com/a/57730055

    First answer was finally not used because Lightning use inspect.getfullargspec
    to get the signature of a function and it doesn't work with wrapped functions
    (it should use inspect.signature instead).

    Parameters
    ----------
    defaults: dict
        Default values for parameters.
        Positional only arguments can also be defaulted in this dictionary.

    Examples
    --------
    >>> @default_args(a=1, b=2, c=3, d=4)
    ... def dummy(a, /, b, c, *, d):
    ...     return a, b, c, d
    >>> dummy()
    (1, 2, 3, 4)
    >>> dummy(0)
    (0, 2, 3, 4)
    >>> dummy(0, d=-1)
    (0, 2, 3, -1)
    >>> dummy(0, 0, 0, 0)
    Traceback (most recent call last):
        ...
    TypeError: dummy() takes from 0 to 3 positional arguments but 4 were given
    >>> dummy(0, 0, b=3)
    Traceback (most recent call last):
        ...
    TypeError: dummy() got multiple values for argument 'b'

    >>> @default_args(b=2, c=3)
    ... def dummy(a, /, b, c, *, d):
    ...     return a, b, c, d
    >>> dummy()
    Traceback (most recent call last):
        ...
    TypeError: dummy() missing 1 required positional argument: 'a'
    >>> dummy(0, 0)
    Traceback (most recent call last):
        ...
    TypeError: dummy() missing 1 required keyword-only argument: 'd'
    >>> dummy(0, d=5)
    (0, 2, 3, 5)

    >>> @default_args(b=2, d=4)
    ... def dummy(a, /, b, c=3, *, d):
    ...     return a, b, c, d
    >>> dummy(1)
    (1, 2, 3, 4)

    >>> @default_args(b=2, d=4)
    ... def dummy(a, /, b, c, *, d):
    ...     return a, b, c, d
    Traceback (most recent call last):
        ...
    SyntaxError: non-default argument c follows default arguments

    >>> @default_args(e=5)
    ... def dummy(a, /, b, c, *, d):
    ...     return a, b, c, d
    Traceback (most recent call last):
        ...
    TypeError: dummy() got default values for unexpected arguments {'e'}
    """

    from inspect import getfullargspec
    from itertools import dropwhile

    def decorator(f):
        f_argspec = getfullargspec(f)

        # Complete default values for positional arguments
        args_defaults = {k: v for k, v in zip(reversed(f_argspec.args), reversed(f.__defaults__ or ()))}
        args_defaults.update((k, defaults[k]) for k in f_argspec.args if k in defaults)
        try:
            f.__defaults__ = tuple(args_defaults[k] for k in dropwhile(lambda k: k not in args_defaults, f_argspec.args))
        except KeyError as err:
            raise SyntaxError(f"non-default argument {err.args[0]} follows default arguments")

        # Complete default values for keyword only arguments
        kwonly_defaults = f.__kwdefaults__ or {}
        kwonly_defaults.update((k, defaults[k]) for k in f_argspec.kwonlyargs if k in defaults)
        f.__kwdefaults__ = kwonly_defaults

        # Unexpected default values
        unexpected_keys = defaults.keys() - set(f_argspec.args) - set(f_argspec.kwonlyargs)
        if unexpected_keys:
            raise TypeError(f"{f.__name__}() got default values for unexpected arguments {unexpected_keys}")

        return f

    return decorator


def get_default_args(f):
    """ Extract argument's default values from a function or a class constructor

    Parameters
    ----------
    f: function or class
        if f is a class, it is replaced by f.__init__

    Returns
    -------
    defaults: dict
        Default values of the arguments of f

    Examples
    --------
    >>> def dummy(a, b=1, /, c=2, *, d, e=4):
    ...     pass
    >>> sorted(get_default_args(dummy).items())
    [('b', 1), ('c', 2), ('e', 4)]
    """
    from inspect import isclass, getfullargspec
    if isclass(f):
        f = f.__init__
    f_argspec = getfullargspec(f)

    # Default values for positional arguments
    defaults = {k: v for k, v in zip(reversed(f_argspec.args), reversed(f.__defaults__ or ()))}

    # Default value for keyword-only arguments
    defaults.update(f.__kwdefaults__ or {})

    return defaults


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

    def dispatch_metrics(self, metrics):
        """ Dispatch metrics (e.g. loss) to log and progress bar """

        def transform_key(key):
            if key == 'loss':
                return 'train_loss'
            else:
                return key

        def format_value(value):
            return f"{value:.2e}" # Fixed width value to avoid shaking progress bar

        # Global metric (default to validation loss)
        if 'val_loss' in metrics and 'metric' not in metrics:
            metrics['metric'] = metrics['val_loss']

        return {
            **metrics,
            'log': {transform_key(key): value for key, value in metrics.items()},
            'progress_bar': {transform_key(key): format_value(value) for key, value in metrics.items()},
        }

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
        import inspect

        # If path if a folder, found last checkpoint from checkpoints subfolder
        checkpoint_path = checkpoint_from_path(checkpoint_path)

        # Forward to the right class if current class is abstract
        if inspect.isabstract(cls):
            import importlib.util
            import torch
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # From https://stackoverflow.com/a/67692
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

