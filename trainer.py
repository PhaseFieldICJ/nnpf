""" Lightning trainer with additional features """

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


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
    from inspect import isclass, signature, Parameter
    if isclass(f):
        f = f.__init__

    return {p.name: p.default for p in signature(f).parameters.values() if p.default != Parameter.empty}


def merge_signature(reference):
    """
    Decorator that merge signature of given function with the decorated one.

    The goal what to include parent class constructor's signature in the derived
    cosntructor's signature so that to allow signature-based inspection function
    to work as on the parent class.

    FIXME: doesn't work well when mixing positional_or_keywork and position_only
    parameters

    Examples
    --------
    >>> class Base:
    ...     def __init__(self, a, b, c=3, d=4):
    ...         print(f"a = {a} ; b = {b} ; c = {c} ; d = {d}")
    >>> class Derived(Base):
    ...     @merge_signature(Base.__init__)
    ...     def __init__(self, e, *args, f=2, g=3, **kwargs):
    ...         print(f"e = {e} ; f = {f} ; g = {g}")
    ...         super().__init__(*args, **kwargs)
    >>> d = Derived(10, 11, 12, c=13, f=14)
    e = 10 ; f = 14 ; g = 3
    a = 11 ; b = 12 ; c = 13 ; d = 4
    >>> import inspect
    >>> inspect.signature(Derived.__init__)
    <Signature (self, e, a, b, c=3, d=4, *, f=2, g=3)>
    """

    from inspect import signature, Parameter
    import itertools

    def decorator(wrapped):
        ref_sgn = signature(reference)
        wrp_sgn = signature(wrapped)
        ref_params = ref_sgn.parameters.values()
        wrp_params = wrp_sgn.parameters.values()
        wrp_names = set(p.name for p in wrp_params)

        positional_only = itertools.chain(
            (p for p in wrp_params if p.kind == Parameter.POSITIONAL_ONLY),
            (p for p in ref_params if p.kind == Parameter.POSITIONAL_ONLY and p.name not in wrp_names))

        positional_or_keyword = itertools.chain(
            (p for p in wrp_params if p.kind == Parameter.POSITIONAL_OR_KEYWORD),
            (p for p in ref_params if p.kind == Parameter.POSITIONAL_OR_KEYWORD and p.name not in wrp_names))

        var_positional = (p for p in ref_params if p.kind == Parameter.VAR_POSITIONAL)

        keyword_only = itertools.chain(
            (p for p in wrp_params if p.kind == Parameter.KEYWORD_ONLY),
            (p for p in ref_params if p.kind == Parameter.KEYWORD_ONLY and p.name not in wrp_names))

        var_keyword = (p for p in ref_params if p.kind == Parameter.VAR_KEYWORD)

        wrapped.__signature__ = ref_sgn.replace(parameters=itertools.chain(
            positional_only,
            positional_or_keyword,
            var_positional,
            keyword_only,
            var_keyword,
        ))
        wrapped.__doc__ = reference.__doc__

        return wrapped

    return decorator


class Trainer(pl.Trainer):
    """
    Lightning trainer with additional features

    * automatic deterministic behavior if seed is given in the command-line
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
    ...     "--default_root_dir", "doctest_logs",
    ...     "--check_val_every_n_epoch", "100",
    ...     "--max_epochs", "1000",
    ... ])
    >>> trainer = Trainer.from_argparse_args(args)

    Checking:
    >>> model_checkpoint = next(c for c in trainer.callbacks if type(c).__name__ == "ModelCheckpoint")
    >>> model_checkpoint.monitor
    'hp_metric'
    >>> model_checkpoint.save_top_k
    1
    >>> model_checkpoint.save_last
    True
    >>> trainer.logger.log_dir
    'doctest_logs/DocTest/Trainer'
    >>> trainer.check_val_every_n_epoch
    100
    >>> trainer.max_epochs
    1000

    Using the constructor:
    >>> trainer = Trainer(name="DocTest", version="Trainer", default_root_dir="doctest_logs", check_val_every_n_epoch=100, max_epochs=1000)

    Checking:
    >>> model_checkpoint = next(c for c in trainer.callbacks if type(c).__name__ == "ModelCheckpoint")
    >>> model_checkpoint.monitor
    'hp_metric'
    >>> model_checkpoint.save_top_k
    1
    >>> model_checkpoint.save_last
    True
    >>> trainer.logger.log_dir
    'doctest_logs/DocTest/Trainer'
    >>> trainer.check_val_every_n_epoch
    100
    >>> trainer.max_epochs
    1000
    """

    @merge_signature(pl.Trainer.__init__)
    def __init__(self, version=None, name=None, **kwargs):

        # Logger
        if kwargs.get('logger', True) is True:
            default_root_dir = kwargs.get('default_root_dir', None)
            kwargs['logger'] = TensorBoardLogger(
                default_root_dir if default_root_dir is not None else "logs",
                name=name,
                version=version,
                default_hp_metric=True,
            )

        # Checkpointer
        if kwargs.get('checkpoint_callback', True) is True:
            kwargs['checkpoint_callback'] = ModelCheckpoint(
                filepath=None,
                monitor='hp_metric',
                save_top_k=1,
                mode='min',
                period=1,
                save_last=True,
            )

        # Create trainer
        super().__init__(
            **kwargs
        )


    @classmethod
    def add_argparse_args(cls, parent_parser, defaults={}):
        """ Add trainer command-line options and experiment version """
        parser = super().add_argparse_args(parent_parser)
        parser.add_argument('--version', help="Experiment version for the logger")
        parser.add_argument('--name', help="Experiment name for the logger")
        parser.set_defaults(**{**get_default_args(Trainer), **defaults})
        return parser


    def train(self):
        # Saving initial state
        path = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        self.save_checkpoint(os.path.join(path, f"epoch={self.current_epoch}.ckpt"))

        super().train()

