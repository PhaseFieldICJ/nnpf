"""
Train a given model
"""

import argparse
from nnpf.nn import get_model_by_name


__all__ = [
    "add_parser",
    "process",
]


def model_name(name):
    """ Returns the name if it matches a valid model class, raises ValueError otherwise """
    try:
        get_model_by_name(name)
    except AttributeError:
        raise ValueError()

    return name


def add_action_parser(parser, parents=[], drill=False):
    """ Adds base options for action parser """

    from . import get_subparsers
    subparsers = get_subparsers(parser)

    base_parser = argparse.ArgumentParser(add_help=not drill)
    base_parser.add_argument(
        "model",
        type=model_name,
        nargs='?' if drill else None,
        help="Model name",
    )

    action_parser = subparsers.add_parser(
        "train",
        description=__doc__,
        help=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        parents=[base_parser] + parents,
    )

    return action_parser


def add_parser(parser):
    """ Add subparser for training action """
    import copy
    from nnpf.trainer import Trainer
    from nnpf.nn import get_model_by_name

    parents = []

    # Trial parsing to check if a model is actually given
    tmp_parser = copy.deepcopy(parser)
    add_action_parser(tmp_parser, drill=True)
    args, _ = tmp_parser.parse_known_args()

    #if args.action == "train" and "model" in args:
    if args.action == "train" and args.model is not None:
        # Add model and training specific options
        Model = get_model_by_name(args.model)

        model_parser = argparse.ArgumentParser(add_help=False)
        model_parser = Trainer.add_argparse_args(model_parser, dict(name=args.model))
        model_parser = Model.add_model_specific_args(model_parser)
        parents.append(model_parser)

    return add_action_parser(parser, parents=parents)


def process(args):
    """ Process command line arguments """

    from nnpf.nn import get_model_by_name
    from nnpf.trainer import Trainer

    # Model, training & fit
    Model = get_model_by_name(args.model)
    model = Model(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

