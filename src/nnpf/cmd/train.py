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


def add_action_parser(parser, parents=[], defaults={}, drill=False, name="train", doc=__doc__):
    """ Adds base options for action parser """

    from . import get_subparsers
    subparsers = get_subparsers(parser)

    base_parser = argparse.ArgumentParser(add_help=not drill)

    # Mandatory model unless available in defaults
    base_parser.add_argument(
        "model",
        type=model_name,
        nargs='?' if drill or 'model' in defaults else None,
        help="Model name",
    )
    base_parser.add_argument("-i", "--interactive", action="store_true", help="Switch to interactive mode at the end of the script")

    # Add config option if not already defined (ugly)
    if all(len([action for action in parent._actions if action.dest == "config"]) == 0 for parent in parents):
        base_parser.add_argument('--config', type=str, help="Load configuration from the given YAML file")

    # Declare action
    action_parser = subparsers.add_parser(
        name,
        description=doc,
        help=doc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        parents=[base_parser] + parents,
    )

    # Override defaults
    action_parser.set_defaults(**defaults)

    return action_parser


def add_parser(parser, drill_parser, name="train", doc=__doc__):
    """ Add subparser for training action """
    import copy
    from nnpf.trainer import Trainer
    from nnpf.nn import get_model_by_name
    from . import ArgumentError

    parents = []
    defaults = {}

    # Trial parsing to check if train is the current action
    # and if a model is actually given
    add_action_parser(drill_parser, drill=True, name=name, doc=doc)
    try:
        args, _ = drill_parser.parse_known_args()
    except ArgumentError:
        pass
    else:
        if args.action == name:
            # Load config if specified
            if args.config is not None:
                import yaml
                with open(args.config, 'r') as fh:
                    defaults = {**defaults, **yaml.safe_load(fh)}

            # Use model from config if necessary and available
            args.model = defaults.get("model", args.model)

            if args.model is not None:
                # Add model and training specific options
                Model = get_model_by_name(args.model)
                model_parser = argparse.ArgumentParser(add_help=False)
                model_parser = Trainer.add_argparse_args(model_parser, dict(name=Model.__name__))
                model_parser = Model.add_model_specific_args(model_parser)
                parents.append(model_parser)

    # Add action parser and set defaults
    action_parser = add_action_parser(parser, parents=parents, defaults=defaults, name=name, doc=doc)
    action_parser.set_defaults(**defaults)
    return action_parser


def process(args):
    """ Process command line arguments """

    from nnpf.nn import get_model_by_name
    from nnpf.trainer import Trainer

    # Model, training & fit
    Model = get_model_by_name(args.model)
    model = Model(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

    # Interactive mode
    if args.interactive:
        from IPython import embed
        embed(colors="neutral")

