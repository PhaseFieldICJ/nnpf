import argparse


__all__ = [
    "actions",
]


def get_subparsers(parser):
    """ Returns first subparser of the given parser """
    return next(action for action in parser._subparsers._actions if isinstance(action, argparse._SubParsersAction))


def actions():
    """ Parse actions for submodules of nnpf.cmd """

    import argparse
    import pkgutil
    import nnpf.cmd
    import importlib

    # Command line arguments parser
    parser = argparse.ArgumentParser(
        description="Neural Network for Phase Field models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )

    # Action subparser
    subparser = parser.add_subparsers(
        dest="action",
        title="Actions",
        description="Action to launch",
        required=True,
    )

    # Let each submodule add the corresponding action parser
    for module_info in pkgutil.iter_modules(nnpf.cmd.__path__):
        name = module_info.name
        module = importlib.import_module(f"nnpf.cmd.{name}")
        action_parser = module.add_parser(parser)

        # Use default value to store the associated process function
        action_parser.set_defaults(process=module.process)

    # Parsing & processing
    args = parser.parse_args()
    args.process(args)


