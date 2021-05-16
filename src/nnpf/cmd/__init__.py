import argparse


__all__ = [
    "actions",
    "ArgumentError",
    "ThrowingArgumentParser",
]


def get_subparsers(parser):
    """ Returns first subparser of the given parser """
    return next(action for action in parser._subparsers._actions if isinstance(action, argparse._SubParsersAction))


class ArgumentError(Exception):
    """
    Exception throw by TrowingArgumentParser

    Also by argparse.ArgumentParser since Python 3.9
    """
    pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    """
    Argument parser that throw an exception instead of exit the program

    Used for premature argument parsing to dynamically add options
    """
    def error(self, message):
        raise ArgumentError(message)


def create_parser(drill=False):
    """
    Helper that create the argument parser depending on the context

    If drill is True, argument parser doesn't exit the program in case
    of errors and no help option is added.
    """
    Parser = ThrowingArgumentParser if drill else argparse.ArgumentParser

    # Command line arguments parser
    parser = Parser(
        description="Neural Network for Phase Field models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=not drill,
    )

    # Action subparser
    subparser = parser.add_subparsers(
        dest="action",
        title="Actions",
        description="Action to launch",
        required=True,
    )

    return parser


def actions():
    """ Parse actions for submodules of nnpf.cmd """

    import argparse
    import pkgutil
    import nnpf.cmd
    import importlib

    parser = create_parser()

    # Let each submodule add the corresponding action parser
    for module_info in pkgutil.iter_modules(nnpf.cmd.__path__):
        name = module_info.name
        module = importlib.import_module(f"nnpf.cmd.{name}")
        action_parser = module.add_parser(parser, create_parser(drill=True))

        # Use default value to store the associated process function
        action_parser.set_defaults(process=module.process)

    # Parsing & processing
    args = parser.parse_args()
    args.process(args)


