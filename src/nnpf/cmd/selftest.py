"""
Launchs self-test of the module
"""

def add_parser(parser, drill_parser):
    """ Add subparser for selftest action """

    import argparse

    from . import get_subparsers
    subparsers = get_subparsers(parser)

    action_parser = subparsers.add_parser(
        "selftest",
        description=__doc__,
        help=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    action_parser.add_argument('module', type=str, nargs='?', default="nnpf", help="Module or submodule to check")

    return action_parser


def process(args):
    """ Process command line arguments """
    import pytest
    import shutil
    import importlib
    module = importlib.import_module(args.module)
    shutil.rmtree("./logs_doctest", ignore_errors=True)
    pytest.main(["--doctest-modules", *module.__path__])

