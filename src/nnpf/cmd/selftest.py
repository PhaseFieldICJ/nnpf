"""
Launchs self-test of the module
"""

def add_parser(parser):
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

    return action_parser


def process(args):
    """ Process command line arguments """
    import pytest
    import nnpf
    pytest.main(["--doctest-modules", *nnpf.__path__])

