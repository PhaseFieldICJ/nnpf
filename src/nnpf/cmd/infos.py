"""
Display informations about a model from a checkpoint
"""

def add_parser(parser, drill_parser):
    """ Add subparser for infos action """
    import argparse

    from . import get_subparsers
    subparsers = get_subparsers(parser)

    action_parser = subparsers.add_parser(
        "infos",
        description=__doc__,
        help=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )

    action_parser.add_argument('checkpoint', type=str, help="Path to the model's checkpoint")
    action_parser.add_argument('--recursive', type=lambda v: bool(int(v)), default=True, help="Display informations about checkpoint founds in hyper-parameters")
    action_parser.add_argument('--use_torch_info', type=lambda v: bool(int(v)), default=True, help="Display detailed informations and memory usage using torchinfo package")
    action_parser.add_argument('--input_size', type=lambda v: [int(s) for s in v.split(',')], default=None, help="Input size for calculating memory usage. Example: 1,1,256,256. If None, use example_input_array shape if available.")
    action_parser.add_argument('--batch_size', type=int, default=None, help="Overwrite batch size in input size")
    action_parser.add_argument('--depth', type=int, default=100, help="Number of nested layers to traverse for detailed informations")
    action_parser.add_argument('--verbose', type=int, default=1, help="Verbose level of torchinfo.summary")

    return action_parser


def process(args):
    """ Process command line arguments """
    from nnpf.nn import display_model_infos

    display_model_infos(
        args.checkpoint,
        args.recursive,
        use_torch_info=args.use_torch_info,
        input_size=args.input_size,
        batch_size=args.batch_size,
        depth=args.depth,
        verbose=args.verbose,
    )

