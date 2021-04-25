def actions():
    import argparse

    # Command line arguments
    parser = argparse.ArgumentParser(description="Neural Network for Phase Field models")
    subparser = parser.add_subparsers(dest="action", title="Actions", description="Action to launch")
    subparser.required = True

    from nnpf.cmd import infos
    infos_parser = subparser.add_parser(
        'infos',
        description=infos.__doc__,
        help=infos.__doc__,
    )
    infos.add_arguments(infos_parser)
    infos_parser.set_defaults(process=infos.process)

    args = parser.parse_args()
    args.process(args)

