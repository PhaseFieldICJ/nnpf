"""
Benchmark a given model
"""

import timeit
import nnpf.cmd.train as train

__all__ = [
    "add_parser",
    "process",
]


def time_it(stmt='pass', setup='pass'):
    timer = timeit.Timer(stmt, setup)
    r = timer.autorange() # Returns (execution count, total time)
    return r[1] / r[0]


def add_parser(parser, drill_parser):
    """ Add subparser for benchmarking action """
    action_parser = train.add_parser(parser, drill_parser, name="bench", doc=__doc__)

    group = action_parser.add_argument_group("Benchmark")
    group.add_argument('--input_size', type=lambda v: [int(s) for s in v.split(',')], default=None, help="Input size. Example: 1,1,256,256. If None, use example_input_array shape if available.")
    group.add_argument('--use_torch_info', type=lambda v: bool(int(v)), default=True, help="Display detailed informations and memory usage using torchinfo package")
    group.add_argument('--depth', type=int, default=100, help="Number of nested layers to traverse for detailed memory usage stimation")
    group.add_argument('--sampling', type=float, default=1e-3, help="Sampling rate when measuring memory usage")

    return action_parser


def process(args):
    """ Process command line arguments """
    from nnpf.nn import get_model_by_name
    from nnpf.trainer import Trainer
    import torch

    # Model, training & fit
    Model = get_model_by_name(args.model)
    model = Model(**vars(args))

    # Fake input
    if args.input_size is None:
        args.input_size = list(model.example_input_array.shape)
        if args.batch_size is not None:
            args.input_size[0] = args.batch_size
    data = torch.rand(args.input_size)

    print()
    print(f"Input shape: {args.input_size}")

    # Benchmarks
    print()
    print("Benchmarks:")
    print("\tInference: ", flush=True, end='')
    model.freeze()
    duration = time_it(lambda: model(data))
    print(f"{duration:.2e} s ({1/duration:.2e} ips)")

    print("\tLoss:      ", flush=True, end='')
    output = model(data)
    target = torch.rand(output.shape)
    duration = time_it(lambda: model.loss(output, target))
    print(f"{duration:.2e} s ({1/duration:.2e} ips)")

    print("\tBackward:  ", flush=True, end='')
    model.unfreeze()
    output = model(data)
    target = torch.rand(output.shape)
    loss = model.loss(output, target)
    duration = time_it(lambda: loss.backward(retain_graph=True))
    print(f"{duration:.2e} s ({1/duration:.2e} ips)")

    if args.use_torch_info:
        try:
            from torchinfo import summary
        except ModuleNotFoundError:
                print("Package torchinfo not installed: missing detailed informations and memory usage!")
        finally:
            infos = summary(model, input_data=data, depth=args.depth, verbose=0)
            print()
            print("Parameters & mult-add:")
            print(f"\tTotal params: {infos.total_params:,}")
            print(f"\tTrainable params: {infos.trainable_params:,}")
            print(f"\tNon-trainable params: {infos.total_params - infos.trainable_params:,}")
            print("\tTotal mult-adds ({}): {:0.2f}".format(*infos.to_readable(infos.total_mult_adds)))

            print()
            print("Estimated memory usage (MiB):")
            print(f"\tInput size: {infos.to_bytes(infos.total_input):0.2f}")
            print(f"\tForward/backward pass size: {infos.to_bytes(infos.total_output):0.2f}")
            print(f"\tParams size: {infos.to_bytes(infos.total_params):0.2f}")
            print(f"\tEstimated total size: {infos.to_bytes(infos.total_input + infos.total_output + infos.total_params):0.2f}")


    print()
    print("Measured memory usage (MiB):")

    try:
        from memory_profiler import memory_usage
    except ModuleNotFoundError:
        print("Package memory_profiler not instralled: missing measured memory usage")
    finally:
        def mem_it(f, *pargs, no_return=False, **kwargs):
            mem = memory_usage((f, pargs, kwargs), interval=args.sampling, timeout=10.)
            print(f"initial: {mem[0]:7.2f} ; final: {mem[-1]:7.2f} ; increment: {mem[-1] - mem[0]:8.2f} ; peak: {max(mem):7.2f} ; delta max: {max(mem) - mem[0]:7.2f}")
            return None if no_return else f(*pargs, **kwargs)

        # Cleaning environment
        import gc
        del model, data, output, target, loss
        gc.collect()

        print("\tModel:             ", flush=True, end='')
        model = mem_it(Model, **vars(args))
        model.freeze()

        print("\tInput data:        ", flush=True, end='')
        data = mem_it(torch.rand, args.input_size)

        print("\tInference:         ", flush=True, end='')
        mem_it(model, data, no_return=True)

        print("\tForward:           ", flush=True, end='')
        model.unfreeze()
        output = mem_it(model, data)

        print("\tLoss:              ", flush=True, end='')
        loss = mem_it(model.loss, output, output)

        print("\tBackward (retain): ", flush=True, end='')
        mem_it(loss.backward, no_return=True, retain_graph=True)

        print("\tBackward:          ", flush=True, end='')
        mem_it(loss.backward, no_return=True)


    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\tMaximum resident set size (MiB): {maxrss / 1024:0.2f}")


