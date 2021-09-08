""" Neural-network toolbox """

import torch


__all__ = [
    "gen_function_layers",
    "ndof",
    "get_model_by_name",
    "display_model_infos",
]


def gen_function_layers(m, n, *activation_fn):
    """ Generates the modules of a R^m -> R^n function model

    Activation layers is interleaved with Linear layers of appropriate
    dimensions and with a bias.

    Parameters
    ----------
    m: int
        Input domain dimension
    n: int
        Output domain dimension
    *activation_fn: many pairs (fn, dim)
        Multiple pairs of activation functions and working dimensions
        for hidden layers

    Returns
    -------
    layers: iterator
        Generator of neural network modules (suitable for torch.nn.Sequential)

    Example
    -------
    >>> torch.nn.Sequential(*gen_function_layers(2, 5, (torch.nn.ELU(), 10), (torch.nn.Tanh(), 7)))
    Sequential(
      (0): Linear(in_features=2, out_features=10, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=10, out_features=7, bias=True)
      (3): Tanh()
      (4): Linear(in_features=7, out_features=5, bias=True)
    )
    """

    curr_dim = m
    for fn, dim in activation_fn:
        yield torch.nn.Linear(curr_dim, dim, bias=True)
        yield fn
        curr_dim = dim

    yield torch.nn.Linear(curr_dim, n, bias=True)


def ndof(model):
    """ Number of Degree Of Freedom of a model

    Parameters
    ----------
    model: object
        A PyTorch model

    Returns
    -------
    count: int
        Number of (unique) parameters of the model

    Examples
    --------
    >>> bound_layer = torch.nn.Linear(2, 3, bias=False)
    >>> middle_layer = torch.nn.Linear(3, 2, bias=True)
    >>> model = torch.nn.Sequential(bound_layer, middle_layer, bound_layer) # Repeated bound layer
    >>> ndof(model) # 2*3 + 3*2 + 2 = 14
    14
    """
    addr_set = set()
    count = 0
    for param in model.state_dict().values():
        # Using address of first element as unique identifier
        if param.data_ptr() not in addr_set:
            count += param.nelement()
            addr_set.add(param.data_ptr())

    return count


def get_model_by_name(name):
    """ Return model class given its name.

    Search in (first found):
    - in the file specified before the last ':' token if found
    - a module (also from the current folder) if the name has a dot
    - global namespace
    - nnpf.nn
    - nnpf.models
    - torch.nn
    """

    # Already a class
    import inspect
    if inspect.isclass(name):
        return name

    from nnpf.utils import fix_path

    # Name is composed of the path and class name
    if name.find(':') > 0: # Ignore if at first position
        splits = name.split(':')
        path, name = fix_path(':'.join(splits[:-1])), splits[-1]
    else:
        path, name = None, name.split(':')[-1]

    # Path is specified
    if path is not None:
        try:
            import importlib.util
            import sys
            # Using magick module name to use appropriate path in Problem.on_save_checkpoint
            spec = importlib.util.spec_from_file_location("IMPORT_FROM_FILE", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return getattr(module, name)
        except:
            pass

    # Name is composed of the module and class name
    if '.' in name:
        import importlib
        import sys
        import os

        parts = name.split('.')
        # FIXME: this is ugly
        sys.path.append(os.getcwd())
        module = importlib.import_module('.'.join(parts[:-1]))
        sys.path.pop()
        return getattr(module, parts[-1])

    # Class available in current global scope
    try:
        return globals()[name]
    except KeyError:
        pass

    # Class defined in nn
    try:
        import nnpf.nn
        return getattr(nnpf.nn, name)
    except AttributeError:
        pass

    # Class defined in models
    try:
        import nnpf.models
        return getattr(nnpf.models, name)
    except AttributeError:
        pass

    # Class defined in PyTorch
    try:
        import torch.nn
        return getattr(torch.nn, name)
    except AttributeError:
        pass

    raise AttributeError(f"Model {name} not found")


def display_model_infos(model_or_path, recursive=True, use_torch_info=True, input_size=None, batch_size=None, depth=4, verbose=1):
    """
    Display the informations about a model

    Parameters
    ----------
    model_or_path: Ligthning module or str
        The model or the path to a checkpoint
    recursive: bool
        If True, search for checkpoint in the hyper-parameters and display associated infos.
    use_torch_info: bool
        If True, display detailed informations and memory usage using torchinfo package.
    input_size: None or list of int
        Input size for calculating memory usage. If None, use example_input_array shape if available.
    batch_size: None or int
        Overwrite batch size in input_size (first dimension).
    depth: int
        Number of nested layers to traverse for detailed informations.
    verbose: int
        Verbose level of torchinfo.summary.

    Examples
    --------

    Training a model:
    >>> from nnpf.models import Reaction
    >>> from nnpf.trainer import Trainer
    >>> trainer = Trainer(default_root_dir="logs_doctest", name="Reaction", version="test_model_infos", max_epochs=1, log_every_n_steps=1)
    >>> model = Reaction(train_N=10, val_N=20, seed=0, num_workers=4)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)

    Displaying informations of this trained model:
    >>> import os
    >>> display_model_infos(os.path.join('logs_doctest', 'Reaction', 'test_model_infos'), use_torch_info=False) # doctest:+ELLIPSIS
    <BLANKLINE>
    Model summary:
        class: Reaction
        class path: ...
        module: nnpf.models.reaction
        problem: ReactionProblem
        ndof: 47
    <BLANKLINE>
    Checkpoint:
        class name: nnpf.models.reaction.Reaction
        class path: None
        checkpoint path: logs_doctest/Reaction/test_model_infos/checkpoints/epoch=0-step=0.ckpt
        epochs: 1
        steps: 1
        current score: 1.655346155166626
        best score: 1.655346155166626
        best path: logs_doctest/Reaction/test_model_infos/checkpoints/epoch=0-step=0.ckpt
        PyTorch-Ligthing version: ...
    <BLANKLINE>
    Model hyper parameters:
        seed: 0
        dt: 6.103515625e-05
        epsilon: 0.0078125
        margin: 0.1
        train_N: 10
        val_N: 20
        batch_size: 10
        batch_shuffle: True
        lr: 0.001
        num_workers: 4
        layer_dims: [8, 3]
        activation: GaussActivation
    <BLANKLINE>
    """

    import inspect
    import torch

    from nnpf.utils import checkpoint_from_path
    from nnpf.problems import Problem

    is_path = isinstance(model_or_path, str)
    if is_path:
        map_location = torch.device("cpu")
        checkpoint_path = checkpoint_from_path(model_or_path)
        model = Problem.load_from_checkpoint(checkpoint_path, map_location=map_location)
        extra_data = torch.load(checkpoint_path, map_location=map_location)

        try:
            best_infos = next(value for callback, value in extra_data['callbacks'].items() if callback.__name__ == 'ModelCheckpoint')
            best_model_score = best_infos.get('best_model_score')
            best_model_path = best_infos.get('best_model_path')
            current_score = best_infos.get('current_score')
        except KeyError:
            # Compatibility with checkpoints generated by previous Lightning versions
            best_model_score = extra_data.get('checkpoint_callback_best_model_score')
            best_model_path = extra_data.get('checkpoint_callback_best_model_path')
            current_score = None
    else:
        model = model_or_path

    model_class = type(model)
    problem_class = model_class.__bases__[0]

    print(f"""
Model summary:
    class: {model_class.__name__}
    class path: {inspect.getfile(model_class)}
    module: {model_class.__module__}
    problem: {problem_class.__name__}
    ndof: {ndof(model)}
""")

    if is_path:
        print(f"""\
Checkpoint:
    class name: {extra_data['class_name']}
    class path: {extra_data['class_path']}
    checkpoint path: {checkpoint_path}
    epochs: {extra_data['epoch']}
    steps: {extra_data['global_step']}
    current score: {current_score}
    best score: {best_model_score}
    best path: {best_model_path}
    PyTorch-Ligthing version: {extra_data.get('pytorch-lightning_version')}
""")

    print("Model hyper parameters:")
    for key, value in model.hparams.items():
        print(f"    {key}: {value}")
    print()

    if use_torch_info:
        print("Details and memory usage:")
        try:
            from torchinfo import summary

            if input_size is None:
                try:
                    input_size = list(model.example_input_array.shape)
                except AttributeError:
                    pass

            if batch_size is not None and input_size is not None:
                input_size[0] = batch_size

            if input_size is not None:
                print(f"Input shape: {input_size}")

            summary(model, input_size=input_size, depth=depth, verbose=verbose)
            print()

        except ModuleNotFoundError:
                print("Package torchinfo not installed: missing detailed informations and memory usage!")

    if recursive:
        for key, value in model.hparams.items():
            if isinstance(key, str) and key.find('checkpoint') >= 0:
                if isinstance(value, str):
                    print()
                    msg = f"Dependant model found in {key}: {value}"
                    print('#' * len(msg))
                    print(msg)
                    display_model_infos(value, recursive)
                else:
                    for i, v in enumerate(value):
                        print()
                        msg = f"Dependant model found in {key}[{i}]: {v}"
                        print('#' * len(msg))
                        print(msg)
                        display_model_infos(v, recursive)

