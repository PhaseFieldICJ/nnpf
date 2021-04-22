""" Neural-network toolbox """

import torch
import numpy as np

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

    Search in global namespace, nn_models and torch.nn.
    """

    # Already a class
    import inspect
    if inspect.isclass(name):
        return name

    # Name is composed of the module and class name
    if '.' in name:
        import importlib
        parts = name.split('.')
        module = importlib.import_module('.'.join(parts[:-1]))
        return getattr(module, parts[-1])

    # Class available in current global scope
    try:
        return globals()[name]
    except KeyError:
        pass

    # Class defined in nn_models
    try:
        import nn_models
        return getattr(nn_models, name)
    except AttributeError:
        pass

    # Class defined in PyTorch
    try:
        import torch.nn
        return getattr(torch.nn, name)
    except AttributeError:
        pass

    raise AttributeError(f"Model {name} not found")



