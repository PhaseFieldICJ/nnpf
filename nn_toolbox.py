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


def reshape_for_torch(x, array_like=None, data_dim=None):
    """
    Reshape a given NumPy array or PyTorch tensor so that to have an additional dimension for color channel

    First dimension is considered as the sample dimension unless data_dim is specified.

    Parameters
    ----------
    x: NumPy array or PyTorch tensor
        The array to be reshaped
    array_like: NumPy array or PyTorch Tensor (optional)
        x dimension will be expanded to match the dimension of array_like before reshaping
        Used to make target dimension match prediction dimension.
    data_dim: int (optional)
        Dimension of the data.
        If the dimension of x is greater than data_dim, the additional dimensions (in first positions)
        will be considered as sample dimensions and flattened.

    Returns
    -------
    y: array of same kind as x
        Reshaped array so that to have the color channel dimension.


    Examples
    --------
    >>> reshape_for_torch(np.random.rand(5, 4)).shape # First dimension is considered as sample dimension by default
    (5, 1, 4)
    >>> reshape_for_torch(np.random.rand(5, 4), data_dim=2).shape
    (1, 1, 5, 4)
    >>> reshape_for_torch(np.random.rand(10, 5, 4)).shape
    (10, 1, 5, 4)
    >>> reshape_for_torch(np.random.rand(10, 10, 5, 4), data_dim=2).shape
    (100, 1, 5, 4)
    >>> reshape_for_torch(np.random.rand(100)).shape # Samples of 1D data are reshaped to 2D to ease manipulation
    (100, 1, 1)
    >>> reshape_for_torch(np.random.rand(100, 1), np.random.rand(5, 1, 1)).shape # Not sure if it is useful
    (100, 1, 1, 1)
    """

    new_shape = tuple(x.shape) # Converted to tuple to handle PyTorch tensor

    if data_dim is not None:
        # Flatten additional dimensions (or add the sampled dimension)
        new_shape = (-1,) + new_shape[-data_dim:]

    if array_like is not None:
        new_shape += (1,) * (array_like.ndim - x.ndim)

    # 1D inputs are transformed into 2D to ease some manipulations later
    new_shape += (1,) * (2 - len(new_shape))

    # Reshape and insert color channel dimension
    return x.reshape(new_shape)[:, None, ...]


def get_model_by_name(name):
    """ Return model class given its name.

    Search in global namespace, nn_models and torch.nn.
    """

    try:
        return globals()[name]
    except KeyError:
        pass

    try:
        import nn_models
        return getattr(nn_models, name)
    except AttributeError:
        pass

    try:
        import torch.nn
        return getattr(torch.nn, name)
    except AttributeError:
        pass

    raise AttributeError(f"Model {name} not found")


def get_derivatives(model, t, order=1):
    """
    Calculates the derivatives of a R -> R^N model up to a given order

    Parameters
    ----------
    model: any
        Model of a R - R^N function
    t: number
        Evaluation point of the derivatives
    order: int
        Maximum derivative order.

    Returns
    -------
    derivatives: list of tensors
        Model derivatives from order 0 to given maximal order.
    """
    output = [model(t)]
    for i in range(order):
        output.append(torch.autograd.grad(output[-1], [t], torch.ones_like(t), create_graph=True)[0])
    return output


def diff(data, n=1, axis=-1):
    """
    Calculate the n-th discrete difference along the given axis.

    Parameters
    ----------
    data: Tensor
        Input tensor
    n: int
        The number of times values are differenced.
    axis: int
        The axis along which the difference is taken.

    Examples
    --------
    >>> a = torch.Tensor([1., 2., 3.5, 4.6, 5.1])
    >>> diff(a)
    tensor([1.0000, 1.5000, 1.1000, 0.5000])
    >>> diff(a, n=2)
    tensor([ 0.5000, -0.4000, -0.6000])

    >>> a = torch.arange(8).reshape(2, 4)
    >>> diff(a, axis=1)
    tensor([[1, 1, 1],
            [1, 1, 1]])
    """

    # Creating slices
    dim = data.dim()
    left_slice = [slice(None)] * dim
    left_slice[axis] = slice(None, -1)
    left_slice = tuple(left_slice)
    right_slice = [slice(None)] * dim
    right_slice[axis] = slice(1, None)
    right_slice = tuple(right_slice)

    # Differences
    for i in range(n):
        data = data[right_slice] - data[left_slice]

    return data


def total_variation_norm(data, dim=None):
    """
    Total variation "norm"

    Parameters
    ----------
    data: Tensor
        The input tensor
    dim: int, tuple, list
        Dimensions along which to calculate the norm.
        See torch.norm (you want to specify it appropriately!)

    Exemples
    --------
    >>> a = torch.arange(8).reshape(2, 4)
    >>> total_variation_norm(a, dim=1)
    tensor([3, 3])
    """
    # Shape of the result
    if dim is None:
        dim = list(range(0, data.dim()))
    elif isinstance(dim, int):
        dim = [dim]
    out_dim = [data.shape[i] for i in range(data.dim()) if i not in dim]

    # Differences along dims
    out = data.new_zeros(out_dim)
    for i in dim:
        out += diff(data, axis=i).abs().sum(dim=dim)

    return out


# TODO: merge with total variation...
def laplacian_norm(data, dim=None):
    # Shape of the result
    if dim is None:
        dim = list(range(0, data.dim()))
    elif isinstance(dim, int):
        dim = [dim]
    out_dim = [data.shape[i] for i in range(data.dim()) if i not in dim]

    # Differences along dims
    out = data.new_zeros(out_dim)
    for i in dim:
        out += diff(data, n=2, axis=i).abs().sum(dim=dim)

    return out


def norm(data, p=2, dim=None):
    """
    Returns the matrix norm or vector norm of a given tensor

    Same as torch.norm with additional total variation "norm"

    Parameters
    ----------
    data: Tensor
        The input tensor
    p: int, float, inf, -inf, 'fro', 'nuc', 'tv'
        The order of the norm. 'tv' for total variation "norm".
        See torch.norm for other choices.
    dim: int, tuple, list
        Dimensions along which to calculate the norm.
        See torch.norm (you want to specify it appropriately!)
    """

    if p == 'tv':
        return total_variation_norm(data, dim)
    elif p == 'laplacian':
        return laplacian_norm(data, dim)
    else:
        return torch.norm(data, p, dim)

