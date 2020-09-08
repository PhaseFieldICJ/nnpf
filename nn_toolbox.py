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


def total_variation_norm(data, power=1, dim=None):
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
    >>> a = torch.arange(8.).reshape(2, 4)
    >>> total_variation_norm(a, dim=1)
    tensor([3., 3.])
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
        out += diff(data, axis=i).abs().pow(power).sum(dim=dim).pow(1/power)

    return out


# TODO: merge with total variation...
def laplacian_norm(data, power=1, dim=None):
    # Shape of the result
    if dim is None:
        dim = list(range(0, data.dim()))
    elif isinstance(dim, int):
        dim = [dim]
    out_dim = [data.shape[i] for i in range(data.dim()) if i not in dim]

    # Differences along dims
    out = data.new_zeros(out_dim)
    for i in dim:
        out += diff(data, n=2, axis=i).abs().pow(power).sum(dim=dim).pow(1/power)

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

    if isinstance(p, str) and len(p) >= 2 and p[:2] == 'tv':
        return total_variation_norm(data, power=float(p[2:]) if len(p) > 2 else 1, dim=dim)
    elif isinstance(p, str) and len(p) >= 9 and p[:9] == 'laplacian':
        return laplacian_norm(data, power=float(p[9:]) if len(p) > 9 else 1, dim=dim)
    else:
        return torch.norm(data, p, dim)


def pad(data, padding, mode='constant', value=0.):
    """
    Pads tensor.

    Fix issue https://github.com/pytorch/pytorch/issues/20981 with circular
    padding when right padding is null.

    Also remove dimension restriction for circular padding (i.e. works on
    any input dimension with any padding length) and is faster than torch
    version.

    See documentation of torch.nn.functional.pad

    Examples
    --------
    >>> x = torch.arange(5.)
    >>> pad(x, (1, 0), mode='circular')
    tensor([4., 0., 1., 2., 3., 4.])
    >>> pad(x, (0, 1), mode='circular')
    tensor([0., 1., 2., 3., 4., 0.])
    >>> pad(x, (2, 1), mode='circular')
    tensor([3., 4., 0., 1., 2., 3., 4., 0.])

    >>> x = torch.rand(2, 3, 4, 5)
    >>> x1 = pad(x, (4, 3, 2, 1), mode='circular')
    >>> x2 = torch.nn.functional.pad(x, (4, 3, 2, 1), mode='circular')
    >>> torch.allclose(x1, x2)
    True

    >>> x = torch.rand(2, 3, 4, 5, 6)
    >>> x1 = pad(x, (6, 5, 4, 3, 2, 1), mode='circular')
    >>> x2 = torch.nn.functional.pad(x, (6, 5, 4, 3, 2, 1), mode='circular')
    >>> torch.allclose(x1, x2)
    True
    """

    if mode != 'circular':
        return torch.nn.functional.pad(data, padding, mode, value)

    # From torch/nn/functional.py#_pad
    assert len(padding) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(padding) // 2 <= data.dim(), 'Padding length too large'
    assert all(max(padding[i], padding[i+1]) <= data.shape[data.dim() - i//2 - 1] for i in range(0, len(padding), 2)), 'Padding greater than the data size'

    for i in range(0, len(padding), 2):
        d = i // 2 + 1
        idxl = [slice(None)] * data.dim()
        idxr = [slice(None)] * data.dim()
        idxl[-d] = slice(data.shape[-d] - padding[i], None)
        idxr[-d] = slice(padding[i+1])
        data = torch.cat((data[idxl], data, data[idxr]), dim=data.dim() - d)

    return data


def conv(data, weight, bias=None, stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1):
    """
    Applies a convolution over an input image composed of several input planes.

    Note: works only in 1D, 2D and 3D.

    Parameters
    ----------
    data: tensor
        Input tensor of shape (minibatch, in_channels, Ni...)
    weight: tensor
        Filters of shape (out_channels, in_channels/groups, Ki...)
    bias: tensor
        Optional tensor of shape (out_channels,)
    stride: int
        Stride of the convolution
    padding: int or tuple
        Zero-padding added to both sides of the input. 'center' to center the kernel
    padding_mode: string
        'zeros', 'reflect', 'replicate' or 'circular'
    dilation: int
        Spacing between kernel elements
    groups: int
        Number of blocked connections from input channels to output channels

    Returns
    -------
    ouput: tensor
        Result of size ((Ni + 2*padding - dilation*(Ki -1) - 1)/stride + 1...)

    Examples
    --------
    >>> weight = torch.tensor([1., 2., 3.]).reshape(1, 1, -1)
    >>> x = torch.arange(10.).reshape(1, 1, -1)

    >>> conv(x, weight, padding='center')
    tensor([[[ 3.,  8., 14., 20., 26., 32., 38., 44., 50., 26.]]])

    >>> conv(x, weight, padding='center', padding_mode='circular')
    tensor([[[12.,  8., 14., 20., 26., 32., 38., 44., 50., 26.]]])
    """

    dim = data.ndim - 2
    assert dim >= 1, "Input must have at least 3 dimensions, including minibatch and channels"
    assert data.ndim == weight.ndim, "Input and weight must have same dimension"

    # Choosing appropriate convolution implementation
    if dim == 1:
        convolution = torch.nn.functional.conv1d
    elif dim == 2:
        convolution = torch.nn.functional.conv2d
    elif dim == 3:
        convolution = torch.nn.functional.conv3d
    else:
        raise ValueError('No convolution implementation in dimension {}'.format(dim))

    # Padding
    if padding == 'center':
        padding = tuple(s//2 for s in weight.shape[2:])
    elif isinstance(padding, int):
        padding = [padding] * dim

    # Padding mode & convolution
    if padding_mode == 'zeros':
        return convolution(data,
                           weight,
                           bias=bias,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups)
    else:
        # See documentation of torch.nn.functional.pad
        data = pad(data,
                   tuple(i for p in padding[::-1] for i in [p, p]),
                   mode=padding_mode)
        return convolution(data,
                           weight,
                           bias=bias,
                           stride=stride,
                           padding=0, # Already padded
                           dilation=dilation,
                           groups=groups)


def fftconv(data, weight, bias=None, padding=0, padding_mode='zeros'):
    """
    Applies a convolution over an input image composed of several input planes, using FFT.
    FIXME: NOT WORKING NOW!!!

    Parameters
    ----------
    data: tensor
        Input tensor of shape (minibatch, in_channels, Ni...)
    weight: tensor
        Filters of shape (out_channels, in_channels, Ki...)
    bias: tensor
        Optional tensor of shape (out_channels,)
    padding: int or tuple
        Zero-padding added to both sides of the input. 'center' to center the kernel
    padding_mode: string
        'zeros', 'reflect', 'replicate' or 'circular'

    Returns
    -------
    ouput: tensor
        Result of size (Ni + 2*padding - Ki + 1...)

    Examples
    --------
    >>> weight = torch.tensor([1., 2., 3.]).reshape(1, 1, -1)
    >>> x = torch.arange(10.).reshape(1, 1, -1)

    >>> xw = fftconv(x, weight, padding='center')
    >>> xw_ref = conv(x, weight, padding='center')
    >>> torch.allclose(xw, xw_ref)
    True

    >>> xw = fftconv(x, weight, padding='center', padding_mode='circular')
    >>> xw_ref = conv(x, weight, padding='center', padding_mode='circular')
    >>> torch.allclose(xw, xw_ref)
    True

    >>> x = torch.rand(1, 1, 10, 11)
    >>> weight = torch.rand(1, 1, 3, 5)
    >>> xw = fftconv(x, weight, padding='center')
    >>> xw_ref = conv(x, weight, padding='center')
    >>> torch.allclose(xw, xw_ref)
    True

    >>> x = torch.rand(4, 2, 5, 6, 7)
    >>> weight = torch.rand(1, 2, 3, 3, 5)
    >>> xw = fftconv(x, weight, padding=(1, 1, 2))
    >>> xw_ref = conv(x, weight, padding=(1, 1, 2))
    >>> torch.allclose(xw, xw_ref)
    True

    """

    # Checking dimensions
    dim = data.ndim - 2
    assert dim >= 1, "Input must have at least 3 dimensions, including minibatch and channels"
    assert data.ndim == weight.ndim, "Input and weight must have same dimension"

    # Padding size
    if padding == 'center':
        padding = tuple(s//2 for s in weight.shape[2:])
    elif isinstance(padding, int):
        padding = [padding] * dim

    # Padding input
    padding = tuple(i for p in padding[::-1] for i in [p, p]) # See torch.nn.functional.pad doc
    if padding_mode == 'zeros':
        data = pad(data, padding)
    elif padding_mode != 'circular':
        data = pad(data, padding, mode=padding_mode)
    print(data)

    # Padding weight
    kernel_size = weight.shape[2:]
    #weight = weight.flip(list(range(2, weight.dim())))
    weight = pad(weight, tuple(i for ds, ws in zip(data.shape[-1:1:-1], weight.shape[-1:1:-1]) for i in (0, ds - ws)))
    weight = weight.roll([-(k // 2 - 1) for k in kernel_size], tuple(range(2, weight.dim())))
    #print(weight.shape)

    # Forward transformation
    data_hat = torch.rfft(data[:, None, ...], dim)
    weight_hat = torch.rfft(weight, dim)
    #print(data_hat.shape)
    #print(weight_hat.shape)

    # Convolution
    # Insert dimension in data to take into account the input/output channels
    output_hat = weight_hat.conj() * data_hat
    #print(output_hat.shape)

    # Backward transformation
    output = torch.irfft(output_hat, dim, signal_sizes=data.shape[2:])
    print(output)
    output = output.sum(dim=2)

    # Truncate
    if padding_mode == 'circular':
        # Using FFT periodicity to pad & trunc only if necessary
        rel_padding = tuple(i for p, k in zip(padding[::-1], kernel_size[::-1])
                              for i in [p - (k - 1) // 2, p - k // 2])
        output = pad(output, tuple(max(0, p) for p in rel_padding), mode='circular')
        indexing = [slice(None)] * 2 + \
                   [slice(max(0, -rel_padding[i]), output.shape[i//2 + 2] - max(0, -rel_padding[i + 1]))
                    for i in range(0, len(rel_padding), 2)[::-1]]
    else:
        indexing = [slice(None)] * 2 + \
                   [slice((kernel_size[i] - 1) // 2, output.shape[i + 2] - kernel_size[i] // 2)
                    for i in range(dim)]

    output = output[indexing].contiguous()

    # Bias
    if bias is not None:
        output += bias # FIXME

    return output
