import torch


__all__ = [
    "get_derivatives",
    "diff",
    "pad",
    "conv",
]


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
    >>> pad(x, (2, -1), mode='circular')
    tensor([3., 4., 0., 1., 2., 3.])
    >>> pad(x, (-2, 1), mode='circular')
    tensor([2., 3., 4., 0.])

    >>> x = torch.rand(2, 3, 4, 5)
    >>> x1 = pad(x, (4, 3, 2, 1), mode='circular')
    >>> x2 = torch.nn.functional.pad(x, (4, 3, 2, 1), mode='circular')
    >>> torch.allclose(x1, x2)
    True

    >>> x = torch.rand(2, 3, 4, 5)
    >>> x1 = pad(x, (4, 3, -2, 1))
    >>> x2 = torch.nn.functional.pad(x, (4, 3, -2, 1))
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
        idxl = [slice(None)] * data.dim() # Left slice
        idxr = [slice(None)] * data.dim() # Right slice
        idxc = [slice(None)] * data.dim() # Center slice (to handle negative padding <=> cropping)
        idxl[-d] = slice(data.shape[-d] - padding[i], None) # Negative padding => empty slice
        idxr[-d] = slice(max(0, padding[i+1])) # Negative padding must be set to 0
        idxc[-d] = slice(max(0, -padding[i]), data.shape[-d] - max(0, -padding[i+1]))
        data = torch.cat((data[idxl], data[idxc], data[idxr]), dim=data.dim() - d)

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


