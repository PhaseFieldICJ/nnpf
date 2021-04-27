import torch
from nnpf.functional import pad

__all__ = ["fftconv"]


def fftconv(data, weight, bias=None, padding=0, padding_mode='zeros'):
    """
    Applies a convolution over an input image composed of several input planes, using FFT.

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

    >>> x = torch.rand(4, 2, 5, 6, 7)
    >>> weight = torch.rand(3, 2, 3, 3, 5)
    >>> bias = torch.rand(3)
    >>> xw = fftconv(x, weight, padding=(1, 1, 2), bias=bias)
    >>> xw_ref = conv(x, weight, padding=(1, 1, 2), bias=bias)
    >>> torch.allclose(xw, xw_ref)
    True

    Massive test:
    >>> import itertools
    >>> N_range = range(29, 33)
    >>> k_range = range(9, 13)
    >>> p_range = range(3, 7)
    >>> m_range = ('zeros', 'reflect', 'circular')
    >>> failed = set()
    >>> for N, k, p, m in itertools.product(N_range, k_range, p_range, m_range):
    ...     x = torch.rand(5, 2, N)
    ...     w = torch.rand(3, 2, k)
    ...     xw = fftconv(x, w, padding=(p,), padding_mode=m)
    ...     xw_ref = conv(x, w, padding=(p,), padding_mode=m)
    ...     if xw.shape != xw_ref.shape or not torch.allclose(xw, xw_ref):
    ...         failed.add((N, k, p, m))
    >>> failed
    set()

    """

    # Checking dimensions
    dim = data.ndim - 2
    assert dim >= 1, "Input must have at least 3 dimensions, including minibatch and channels"
    assert data.ndim == weight.ndim, "Input and weight must have same dimension"

    # Checking padding mode
    assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular'], "Unsupported padding mode"
    if padding_mode == 'zeros':
        padding_mode = 'constant'

    # Padding size
    if padding == 'center':
        padding = tuple(s//2 for s in weight.shape[2:])
    elif isinstance(padding, int):
        padding = [padding] * dim

    # Padding input
    # Not needed for circular padding, using periodicity of FFT
    if padding_mode != 'circular':
        # See torch.nn.functional.pad doc about the reverse order of the padding spec
        input_padding = tuple(i for p in padding[::-1] for i in [p, p])
        data = pad(data, input_padding, mode=padding_mode)

    # Return next odd
    # Used to virtually pad to the right kernel with even size
    def next_odd(n):
        return n if n % 2 else n + 1

    # Padding weight and flip axes
    # Flipping axes is faster than taking conjugate of data_hat before multiplication
    #   because weight size is often lower than data size
    kernel_size = weight.shape[2:]
    weight_padding = [i for n, k in zip(data.shape[:1:-1], kernel_size[::-1])
                        for i in [(n - next_odd(k)) // 2, n - k - (n - next_odd(k)) // 2]]
    weight = pad(weight, weight_padding).flip(tuple(range(2, weight.ndim)))

    # Forward transformation
    # Insert dimension in data to take into account the input/output channels
    data_hat = torch.view_as_complex(torch.rfft(data[:, None, ...], dim))
    weight_hat = torch.view_as_complex(torch.rfft(weight, dim))

    # Convolution
    output_hat = torch.view_as_real(data_hat * weight_hat)

    # Backward transformation
    from .domain import ifftshift
    output = torch.irfft(output_hat, dim, signal_sizes=data.shape[2:])
    output = output.sum(dim=2) # Summing input channels
    output = ifftshift(output, axes=range(data.ndim - dim , data.ndim))

    # Output padding or truncating
    if padding_mode == 'circular':
        output_padding = [i for p, k in zip(padding[::-1], kernel_size[::-1])
                            for i in [p - k // 2, p - (k - 1 - k // 2)]]
    else:
        output_padding = [i for k in kernel_size[::-1]
                            for i in [- (k // 2), - (k - 1 - k // 2)]]
    output = pad(output, output_padding, mode=padding_mode)

    # Bias
    if bias is not None:
        output += bias.reshape((-1,) + (1,) * dim)

    return output

