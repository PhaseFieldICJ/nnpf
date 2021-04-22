"""
Domain related FFT functions

Note: some functions are heavily inspired/copied from numpy equivalent functions
"""

import torch

__all__ = ["fftfreq", "rfftfreq", "fftshift", "ifftshift"]


def fftfreq(n, d=0.1, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies for complex <-> complex transformations

    Parameters
    ----------
    n: int
        Window length
    d: float
        Sample spacing (inverse of the sampling rate)

    Returns
    -------
    f: Tensor
        Array of length n containing the sample frequencies

    Note
    ----
    Copied from Numpy source

    Examples
    --------
    >>> freq = fftfreq(8, d=0.1)
    >>> freq
    tensor([ 0.0000,  1.2500,  2.5000,  3.7500, -5.0000, -3.7500, -2.5000, -1.2500])
    """

    val = 1.0 / (n * d)
    results = torch.empty(n, dtype=torch.int, device=device)
    N = (n - 1) // 2 + 1
    p1 = torch.arange(0, N, dtype=torch.int, device=device)
    results[:N] = p1
    p2 = torch.arange(-(n // 2), 0, dtype=torch.int, device=device)
    results[N:] = p2
    return results * val


def rfftfreq(n, d=0.1, device=None):
    """
    Return the Discrete Fourier Transform sample frequencies for real <-> complex transformations

    Parameters
    ----------
    n: int
        Window length
    d: float
        Sample spacing (inverse of the sampling rate)

    Returns
    -------
    f: Tensor
        Array of length n containing the sample frequencies

    Note
    ----
    Copied from Numpy source

    Examples
    --------
    >>> freq = rfftfreq(8, d=0.1)
    >>> freq
    tensor([0.0000, 1.2500, 2.5000, 3.7500, 5.0000])
    """

    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = torch.arange(0, N, dtype=torch.int, device=device)
    return results * val


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : Tensor
        The shifted tensor.

    Examples
    --------
    >>> freqs = fftfreq(8, 0.1)
    >>> freqs
    tensor([ 0.0000,  1.2500,  2.5000,  3.7500, -5.0000, -3.7500, -2.5000, -1.2500])
    >>> fftshift(freqs)
    tensor([-5.0000, -3.7500, -2.5000, -1.2500,  0.0000,  1.2500,  2.5000,  3.7500])

    Note
    ----
    Almost copied from Numpy source
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,) * x.ndim
    else:
        axes = tuple(axes)

    shift = [x.shape[ax] // 2 for ax in axes]

    return torch.roll(x, shift, axes)


def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : Tensor
        The shifted tensor.

    Examples
    --------
    >>> freqs = fftfreq(8, 0.1)
    >>> freqs
    tensor([ 0.0000,  1.2500,  2.5000,  3.7500, -5.0000, -3.7500, -2.5000, -1.2500])
    >>> ifftshift(fftshift(freqs))
    tensor([ 0.0000,  1.2500,  2.5000,  3.7500, -5.0000, -3.7500, -2.5000, -1.2500])

    Note
    ----
    Almost copied from Numpy source
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,) * x.ndim
    else:
        axes = tuple(axes)

    shift = [-(x.shape[ax] // 2) for ax in axes]

    return torch.roll(x, shift, axes)

