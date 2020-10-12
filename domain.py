"""
A spatial and frequency discretized domain wrapper with associated utils
for real <-> complex discrete Fourier transformations (using FFT).

Note: some functions are heavily inspired/copied from numpy equivalent functions
"""

import torch
import pytorch_lightning as pl

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


class Domain:
    """
    A spatial and frequency discretized domain wrapper with associated real <-> complex Fast Fourier Transform

    FFT works only for dimensions <= 3

    Examples
    --------

    >>> d = Domain([(0, 1)], 11)
    >>> d.X
    (tensor([0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
            0.9000, 1.0000]),)
    >>> d.K
    (tensor([0., 1., 2., 3., 4., 5.]),)
    >>> d = Domain([(0, 1), (-1, 1)], [5, 5])
    >>> d.X[0]
    tensor([[0.0000],
            [0.2500],
            [0.5000],
            [0.7500],
            [1.0000]])
    >>> d.X[1]
    tensor([[-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]])
    >>> d.K[0]
    tensor([[ 0.],
            [ 1.],
            [ 2.],
            [-2.],
            [-1.]])
    >>> d.K[1]
    tensor([[0.0000, 0.5000, 1.0000]])
    >>> _ = torch.manual_seed(0)
    >>> a = torch.rand(d.spatial_shape)
    >>> torch.allclose(a, d.ifft(d.fft(a)))
    True
    """

    def __init__(self, bounds, N, device=None):
        """
        Constructor

        Parameters
        ----------
        bounds: iterable of pairs of float
            Bounds of the domain
        N: int or iterable of int
            Number of discretization points
        """

        # Bounds and discretization
        self.bounds = [(a, b) for a, b in bounds]
        self.dim = len(self.bounds)
        if type(N) == int:
            self.N = [N] * self.dim
        else:
            self.N = tuple(N)
        if len(self.N) == 1:
            self.N = self.N * self.dim

        self.device = device
        self.spatial_shape = self.N
        self.freq_shape = tuple((*self.N[:-1], self.N[-1] // 2 + 1))

    def _broadcast_shape(self, i):
        return (1,) * i + (-1,) + (1,) * (self.dim - i - 1)

    @property
    def X(self):
        """ Spatial coordinates """
        return tuple(torch.linspace(a, b, n, device=self.device).reshape(self._broadcast_shape(i)) for i, ((a, b), n) in enumerate(zip(self.bounds, self.N)))

    @property
    def dX(self):
        """ Space steps """
        return torch.Tensor([(b - a) / n for (a, b), n in zip(self.bounds, self.N)])

    @property
    def K(self):
        """ Frequency coordinates """
        k = [fftfreq(n, (b - a) / n, device=self.device) for (a, b), n in zip(self.bounds[:-1], self.N[:-1])] \
             + [rfftfreq(self.N[-1], (self.bounds[-1][1] - self.bounds[-1][0]) / self.N[-1], device=self.device)]
        return tuple(k.reshape(self._broadcast_shape(i)) for i, k in enumerate(k))

    def _check_real_shape(self, u):
        assert u.shape[-self.dim:] == self.spatial_shape, "Input shape doesn't match domain shape"

    def _check_complex_shape(self, u):
        assert u.shape[-1] == 2, "Input doesn't seems to be in complex format"
        assert u.shape[-(self.dim + 1):-1] == self.freq_shape, "Input shape doesn't match domain shape"

    def fft(self, u):
        """ Real -> Complex FFT with batch and channel support """
        return torch.view_as_complex(torch.rfft(u, self.dim, normalized=False, onesided=True))

    def ifft(self, u):
        """ Complex -> Real FFT with batch and channel support """
        return torch.irfft(torch.view_as_real(u), self.dim, normalized=False, onesided=True, signal_sizes=self.spatial_shape)

    def freq_shift(self, u):
        """
        Shift the zero-frequency component to the center of the spectrum, with batch and channel support

        Parameters
        ----------
        u: Tensor
            Input tensor
        """
        assert u.shape[-self.dim:] == self.freq_shape, "Input shape doesn't match domain shape"
        return fftshift(u, axes=range(u.ndim - self.dim, u.ndim - 1)) # Do not shift last dimensin (see rfftfreq)

    def freq_ishift(self, u):
        """
        The inverse of `freq_shift`. Although identical for even-length `x`, the
        functions differ by one sample for odd-length `x`.

        Parameters
        ----------
        u: Tensor
            Input tensor
        """
        assert u.shape[-self.dim:] == self.freq_shape, "Input shape doesn't match domain shape"
        return ifftshift(u, axes=range(u.ndim - self.dim, u.ndim - 1)) # Do not shift last dimensin (see rfftfreq)

    def spatial_shift(self, u):
        """
        Shift the origin point (index 0) and corners to the center of the array, with batch and channel support

        Parameters
        ----------
        u: Tensor
            Input tensor
        """
        self._check_real_shape(u)
        return fftshift(u, axes=range(u.ndim - self.dim, u.ndim))

    def spatial_ishift(self, u):
        """
        The inverse of `spatial_shift`. Although identical for even-length `x`, the
        functions differ by one sample for odd-length `x`.

        Parameters
        ----------
        u: Tensor
            Input tensor
        """
        self._check_real_shape(u)
        return ifftshift(u, axes=range(u.ndim - self.dim, u.ndim))

    def __repr__(self):
        return f"Domain( {self.bounds}, {self.N} )"

    def __eq__(self, other):
        return self.bounds == other.bounds and self.N == other.N


