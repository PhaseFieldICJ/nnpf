"""
A spatial and frequency discretized domain wrapper with associated utils
for real <-> complex discrete Fourier transformations (using FFT).
"""

import torch
import torch.fft
from nnpf.fft.domain import *

__all__ = ["Domain"]

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
    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2500, 0.2500, 0.2500, 0.2500, 0.2500],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.7500, 0.7500, 0.7500, 0.7500, 0.7500],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
    >>> d.X[1]
    tensor([[-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
            [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
            [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
            [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000],
            [-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]])
    >>> d.dX
    tensor([0.2500, 0.5000])
    >>> d.K[0]
    tensor([[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [-2., -2., -2.],
            [-1., -1., -1.]])
    >>> d.K[1]
    tensor([[0.0000, 0.5000, 1.0000],
            [0.0000, 0.5000, 1.0000],
            [0.0000, 0.5000, 1.0000],
            [0.0000, 0.5000, 1.0000],
            [0.0000, 0.5000, 1.0000]])
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
        """
        Spatial coordinates

        Notes
        -----
        It actually returns a view on repetitions of 1D tensors so that it is memory efficient.
        Thus, returned tensors should not be modified!
        """
        return torch.meshgrid(*(torch.linspace(a, b, n, device=self.device) for (a, b), n in zip(self.bounds, self.N)))

    @property
    def dX(self):
        """ Space steps """
        return torch.Tensor([(b - a) / (n - 1) for (a, b), n in zip(self.bounds, self.N)])

    def index(self, *X):
        """ Returns discretization indexes from given spatial coordinates

        Examples
        --------
        >>> from nnpf.domain import Domain
        >>> domain = Domain([[-1, 2], [0.5, 3]], [256, 128])
        >>> domain.index(-0.2, 1.2)
        (tensor(68), tensor(36))
        >>> domain[68, 36]
        (tensor(-0.2000), tensor(1.2087))
        >>> all(torch.equal(x1, x2) for x1, x2 in zip(domain.X, domain[domain.index(domain.X)]))
        True
        """
        if len(X) == 1 and isinstance(X[0], tuple):
            X = X[0]

        return tuple(((torch.as_tensor(x) - a) / dx + 0.5).long() for x, (a, b), dx in zip(X, self.bounds, self.dX))

    def __getitem__(self, idx):
        """ Array indexing for spatial coordinates (same as using X property) """
        return tuple(x[idx] for x in self.X)

    @property
    def K(self):
        """
        Frequency coordinates

        Notes
        -----
        It actually returns a view on repetitions of 1D tensors so that it is memory efficient.
        Thus, returned tensors should not be modified!
        """
        k = [fftfreq(n, (b - a) / n, device=self.device) for (a, b), n in zip(self.bounds[:-1], self.N[:-1])] \
             + [rfftfreq(self.N[-1], (self.bounds[-1][1] - self.bounds[-1][0]) / self.N[-1], device=self.device)]
        return torch.meshgrid(*k)

    def _check_real_shape(self, u):
        assert u.shape[-self.dim:] == torch.Size(self.spatial_shape), "Input shape doesn't match domain shape"

    def _check_complex_shape(self, u):
        assert u.shape[-1] == 2, "Input doesn't seems to be in complex format"
        assert u.shape[-(self.dim + 1):-1] == torch.Size(self.freq_shape), "Input shape doesn't match domain shape"

    def fft(self, u):
        """ Real -> Complex FFT with batch and channel support """
        return torch.fft.rfftn(u, s=self.spatial_shape)

    def ifft(self, u):
        """ Complex -> Real FFT with batch and channel support """
        return torch.fft.irfftn(u, s=self.spatial_shape)

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


