import torch
import math

__all__ = [
    "heat_kernel_freq",
    "heat_kernel_spatial",
]


def heat_kernel_freq(domain, dt):
    """ Return the discretizes heat kernel in frequency domain.

    Parameters
    ----------
    domain: domain.Domain
        The discretized domain
    dt: float
        Time step

    Returns
    -------
    kernel: Tensor
        The discretized heat kernel (complex format)
    """
    kernel = torch.exp(-4. * math.pi**2 * sum(k**2 for k in domain.K) * dt)
    return kernel * (1 + 0j) # FIXME: to complex format


def heat_kernel_spatial(domain, dt, truncate=None):
    """ Return the discretizes heat kernel in spatial domain.

    Parameters
    ----------
    domain: domain.Domain
        The discretized domain
    dt: float
        Time step
    truncate: int or (int, int)
        Truncate kernel to have the given size.

    Returns
    -------
    kernel: numpy.array
        The discretized heat kernel.
    """
    kernel_freq = heat_kernel_freq(domain, dt)
    kernel_spatial = domain.spatial_shift(domain.ifft(kernel_freq))

    # Truncating kernel
    if truncate is not None:
        if type(truncate) == int:
            truncate = [truncate] * domain.dim

        indexing = tuple(slice(n//2 - t//2, n//2 + (t - t//2), None) for n, t in zip(domain.N, truncate))
        kernel_spatial = kernel_spatial[indexing]

    return kernel_spatial

