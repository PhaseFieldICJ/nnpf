import torch

from .operators import diff

__all__ = [
    "total_variation_norm",
    "laplacian_norm",
    "norm",
]


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


