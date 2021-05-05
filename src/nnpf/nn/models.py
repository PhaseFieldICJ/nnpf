""" Machine learning models """

import torch
from torch.nn import Module, Linear, Sequential

from nnpf.nn.utils import gen_function_layers


__all__ = [
    "Function",
    "GaussActivation",
    "Parallel",
    "LinearChannels",
    "Residual",
]


def Function(m, n, *activation_fn):
    """
    Model a R^m->R^n function by a multiple linear layers neural network
    with custom activation functions.

    Parameters
    ----------
    m: int
        Input domain dimension
    n: int
        Output domain dimension
    *activation_fn: many pairs (fn, dim)
        Multiple pairs of activation functions and working dimensions for hidden layers
    """

    layers = gen_function_layers(m, n, *activation_fn)
    return torch.nn.Sequential(*layers)


class GaussActivation(Module):
    """ Activation function based on a Gaussian """
    def forward(self, x):
        return torch.exp(-(x**2))


class Parallel(Sequential):
    """ A parallel container.

    Modules will be stacked in a parallel container, each module working
    on a different channel (input's channel dimension is expanded if needed)
    and outputing in the same channel.

    Note: also see Sequential documentation for more construction capablities

    Parameters
    ----------
    modules: iterable
        The modules

    Examples
    --------
    >>> from torch.nn import Sequential

    >>> model = Sequential(
    ...     Parallel(Linear(2, 4), Linear(2, 4), Linear(2, 4)),
    ...     LinearChannels(3, 1, bias=False))
    >>> data = torch.rand(10, 1, 2)
    >>> output = model(data)
    >>> target = sum(model[1].weight[0, i] * module(data) for i, module in enumerate(model[0]))
    >>> torch.allclose(output, target)
    True

    >>> model = Sequential(
    ...     LinearChannels(1, 3, bias=False),
    ...     Parallel(Linear(2, 4), Linear(2, 4), Linear(2, 4)),
    ...     LinearChannels(3, 1, bias=False))
    >>> data = torch.rand(10, 1, 2)
    >>> output = model(data)
    >>> target = sum(model[2].weight[0, i] * module(model[0].weight[i, 0] * data) for i, module in enumerate(model[1]))
    >>> torch.allclose(output, target)
    True
    """
    def forward(self, data):
        """ Apply the model on data

        Parameters
        ----------
        data: Tensor of size (N, C, ...)
            Input with a number of channels C equal to 1 or the number
            of modules in this Parallel container.

        Returns
        -------
        output: Tensor of size (N, M, ...)
            Output with a number of channels M equal to the number of modules
            in this Parallel container.
        """
        data = data.expand(data.shape[0], len(self), *data.shape[2:])
        return torch.cat(
            tuple(module(data[:, [i], ...]) for i, module in enumerate(self)),
            dim=1)


class LinearChannels(Linear):
    """
    Applies a linear transformation to the channels of the incoming data

    Parameters
    ----------
    in_channels: int
        number of input channels
    out_channels: int
        number of output channels
    bias: bool
        If set to False, the layer will not learn an additive bias. Default: True
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, bias)

    def forward(self, data):
        perm = [0, -1] + list(range(2, data.ndim - 1)) + [1]
        return super().forward(data.permute(*perm)).permute(*perm)


class Residual(Module):
    """
    A residual block
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        out = self.model(data)
        out += data
        return out


