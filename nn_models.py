""" Machine learning models """

import torch
from torch.nn import Module

import nn_toolbox

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

    layers = nn_toolbox.gen_function_layers(m, n, *activation_fn)
    return torch.nn.Sequential(*layers)



class GaussActivation(Module):
    """ Activation function based on a Gaussian """
    def forward(self, x):
        return torch.exp(-(x**2))


class LieSplitting(Module):
    """ Models a Lie splitting composed of 2 given models """

    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B

    def forward(self, x):
        return self.A(self.B(x))


class StrangSplitting(Module):
    """ Models a Strang splitting composed of 2 given models """

    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B

    def forward(self, x):
        return self.A(self.B(self.A(x)))


