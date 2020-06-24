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


class ConvolutionArray(torch.nn.Module):
    """ Model a discrete convolution kernel as an array """

    def __init__(self, kernel_size, in_channels=1, out_channels=1, stride=1, padding='center', padding_mode='zeros', dilation=1, groups=1, bias=False):
        """
        Parameters:
        -----------
        kernel_size: int or tuple
            Size of the convolution kernel (a list for 2D and 3D convolutions)
        in_channels: int
            Number of channels in the input image
        out_channels: int
            Number of channels produced by the convolution
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
        bias: bool
             If True, adds a learnable bias to the output.

        """

        super().__init__()

        # Default values and sanity checks
        if type(kernel_size) == int:
            kernel_size = kernel_size,

        dim = len(kernel_size)

        if padding == 'center':
            padding = tuple(s//2 for s in kernel_size)
        elif type(padding) == int:
            padding = padding,

        # Arguments
        args = (in_channels, out_channels, kernel_size)
        kwargs = {'stride': stride, 'padding': padding, 'padding_mode': padding_mode, 'dilation': dilation, 'groups': groups, 'bias': bias}

        # Choosing appropriate convolution implementation
        if dim == 1:
            self.convolution = torch.nn.Conv1d(*args, **kwargs)
        elif dim == 2:
            self.convolution = torch.nn.Conv2d(*args, **kwargs)
        elif dim == 3:
            self.convolution = torch.nn.Conv3d(*args, **kwargs)
        else:
            raise ValueError('No convolution implementation in dimension {}'.format(dim))

    def forward(self, x):
        return self.convolution(x)

    @property
    def weight(self):
        """ Return the discretize kernel values """
        return self.convolution.weight

    @weight.setter
    def weight(self, new_weight):
        """ Set the discretize kernel values

        Note
        ----
        Prefer modifying using `self.weight[:] = new_weight` syntax
        in order to keep original data type.
        """
        self.convolution.weight = new_weight

    @property
    def bias(self):
        """ Return the bias """
        return self.convolution.bias

    @bias.setter
    def bias(self, new_bias):
        """ Set the bias

        Note
        ----
        Prefer modifying using `self.bias[:] = new_bias` syntax
        in order to keep original data type.
        """
        self.Convolution.bias = new_bias


class ConvolutionFunction(torch.nn.Module):
    """
    Model a discrete convolution kernel as the discretization of a function

    FIXME: WIP
    """

    def __init__(self, function, kernel_size, in_channels=1, out_channels=1, stride=1, padding='center', dilation=1, groups=1, bias=False):
        """
        Parameters:
        -----------
        function: class
            model of a R^n to R function (n being the dimension of kernel_size)
        kernel_size: int or tuple
            number of points of discretization (may be a list for nD convolution)
        in_channels: int
            Number of channels in the input image
        out_channels: int
            Number of channels produced by the convolution
        stride: int
            Stride of the convolution
        padding: int or tuple
            Zero-padding added to both sides of the input. 'center' to center the kernel
        dilation: int
            Spacing between kernel elements
        groups: int
            Number of blocked connections from input channels to output channels
        bias: bool or Tensor
            Optional bias of shape (out_channels). If True, adds a learnable bias to the output.
        """

        super().__init__()

        self.function = function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Kernel size
        if type(kernel_size) == int:
            self.kernel_size = kernel_size,
        else:
            self.kernel_size = kernel_size

        # Padding
        dim = len(self.kernel_size)
        if padding == 'center':
            self.padding = tuple(s//2 for s in self.kernel_size)
        elif type(padding) == int:
            self.padding = padding,
        else:
            self.padding = padding

        # Bias
        if bias == True:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        elif bias == False:
            self.bias = None
        else:
            self.bias = bias
            self.out_channels = bias.shape[0]

        # Choosing appropriate convolution implementation
        if dim == 1:
            self.convolution = torch.nn.functional.conv1d
        elif dim == 2:
            self.convolution = torch.nn.functional.conv2d
        elif dim == 3:
            self.convolution = torch.nn.functional.conv3d
        else:
            raise ValueError('No convolution implementation in dimension {}'.format(dim))

        # Generates the N*dim array of all x_i, y_j, ...
        # with N = prod(kernel_size)
        # 1) index range along each axis
        # 2) repeat using meshgrid along remaining axis
        # 3) flatten each index array
        # 4) concatenate each index vectors
        device = next(self.function.parameters()).device
        #tmp = torch.meshgrid(*(torch.arange(s, dtype=torch.float, device=device) - (s-1)/2 for s in self.kernel_size))
        #self.pos = torch.cat((torch.flatten(index) for index in tmp), dim=1).view(-1, 1, dim)
        self.pos = torch.tensor(flat_meshgrid(*(np.arange(s) - (s-1)/2 for s in self.kernel_size)), dtype=torch.float, device=device)

    def forward(self, x):
        return self.convolution(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)

    @property
    def weight(self):
        """ Return the discretized kernel values """
        return self.function(self.pos).view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)


