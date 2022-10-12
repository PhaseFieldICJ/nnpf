""" Machine learning models (convolution) """

import torch
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd

from nnpf.functional import conv
from nnpf.fft import fftconv


__all__ = [
    "ConvolutionArray",
    "FFTConvolutionArray",
    "ConvolutionFunction",
]


class ConvolutionArray(_ConvNd):
    """
    Model a discrete convolution kernel as an array

    Examples
    --------

    >>> _ = torch.set_grad_enabled(False)

    >>> conv = ConvolutionArray(3)
    >>> conv.weight[:] = torch.tensor([1., 1., 0.])
    >>> x = torch.arange(10.)[None, None, ...]
    >>> conv(x)
    tensor([[[ 0.,  1.,  3.,  5.,  7.,  9., 11., 13., 15., 17.]]])

    >>> conv = ConvolutionArray(3, padding_mode='circular')
    >>> conv.weight[:] = torch.tensor([1., 1., 0.])
    >>> x = torch.arange(10.)[None, None, ...]
    >>> conv(x)
    tensor([[[ 9.,  1.,  3.,  5.,  7.,  9., 11., 13., 15., 17.]]])
    """

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

        # Expand single value to tuple of given size
        from torch.nn.modules.utils import _ntuple

        # Kernel size determines the dimension
        kernel_size = _ntuple(1)(kernel_size)
        ntuple = _ntuple(len(kernel_size))

        # Kernel size must have odd size otherwise the convolution result will have a different size than the input.
        assert all(k % 2 == 1 for k in kernel_size), "Kernel must have odd size!"

        # Padding
        if padding == 'center':
            padding = tuple(s//2 for s in kernel_size)
        elif type(padding) == int:
            padding = ntuple(padding)

        # Using ConvNd base class from PyTorch to create/init weight & bias
        stride = ntuple(stride)
        dilation = ntuple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, ntuple(0), groups, bias, padding_mode)


    def forward(self, x):
        return conv(x, self.weight, self.bias, self.stride, self.padding, self.padding_mode, self.dilation, self.groups)


class FFTConvolutionArray(_ConvNd):
    """
    Model a fft convolution kernel as an array

    Examples
    --------

    >>> _ = torch.set_grad_enabled(False)

    >>> conv = FFTConvolutionArray(3)
    >>> conv.weight[:] = torch.tensor([1., 1., 0.])
    >>> x = torch.arange(10.)[None, None, ...]
    >>> torch.allclose(conv(x), torch.tensor([[[ 0.,  1.,  3.,  5.,  7.,  9., 11., 13., 15., 17.]]]))
    True

    >>> conv = FFTConvolutionArray(3, padding_mode='circular')
    >>> conv.weight[:] = torch.tensor([1., 1., 0.])
    >>> x = torch.arange(10.)[None, None, ...]
    >>> torch.allclose(conv(x), torch.tensor([[[ 9.,  1.,  3.,  5.,  7.,  9., 11., 13., 15., 17.]]]))
    True
    """

    def __init__(self, kernel_size, in_channels=1, out_channels=1, padding='center', padding_mode='zeros', bias=False):
        """
        Parameters:
        -----------
        kernel_size: int or tuple
            Size of the convolution kernel (a list for 2D and 3D convolutions)
        in_channels: int
            Number of channels in the input image
        out_channels: int
            Number of channels produced by the convolution
        padding: int or tuple
            Zero-padding added to both sides of the input. 'center' to center the kernel
        padding_mode: string
            'zeros', 'reflect', 'replicate' or 'circular'
        bias: bool
             If True, adds a learnable bias to the output.

        """

        # Expand single value to tuple of given size
        from torch.nn.modules.utils import _ntuple

        # Kernel size determines the dimension
        kernel_size = _ntuple(1)(kernel_size)
        ntuple = _ntuple(len(kernel_size))

        # Kernel size must have odd size otherwise the convolution result will have a different size than the input.
        assert all(k % 2 == 1 for k in kernel_size), "Kernel must have odd size!"

        # Padding
        if padding == 'center':
            padding = tuple(s//2 for s in kernel_size)
        else:
            padding = ntuple(padding)

        # Using ConvNd base class from PyTorch to create/init weight & bias
        super().__init__(
            in_channels, out_channels, kernel_size, ntuple(1), padding, ntuple(1),
            False, ntuple(0), 1, bias, padding_mode)

    def forward(self, x):
        return fftconv(x, self.weight, self.bias, self.padding, self.padding_mode)



###############################################################################

class ConvolutionFunction(Module):
    """
    Model a discrete convolution kernel as the discretization of a function

    FIXME: WIP, not tested
    """

    def __init__(self, function, kernel_size, in_channels=1, out_channels=1, stride=1, padding='center', padding_mode='zeros', dilation=1, groups=1, bias=False):
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
        padding_mode: string
            'zeros', 'reflect', 'replicate' or 'circular'
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
        self.padding_mode = padding_mode

        # Kernel size
        if type(kernel_size) == int:
            self.kernel_size = kernel_size,
        else:
            self.kernel_size = kernel_size

        # Kernel size must have odd size otherwise the convolution result will have a different size than the input.
        assert all(k % 2 == 1 for k in self.kernel_size), "Kernel must have odd size!"

        # Padding
        dim = len(self.kernel_size)
        if padding == 'center':
            self.padding = tuple(s//2 for s in self.kernel_size)
        elif type(padding) == int:
            self.padding = [padding] * dim
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
        #device = next(self.function.parameters()).device
        #tmp = torch.meshgrid(*(torch.arange(s, dtype=torch.float, device=device) - (s-1)/2 for s in self.kernel_size), indexing='ij')
        #self.pos = torch.cat((torch.flatten(index) for index in tmp), dim=1).view(-1, 1, dim)
        #self.pos = torch.tensor(flat_meshgrid(*(np.arange(s) - (s-1)/2 for s in self.kernel_size)), dtype=torch.float, device=device)
        self.pos = torch.Tensor(flat_meshgrid(*(np.arange(s) - (s-1)/2 for s in self.kernel_size)))

    def forward(self, x):
        if self.padding_mode == 'zeros':
            return self.convolution(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups)
        else:
            return self.convolution(
                torch.nn.functional.pad(x, tuple(i for p in self.padding[::-1] for i in [p, p]), mode=self.padding_mode), # See documentation of torch.nn.functional.pad
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=0, # Already padded
                dilation=self.dilation,
                groups=self.groups)


    @property
    def weight(self):
        """ Return the discretized kernel values """
        return self.function(self.pos).view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)


