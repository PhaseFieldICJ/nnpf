import numpy as np

__all__ = [
    "reshape_for_torch",
    "flat_coords_array",
    "flat_meshgrid",
]


def reshape_for_torch(x, array_like=None, data_dim=None):
    """
    Reshape a given NumPy array or PyTorch tensor so that to have an additional dimension for color channel

    First dimension is considered as the sample dimension unless data_dim is specified.

    Parameters
    ----------
    x: NumPy array or PyTorch tensor
        The array to be reshaped
    array_like: NumPy array or PyTorch Tensor (optional)
        x dimension will be expanded to match the dimension of array_like before reshaping
        Used to make target dimension match prediction dimension.
    data_dim: int (optional)
        Dimension of the data.
        If the dimension of x is greater than data_dim, the additional dimensions (in first positions)
        will be considered as sample dimensions and flattened.

    Returns
    -------
    y: array of same kind as x
        Reshaped array so that to have the color channel dimension.


    Examples
    --------
    >>> reshape_for_torch(np.random.rand(5, 4)).shape # First dimension is considered as sample dimension by default
    (5, 1, 4)
    >>> reshape_for_torch(np.random.rand(5, 4), data_dim=2).shape
    (1, 1, 5, 4)
    >>> reshape_for_torch(np.random.rand(10, 5, 4)).shape
    (10, 1, 5, 4)
    >>> reshape_for_torch(np.random.rand(10, 10, 5, 4), data_dim=2).shape
    (100, 1, 5, 4)
    >>> reshape_for_torch(np.random.rand(100)).shape # Samples of 1D data are reshaped to 2D to ease manipulation
    (100, 1, 1)
    >>> reshape_for_torch(np.random.rand(100, 1), np.random.rand(5, 1, 1)).shape # Not sure if it is useful
    (100, 1, 1, 1)
    """

    new_shape = tuple(x.shape) # Converted to tuple to handle PyTorch tensor

    if data_dim is not None:
        # Flatten additional dimensions (or add the sampled dimension)
        new_shape = (-1,) + new_shape[-data_dim:]

    if array_like is not None:
        new_shape += (1,) * (array_like.ndim - x.ndim)

    # 1D inputs are transformed into 2D to ease some manipulations later
    new_shape += (1,) * (2 - len(new_shape))

    # Reshape and insert color channel dimension
    return x.reshape(new_shape)[:, None, ...]


###############################################################################
# Testing

import numpy as np

def flat_coords_array(*args):
    """ Flatten and stack array of coordinates (e.g. meshgrid output) """
    return np.stack(tuple(np.ravel(coords) for coords in args), axis=1)


def flat_meshgrid(*args):
    """ Flatten and stack meshgrid's output to be used as model input """
    return flat_coords_array(*np.meshgrid(*args))

