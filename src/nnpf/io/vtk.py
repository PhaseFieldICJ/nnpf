__all__ = ["save_vtk_image"]


def save_vtk_image(name, data, domain=None, field_name="u"):
    """
    Save an image (ie a nD array) in compressed VTI format.

    Parameters
    ----------
    name: str
        File name (extension not needed)
    data: Tensor, Numpy array of list of
        2D or 3D tensor(s)
    domain: nnpf.domain.Domain
        Discrete domain. If not given, spacing of 1 is used.
    field_name: str or list of str
        Name of the field(s) associated to the data
    """
    if domain is None:
        spacing = [1] * 3
        origin = [0] * 3
    else:
        spacing = list(domain.dX.cpu())
        origin = [a for a, b in domain.bounds]
        spacing += [1] * (3 - len(spacing))
        origin += [0] * (3 - len(origin))

    from tvtk.api import tvtk, write_data
    img = tvtk.ImageData(
        spacing=spacing,
        origin=origin,
    )

    import numpy as np
    import torch

    # Converting to list if only one data is given
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = [data]
        field_name = [field_name]

    shape = None
    for d, n in zip(data, field_name):
        if isinstance(d, torch.Tensor):
            i = img.point_data.add_array(d.detach().cpu().numpy().flatten())
        elif isinstance(d, np.ndarray):
            i = img.point_data.add_array(d.flatten())
        else:
            raise ValueError(f"Invalid data type {type(d).__name__}")

        img.point_data.get_array(i).name = n

        if shape is None:
            shape = list(d.shape)
        elif shape != list(d.shape):
            raise ValueError(f"Data of shape {list(d.shape)} found but {shape} expected")


    img.dimensions = shape + [1] * (3 - len(shape))

    write_data(img, name, compressor=tvtk.ZLibDataCompressor())

