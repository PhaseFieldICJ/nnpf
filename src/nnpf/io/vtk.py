__all__ = ["save_vtk_image"]


def save_vtk_image(name, data, domain, field_name="u"):
    """
    Save an image (ie a nD array) in compressed VTI format.

    Parameters
    ----------
    name: str
        File name (extension not needed)
    data: Tensor
        2D or 3D tensor
    domain: nnpf.domain.Domain
        Discrete domain
    field_name: str
        Name of the field associated to the data
    """
    from tvtk.api import tvtk, write_data
    img = tvtk.ImageData(
        spacing=domain.dX.numpy(),
        origin=[a for a, b in domain.bounds]
    )
    img.point_data.scalars = data.detach().numpy().flatten()
    img.point_data.scalars.name = field_name
    img.dimensions = list(u.shape)
    write_data(img, name, compressor=tvtk.ZLibDataCompressor())

