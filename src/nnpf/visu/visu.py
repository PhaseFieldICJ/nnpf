""" Visualization tools """

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.fft
from torch.distributions.utils import broadcast_all
import numpy as np


__all__ = ["ImShow", "KernelShow", "KernelFreqShow", "KernelCumSumShow",
           "DiffusionIsotropyShow", "ContourShow", "distance_to_img",
           "DistanceShow", "PhaseFieldShow", "AnimWriter"]


def get_axe_fig(ax=None, fig=None):
    """ Default axe and figure """
    ax = ax or plt.gca()
    fig = fig or plt.gcf()
    return ax, fig


class ImShow:
    """ Show image with right orientation and origin at lower left corner

    Parameters
    ----------
    img: torch.tensor
        The image values
    X: tuple or None
        If shape_or_dist is a shape, X are the point coordinates
    extent: tuple/list or None
        Domain extent. If None, calculated from X (if given)
    """
    def __init__(self, img, X=None, extent=None, ax=None, fig=None, *args, **kwargs):
        # Extent
        if X is not None and extent is None:
            extent = [X[0].min(), X[0].max(), X[1].min(), X[1].max()]

        self.ax, self.fig = get_axe_fig(ax, fig)
        self.graph = self.ax.imshow(self._get_img(img), *args, origin="lower", extent=extent, **kwargs)

    def update(self, img):
        self.graph.set_array(self._get_img(img))
        return self.graph,

    @property
    def mappable(self):
        """ Mappable object to give to colorbar """
        return self.graph

    def _get_img(self, img):
        return img.squeeze().transpose(0, 1)


def get_weight(weight_or_module):
    """ Return weight whatever input is the weight or a module """
    try:
        return weight_or_module.weight
    except AttributeError:
        return weight_or_module


def weight_center(weight_or_module):
    """ Gravity center of the given weight """
    weight = get_weight(weight_or_module).squeeze()
    weight_coords = torch.meshgrid(*(torch.arange(n) for n in weight.shape))
    return [(coords * weight).sum() / weight.sum() for coords in weight_coords]


class KernelShow(ImShow):
    """ Show convolution kernel

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf.functional import heat_kernel_spatial
    >>> import matplotlib.pyplot as plt
    >>> d = Domain([[-1, 1], [-1, 1]], [1024, 1024])
    >>> kernel = heat_kernel_spatial(d, dt=1e-2)
    >>> fig = plt.figure(figsize=[8, 8])
    >>> im = KernelShow(kernel, disp_center=True)
    >>> cb = plt.colorbar(im.mappable)
    >>> plt.savefig("doctest_KernelShow.png")
    >>> plt.pause(0.5)
    """
    def __init__(self, weight_or_module, ax=None, fig=None, disp_center=False):
        self.disp_center = disp_center

        weight = self._get_weight(weight_or_module)
        super().__init__(weight, ax, fig)

        if self.disp_center:
            self.ax.plot(*weight_center(torch.ones_like(weight)), '+r', markersize=10, markeredgewidth=2, alpha=0.25)
            self.cross = self.ax.plot(*weight_center(weight), '+r', markersize=10, markeredgewidth=2)

    def update(self, weight_or_module):
        weight = self._get_weight(weight_or_module)
        graphs = super().update(weight)
        if self.disp_center:
            self.cross[0].set_data(*weight_center(weight))
            return graphs + (self.cross,)
        else:
            return graphs

    def _get_weight(self, weight_or_module):
        weight = get_weight(weight_or_module).squeeze()
        assert weight.ndim == 2, "Only for 2D input"
        return weight


class KernelFreqShow(ImShow):
    """ Show convolution kernel in frequency space

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf.functional import heat_kernel_spatial
    >>> import matplotlib.pyplot as plt
    >>> d = Domain([[-1, 1], [-1, 1]], [1024, 1024])
    >>> kernel = heat_kernel_spatial(d, dt=1e-5)
    >>> fig = plt.figure(figsize=[8, 8])
    >>> im = KernelFreqShow(kernel)
    >>> cb = plt.colorbar(im.mappable)
    >>> plt.savefig("doctest_KernelFreqShow.png")
    >>> plt.pause(0.5)
    """
    def __init__(self, weight_or_module, ax=None, fig=None):
        super().__init__(self._get_array(weight_or_module), ax, fig)

    def update(self, weight_or_module):
        return super().update(self._get_array(weight_or_module))

    def _get_array(self, weight_or_module):
        weight = get_weight(weight_or_module).squeeze()
        assert weight.ndim == 2, "Only for 2D input"
        return torch.fft.fftn(weight[None, None, ...], s=weight.shape).abs()


class KernelCumSumShow(ImShow):
    """ Show kernel cumulative sum

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf.functional import heat_kernel_spatial
    >>> import matplotlib.pyplot as plt
    >>> d = Domain([[-1, 1], [-1, 1]], [1024, 1024])
    >>> kernel = heat_kernel_spatial(d, dt=1e-2)
    >>> fig = plt.figure(figsize=[8, 8])
    >>> im = KernelCumSumShow(kernel)
    >>> cb = plt.colorbar(im.mappable)
    >>> plt.savefig("doctest_KernelCumSumShow.png")
    >>> plt.pause(0.5)
    """
    def __init__(self, weight_or_module, ax=None, fig=None):
        super().__init__(self._get_array(weight_or_module), ax, fig)

    def update(self, weight_or_module):
        return super().update(self._get_array(weight_or_module))

    def _get_array(self, weight_or_module):
        weight = get_weight(weight_or_module).squeeze()
        for d in range(weight.dim()):
            weight = torch.cumsum(weight, dim=d)
        return weight


class DiffusionIsotropyShow:
    """ Illustrate diffusion operator isotropy on a sphere for Lp-norm """
    def __init__(self, operator, ax=None, fig=None, p=2, N=10):
        from nnpf.shapes import sphere
        from nnpf.functional.phase_field import profil

        domain = operator.domain
        center = [0.5 * (a + b) for a, b in domain.bounds]
        diameter = max(b - a for a, b in domain.bounds)
        epsilon = max(dx for dx in domain.dX) * 4
        self.circle = profil(sphere(diameter/3, center, p=p)(*domain.X), epsilon)
        self.N = N

        X = broadcast_all(*domain.X)
        self.ax, self.fig = get_axe_fig(ax, fig)
        self.ax.contour(*X, self.circle, [0.5,], alpha=0.5)
        self.contours = self.ax.contour(*X, self._get_after(operator), [0.5,], linestyles='dashed')
        self.ax.set_aspect('equal')

    def update(self, operator):
        X = broadcast_all(*operator.domain.X)
        for path_coll in self.contours.collections:
            self.ax.collections.remove(path_coll)
        self.contours = self.ax.contour(*X, self._get_after(operator), [0.5,], linestyles='dashed')
        return self.contours

    def _get_after(self, operator):
        after = self.circle
        for i in range(self.N):
            after = operator(after[None, None, ...]).squeeze() / get_weight(operator).sum()
        return after


class ContourShow:
    """ Show contours of a given field

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> import matplotlib.pyplot as plt
    >>> import torch
    >>> d = Domain([[-1, 1], [-1, 1]], [1024, 1024])
    >>> data = torch.cos(5 * d.X[0]) * torch.sin(4 * d.X[1])
    >>> fig = plt.figure(figsize=[8, 8])
    >>> im = ImShow(data, X=d.X)
    >>> cb = plt.colorbar(im.mappable)
    >>> ct = ContourShow(data, torch.linspace(-1, 1, 11), X=d.X, colors='black')
    >>> plt.savefig("doctest_ContourShow.png")
    >>> plt.pause(0.5)
    """
    def __init__(self, data, levels, X=None, ax=None, fig=None, **kwargs):

        if X is None:
            def create_contour(data):
                return self.ax.contour(data, levels, **kwargs)
        else:
            X = broadcast_all(*X)
            def create_contour(data):
                return self.ax.contour(X[0], X[1], data, levels, **kwargs)

        self.ax, self.fig = get_axe_fig(ax, fig)
        self.create_contour = create_contour
        self.contours = self.create_contour(data)

    def update(self, data):
        for path_coll in self.contours.collections:
            self.ax.collections.remove(path_coll)
        self.contours = self.create_contour(data)



def distance_to_img(shape_or_dist, X=None, scale=1., in_color=[0.6, 0.8, 1.0], out_color=[0.9, 0.6, 0.3]):
    """ Transform a 2D shape or distance function to an image

    Parameters
    ----------
    shape_or_dist: shape or torch.tensor
        Shape definition (X needed) or directly the signed distance field
    X: tuple or None
        If shape_or_dist is a shape, X are the point coordinates
    scale: real
        Scale of the visualization
    in_color, out_color: list of 3 floats in [0, 1]
        Inside and outside color
    """
    def smoothstep(a, b, x):
        x = torch.clamp((x - a) / (b - a), 0., 1.)
        return x.square() * (3 - 2. * x)

    def mix(a, b, r):
        return a + (b - a) * r

    # Color from Inigo Quilez
    # See e.g. https://www.shadertoy.com/view/3t33WH
    def color(dist):
        adist = dist[..., None].abs()
        col = torch.where(dist[..., None] < 0., dist.new(in_color), dist.new(out_color))
        col *= 1.0 - (-9.0 / scale * adist).exp()
        col *= 1.0 + 0.2 * torch.cos(128.0 / scale * adist)
        return mix(col, dist.new_ones(3), 1.0 - smoothstep(0., scale * 0.015, adist))

    # Calculating distance
    if not torch.is_tensor(shape_or_dist):
        shape_or_dist = shape_or_dist(*X)
    shape_or_dist = shape_or_dist.squeeze()

    # Image
    return color(shape_or_dist).clamp(0., 1.)


class DistanceShow(ImShow):
    """ Display a 2D shape or distance function

    Parameters
    ----------
    shape_or_dist: shape or torch.tensor
        Shape definition (X needed) or directly the signed distance field
    X: tuple or None
        If shape_or_dist is a shape, X are the point coordinates
    scale: real
        Scale of the visualization
    in_color, out_color: list of 3 floats in [0, 1]
        Inside and outside color

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf.shapes import periodic, union, sphere
    >>> import matplotlib.pyplot as plt
    >>> d = Domain([[-1, 1], [-1, 1]], [1024, 1024])
    >>> s = periodic(union(sphere(0.5, [0, 0]), sphere(0.3, [0.4, 0.3])), d.bounds)
    >>> fig = plt.figure(figsize=[8, 8])
    >>> im = DistanceShow(s, d.X)
    >>> plt.savefig("doctest_DistanceShow.png")
    >>> plt.pause(0.5)
    """
    def __init__(self, shape_or_dist, X=None, scale=1., in_color=[0.6, 0.8, 1.0], out_color=[0.9, 0.6, 0.3], **kwargs):
        # Distance to image
        get_array = lambda s: distance_to_img(s, X, scale, in_color, out_color)

        super().__init__(get_array(shape_or_dist), X=X, **kwargs)
        self._get_array = get_array

    def update(self, shape_or_dist):
        super().update(self._get_array(shape_or_dist))


def get_frame(fig=None, **savefig_kwargs):
    """ Returns given figure as image """
    if fig is None:
        fig = plt.gcf()

    # From https://stackoverflow.com/a/61443397
    from io import BytesIO
    with BytesIO() as buff:
        fig.savefig(buff, format="raw", **savefig_kwargs)
        buff.seek(0)
        return np.frombuffer(buff.getvalue(), dtype=np.uint8) \
                .reshape(fig.canvas.get_width_height()[::-1] + (-1,))


class PhaseFieldShow(ImShow):
    """ Show one or more phase field functions

    Parameters
    ----------
    fields: Tensors
        Phase fields
    X: Tensors
        Grid point coordinates
    cmap: str or Colormap
        Colormap to use
    fmin, fmax: float
        Minimum (far) and maximum (near) value of one field.
        Phase field will be rescaled to [0, 1].

    Example
    -------
    >>> from nnpf.domain import Domain
    >>> from nnpf.shapes import periodic, sphere, subtraction
    >>> from nnpf.functional import profil
    >>> import matplotlib.pyplot as plt
    >>> d = Domain([[-1, 1], [-1, 1]], [1024, 1024])
    >>> epsilon = 0.01
    >>> u1 = profil(periodic(sphere(0.5, [0, 0]), d.bounds)(*d.X), epsilon)
    >>> u2 = profil(periodic(subtraction(sphere(0.3, [0.4, 0.3]), sphere(0.5, [0, 0])), d.bounds)(*d.X), epsilon)
    >>> fig = plt.figure(figsize=[8, 8])
    >>> img = PhaseFieldShow(u1, u2, 1 - u1 - u2, X=d.X)
    >>> plt.savefig("doctest_PhaseFieldShow.png")
    >>> plt.pause(0.5)
    """
    def __init__(self, *fields, X=None, cmap=plt.get_cmap("gist_rainbow"), fmin=0., fmax=1., **kwargs):
        self.fmin = fmin
        self.fmax = fmax
        super().__init__(self._get_array(*fields),
                         X=X,
                         cmap=cmap,
                         vmin=0, vmax=max(1., len(fields) - 1),
                         **kwargs)

    def update(self, *fields):
        return super().update(self._get_array(*fields))

    def _rescale(self, fields):
        return [(u - self.fmin) / (self.fmax - self.fmin) for u in fields]

    def _get_array(self, *fields):
        fields = self._rescale(fields)
        if len(fields) == 1:
            fields = 1. - fields[0], fields[0]
        ampl_fn = lambda u: 1. - (1. + torch.cos(np.pi * u)) / 2
        return sum(i * ampl_fn(u) for i, u in enumerate(fields))


class AnimWriter:
    """ Context manager for an animation writer """

    def __init__(self, file_uri, file_format=None, file_mode='?', do_nothing=False, fps=25, **kwargs):
        """ Constructor

        Parameters
        ----------
        file_uri: file name or file objet
            See documentation of imageio.get_writer
        file_format: str
            Format of the file. None to deduce it from the file name.
        file_mode: str
            See documentation of imageio.get_writer
        do_nothing: bool
            If True, this writer do not generate the requested file.
        fps: int
            Frame per second
        **kwargs: dict
            Additional parameters passed to imageio.get_writer
        """

        self.do_nothing = do_nothing
        if not do_nothing:
            import imageio
            self.writer = imageio.get_writer(file_uri,
                                             format=file_format,
                                             mode=file_mode,
                                             fps=fps,
                                             **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def add_frame(self, fig_or_img=None):
        if self.do_nothing:
            return

        if isinstance(fig_or_img, matplotlib.figure.Figure):
            self.writer.append_data(get_frame(fig))
        elif fig_or_img is None:
            self.writer.append_data(get_frame(plt.gcf()))
        else:
            self.writer.append_data(fig_or_img)

    def close(self):
        if not self.do_nothing:
            self.writer.close()

