""" Visualization tools """

import matplotlib.pyplot as plt
import torch
from torch.distributions.utils import broadcast_all
import numpy as np

def get_axe_fig(ax=None, fig=None):
    """ Default axe and figure """
    ax = ax or plt.gca()
    fig = fig or plt.gcf()
    return ax, fig


class ImShow:
    """ Show image with right orientation and origin at lower left corner """
    def __init__(self, img, ax=None, fig=None, *args, **kwargs):
        ax, fig = get_axe_fig(ax, fig)
        self.graph = ax.imshow(self._get_img(img), origin="lower", *args, **kwargs)

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
    """ Show convolution kernel """
    def __init__(self, weight_or_module, ax=None, fig=None, disp_center=False):
        self.disp_center = disp_center

        weight = self._get_weight(weight_or_module)
        ax, fig = get_axe_fig(ax, fig)
        super().__init__(weight, ax, fig)

        if self.disp_center:
            ax.plot(*weight_center(torch.ones_like(weight)), '+r', markersize=10, markeredgewidth=2, alpha=0.25)
            self.cross = ax.plot(*weight_center(weight), '+r', markersize=10, markeredgewidth=2)

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
    """ Show convolution kernel in frequency space """
    def __init__(self, weight_or_module, ax=None, fig=None):
        super().__init__(self._get_array(weight_or_module), ax, fig)

    def update(self, weight_or_module):
        return super().update(self._get_array(weight_or_module))

    def _get_array(self, weight_or_module):
        weight = get_weight(weight_or_module).squeeze()
        assert weight.ndim == 2, "Only for 2D input"
        return torch.view_as_complex(torch.fft(weight[None, None, ..., None] * torch.tensor([1., 0]), weight.ndim)).abs()


class KernelCumSumShow(ImShow):
    """ Show kernel cumulative sum """
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
        from shapes import sphere
        from phase_field import profil

        domain = operator.domain
        center = [0.5 * (a + b) for a, b in domain.bounds]
        diameter = max(b - a for a, b in domain.bounds)
        epsilon = max(dx for dx in domain.dX) * 4
        self.circle = profil(sphere(diameter/3, center, p=p)(*domain.X), epsilon)
        self.N = N

        X = broadcast_all(*domain.X)
        self.ax, fig = get_axe_fig(ax, fig)
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
    extent: tuple/list or None
        Domain extent. If None, calculated from X (if given)
    in_color, out_color: list of 3 floats in [0, 1]
        Inside and outside color

    Example
    -------
    >>> from domain import Domain
    >>> from shapes import periodic, union, sphere
    >>> d = Domain([[-1, 1], [-1, 1]], [256, 256])
    >>> s = periodic(union(sphere(0.5, [0, 0]), sphere(0.3, [0.4, 0.3])), d.bounds)
    >>> im = DistanceShow(s, d.X)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    def __init__(self, shape_or_dist, X=None, scale=1., extent=None, in_color=[0.6, 0.8, 1.0], out_color=[0.9, 0.6, 0.3], **kwargs):
        # Extent
        if X is not None and extent is None:
            extent = [X[0].min(), X[0].max(), X[1].min(), X[1].max()]

        # Distance to image
        get_array = lambda s: distance_to_img(s, X, scale, in_color, out_color)

        super().__init__(get_array(shape_or_dist), extent=extent, **kwargs)
        self._get_array = get_array

    def update(self, shape_or_dist):
        super().update(self._get_array(shape_or_dist))


def get_frame(fig=None):
    """ Returns given figure as image """
    canvas = (fig or plt.gcf()).canvas
    return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8) \
             .reshape(canvas.get_width_height()[::-1] + (3,))


class PhaseFieldShow(ImShow):
    """ Show one or more phase field functions """
    def __init__(self, *fields, X=None, extent=None, **kwargs):
        # Extent
        if X is not None and extent is None:
            extent = [X[0].min(), X[0].max(), X[1].min(), X[1].max()]

        super().__init__(self._get_array(*fields),
                         extent=extent,
                         cmap=plt.get_cmap("gist_rainbow"),
                         vmin=0, vmax=max(1., len(fields) - 1),
                         **kwargs)

    def update(self, *fields):
        return super().update(self._get_array(*fields))

    def _get_array(self, *fields):
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

    def add_frame(self, fig=None):
        if not self.do_nothing:
            self.writer.append_data(get_frame(fig))

    def close(self):
        if not self.do_nothing:
            self.writer.close()

