import torch
from nnpf.functional import heat_kernel_freq


__all__ = [
    "HeatSolution",
]


class HeatSolution:
    """
    Exact solution to the heat equation
    """

    def __init__(self, domain, dt):
        """ Constructor

        Parameters
        ----------
        domain: domain.Domain
            Spatial domain
        dt: float
            Time step.
        """

        self.domain = domain
        self.dt = dt
        self.kernel = heat_kernel_freq(self.domain, self.dt)

    def __call__(self, u):
        """
        Returns u(t + dt) from u(t)

        Support batch and channels
        """
        return self.domain.ifft(self.kernel * self.domain.fft(u))

