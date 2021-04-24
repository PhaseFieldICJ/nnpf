import torch


__all__ = [
    "ReactionSolution",
]


class ReactionSolution:
    """
    Exact solution for the reaction operator of the Allen-Cahn equation
    """

    def __init__(self, epsilon=2/2**8, dt=None):
        """ Constructor

        Parameters
        ----------
        epsilon: float
            Interface sharpness in phase field model
        dt: float
            Time step. epsilon**2 if None.
        """
        self.epsilon = epsilon
        self.dt = dt or self.epsilon**2

    def __call__(self, u):
        """ Returns u(t + dt) from u(t) """
        result = torch.empty_like(u)
        dt, epsilon = self.dt, self.epsilon

        def helper(u):
            return torch.as_tensor(-dt / epsilon**2).exp() * u * (1 - u) / (1 - 2 * u)**2

        result[u == 0.5] = 0.5

        a = torch.sqrt(1 + 4 * helper(u[u < 0.5]))
        result[u < 0.5] = 1 - (a + 1) / (2 * a)

        a = torch.sqrt(1 + 4 * helper(u[u > 0.5]))
        result[u > 0.5] = (a + 1) / (2 * a)

        return result


