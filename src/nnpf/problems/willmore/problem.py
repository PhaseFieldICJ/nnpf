"""
Base module and utils for the Willmore equation learning problem
"""

import torch

from nnpf.problems import MeanCurvatureProblem
from nnpf.functional.phase_field import profil, iprofil


__all__ = [
    "WillmoreProblem",
]


class WillmoreProblem(MeanCurvatureProblem):
    """
    Base class for the Willmore equation problem

    Features the train/validation data, and the metric.

    See documentation of mean_curvature_problem.MeanCurvatureProblem
    """

    @staticmethod
    def stability_dt(epsilon):
        """ Maximum allowed time step that guarantee stability of the scheme """
        return epsilon**4

    @staticmethod
    def profil(dist, epsilon):
        """ Solution profil from distance field """
        return profil(dist, epsilon)

    @staticmethod
    def iprofil(dist, epsilon):
        """ Distance field from solution profil """
        return iprofil(dist, epsilon)

    @staticmethod
    def sphere_radius(r0, t):
        """ Time evolution of the radius of the sphere """
        r0 = torch.as_tensor(r0)
        t = torch.as_tensor(t)
        return (r0**4 + 2 * t).max(t.new_zeros(())).pow(1/4)


    def check_sphere_mass(self, radius=[0.1, 0.2, 0.3, 0.4], num_steps=100, center=None, progress_bar=False):
        """
        Check an Willmore model by measuring sphere volume decreasing

        Note: Remember to freeze the model if gradient calculation is not needed!!!

        See documentation of MeanCurvatureProblem.check_sphere_volume
        """
        return super().check_sphere_mass(radius, num_steps, center, progress_bar)


