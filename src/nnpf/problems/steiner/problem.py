"""
Base module and utils for the Steiner learning problem
"""

import torch
from torch.utils.data import DataLoader

from nnpf.problems import MeanCurvatureProblem
from nnpf.functional.phase_field import dprofil, idprofil


__all__ = [
    "SteinerProblem",
]


class SteinerProblem(MeanCurvatureProblem):
    """
    Base class for the Steiner problem

    Features the train/validation data, and the metric.

    See documentation of mean_curvature_problem.MeanCurvatureProblem
    """

    @staticmethod
    def stability_dt(epsilon):
        """ Maximum allowed time step that guarantee stability of the scheme """
        return epsilon**2

    @staticmethod
    def profil(dist, epsilon):
        """ Solution profil from distance field """
        return dprofil(dist, epsilon)

    @staticmethod
    def iprofil(dist, epsilon):
        """ Distance field from solution profil """
        return idprofil(dist, epsilon)

    @staticmethod
    def sphere_radius(r0, t):
        """ Time evolution of the radius of the sphere """
        r0 = torch.as_tensor(r0)
        t = torch.as_tensor(t)
        return (r0**2 - 2 * t).max(t.new_zeros(())).sqrt()

    def check_sphere_mass(self, radius=0.45, num_steps=None, center=None, progress_bar=False):
        """
        Check an Steiner model by measuring sphere volume decreasing

        Note: Remember to freeze the model if gradient calculation is not needed!!!

        See documentation of MeanCurvatureProblem.check_sphere_volume
        """
        return super().check_sphere_mass(radius, num_steps, center, progress_bar)



