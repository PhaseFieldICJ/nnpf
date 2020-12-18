"""
Base module and utils for the Allen-Cahn equation learning problem
"""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import math

import mean_curvature_problem as mcp
from domain import Domain
from phase_field import profil, iprofil
import nn_toolbox
import shapes


def check_sphere_mass(*args, **kwargs):
    """
    Check an Allen-Cahn model by measuring the decreasing of the mass of the profil associated to a sphere.

    See documentation of mean_curvature_problem.check_sphere_volume
    """
    return mcp.check_sphere_volume(AllenCahnProblem.sphere_radius, AllenCahnProblem.profil, *args, **kwargs)


class ACSphereLazyDataset(mcp.MCSphereLazyDataset):
    """
    Dataset of spheres for Allen-Cahn problem, with samples generated at loading.

    See documentation of mean_curvature_problem.MCSphereLazyDataset
    """
    def __init__(self, X, radius, center, epsilon, dt, lp=2, steps=1, reverse=False):
        super().__init__(
            AllenCahnProblem.sphere_dist,
            AllenCahnProblem.profil,
            X, radius, center, epsilon, dt, lp, steps, reverse)


class ACSphereDataset(TensorDataset):
    """
    Dataset of spheres for Allen-Cahn problem (non-lazy version).

    See documentation of ACSphereLazyDataset
    """
    def __init__(self, *args, **kwargs):
        ds = ACSphereLazyDataset(*args, **kwargs)
        super().__init__(*ds[:])


class AllenCahnProblem(mcp.MeanCurvatureProblem):
    """
    Base class for the Allen-Cahn equation problem

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
        return (r0**2 - 2 * t).max(t.new_zeros(())).sqrt()

    def check_sphere_mass(self, radius=0.45, num_steps=None, center=None, progress_bar=False):
        """
        Check an Allen-Cahn model by measuring sphere volume decreasing

        Note: Remember to freeze the model if gradient calculation is not needed!!!

        See documentation of MeanCurvatureProblem.check_sphere_volume
        """
        return super().check_sphere_mass(radius, num_steps, center, progress_bar)




