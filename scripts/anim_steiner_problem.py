#!/usr/bin/env python3
"""
Animation for the mean curvature based-equation learning problem
"""

import argparse
import imageio
import torch
import tqdm
import math
from functools import reduce
from distutils.util import strtobool

from nnpf.problems import MeanCurvatureProblem
import nnpf.shapes as shapes
import nnpf.visu as visu

# Command-line arguments
parser = argparse.ArgumentParser(
    description="Spheres evolution compared to the solution of the mean curvature flow",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("checkpoint", type=str, help="Path to the model's checkpoint")
parser.add_argument("--no_save", action="store_true", help="Don't save the animation")
parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance used as a stop criteron")
parser.add_argument("--max_it", type=int, default=-1, help="Maximum number of calculated iterations (-1 for illimited")
parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of rendered frames (-1 for illimited)")
parser.add_argument("--max_duration", type=float, default=-1, help="Maximum duration of the animation (-1 for illimited")
parser.add_argument("--scalars", type=str, choices=["u", "dist", "check"], nargs='*', default=["dist"], help="Scalars to be displayed")
parser.add_argument("--scale", type=float, default=1., help="Initial shape scale")
parser.add_argument("--shape", type=str, choices=["one", "two", "three"], default="one", help="Initial shape")
parser.add_argument("--lp_shape", type=float, help="lp norm used to define the initial shapes (default to model.hparams.lp)")
parser.add_argument("--lp_check", type=float, help="lp norm used to check the distance (default to model.hparams.lp)")
parser.add_argument("--offscreen", action="store_true", help="Don't display the animation (but still saving")
parser.add_argument("--gpu", action="store_true", help="Evaluation model on your GPU")
parser.add_argument("--display_step", type=int, default=1, help="Render frame every given number")
parser.add_argument("--fps", type=int, default=25, help="Frame per second in the saved animation")
parser.add_argument("--figsize", type=int, default=[6, 6], nargs=2, help="Figure size in inches")
parser.add_argument("--revert", type=lambda s:bool(strtobool(s)), nargs='?', const=True, default=False, help="Revert inside and outside of the phase")
parser.add_argument("--output", type=str, default="anim.avi", help="File name of the generated animation")
parser.add_argument("--npoints", type=int, default=3, help="Number of points in Steiner problem")
parser.add_argument("--seed", type=int, default=0, help="Seed when generating fixed points")
parser.add_argument("--random_scale", type=float, default=1., help="Scale of the point's positions")

args = parser.parse_args()

if args.max_duration < 0.:
    args.max_duration = float('inf')

# Matplotlib rendering backend
if args.offscreen:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

# Loading model
model = MeanCurvatureProblem.load_from_checkpoint(args.checkpoint, map_location=torch.device("cpu"))
model.freeze()

if args.gpu:
    model.cuda()

domain = model.domain

# Defining initial shape
bounds = domain.bounds
domain_extent = [b[1] - b[0] for b in bounds]
domain_diameter = min(domain_extent)
args.lp_shape = args.lp_shape or model.hparams.lp
args.lp_check = args.lp_check or model.hparams.lp

def radius(r, scale):
    return scale * r * domain_diameter

def pos(X, scale):
    #return [b[0] + (0.5 + scale * (x - 0.5)) * (b[1] - b[0]) for x, b in zip(X, bounds)]
    return [b[0] + x * (b[1] - b[0]) for x, b in zip(X, bounds)]

# Shape
if args.shape == "one":
    spheres = [(0.3, 0.5, 0.5)]

elif args.shape == "two":
    spheres = [(0.1, 0.2, 0.2), (0.2, 0.7, 0.7)]

elif args.shape == "three":
    spheres = [(0.1, 0.2, 0.2), (0.2, 0.3, 0.7), (0.05, 0.7, 0.3)]

s = shapes.union(*(shapes.sphere(radius(p[0], args.scale), pos(p[1:], args.scale), args.lp_shape) for p in spheres))
#s = shapes.intersection(s, shapes.segment([0.5, 0], [0.5, 1]))
dist_sol = lambda t: reduce(torch.min, [shapes.sphere(model.sphere_radius(radius(p[0], args.scale), t), pos(p[1:], args.scale), args.lp_shape)(*domain.X) for p in spheres])

# Periodizing
s = shapes.periodic(s, bounds)

# Phase field
u = model.profil(s(*domain.X), model.hparams.epsilon)
u = torch.min(u, model.profil(shapes.segment([0.5, 0.1],[0.5, 0.9])(*domain.X), model.hparams.epsilon))
u = torch.min(u, model.profil(shapes.segment([0.1, 0.5],[0.9, 0.5])(*domain.X), model.hparams.epsilon))
if args.revert:
    u = 1. - u

def dot_pos(theta):
    return [
        pos([0.5, 0.5], args.scale)[0] + radius(0.3, args.scale) * math.cos(theta),
        pos([0.5, 0.5], args.scale)[1] + radius(0.3, args.scale) * math.sin(theta)
    ]
import random
random.seed(args.seed)
dots = shapes.union(*(shapes.translation(shapes.dot(args.lp_shape), dot_pos((i + args.random_scale * (random.random() - 0.5)) * 2 * math.pi / args.npoints )) for i in range(args.npoints)))
dots_u = model.profil(dots(*domain.X), model.hparams.epsilon)

# Graph
scale = 0.25 * max(b[1] - b[0] for b, n in zip(domain.bounds, domain.N))
extent = [*domain.bounds[0], *domain.bounds[1]]
interpolation = "kaiser"

fig = plt.figure(figsize=(args.figsize[0] * len(args.scalars), args.figsize[0]))
subplots = []

if "u" in args.scalars:
    plt.subplot(1, len(args.scalars), len(subplots) + 1)
    def data_from(u):
        return u.cpu()
    graph = visu.PhaseFieldShow(data_from(u), extent=extent, interpolation=interpolation)
    contour = visu.ContourShow(dist_sol(0.).cpu(), [0.], X=[x.cpu() for x in domain.X], colors='red')
    subplots.append((data_from, graph, contour))
    plt.title("u")

if "dist" in args.scalars:
    plt.subplot(1, len(args.scalars), len(subplots) + 1)
    def data_from(u):
        return model.iprofil(u, model.hparams.epsilon).cpu()
    graph = visu.DistanceShow(data_from(u), scale=scale, extent=extent, interpolation=interpolation)
    contour = visu.ContourShow(dist_sol(0.).cpu(), [0.], X=[x.cpu() for x in domain.X], colors='red')
    subplots.append((data_from, graph, contour))
    plt.title("distance")

if "check" in args.scalars:
    plt.subplot(1, len(args.scalars), len(subplots) + 1)
    def data_from(u):
        return shapes.check_dist(model.iprofil(u, model.hparams.epsilon).cpu(), domain.dX, p=args.lp_check)
    graph = visu.ImShow(data_from(u), extent=extent, interpolation=interpolation, vmin=0, vmax=2, cmap='seismic')
    contour = visu.ContourShow(dist_sol(0.).cpu(), [0.], X=[x.cpu() for x in domain.X], colors='red')
    subplots.append((data_from, graph, contour))
    #plt.colorbar(graph.mappable)
    plt.title("distance check")

title = fig.suptitle(f"t = 0 ; it = 0")
plt.tight_layout()
plt.pause(1)

with visu.AnimWriter(args.output, fps=args.fps, do_nothing=args.no_save) as anim:

    for i in range(25):
        anim.add_frame()

    last_diff = [args.tol + 1] * 25

    with tqdm.tqdm() as pbar:
        while max(last_diff) > args.tol and pbar.n != args.max_it and pbar.n != args.max_frames * args.display_step and pbar.n / args.display_step / args.fps < args.max_duration:
            last_u = u.clone()
            u = model(u[None, None, ...])[0, 0, ...]
            u = torch.min(u, dots_u)

            if pbar.n % args.display_step == 0:
                for data_from, graph, contour in subplots:
                    graph.update(data_from(u))
                    contour.update(dist_sol(pbar.n * model.hparams.dt).cpu())
                title.set_text(f"t = {pbar.n*model.hparams.dt:.5} ; it = {pbar.n}")
                plt.pause(0.01)

                anim.add_frame()

            vol = model.domain.dX.prod() * u.sum()
            last_diff[1:] = last_diff[:-1]
            last_diff[0] = (u - last_u).norm().item()

            pbar.update(1)
            pbar.set_postfix({
                'volume': vol.item(),
                'diff': last_diff[0],
                'max diff': max(last_diff),
                't': pbar.n * model.hparams.dt,
                'frames': pbar.n // args.display_step,
                'duration': pbar.n / args.display_step / args.fps,
            })

