#!/usr/bin/env python3
"""
Draft script that save animation of a model applied to a phase field function
"""

from problem import Problem
import shapes
import visu
import phase_field as pf
import argparse
import imageio
import torch
import tqdm
import math

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to the model's checkpoint")
parser.add_argument("--no_save", action="store_true", help="Don't save the animation")
parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance used as a stop criteron")
parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of calculated frames (-1 for illimited)")
parser.add_argument("--no_dist", action="store_true", help="Display phase field instead of distance")
parser.add_argument("--scale", type=float, default=1., help="Initial shape scale")
parser.add_argument("--shape", type=str, choices=["bunch", "one", "two", "three", "mixed_one", "mixed_two"], default="bunch", help="Initial shape")
parser.add_argument("--offscreen", action="store_true", help="Don't display the animation (but still saving")
parser.add_argument("--gpu", action="store_true", help="Evaluation model on your GPU")
parser.add_argument("--display_step", type=int, default=1, help="Render frame every given number")

args = parser.parse_args()

# Matplotlib rendering backend
if args.offscreen:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

# Loading model
model = Problem.load_from_checkpoint(args.checkpoint, map_location=torch.device("cpu"))
model.freeze()

if args.gpu:
    model.cuda()

# Defining initial shape
bounds = model.domain.bounds
domain_extent = [b[1] - b[0] for b in bounds]
domain_diameter = min(domain_extent)

def radius(r):
    return r * domain_diameter

def pos(X):
    return [b[0] + x * (b[1] - b[0]) for x, b in zip(X, bounds)]

periodic_bounds = bounds.copy()

# Shape
if args.shape == "bunch":
    s = shapes.translation(
            shapes.union(
                shapes.sphere(radius(0.3), pos([0.5, 0.5])),
                shapes.sphere(radius(0.2), pos([0.7, 0.8])),
                shapes.translation(shapes.box([radius(0.1), radius(0.08)]), pos([0.2, 0.2])),

            ), pos([0., 0.2]))

elif args.shape == "one":
    s = shapes.sphere(radius(0.3), pos([0.5, 0.5]))

elif args.shape == "two":
    r = 0.2
    s = shapes.union(
        shapes.sphere(radius(r), pos([0.5 - r, 0.5])),
        shapes.sphere(radius(r), pos([0.5 + r, 0.5])),
    )

elif args.shape == "three":
    r = 0.2
    R = 2 * r * math.sqrt(3) / 3.
    theta = [i * 2. * math.pi / 3. for i in range(3)]
    s = shapes.union(*[shapes.sphere(radius(r), pos([0.5 + R * math.sin(t), 0.5 + R * math.cos(t)])) for t in theta])

elif args.shape == "mixed_one":
    r = 0.3
    s = shapes.symmetric_difference(
        shapes.translation(shapes.beam(1, domain_extent[1] / 2), pos([0.5, 0.25])),
        shapes.sphere(radius(r), pos([0.5, 0.5])))

    periodic_bounds[0] = None

elif args.shape == "mixed_two":
    r = 0.2
    s = shapes.symmetric_difference(
        shapes.translation(shapes.beam(1, domain_extent[1] / 2), pos([0.5, 0.25])),
        shapes.union(
            shapes.sphere(radius(r), pos([0.5 - r, 0.5])),
            shapes.sphere(radius(r), pos([0.5 + r, 0.5]))))

    periodic_bounds[0] = None

# Scaling at domain center
s = shapes.translation(shapes.scaling(shapes.translation(s, pos([-0.5, -0.5])), args.scale), pos([0.5, 0.5]))

# Periodizing
s = shapes.periodic(s, periodic_bounds)

# Phase field
u = pf.profil(s(*model.domain.X), model.hparams.epsilon)

# Graph
scale = 0.25 * max(b[1] - b[0] for b, n in zip(model.domain.bounds, model.domain.N))
extent = [*model.domain.bounds[0], *model.domain.bounds[1]]
interpolation = "kaiser"

plt.figure(figsize=(6, 6))

if args.no_dist:
    def data_from(u):
        return u.cpu()
    graph = visu.PhaseFieldShow(data_from(u), extent=extent, interpolation=interpolation)
else:
    def data_from(u):
        return pf.iprofil(u, model.hparams.epsilon).cpu()
    graph = visu.DistanceShow(data_from(u), scale=scale, extent=extent, interpolation=interpolation)


title = plt.title(f"t = 0 ; it = 0")
plt.tight_layout()
plt.pause(1)

with visu.AnimWriter('anim.avi', fps=25, do_nothing=args.no_save) as anim:

    for i in range(25):
        anim.add_frame()

    last_diff = [args.tol + 1] * 25

    with tqdm.tqdm() as pbar:
        while max(last_diff) > args.tol and pbar.n != args.max_frames:
            last_u = u.clone()
            u = model(u[None, None, ...])[0, 0, ...]

            if pbar.n % args.display_step == 0:
                graph.update(data_from(u))
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
                'max diff': max(last_diff)
            })
