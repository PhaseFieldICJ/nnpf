#!/usr/bin/env python3
"""
Draft script that save animation of Allen-Cahn model evolution
"""

from allen_cahn_problem import AllenCahnProblem
import shapes
import visu
import phase_field as pf
import matplotlib.pyplot as plt
import argparse
import imageio
import numpy as np
import torch

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to the model's checkpoint")
parser.add_argument("--no_save", action="store_true", help="Don't save the animation")
parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance used as a stop criteron")
parser.add_argument("--no_dist", action="store_true", help="Display phase field instead of distance")

args = parser.parse_args()

# Loading model
model = AllenCahnProblem.load_from_checkpoint(args.checkpoint, map_location=torch.device("cpu"))
model.freeze()

# Defining initial shape
bounds = model.domain.bounds
domain_diameter = min(b[1] - b[0] for b in bounds)

def radius(r):
    return r * domain_diameter

def pos(X):
    return [b[0] + x * (b[1] - b[0]) for x, b in zip(X, bounds)]

s = shapes.periodic(
        shapes.translation(
        shapes.union(
            shapes.sphere(radius(0.3), pos([0.5, 0.5])),
            shapes.sphere(radius(0.2), pos([0.7, 0.8])),
            shapes.translation(shapes.box([radius(0.1), radius(0.08)]), pos([0.2, 0.2])),

        ), pos([0., 0.2])),
        model.domain.bounds)

u = pf.profil(s(*model.domain.X), model.hparams.epsilon)


# Graph
scale = 0.25 * max(b[1] - b[0] for b, n in zip(model.domain.bounds, model.domain.N))
extent = [*model.domain.bounds[0], *model.domain.bounds[1]]
interpolation = "kaiser"

plt.figure(figsize=(6, 6))

if args.no_dist:
    def data_from(u):
        return u
    graph = visu.PhaseFieldShow(data_from(u), extent=extent, interpolation=interpolation)
else:
    def data_from(u):
        return pf.iprofil(u, model.hparams.epsilon)
    graph = visu.DistanceShow(data_from(u), scale=scale, extent=extent, interpolation=interpolation)


title = plt.title(f"t = 0 ; it = 0")
plt.tight_layout()
plt.pause(1)

with visu.AnimWriter('anim.avi', fps=25, do_nothing=args.no_save) as anim:

    for i in range(25):
        anim.add_frame()

    last_error = [0] * 25

    for i in range(10000):
        last_u = u.clone()
        u = model(u[None, None, ...])[0, 0, ...]

        graph.update(data_from(u))
        title.set_text(f"t = {i*model.hparams.dt:.5} ; it = {i}")
        plt.pause(0.01)

        anim.add_frame()

        last_error[1:] = last_error[:-1]
        last_error[0] = (u - last_u).norm()
        if max(last_error) <= args.tol:
            break

