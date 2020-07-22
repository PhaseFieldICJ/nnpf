#!/usr/bin/env python3
"""
Draft script that draw volume evolution and comparison to exact solution

FIXME: should be in allen_cahn_problem directly!
"""

from allen_cahn_problem import AllenCahnProblem
import torch
import math
import matplotlib.pyplot as plt
import argparse

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to the model's checkpoint")
args = parser.parse_args()

# Loading model
model = AllenCahnProblem.load_from_checkpoint(args.checkpoint)
model.freeze()

# Exact volume
domain_diameter = min(b[1] - b[0] for b in model.domain.bounds)
radius = 0.45 * domain_diameter
a, b = model.check_sphere_volume(radius=radius)
t = torch.arange(0, a.shape[0]) * model.hparams.dt # FIXME
r = (radius**2 - 2*t).sqrt()
c = math.pi * r.square()

# Graphs
plt.plot(t, a, label='model');
plt.plot(t, b, label='solution');
plt.plot(t, c, ':', label='exact');
plt.xlabel('t');
plt.ylabel('vol');
plt.legend();
plt.tight_layout()
plt.show()

