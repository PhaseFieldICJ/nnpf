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
parser.add_argument("--interactive", action="store_true", help="Switch to interactive mode at the end of the script")
args = parser.parse_args()

# Loading model
model = AllenCahnProblem.load_from_checkpoint(args.checkpoint, map_location=torch.device("cpu"))
model.freeze()

# Exact volume
domain_diameter = min(b[1] - b[0] for b in model.domain.bounds)
radius = 0.45 * domain_diameter
dim = len(model.hparams.bounds)
lp = model.hparams.lp if "lp" in model.hparams else 2

model_vol, solution_vol = model.check_sphere_volume(radius=radius, progress_bar=True)
t = torch.arange(0, model_vol.shape[0]) * model.hparams.dt # FIXME
r = (radius**2 - 2*t).sqrt()
exact_vol = 2**dim * math.gamma(1 + 1 / lp)**dim / math.gamma(1 + dim / lp) * r**dim

# Graphs
def disp_volume():
    plt.plot(t, model_vol, label='model')
    plt.plot(t, solution_vol, '-.', label='solution')
    plt.plot(t, exact_vol, ':', label='exact')
    plt.xlabel('t')
    plt.ylabel('vol')
    plt.legend()
    plt.grid()
    plt.title("Volume of a sphere")
    plt.tight_layout()
    plt.show()

def disp_error():
    plt.plot(t, model_vol - exact_vol, label='model')
    plt.plot(t, solution_vol - exact_vol, '-.', label='solution')
    plt.plot(t, exact_vol - exact_vol, ':', label='exact')
    plt.xlabel('t')
    plt.ylabel('vol')
    plt.legend()
    plt.grid()
    plt.title('Error to exact volume')
    plt.title("Error of the volume of a sphere")
    plt.tight_layout()
    plt.show()

disp_volume()
disp_error()

# Interactive mode
if args.interactive:
    #import code
    #code.interact(local=locals())
    from IPython import embed
    embed(colors="neutral")
