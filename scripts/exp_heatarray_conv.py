#!/usr/bin/env python3

import itertools
import argparse

from heat_array_model import HeatArray
from trainer import Trainer

# Experiment configuration
init_list = ['zeros']
loss_norms_list = [[(1, 1)], [(2, 1)], [(float('inf'), 1)]]
loss_power_list = [1, 2, 3, 4]
lr_exp_list = [2, 3, 4, 5]
seed = 888

# Command-line arguments
parser = argparse.ArgumentParser(
    description="Experiment on the loss formulation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = Trainer.add_argparse_args(parser)
parser = HeatArray.add_model_specific_args(parser)
args = parser.parse_args()

kwargs = vars(args)
kwargs['seed'] = kwargs['seed'] or seed

# Main loop
for (init, loss_norms, loss_power, lr_exp) in itertools.product(init_list, loss_norms_list, loss_power_list, lr_exp_list):

    kwargs.update({
        'init': init,
        'loss_norms': loss_norms,
        'loss_power': loss_power,
        'lr': 10**(-lr_exp),
        'version': f"{init}_l{loss_norms[0][0]}_p{loss_power}_lr{lr_exp}",
    })

    print(f"version: {kwargs['version']}")

    model = HeatArray(**kwargs)
    trainer = Trainer.from_argparse_args(args, "HeatArrayExperiment")
    trainer.fit(model)

