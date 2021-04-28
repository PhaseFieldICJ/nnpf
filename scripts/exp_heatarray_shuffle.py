#!/usr/bin/env python3

import itertools
import argparse

from heat_array_model import HeatArray
from trainer import Trainer

# Experiment configuration
loss_norms_list = [[(1, 1)], [(2, 1)]]
loss_power_list = [1, 2]
lr_exp_list = [3, 4]
batch_suffle_list = [0, 1]
seed = 888

# Command-line arguments
parser = argparse.ArgumentParser(
    description="Experiment on batch suffle effect",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = Trainer.add_argparse_args(parser)
parser = HeatArray.add_model_specific_args(parser)
args = parser.parse_args()
kwargs = vars(args)

kwargs = vars(args)
kwargs['seed'] = kwargs['seed'] or seed

# Main loop
for (loss_norms, loss_power, lr_exp, batch_shuffle) in itertools.product(loss_norms_list, loss_power_list, lr_exp_list, batch_suffle_list):

    kwargs.update({
        'loss_norms': loss_norms,
        'loss_power': loss_power,
        'lr': 10**(-lr_exp),
        'batch_shuffle': bool(batch_shuffle),
        'version': f"norm_l{loss_norms[0][0]}_pow{loss_power}_lr{lr_exp}_shuffle{batch_shuffle}",
    })

    print(f"version: {kwargs['version']}")

    model = HeatArray(**kwargs)
    trainer = Trainer.from_argparse_args(args, "HeatArrayExperimentShuffle")
    trainer.fit(model)


