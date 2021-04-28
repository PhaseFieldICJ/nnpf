#!/usr/bin/env python3
"""
Update checkpoint so that to be compatible with new ConvolutionArray
and FFTConvolutionArray
"""

from collections import OrderedDict
import re
import torch


def fix_convolution(data):
    """ Update field name of (FFT)ConvolutionArray """

    def fix_field_name(name):
        return re.sub(r'(^|\.)convolution\.(weight|bias)',
                      r'\1\2',
                      name)

    state_dict = OrderedDict()
    for key, value in data['state_dict'].items():
        state_dict[fix_field_name(key)] = value
    data['state_dict'] = state_dict

    return data


def fix_allen_cahn_splitting(data):
    """ Move checkpoint from AllenCahnDR to AllenCahnSplitting """
    if data.get('class_name') != 'AllenCahnDR':
        return data

    data['class_name'] = 'AllenCahnSplitting'
    data['class_path'] = 'allen_cahn_splitting.py'
    data['hyper_parameters']['checkpoints'] = [
        data['hyper_parameters']['heat_checkpoint'],
        data['hyper_parameters']['reaction_checkpoint'],
    ]
    del data['hyper_parameters']['heat_checkpoint']
    del data['hyper_parameters']['reaction_checkpoint']

    def fix_field_name(name):
        name = re.sub(r'^heat', r'operators.0', name)
        name = re.sub(r'^reaction', r'operators.1', name)
        return name

    state_dict = OrderedDict()
    for key, value in data['state_dict'].items():
        state_dict[fix_field_name(key)] = value
    data['state_dict'] = state_dict

    return data


def fix_willmore_parallel(data):
    """ Update hyperparameters of WillmoreParallel

    init -> kernel_init
    layer_dims -> reaction_layers
    activation -> reaction_activation
    depth -> scheme_layers
    width -> scheme_repeat
    """
    if data.get('class_name') != 'WillmoreParallel':
        return data

    def rename_hparams(before, after, transformation=lambda v: v):
        if before in data['hyper_parameters']:
            data['hyper_parameters'][after] = transformation(data['hyper_parameters'].pop(before))

    rename_hparams('init', 'kernel_init')
    rename_hparams('layer_dims', 'reaction_layers')
    rename_hparams('activation', 'reaction_activation')
    rename_hparams('depth', 'scheme_layers', lambda v: [v])
    rename_hparams('width', 'scheme_repeat')

    def fix_field_name(name):
        name = re.sub(r'\.parallel\.', r'.parallel0.', name)
        return name

    state_dict = OrderedDict()
    for key, value in data['state_dict'].items():
        state_dict[fix_field_name(key)] = value
    data['state_dict'] = state_dict

    return data


def update_checkpoint_content(data):
    """ Update given checkpoint content """
    fix_convolution(data)
    fix_allen_cahn_splitting(data)
    fix_willmore_parallel(data)
    return data


def update_checkpoint(file_name):
    """ Update checkpoint with given path """
    data = torch.load(file_name, map_location=torch.device("cpu"))
    update_checkpoint_content(data)
    torch.save(data, file_name)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description="Update checkpoint(s) so that to be compatible with new (FFT)ConvolutionArray and some other new stuffs")
    parser.add_argument('path', type=str, help="Checkpoint or folder containing checkpoints")
    args = parser.parse_args()

    import os
    if os.path.isfile(args.path):
        update_checkpoint(args.path)
    else:
        import glob
        for path in glob.glob(os.path.join(args.path, "**", "*.ckpt"), recursive=True):
            print(path)
            update_checkpoint(path)

