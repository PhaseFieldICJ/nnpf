#!/usr/bin/env python3
"""
Update checkpoint so that to be compatible with new ConvolutionArray
and FFTConvolutionArray
"""

from collections import OrderedDict
import re
import torch

def fix_field_name(name):
    """ Given a field name, fix it if needed and return it """
    return re.sub(r'(^|\.)convolution\.(weight|bias)',
                  r'\1\2',
                  name)

def update_checkpoint_content(data):
    """ Update given checkpoint content """
    state_dict = OrderedDict()
    for key, value in data['state_dict'].items():
        state_dict[fix_field_name(key)] = value
    data['state_dict'] = state_dict
    return data

def update_checkpoint(file_name):
    """ Update checkpoint with given path """
    data = torch.load(file_name, map_location=torch.device("cpu"))
    update_checkpoint_content(data)
    torch.save(data, file_name)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description="Update checkpoint(s) so that to be compatible with new (FFT)ConvolutionArray")
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

