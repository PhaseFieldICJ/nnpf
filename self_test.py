#!/usr/bin/env python3

import doctest
import sys
import importlib
import time

modules = [
    'domain',
    'nn_models',
    'nn_toolbox',
    'shapes',
    'splitting',
    'visu',
    'trainer',
    'problem',
    'reaction_model',
    'reaction_problem',
    'heat_problem',
    'heat_array_model',
    'mean_curvature_problem',
    'allen_cahn_problem',
    'allen_cahn_splitting',
    'willmore_problem',
    'exp_willmore_parallel',
    'model_infos',
]

# Cleaning log folder
import shutil
try:
    shutil.rmtree("logs_doctest")
except FileNotFoundError:
    pass

failure_count, test_count = 0, 0

for mod_name in modules:
    print("#" * 80)
    print(f"# {mod_name}")

    tic = time.time()
    module = importlib.import_module(mod_name)
    curr_failure_count, curr_test_count = doctest.testmod(module)
    failure_count += curr_failure_count
    test_count += curr_test_count
    toc = time.time()

    if curr_test_count == 0:
        print(f"-> No test", end='')
    else:
        if curr_failure_count == 0:
            print(f"-> {curr_test_count} tests passed successfully", end='')
        else:
            print(f"-> /!\ {curr_failure_count}/{curr_test_count} tests fail", end='')

    print(f" ({toc - tic:.3f}s elapsed)")
    print()

print()

if failure_count == 0:
    print(f"OK: {test_count} tests passed successfully")
    sys.exit(0)
else:
    print(f"KO: {failure_count}/{test_count} tests fail")
    sys.exit(1)

