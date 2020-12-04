#!/usr/bin/env python3

import doctest
import sys
import importlib

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
    'allen_cahn_problem',
    'allen_cahn_splitting',
    'willmore_problem',
    'exp_willmore_parallel',
    'model_infos',
]

# Cleaning log folder
import shutil
shutil.rmtree("logs_doctest")

failure_count, test_count = 0, 0

for mod_name in modules:
    print("#" * 80)
    module = importlib.import_module(mod_name)
    curr_failure_count, curr_test_count = doctest.testmod(module)
    failure_count += curr_failure_count
    test_count += curr_test_count

    if curr_test_count == 0:
        print(f"-> {mod_name}: No test")
    else:
        if curr_failure_count == 0:
            print(f"-> {mod_name}: {curr_test_count} tests passed successfully")
        else:
            print(f"-> {mod_name}: /!\ {curr_failure_count}/{curr_test_count} tests fail")

    print()

print()

if failure_count == 0:
    print(f"OK: {test_count} tests passed successfully")
    sys.exit(0)
else:
    print(f"KO: {failure_count}/{test_count} tests fail")
    sys.exit(1)

