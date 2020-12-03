#!/usr/bin/env python3
"""
Display information about a model
"""


def display_model_infos(model_or_path, recursive=True):
    """
    Display the informations about a model

    Parameters
    ----------
    model_or_path: Ligthning module or str
        The model or the path to a checkpoint
    recursive: bool
        If True, search for checkpoint in the hyper-parameters and display associated infos.

    Examples
    --------
    >>> display_model_infos("logs_doctest/Reaction/test0")
    <BLANKLINE>
    Model summary:
        class: Reaction
        problem: ReactionProblem
        ndof: 47
        checkpoint path: logs_doctest/Reaction/test0/checkpoints/epoch=0.ckpt
        epochs: 1
        steps: 1
        best score: 0
        best path: 
    <BLANKLINE>
    Model hyper parameters:
        seed: None
        dt: 6.103515625e-05
        epsilon: 0.0078125
        margin: 0.1
        Ntrain: 100
        Nval: 1000
        batch_size: 10
        batch_shuffle: True
        lr: 0.001
        layer_dims: [8, 3]
        activation: GaussActivation
    """

    import nn_toolbox
    import problem
    import inspect
    import torch

    is_path = isinstance(model_or_path, str)
    if is_path:
        map_location = torch.device("cpu")
        checkpoint_path = problem.checkpoint_from_path(model_or_path)
        model = problem.Problem.load_from_checkpoint(checkpoint_path, map_location=map_location)
        extra_data = torch.load(checkpoint_path, map_location=map_location)

        try:
            best_infos = next(value for callback, value in extra_data['callbacks'].items() if callback.__name__ == 'ModelCheckpoint')
            best_model_score = best_infos.get('best_model_score')
            best_model_path = best_infos.get('best_model_path')
        except KeyError:
            # Compatibility with checkpoints generated by previous Lightning versions
            best_model_score = extra_data.get('checkpoint_callback_best_model_score')
            best_model_path = extra_data.get('checkpoint_callback_best_model_path')
    else:
        model = model_or_path

    model_class = type(model)
    problem_class = model_class.__bases__[0]

    print(f"""
Model summary:
    class: {model_class.__name__}
    problem: {problem_class.__name__}
    ndof: {nn_toolbox.ndof(model)}""")

    if is_path:
        print(f"""\
    checkpoint path: {checkpoint_path}
    epochs: {extra_data['epoch']}
    steps: {extra_data['global_step']}
    best score: {best_model_score}
    best path: {best_model_path}
""")

    print("Model hyper parameters:")
    for key, value in model.hparams.items():
        print(f"    {key}: {value}")

    if recursive:
        for key, value in model.hparams.items():
            if isinstance(key, str) and key.find('checkpoint') >= 0:
                if isinstance(value, str):
                    print()
                    msg = f"Dependant model found in {key}: {value}"
                    print('#' * len(msg))
                    print(msg)
                    display_model_infos(value, recursive)
                else:
                    for i, v in enumerate(value):
                        print()
                        msg = f"Dependant model found in {key}[{i}]: {v}"
                        print('#' * len(msg))
                        print(msg)
                        display_model_infos(v, recursive)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
            description="Display informations about a model from a checkpoint",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', type=str, help="Path to the model's checkpoint")
    parser.add_argument('--recursive', type=lambda v: bool(int(v)), default=True, help="Display informations about checkpoint founds in hyper-parameters")
    args = parser.parse_args()

    display_model_infos(args.checkpoint, args.recursive)

