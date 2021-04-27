###############################################################################
# Command-line interface
###############################################################################

if __name__ == "__main__":

    import problem
    import heat_problem

    # Command-line arguments
    parser = argparse.ArgumentParser(
            description="Evaluation of a model for the heat equation problem",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', type=str, help="Path to the model's checkpoint")
    parser.add_argument('--spheres', type=int, nargs='+', default=[1], help="Number (or interval) of spheres")
    parser.add_argument('--radius', type=float, nargs='+', default=[0.2], help="Radius (or interval) of the spheres")
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.01], help="Interface sharpness (or interval) of the phase field")
    parser.add_argument('--steps', type=int, default=1, help="Number of evolution steps")
    args = parser.parse_args()

    # Expanding scalar to bounds
    args.spheres += args.spheres * (2 - len(args.spheres))
    args.radius += args.radius * (2 - len(args.radius))
    args.epsilon += args.epsilon * (2 - len(args.epsilon))

    # Loading model
    checkpoint_path = problem.checkpoint_from_path(args.checkpoint)
    extra_data = torch.load(checkpoint_path)
    model = heat_problem.HeatProblem.load_from_checkpoint(args.checkpoint)
    model.freeze()

    # Some informations about the model
    print(f"""
Model summary:
    checkpoint path: {checkpoint_path}
    epochs: {extra_data['epoch']}
    steps: {extra_data['global_step']}
    best score: {extra_data['checkpoint_callback_best_model_score']}
    ndof: {nn_toolbox.ndof(model)}
""")

    print("Model hyper parameters:")
    for key, value in model.hparams.items():
        print(f"    {key}: {value}")

    # Exact solution
    exact_sol = heat_problem.HeatSolution(model.domain, model.hparams.dt)

    # Generate initial data
    exact_data = generate_phase_field_union(
            1,
            lambda num_samples: generate_sphere_phase(num_samples,
                                                      model.domain,
                                                      args.radius,
                                                      args.epsilon),
            args.spheres)
    model_data = exact_data.clone()

    # Display
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=args.steps + 1, ncols=3)

    for i in range(args.steps + 1):
        # Exact solution
        axes[i, 0].imshow(exact_data[0, 0, ...])
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Solution')

        # Model solution
        axes[i, 1].imshow(model_data[0, 0, ...])
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Model')

        # Difference
        im = axes[i, 2].imshow((exact_data - model_data)[0, 0, ...])
        fig.colorbar(im, ax=axes[i, 2])
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Difference')

        # Evolution
        exact_data = exact_sol(exact_data)
        model_data = model(model_data)

    fig.tight_layout(pad=0, w_pad=-10, h_pad=0)
    plt.show()


