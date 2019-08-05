"""
simple convergence
"""
import matplotlib
matplotlib.use('macosx')
import numpy as np
import matplotlib.pyplot as plt

from psyclab.learning import pi_squared as pi2


def convergence_test(cost_function, n_parameters, n_dimensions, tolerance=0.01, noise_gain=1.0, plots=False, printout=True, parallel=False):

    basis_functions = np.stack([np.identity(n_parameters)] * n_dimensions)
    noise_covariance = np.stack([np.identity(n_parameters) * noise_gain] * n_dimensions)
    theta = np.random.randn(n_dimensions, n_parameters)

    n_rollouts = int(n_parameters * 1.5)
    iterations = 0

    if plots:
        plt.ion()
        fig, ax = plt.subplots(ncols=1, figsize=(12, 8))
        lines = []
        for t in theta:
            line, = ax.plot(t)
            lines.append(line)
        plt.show()

    error = np.sum(cost_function(theta))

    while error > tolerance or iterations == 0:

        noise = [pi2.generate_noise(theta, noise_covariance) for _ in range(n_rollouts)]
        costs = np.vstack([cost_function([th + e for th, e in zip(theta, n)]) for n in noise])

        dtheta, _, noise_covariance = pi2.compute_update(
            theta, noise, costs,
            basis_functions=basis_functions,
            noise_covariance=noise_covariance,
            control_cost_weight=0.0,
            noise_update_alpha=0.25,
        )

        theta += dtheta

        error = np.sum(cost_function(theta))
        iterations += 1

        if plots:

            for t, line in zip(theta, lines):
                line.set_data(np.arange(n_parameters), t)

            fig.canvas.draw()
            # plt.draw()
            plt.pause(0.001)

        if printout:
            s = np.sum([np.dot(np.matrix(cov).dot(np.ones(n_parameters)), np.ones(n_parameters)) for cov in noise_covariance])
            print('  {:03d}  |  {:04g}\t|\t{:04f}'.format(iterations, error, s))

    return


def main(n_parameters=10, n_dimensions=1, noise_gain=1.0, plots=True, printout=True, parallel=False, seed=None):

    # learn a sine wave in the parameters
    def sin_costs(theta):
        return np.sum([np.square(t - np.sin(np.linspace(0, 2.*np.pi, t.size))) for t in theta], axis=0)

    if seed is not None:
        np.random.seed(seed)

    convergence_test(sin_costs, n_parameters, n_dimensions, 0.01, noise_gain=noise_gain, plots=plots, printout=printout, parallel=parallel)


if __name__ == "__main__":
    main(plots=True, printout=True)
