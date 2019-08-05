# -*- coding: utf-8 -*-
"""
Policy Improvement through Path Integrals

These functions implement policy improvement through path integrals (pi-squared)
explanation and derivation may be found here


References
----------
.. [1] Evangelos Theodorou, Jonas Buchli, and Stefan Schaal. 2010.
    A Generalized Path Integral Control Approach to Reinforcement Learning.
    J. Mach. Learn. Res. 11 (December 2010), 3137-3181.

"""

import numpy as np


def compute_update(
    parameters, noise, state_costs,
    basis_functions=None, control_costs=None,
    noise_covariance=None, noise_update_alpha=0.1, noise_decay=0.95,
    control_cost_weight=1.0, kernel_weighted=True,
):
    """
    Given a set of noisy parameters for a policy and their costs computed
    for an execution of each set of parameters; compute updates to the policy
    parameters using the pi-squared algorithm

    d - number of dimensions
    n - number of time steps
    m - number of parameters per dimension
    r - number of rollouts

    Parameters
    ----------
    parameters : numpy.ndarray
        set of initial policy parameter vectors to be updated (d, m)
    noise : numpy.ndarray:
        vectors of noise added to the parameters for each rollout (d, r, m)
    state_costs : numpy.ndarray
        state costs for each rollout
    basis_functions : numpy.ndarray
        basis function matrices for each dimension, shaped (d, m, n)
    control_costs : numpy.ndarray, optional
        control cost matrices for each dimension, used to shape the distribution
        of exploration noise and as a regularization term in the cost function
        defaults to identity (d, m, m)
    noise_covariance : numpy.ndarray, optional
        covariance matrices used to generate multivariate normal noise (d, m, m)
    noise_update_alpha : float
        hyper-parameter used to smooth the updates to noise covariance
    control_cost_weight : float
        weighting of control costs in computing updates
    kernel_weighted : bool
        if True, weight the updates based on the level of kernel activation


    Returns
    -------
    updates : numpy.ndarray
        computed updates to the policy parameterization
    probabilities : numpy.ndarray
        computed probabilities associated with each rollouts' timesteps
    new_covariance : numpy.ndarray
        new noise covariance matrices for each dimension, only returned if the
        parameter noise_covariance is supplied
    """

    d, m = parameters.shape
    r, n = state_costs.shape

    if basis_functions is None:
        basis_functions = [np.ones((n, m)) * (1.0 / m) for _ in range(m)]

    if len(noise) != d:
        noise = [np.vstack(n) for n in zip(*noise)]

    if control_costs is None:
        control_costs = [np.identity(theta.size) for theta in parameters]

    time_step_weights = [
        compute_time_step_weights(basis, kernel_weighted=kernel_weighted)
        for basis in basis_functions
    ]

    # form matrices used to project into low cost spaces
    projection_matrices = map(compute_projection_matrices, control_costs, basis_functions)
    projected_noise = list(map(compute_projected_noise, noise, projection_matrices))

    # compute the total cost-to-goal for each of the rollouts
    costs = np.sum([
        compute_control_costs(theta, eps, R, control_cost_weight)
        for theta, eps, R in zip(parameters, projected_noise, control_costs)
    ])

    # flip to last time step first, perform cumulative sum, then flip back around
    sum_costs = np.fliplr(np.cumsum(np.fliplr(costs + state_costs), axis=1))
    probabilities = compute_probabilities(sum_costs)

    updates = np.stack([
        compute_parameter_updates(probabilities, pnoise, weights)
        for pnoise, weights in zip(projected_noise, time_step_weights)
    ])

    if noise_covariance is None:
        return updates, probabilities

    # use covariance matrix adaptation to update the noise covariance matrix
    covariance_update = [
        compute_covariance_update(eps, probabilities, weights)
        for eps, weights in zip(noise, time_step_weights)
    ]

    new_covariance = [
        (noise_update_alpha * A + (1.0 - noise_update_alpha) * B) * noise_decay
        for A, B in zip(covariance_update, noise_covariance)
    ]

    return updates, probabilities, new_covariance


def generate_noise(parameters, covariance=None, gain=1.0):
    """
    Generate new noise vectors from a multivariate gaussian distribution

    Parameters
    ----------
    parameters : numpy.ndarray
        vectors of parameters, used to pick size of noise vectos
    covariance : numpy.ndarray, optional
        covariance matrices used to shape the noise distribution, defaults to
        identity matrices

    Returns
    -------
    numpy.ndarray
        vectors of multivariate normal noise
    """

    if covariance is None:
        return [
            np.random.randn(*theta.shape) * gain
            for theta in parameters
        ]

    return [
        np.random.multivariate_normal(np.zeros(cov.shape[0]), cov)
        for cov in covariance
    ]


def compute_projection_matrices(control_costs, basis_functions):
    """
    compute the projection matrices for one dimensions
    m - number of parameters in dimension
    n - number of time steps

    ..math::  M_t = R^-1 * (g * g') / g' * R^-1 * g

    Parameters
    ----------
    control_costs : numpy.ndarray
        [m x m]
    basis_functions : numpy.ndarray, (n, m)
        activation levels of each parameter across time steps

    Returns
    -------
    numpy.ndarray
        set of matrices used to project the noise into a low-cost parameter space
    """

    Ri = np.linalg.inv(control_costs)
    return [
        np.dot(Ri, np.outer(g, g)) / (np.dot(g, np.dot(Ri, g.T)) + 1.0e-7)
        for g in basis_functions
    ]


def compute_projected_noise(noise, projection_matrices):
    """
    compute the noise projected into the inverse control costs
    this means we are exploring a low-costs subspace

    Parameters
    ----------
    noise : numpy.ndarray
    projection_matrices : numpy.ndarray
        set of matrices used to project the noise into a low-cost parameter space

    Returns
    -------
    numpy.ndarray
        noise from each rollout projected into low parameter cost space
    """

    return np.dstack([np.vstack([np.dot(M, eps) for M in projection_matrices]) for eps in noise])


def compute_control_costs(theta, projected_noise, control_costs, control_cost_weight=1.0):
    """
    compute the total cost of each rollout to finish from each time point

    Parameters
    ----------
    theta : numpy.ndarray, (m,)
        policy parameters for a single dimension
    projected_noise : numpy.ndarray, (r, m)
        noise from each rollout projected into low parameter cost space
    control_costs : numpy.ndarray, (m, m)
        matrix of control costs (m, m)
    control_cost_weight: float
        weight used to penalize large control parameter values

    Returns
    -------
    numpy.ndarray, (r, n)
        costs of control for each rollout over the full policy
    """

    return np.vstack([
        np.array([
            np.dot(np.dot(theta + e, control_costs), theta + e) * control_cost_weight
            for e in pn.T])
        for pn in projected_noise.T
    ])


def compute_probabilities(costs):
    """
    compute the relative probabilities based on cost-to-goal from each time step
    in each rollout

    Parameters
    ----------
    costs : numpy.ndarray, (r, n)
        total summed cost-to-goal for each time step in each rollout

    Returns
    -------
    numpy.ndarray, (r, n)
        relative probabilities of each time step

    """

    min_costs = costs.min(axis=0)
    max_costs = costs.max(axis=0) + 1.0e-3
    diff_costs = (max_costs - min_costs)
    diff_costs[diff_costs <= 0.0] = 1.0e-6

    h = 10.0   # scaling parameter
    exp_costs = np.exp(-h * (costs - min_costs) / (diff_costs))
    return exp_costs / np.sum(exp_costs, axis=0)


def compute_parameter_updates(probabilities, projected_noise, time_step_weights):
    """
    compute the updates for parameter values in a single dimension
    m - number of parameters in dimension
    n - number of time steps

    Parameters
    ----------
    probabilities : numpy.ndarray, (r, n)
        the relative probabilities computed for each time step and rollout
    projected_noise : numpy.ndarray, (m, n, r)
        noise from each rollout projected into low parameter cost space
    time_step_weights : numpy.ndarray, (n,)
        weights used to bias parameter updates to specific time steps

    Returns
    -------
    numpy.ndarray, (m,)
        updates to the parameters of a policy
    """

    dttheta = np.vstack([np.sum(P * e, axis=1) for P, e in zip(probabilities.T, projected_noise)])
    return np.dot(dttheta.T, time_step_weights)


def compute_covariance_update(noise, probabilities, time_step_weights=None):
    """
    compute a new covariance matrix to generate noise for future rollouts
    essentially a probability weighted outer product of current noise

    Parameters
    ----------
    probabilities : numpy.ndarray, (r, n)
        the relative probabilities computed for each time step and rollout
    projected_noise : numpy.ndarray, (m, n, r)
        noise from each rollout projected into low parameter cost space
    time_step_weights : numpy.ndarray, (n,)
        weights used to bias parameter updates to specific time steps

    Returns
    -------
    numpy.ndarray, (m x m)
        covariance matrix used to shape the distribution of noise for rollouts

    """

    if time_step_weights is None:
        _, n = probabilities.shape
        time_step_weights = np.arange(n, 0, -1) / ((n / 2.) * (n + 1.))

    return np.sum([
        np.sum([
            np.outer(e, e) * p * w
            for p, w in zip(pr, time_step_weights)
        ], axis=0)
        for e, pr in zip(noise, probabilities)
    ], axis=0)


def compute_time_step_weights(basis_functions, kernel_weighted=True):
    """
    Determine a weighting for each time step of the problem

    Parameters
    ----------
    basis_functions : numpy.ndarray
        activation levels of each parameter across time steps

    Returns
    -------
    numpy.ndarray, (n,)
        weights
    """

    if basis_functions is not None and kernel_weighted:
        return np.sum(basis_functions, axis=1) / np.sum(basis_functions)

    n, _ = basis_functions.shape
    return np.arange(n, 0, -1) / ((n + 1.) * (n / 2.))
