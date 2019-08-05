#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/movement/primitives/primitive.py
Vince Enachescu 2018


Ijspeert et. al (2003) Learning attractor landscapes for learning motor primitives.
"""

import numpy as np
# from numpy.core._multiarray_umath import dtype

from psyclab.utilities.errors import DimensionMismatch, InvalidParameterValue


class DynamicMovementPrimitive(object):
    """
    Base class for a dynamic movement primitive
    """

    _canonical_system = None
    _transformation_system = None

    def __init__(self, n_dimensions=1, n_basis=10, duration=0.5, goal=None, **kwargs):

        if n_dimensions <= 0:
            raise InvalidParameterValue('number of dimensions must be greater than zero')

        self.canonical = self._canonical_system(**kwargs)
        self.transformation = [
            self._transformation_system(self.canonical, n_basis)
            for _ in range(n_dimensions)
        ]

        self.duration = duration

        if goal is not None:
            self.goal = goal

    @property
    def parameters(self):
        """ Parameters used to shape the forcing term of the primitive """
        return np.stack([trans.weights for trans in self.transformation])

    @parameters.setter
    def parameters(self, values):
        """ Set the parameters used to shape the forcing term of the primitive """

        if len(values) != self.n_dimensions:
            if len(values) % self.n_dimensions != 0:
                raise DimensionMismatch('parameters', len(values), self.n_dimensions)
            values = np.split(values, self.n_dimensions)

        for weights, trans in zip(values, self.transformation):
            trans.weights = weights.astype(np.float32)
            trans.compute_basis()

    @property
    def n_dimensions(self):
        return len(self.transformation)

    @property
    def n_parameters(self):
        """ Number of parameters / basis functions per dimension """
        return [len(trans) for trans in self.transformation]

    @property
    def duration(self):
        """ Length of time in seconds of the base primitive duration """
        return self.canonical.duration

    @duration.setter
    def duration(self, duration):
        """
        Set the base primitive duration

        Parameters
        ----------
        duration : float
            Time in seconds
        """

        if not isinstance(duration, float) or duration <= 0.0:
            raise InvalidParameterValue('movement duration', duration)

        self.canonical.duration = duration

    @property
    def goal(self):
        """ Target coordinates for the end of the movement primitive """
        return [trans.goal for trans in self.transformation]

    @goal.setter
    def goal(self, goal):
        """ set the target end coordinates for the discrete movement primitive

        Parameters
        ----------
        goal : float, array_like
            specify the target coordinate for the end of the motion, if only a
            float is given, its values will be used for all dimensions
        """

        if len(goal) != self.n_dimensions:
            raise DimensionMismatch('goal', len(goal), self.n_dimensions)

        for trans, g in zip(self.transformation, goal):
            trans.goal = g

        self.canonical.reset()

    @property
    def state(self):
        """
        The current full state of the movement primitive, including canonical
        phase variable and its derivative.
        """

        data = self.canonical.state
        states = [trans.state for trans in self.transformation]
        data.update({
            name: np.array([s[name] for s in states])
            for name in states[0].keys()
        })
        data.update(self.canonical.state)
        return data

    @property
    def y(self):
        return np.array([trans.y for trans in self.transformation])

    @property
    def yd(self):
        return np.array([trans.yd for trans in self.transformation])

    @property
    def f(self):
        return np.array([trans.f for trans in self.transformation])

    def step(self, time_step):
        """
        Step the movement primitive forward by a small time step

        Parameters
        ----------
        time_step : float
            Length of time in seconds to step the dynamical systems forward
        """

        self.canonical.step(time_step)
        y, yd, ydd = zip(*[trans.step(time_step) for trans in self.transformation])
        return np.array(y), np.array(yd), np.array(ydd)

    def reset(self, y=None):
        """
        Reset the movement primitive to its initial state

        Parameters
        ----------
        y : array_like
            Initial coordinates to reset the state to

        Raises
        ------
        DimensionMismatch
            If the parameter `y` provided is not the same dimensionality
        """

        if y is None:
            y = [trans.y0 for trans in self.transformation]
        elif len(y) != self.n_dimensions:
            raise DimensionMismatch('state dimensions', len(y), self.n_dimensions)

        self.canonical.reset()
        [trans.reset(y=yi) for yi, trans in zip(y, self.transformation)]

    def basis_functions(self, duration=0.5, time_step=0.01):
        """
        Return the functions that form the basis used to parameterize the
        movement primitive

        Parameters
        ----------
        duration : float
            length of time in seconds for the whole primitive
        time_step : float
            length of time in seconds between time steps

        Returns
        -------
        array_like
            basis_functions d x [p x t]
        """

        x, _ = self.canonical.trajectory(duration=duration, time_step=time_step)
        return np.stack([trans.basis(x) for trans in self.transformation])

    def generate_trajectory(self, duration=0.5, time_step=0.01, y0=None, noise=None):
        """ generate a trajectory of the primitive based on the current parameters

        Parameters
        ----------
        duration : float
            Length of time in seconds of the movement trajectory to generate
        time_step : float
            Length of time in seconds between steps
        y0 : array_like
            Starting coordinates

        Returns
        -------
        y, yd, ydd : array_like, (Nd, Nt)
            Trajectory from the movement primitive
        """

        n_time_steps = int(round(duration / time_step))
        parameters = self.parameters

        if noise is not None:
            if len(noise) != len(parameters):
                raise DimensionMismatch('noise', len(noise), len(parameters))

            self.parameters = [np.array(n + p) for n, p in zip(noise, parameters)]

        self.reset(y=y0)
        y, yd, ydd = zip(*[self.step(time_step) for _ in range(n_time_steps)])

        if noise is not None:
            self.parameters = parameters

        return np.stack(y, axis=1), np.stack(yd, axis=1), np.stack(ydd, axis=1)

    def fit_trajectory(self, y, yd=None, ydd=None, time_step=0.01):
        """ attempt to solve for a parameterization that best fits a given
        trajectory

        Parameters
        ----------
        y : array_like
            Trajectory position over time used to fit parameters (Nd, Nt)
        yd, ydd : array_like, optional
            Velocity and acceleration vectors used to fit parameters
        time_step : float
            Length of time in seconds between steps
        """

        n_dimensions, n_time_steps = y.shape
        duration = n_time_steps * time_step

        if yd is None:
            yd = np.pad(np.diff(y, n=1, axis=1), [[0, 0], [1, 0]], 'constant')

        if ydd is None:
            ydd = np.pad(np.diff(yd, n=1, axis=1), [[0, 0], [1, 0]], 'constant')

        self.duration = duration
        x, _ = self.canonical.trajectory(duration=duration, time_step=time_step)
        for i, trans in enumerate(self.transformation):
            trans.fit(x, y[i], yd[i], ydd[i])

    def plot(self, axs, duration=1.0, time_step=0.01, y0=None, color='C0', **kwargs):
        """
        Generate a trajectory from primitive and plot the results

        Parameters
        ----------
        axs : subplots
            Subplot axes to plot the movement primitive on
        duration : float
            Length of time in seconds of the movement trajectory to generate
        time_step : float
            Length of time in seconds between steps
        y0 : array_like
            Starting coordinates
        """

        x, xd, xdd = self.generate_trajectory(duration=duration, time_step=time_step, y0=y0)
        t = np.linspace(0, duration, max(x.shape))

        for i, y in enumerate((x, xd, xdd)):
            for j, v in enumerate(y):
                axs[j, i].plot(t, v, color=color, **kwargs)

        return axs, (x, xd, xdd)

    def write_file(self, file_path):
        """ Write the movement primitive parameters to a numpy format file """
        np.save(file_path, self.parameters.astype(np.float32))

    @classmethod
    def from_parameters(cls, parameters, goal=None):
        """
        Create a movement primitive from a numpy format file

        Parameters
        ----------
        parameters : np.ndarray
        """

        n_dimensions, n_basis = parameters.shape
        primitive = cls(n_dimensions=n_dimensions, n_basis=n_basis, goal=goal)
        primitive.parameters = parameters
        return primitive

    @classmethod
    def from_file(cls, file_path):
        """
        Create a movement primitive from a numpy format file

        Parameters
        ----------
        file_path : str
            Path to the ``.npy`` formatted file containing the parameters
        """

        parameters = np.load(file_path)
        n_dimensions, n_basis = parameters.shape
        primitive = cls(n_dimensions=n_dimensions, n_basis=n_basis)
        primitive.parameters = parameters
        return primitive

    @classmethod
    def from_trajectory(cls, y, yd=None, ydd=None, duration=None, time_step=None):
        """
        Create a movement primitive from a desired trajectory

        Parameters
        ----------
        y : array_like

        yd, ydd : array_like, optional

        duration : float

        time_step : float

        """
        # TODO
        pass

    def __len__(self):
        return len(self.transformation)

    def __iter__(self):
        for transformation in self.transformation:
            yield transformation

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return self.transformation[index]
