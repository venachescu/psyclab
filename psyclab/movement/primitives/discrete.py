#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/movement/primitives/discrete.py
Vince Enachescu 2018
"""

import numpy as np

from .canonical import CanonicalSystem
from .transformation import TransformationSystem
from .primitive import DynamicMovementPrimitive


class DiscreteCanonicalSystem(CanonicalSystem):
    """
    A canonical system used to represent a discrete movement primitive

    Parameters
    ----------
    second_order : bool
        choose the order of the time parameterization

    """

    _variables_ = ('x', 'v', 'xd', 'vd')

    def __init__(self, second_order=False):

        CanonicalSystem.__init__(self, second_order=second_order)

        # discrete system state variables
        self.v = 0.0
        self.vd = 0.0

    @property
    def duration(self):
        """ Current length of time in seconds of the executed primitive """
        return self._duration

    @duration.setter
    def duration(self, value):
        """ Set the length of time in seconds, used to stretch the primitive """
        self._duration = value
        self.tau = 0.5 / value

    def step(self, time_step=0.01):
        """
        Step the dynamical equations forward by a small time increment

        Parameters
        ----------
        time_step : float
            Time step in seconds
        """

        if self.second_order:
            self.vd = self.alpha_v * (- self.x * self.beta_v - self.v) * self.tau
            self.xd = self.v * self.tau

        else:
            self.xd = - self.alpha_x * self.x * self.tau
            self.vd = self.xd

        self.x += self.xd * time_step
        self.v += self.vd * time_step

    def reset(self):
        """ Reset the canonical system to the initial state """

        self.v = 0.0
        self.x = 1.0
        self.vd = 0.0
        self.xd = 0.0


class DiscreteTransformationSystem(TransformationSystem):
    """
    Discrete Transformation system - represents one dimension of a point-to-point
    movement primitive

    """

    _variables_ = ('y', 'yd', 'ydd', 'z', 'zd', 'g', 'gd', 'f')

    def __init__(self, canon, n_basis_functions=10):

        TransformationSystem.__init__(self, canon, n_basis_functions=n_basis_functions)

        self.g = 1.0
        self.gd = 0.0

        self.delta = 1.0
        self.goal = 1.0

        # centers of the basis functions and their phase velocity and bandwidth
        self.compute_basis()

    def compute_basis(self):
        """ Compute the center locations of the basis functions """

        t = np.linspace(0, 1.0, self.weights.size)
        if self.canon.second_order:
            alpha = self.canon.alpha_z / 2.0
            c = (1.0 + alpha * t) * np.exp(-alpha * t)
            cd = -alpha * c + alpha * np.exp(-alpha * t)

        else:
            c = np.exp(-self.canon.alpha_x / 2.0 * t)
            cd = -self.canon.alpha_x / 2.0 * c

        h = np.power(np.diff(c) * 0.55, 2.0)
        h = 1.0 / np.append(h, h[-1])
        self.c, self.cd, self.h = c, cd, h

    @property
    def goal(self):
        """ Target end coordinates """
        return self._goal

    @goal.setter
    def goal(self, goal):
        """ Set the target """

        # if goal is set to zero, set scale variable to 1.0
        if abs(self.y0 - goal) < 1.0e-6:
            scale = 1.0
        # make sure that (y0 - goal) doesn't equal 0
        elif abs(self.y0 - goal) < 1.0e-3:
            goal += np.sign(goal) * 1.0e-3
            scale = (goal - self.y0) / self.delta
        else:
            scale = (goal - self.y0) / self.delta

        self.g = goal
        self.scale = scale
        self._goal = goal

    def basis(self, x=None):
        """
        Compute the activations of the basis functions

        Parameters
        ----------
        x : array_like, float, optional
            Phase variable from canonical system used to specify time(s)
            along the primitive to compute the basis functions, returns
            the current values by default
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]
        elif x is None:
            x = self.canon.x
        return np.exp(-0.5 * np.power(x - self.c, 2.0) * self.h)

    def step(self, time_step):
        """
        Step forward the dynamical equations in the transformation system

        Parameters
        ----------
        time_step : float
            Time in seconds to step the dynamic system forward
        """

        x = self.canon.x
        psi = self.basis(x)
        self.f = x * (np.dot(psi, self.weights)) / (psi.sum() + 1e-9) * self.scale
        # print(x, np.dot(psi, self.weights) / (psi.sum() + 1e-9), self.scale)

        self.zd = (self.canon.alpha_z *
                   (self.canon.beta_z * (self.g - self.y) - self.z) +
                   self.f + self.coupling) * self.tau
        self.yd = self.z * self.tau
        self.ydd = self.zd * self.tau
        self.y += self.yd * time_step
        self.z += self.zd * time_step

        # goal dynamics
        self.gd = self.canon.alpha_g * (self._goal - self.g)
        self.g += self.gd * time_step

        return self.y, self.yd, self.ydd

    def fit(self, x, y, yd, ydd):
        """ Fit the parameters of the forcing function to a given trajectory """

        self.reset(y=y[0])

        self.goal = y[-1]
        self.delta = self.goal - self.y0

        self.amplitude = (np.max(y) - np.min(y))
        self.scale = 1.0

        x_psi = x[:, None] * self.basis(x)

        f_target = ydd / np.power(self.tau, 2.0) - self.canon.alpha_z * \
            (self.canon.beta_z * (self.g - y) - yd / self.tau) / \
            self.scale

        ones = np.ones_like(self.weights).T
        self.weights = np.sum(np.outer(ones, x * f_target) * x_psi.transpose(), 1) / \
            (np.sum(np.outer(ones, np.power(x, 2.0)) * x_psi.transpose(), 1) + 1e-9)

    def reset(self, y=None):
        """ Reset the transformation system to its initial state """

        if y is not None:
            self.y0 = y

        self.y = self.y0
        self.yd = 0.0
        self.ydd = 0.0

        self.z = 0.0
        self.zd = 0.0

        self.f = 0.0
        self.g = self.goal
        self.gd = 0.0


class DiscreteMovement(DynamicMovementPrimitive):
    """
    A discrete movement primitive, i.e. a point-to-point motion.

    Parameters
    ----------
    n_dimensions : int
        number of independent dimensions
    n_basis : int
        number of basis functions per dimension
    duration : float
        time in seconds of the base primitive
    goal : array_like
        target coordinates for the end of the motion

    Attributes
    ----------


    Raises
    ------
    InvalidParameterValue
    """

    _canonical_system = DiscreteCanonicalSystem
    _transformation_system = DiscreteTransformationSystem
