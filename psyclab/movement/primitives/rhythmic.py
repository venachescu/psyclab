#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/movement/primitives/rhythmic.py
Vince Enachescu 2018
"""

import numpy as np

from .canonical import CanonicalSystem
from .transformation import TransformationSystem
from .primitive import DynamicMovementPrimitive


class RhythmicCanonicalSystem(CanonicalSystem):
    """
    A canonical system used to represent a rhythmic movement primitive
    """

    _variables = ('x', 'xd')

    def __init__(self, **kwargs):

        CanonicalSystem.__init__(self, second_order=False, **kwargs)

        self.omega = 2 * np.pi
        self.reset()

    def step(self, time_step=0.01):
        """
        Step the dynamical equations forward by a small time increment

        Parameters
        ----------
        time_step : float
            Time step in seconds
        """

        self.xd = self.omega * self.tau
        self.x += self.xd * time_step

    @property
    def duration(self):
        """ Current length of time in seconds of the executed primitive """
        return self._duration

    @duration.setter
    def duration(self, value):
        """ Set the length of time in seconds, used to stretch the primitive """

        self._duration = value
        self.tau = 1.0

    def reset(self):
        """ Reset the canonical system to the initial state """

        self.x = 0.0
        self.xd = 0.0


class RhythmicTransformationSystem(TransformationSystem):
    """

    """

    _variables_ = ('y', 'yd', 'ydd', 'z', 'zd', 'g', 'gd')

    def __init__(self, canonical, n_basis_functions=10):

        TransformationSystem.__init__(self, canonical, n_basis_functions=n_basis_functions)

        self.baseline = 0.0
        self.ym = 0.0
        self.compute_basis()

    def compute_basis(self):
        """ Compute the size and spacing of the basis functions """

        h = np.ones(self.n_basis) * 0.1 * (self.n_basis ** 2)
        c = np.linspace(0, 2 * np.pi, self.n_basis + 1)[:-1]
        cd = np.zeros(self.n_basis)
        self.h, self.c, self.cd = h, c, cd

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
        return np.exp(self.h * (np.cos(x - self.c) - 1))

    def step(self, time_step=0.01):
        """
        Step forward the dynamical equations in the transformation system

        Parameters
        ----------
        time_step : float
            Time in seconds to step the dynamic system forward
        """

        x = self.canon.x
        psi = self.basis(x)
        self.f = (np.dot(psi, self.weights)) / (psi.sum() + 1e-9) * self.scale

        self.zd = (self.canon.alpha_z *
                   (self.canon.beta_z * (self.ym - self.y) - self.z) +
                   self.f + self.coupling) * self.tau
        self.yd = self.z * self.tau
        self.ydd = self.zd * self.tau
        self.y += self.yd * time_step
        self.z += self.zd * time_step

        return self.y, self.yd, self.ydd

    def fit(self, x, y, yd, ydd):
        """ Fit the parameters of the forcing function to a given trajectory """

        self.reset(y=y[0])
        self.baseline = np.mean(y, axis=0)
        self.amplitude = np.max(y) - np.min(y)
        self.scale = 1.0

        # f_target = ydd / np.power(self.tau, 2.0) - self.canon.alpha_z * \
                   # (self.canon.beta_z * (self.ym - y) - yd / self.tau) / self.scale

        f_target = ydd - self.canon.alpha_z * (self.canon.beta_z * (self.ym - y) - yd) / self.scale

        psi = self.basis(x).transpose()
        ones = np.ones_like(self.weights)
        self.weights = np.sum(np.outer(ones, f_target) * psi, 1) / (np.sum(psi, 1) + 1e-9)
        # (np.sum(np.outer(ones, ones) * psi.transpose(), 1) + 1e-9)

    def reset(self, y=None):
        """ Reset the transformation system to its initial state """

        if y is not None:
            self.y0 = y

        self.y = y
        self.yd = 0.0
        self.ydd = 0.0

        self.z = 0.0
        self.zd = 0.0

        self.f = 0.0


class RhythmicMovement(DynamicMovementPrimitive):
    """
    A rhythmic movement primitive, i.e. a cyclic movement like a gait

    Parameters
    ----------
    duration : float
        time in seconds of the base primitive
    n_dimensions : int
        number of independent dimensions
    n_basis : int
        number of basis functions per dimension
    goal : array_like
        target coordinates for the end of the motion

    Attributes
    ----------
    """

    _canonical_system = RhythmicCanonicalSystem
    _transformation_system = RhythmicTransformationSystem
