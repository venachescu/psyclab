#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/movement/primitives/transformation.py
Vince Enachescu 2018
"""

import numpy as np

from .canonical import CanonicalSystem
from psyclab.utilities.errors import InvalidParameterValue


class TransformationSystem(object):
    """
    Transformation system - represents one dimension of a movement primitive

    Parameters
    ----------
    canonical : CanonicalSystem
        Canonical system to couple transformation systems across dimensions
    n_basis_functions : int
        Number of basis functions and parameters used to represent system

    Raises
    ------
    InvalidParameterValue
        Make sure to pass a CanonicalSystem object as the first argument
    """

    _variables_ = tuple()

    def __init__(self, canonical, n_basis_functions=10):

        # (observable / external) state variables
        self.y = 0.0
        self.yd = 0.0
        self.ydd = 0.0

        # initial state
        self.y0 = 0.0

        # (internal) state variables
        self.z = 0.0
        self.zd = 0.0

        # linear forcing function term
        self.f = 0.0
        self.a = 0.0

        self.amplitude = 1.0
        self.scale = 1.0
        self.coupling = 0.0

        if not isinstance(canonical, CanonicalSystem):
            raise InvalidParameterValue('need a canonical system!')

        self.canon = canonical

        # basis values and weights
        self.weights = np.zeros(n_basis_functions, dtype=np.float32)

    @property
    def state(self):
        """ Current state of the transformation system variables """
        return {name: getattr(self, name) for name in self._variables_}

    def step(self, time_step):
        """
        Step forward the dynamical equations in the transformation system

        Parameters
        ----------
        time_step : float
            time in seconds to step the dynamic system forward
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def tau(self):
        return self.canon.tau

    def reset(self, y=None):
        """ Reset the canonical system to the initial state """
        raise NotImplementedError

    @property
    def n_basis(self):
        """ Number of basis functions and parameters used in this system """
        return self.weights.size

    def __len__(self):
        """ length of this object is the number of parameters """
        return self.weights.size
