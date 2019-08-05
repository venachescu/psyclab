#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/movement/primitives/canonical.py
Vince Enachescu 2018
"""

import numpy as np


class CanonicalSystem(object):
    """
    Base class of canonical system used to represent a movement primitive,
    the canonical system parameterizes and binds the temporal characters
    across all the transformation dimensions

    Parameters
    ----------
    second_order : bool
        choose the order of the time parameterization

    """

    _variables_ = tuple()

    def __init__(self, duration=0.5, second_order=False):

        # main phase variable of the canonical system and its derivative
        self.x = 0.0
        self.xd = 0.0

        self.c = 1.0
        self.tau = 1.0
        self.duration = duration

        # constants used to specify a critically damped system
        self.alpha_z = 25.0
        self.beta_z = self.alpha_z / 4.0
        self.alpha_g = self.alpha_z / 2.0
        self.alpha_x = self.alpha_z / 3.0
        self.alpha_v = self.alpha_z
        self.beta_v = self.beta_z

        self.second_order = second_order

    @property
    def duration(self):
        """ Current length of time in seconds of the executed primitive """
        raise NotImplementedError

    @duration.setter
    def duration(self, value):
        """ Set the length of time in seconds, used to stretch the primitive """
        raise NotImplementedError

    @property
    def state(self):
        """ Return the full state of the canonical system """
        return {name: getattr(self, name) for name in self._variables_}

    def step(self, time_step=0.01):
        """
        Step the dynamical equations forward by a small time increment

        Parameters
        ----------
        time_step : float
            Time step in seconds
        """
        raise NotImplementedError

    def reset(self):
        """ Reset the canonical system to the initial state """
        raise NotImplementedError

    def trajectory(self, duration=0.5, time_step=0.01):
        """
        Generate a rollout of the canonical system

        Parameters
        ----------
        duration : float
            Length of time in seconds of the movement primitive's rollout
        time_step :
            Length of time in seconds between simulation steps

        Returns
        -------
        x, xd : numpy.ndarray
            Vector of the phase variable and its derivative
        """

        self.duration = duration
        n_time_steps = int(duration / time_step)
        x, xd = np.zeros(n_time_steps), np.zeros(n_time_steps)

        self.reset()
        for t in range(n_time_steps):
            self.step(time_step)
            x[t] = self.x
            xd[t] = self.xd

        return x, xd
