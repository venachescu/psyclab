#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/movement/movement_field.py
Vince Enachescu 2019
"""

import numpy as np

from psyclab.movement.primitives import DiscreteMovement


class MovementField:
    """
    A collection of movement primitives that span a particular dimension in planning space

    """

    def __init__(self, primitives=91, centers=None, duration=1.0, **kwargs):

        if isinstance(primitives, int):
            self.primitives = self.make_primitives(n_primitives=primitives, duration=duration, **kwargs)
        else:
            self.primitives = primitives

        if centers is None:
            centers = np.arange(len(self.primitives)) - len(self.primitives) // 2

        self.centers = np.array(centers)

        self._duration = duration

    def step(self, time_step=0.01):
        """ Step forward the movement field simulation """

        for primitive in self.primitives:
            primitive.step(time_step=time_step)

    def reset(self, *args, **kwargs):
        """ Reset the actions and sensory fields as well as all primitives """

        for primitive in self.primitives:
            primitive.reset(*args, **kwargs)

    @property
    def y(self):
        return np.stack([primitive.y for primitive in self.primitives])

    @property
    def yd(self):
        return np.stack([primitive.yd for primitive in self.primitives])

    @property
    def f(self):
        return np.stack([primitive.f for primitive in self.primitives])

    @property
    def parameters(self):
        return np.stack([primitive.parameters for primitive in self.primitives])

    @property
    def goals(self):
        return np.stack([primitive.goal for primitive in self.primitives])

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        for primitive in self.primitives:
            primitive.duration = value
        self._duration = 0.0

    def generate_trajectories(self, duration=0.5, time_step=0.01, y0=None):
        """ Generate a set of trajectories, one from each of the primitives """

        return np.stack([
            primitive.generate_trajectory(duration=duration, time_step=time_step, y0=y0)
            for primitive in self.primitives
        ])

    def __len__(self):
        return len(self.primitives)

    def __iter__(self):
        for primitive in self.primitives:
            yield primitive

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return self.primitives[index]

    @classmethod
    def from_file(cls, file_path, **kwargs):
        """
        Create a MovementField from the parameters stored in a file

        :param file_path:
        :return:
        """

        parameters, goals, centers = tuple(np.load(file_path).values())
        return cls.from_parameters(parameters, goals=goals, centers=centers, **kwargs)

    @classmethod
    def from_parameters(cls, parameters, goals=None, centers=None, **kwargs):

        n_primitives, n_dimensions, n_parameters = parameters.shape
        primitives = cls.make_primitives(n_primitives=n_primitives, n_dimensions=n_dimensions, n_parameters=n_parameters, goals=goals)
        for primitive, values in zip(primitives, parameters):
            primitive.parameters = values

        return cls(primitives, centers=centers, **kwargs)

    def write_file(self, file_path):
        """
        Write the current parameters of the primitives to a file

        :param file_path:
        :return:
        """

        np.savez(file_path, self.parameters, self.goals, self.centers)

    @staticmethod
    def make_primitives(n_primitives=91, n_dimensions=2, n_parameters=25, goals=None, duration=1.0):
        """
        Create a set of movement primitives

        """

        if goals is None:
            values = [(x,) + (1.0,) * (n_dimensions - 1) for x in np.linspace(-1.0, 1.0, n_primitives)]
        elif len(goals) == n_dimensions and len(goals) != n_primitives:
            values = [goals] * n_primitives
        else:
            values = goals

        return [DiscreteMovement(n_dimensions=n_dimensions, n_basis=n_parameters, goal=goal, duration=duration) for goal in values]


if __name__ == "__main__":

    # import matplotlib
    # matplotlib.use('macosx')

    import matplotlib.pyplot as plt

    field = MovementField()

    Y, Yd = [], []
    for t in np.linspace(0, 1.0, 100):
        field.step()
        Y.append(field.y)
        Yd.append(field.yd)


    Y, Yd = np.stack(Y), np.stack(Yd)

    field.write_file('fieldtest1.npz')
    new_field = MovementField.from_file('fieldtest1.npz')
