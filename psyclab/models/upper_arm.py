#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/models/upper_arm.py
Vince Enachescu 2019
"""

import numpy as np

from psyclab.models.arm2d import Arm2DModel
from psyclab.muscle.brown import BrownMuscleModel


class UpperArm(Arm2DModel):
    """
    Model of the muscles and limbs of the upper arm, 2 joints and 6 muscles.

    Brown - Loeb Muscle Model
    Simple 2D Planar Arm Model

    Muscles

    1) Elbow flexors
        ('biceps, brachialis, brachioradialis'), ('elbow flexors'),
    2) Elbow extensors
        ('tricepts lateral, anconeus'), ('elbow extensors'),
    3) Shoulder flexors
        ('deltoid anterior'), ('shoulder flexors'),
    4) Shoulder extensors
        ('deltoid posterior'), ('shoulder extensors'),
    5) Biarticulate flexors
        (), ('biarticulate flexors'),
    6) Biarticulate extensors
        (), ('biarticulate extensors')

    [1]
    [2]
    [3]
    """

    radius_humerus = 0.025
    radius_ulna = 0.025
    radius_elbow = 0.02
    radius_shoulder = 0.04

    # (origin, insertion)
    lengths = (
        (0.14, radius_elbow),
        (0.14, radius_elbow),
        (radius_shoulder, 0.10),
        (radius_shoulder, 0.10),
        (radius_shoulder, radius_elbow),
        (radius_shoulder, radius_elbow)
    )

    # limits in normalized muscle units
    limits = (
        (0.6, 1.1),
        (0.8, 1.24),
        (0.7, 1.2),
        (0.7, 1.1),
        (0.6, 1.1),
        (0.85, 1.2)
    )

    pcsa = (18, 14, 22, 12, 5, 10)

    def __init__(self, dt=0.001, **kwargs):

        Arm2DModel.__init__(self, dt=dt, **kwargs)

        lengths = self.muscle_lengths(self.q)
        self.muscles = [
            BrownMuscleModel(ml, pcsa, limits)
            for ml, pcsa, limits in zip(lengths, self.pcsa, self.limits)
        ]

    def step(self, neural_input, dt):

        lengths = self.muscle_lengths(self.q)

        for (muscle, ml, u) in zip(self.muscles, lengths, neural_input):
            muscle.step(ml, u)

        u = self.muscle_torque
        print('torque \t', u)
        return self.apply_torque(self.muscle_torque, dt)

    @property
    def muscle_torque(self):
        return self.moment_arms(self.q).T @ self.muscle_tension

    @property
    def muscle_tension(self):
        return np.array([muscle.T for muscle in self.muscles])

    def muscle_lengths(self, q):

        ml1 = (np.linalg.norm(self.lengths[0]) - 2 * np.prod(self.lengths[0]) +
         ((np.pi / 4.0 - q[1]) * self.radius_elbow if q[1] <= np.pi / 4.0 else 0.0))

        ml2 = (np.linalg.norm(self.lengths[1]) - 2 * np.prod(self.lengths[1]) +
         ((np.pi / 4.0 - q[1]) * self.radius_elbow if q[1] <= np.pi / 4.0 else 0.0))


        phi1 = np.pi / 3.0
        if q[0] < np.pi / 3.0:
            ml3 = (np.linalg.norm(self.lengths[2]) + (np.pi / 3.0 - q[0]) * self.radius_shoulder)
        else:
            ml3 = np.linalg.norm(self.lengths[2]) - 2 * np.prod(self.lengths[2]) * np.cos(2 * np.pi / 3 - q[0])

        phi2 = q[0] + np.pi / 6
        if phi2 >= np.pi / 2.0:
            ml4 = np.linalg.norm(self.lengths[3]) + self.radius_shoulder * (phi2 - np.pi / 2)
        else:
            ml4 = np.linalg.norm(self.lengths[3]) - 2 * np.prod(self.lengths[3]) * np.cos(phi2)

        # biarticulate flexors
        ml5 = self.l1 - q[0] * self.lengths[4][0] - q[1] * self.lengths[4][1]

        # biarticulate extensors
        ml6 = self.l1 + q[0] * self.lengths[5][0] + q[1] * self.lengths[5][1]

        return np.array((ml1, ml2, ml3, ml4, ml5, ml6))

    @staticmethod
    def moment_arms(q):
        """
        Compute the moment arm matrix

        :param q: angles of the shoulder and elbow joints
        """

        m1 = m52 = 4.0 - 2.0 * np.cos(q[1])
        m51 = 4.0 - 2.0 * np.cos([0])
        m3 = 5.0 - 2.0 * np.cos(q[0])
        return np.matrix([
            [0.0,    m1],
            [0.0,  -2.0],
            [m3,    0.0],
            [-5.0,  0.0],
            [m51,   m52],
            [-4.0, -2.5]
        ]) / 100.0

    @property
    def T(self):
        return np.array([muscle.T for muscle in self.muscles])

    @property
    def n_muscles(self):
        return len(self.muscles)


if __name__ == "__main__":

    import matplotlib
    matplotlib.use('MacOSX')

    import matplotlib.pyplot as plt
    from matplotlib import animation

    arm2d = UpperArm()
    dt = 0.01

    # t = 0.0
    u = np.abs(np.random.randn(6)) * 0.0001

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=4, mew=5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        """initialize animation"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        """perform animation step"""
        global arm2d, dt
        arm2d.step(u, dt)
        # print(arm2d.muscle_lengths(arm2d.q))

        line.set_data(*arm2d.position())
        time_text.set_text('time = %.2f' % arm2d.t)
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=range(2), interval=25, blit=True, init_func=init)
    fig.show()
