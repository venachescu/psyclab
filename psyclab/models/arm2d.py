#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/models/arm2d.py
Vince Enachescu 2019
"""

import math
import numpy as np


class Arm2DModel(object):
    """
    2 arm, 2 fast, 2 furious
    simple 2 joint, 2 link arm muscle
    inherits parameters from sl's arm2D

    [1]
    """

    # initial joint angles
    init_q = (0.75613, 1.8553)

    # length of links (upper arm, forearm)
    length = (0.31, 0.34)

    # mass of arm segments
    mass = (1.93, 1.52)

    # inertia of segments
    inertia = (0.025, 0.075)

    # distance along segment from joint to center of mass
    distance = (0.11, 0.16)

    # matrix of joint frictions
    joint_friction_matrix = np.matrix([
        [0.05, 0.025],
        [0.025, 0.05]
    ])

    def __init__(self, dt=0.01, init_q=None):
        """
        dt float: the timestep for simulation
        """

        self.t = 0.0
        self.dt = dt

        # create mass matrices at COM for each link
        self.q = np.array(self.init_q)
        self.qd = np.zeros(self.n_dofs)
        self.u = np.zeros(self.n_dofs)

        self.n_cartesian = 2

    def positions(self, q=None, z=False):
        """
        Compute cartesian coordinates for the joint positions and hand

        :param q:
        :return:
        """

        q = self.q if q is None else q
        x = np.cumsum([0, self.l1 * np.cos(q[0]), self.l2 * np.cos(q[0] + q[1])])
        y = np.cumsum([0, self.l1 * np.sin(q[0]), self.l2 * np.sin(q[0] + q[1])])

        if not z:
            return np.array([x, y])

        return np.concatenate((np.array([x, y]), np.zeros((1, 3))))

    def step(self, u=None, dt=None):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly.

        u np.array: the control signal to apply
        dt float: the timestep
        """

        dt = dt if dt is not None else self.dt
        u = u if u is not None else np.zeros(self.n_dofs)

        qdd = np.linalg.inv(self.M) @ (u - self.C - self.B @ self.qd).T
        self.qd += np.array(qdd).flatten() * dt
        self.q += self.qd * dt
        self.t += dt

    def inertia_matrix(self, q):
        """

        :return:
        """
        a1 = self.i1 + self.i2 + self.m2 * (self.l1 ** 2)
        a2 = self.m2 * self.l1 * self.s2
        a3 = self.i2
        return np.matrix([
            [a1 + 2 * a2 * np.cos(q[1]), a3 + a2 * np.cos(q[1])],
            [a3 + a2 * np.cos(q[1]), a3]
        ])

    def coriolis_vector(self, q, qd):
        """

        :param q:
        :param qd:
        :return:
        """
        return np.array((-qd[1] * (2 * qd[0] + qd[1]), qd[0])) * self.m2 * (self.l1 ** 2) * np.sin(q[1])

    def reset(self, q=None):
        """

        :param q:
        :return:
        """

        if q is None:
            q = self.init_q

        assert len(q) == self.n_dofs
        self.q = np.copy(q)
        self.qd = np.zeros(self.n_dofs)
        self.u = np.zeros(self.n_dofs)
        self.t = 0.0

    @property
    def M(self):
        return self.inertia_matrix(self.q)

    @property
    def B(self):
        return self.joint_friction_matrix

    @property
    def C(self):
        return self.coriolis_vector(self.q, self.qd)

    @property
    def x(self):
        return self.positions()[:, -1]

    @property
    def L(self):
        return np.array(self.length)

    @property
    def m1(self):
        """ Mass of the upper arm (from shoulder to elbow) """
        return self.mass[0]

    @property
    def m2(self):
        """ Mass of the forearm (from elbow to wrist) """
        return self.mass[1]

    @property
    def l1(self):
        """ Length of upper arm """
        return self.length[0]

    @property
    def l2(self):
        """ Length of forearm """
        return self.length[1]

    @property
    def i1(self):
        """ """
        return self.inertia[0]

    @property
    def i2(self):
        """ """
        return self.inertia[1]

    @property
    def s1(self):
        """ """
        return self.distance[0]

    @property
    def s2(self):
        """ """
        return self.distance[1]

    @property
    def n_dofs(self):
        return len(self.q)


if __name__ == "__main__":

    import matplotlib
    matplotlib.use('MacOSX')

    import matplotlib.pyplot as plt
    from matplotlib import animation

    arm2d = Arm2DModel(dt=0.01)

    u = np.random.randn(2)
    print(u)

    arm2d.step()

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
        global arm2d
        arm2d.step(u=np.random.rand(arm2d.n_dofs))

        line.set_data(*arm2d.positions())
        time_text.set_text('time = %.2f' % arm2d.t)
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=None, interval=25, blit=True, init_func=init)
    fig.show()
