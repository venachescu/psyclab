#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from psyclab.movements.dmp.discrete import DiscreteMovementPrimitive, DiscreteCanonicalSystem, DiscreteTransformationSystem


class TestDiscreteCanonicalSystem(unittest.TestCase):

    def setUp(self):
        self.canonical = DiscreteCanonicalSystem()

    def tearDown(self):
        pass

    def test_step(self):
        self.canonical.step(0.01)
        self.assertTrue(self.canonical.x < 1.0)

    def test_state(self):
        self.canonical.reset()
        self.assertEqual(self.canonical.state, {'x': 1.0, 'v': 0.0, 'vd': 0.0, 'xd': 0.0})

    def test_reset(self):
        self.canonical.step(0.01)
        self.canonical.reset()
        self.assertEqual(self.canonical.x, 1.0)
        self.assertEqual(self.canonical.v, 0.0)
        self.assertEqual(self.canonical.xd, 0.0)
        self.assertEqual(self.canonical.vd, 0.0)


class TestDiscreteTransformationSystem(unittest.TestCase):

    def setUp(self):
        self.transformation = DiscreteTransformationSystem()
        pass

    def tearDown(self):
        pass


class TestDiscreteMovementPrimitive(unittest.TestCase):

    duration = 0.75
    n_dimensions = 2
    n_basis = 12
    goal = np.array([-1.5, 3.14])

    def setUp(self):
        self.primitive = DiscreteMovementPrimitive(
            duration=self.duration,
            n_dimensions=self.n_dimensions,
            n_basis=self.n_basis,
            goal=self.goal
        )

    def tearDown(self):
        pass

    def test_attributes(self):

        self.assertEqual(self.n_dimensions, self.primitive.n_dimensions)
        self.assertEqual(len(self.primitive.transformation[0]), self.n_basis)
        self.assertTrue((self.primitive.goal == self.goal).all())
        self.assertEqual(self.duration, self.primitive.duration)

    def test_reset(self):

        reset_state = {
            'x': 1.0, 'v': 0.0, 'xd': 0.0, 'vd': 0.0,
            'y': np.array([0., 0.]), 'yd': np.array([0., 0.]), 'ydd': np.array([0., 0.]),
            'z': np.array([0., 0.]), 'zd': np.array([0., 0.]),
            'g': self.goal, 'gd': np.array([0., 0.]), 'f': np.array([0., 0.])
        }

        self.primitive.step(0.1)
        self.assertNotEqual(self.primitive.canonical.x, 1.0)

        self.primitive.reset()
        for key, value in self.primitive.state.items():
            if isinstance(value, float):
                self.assertEqual(reset_state[key], value)
            else:
                self.assertTrue((reset_state[key] == value).all())

    def test_generate_trajectory(self):

        self.primitive.goal = self.goal

        # clean trajectory
        x, xd, xdd = self.primitive.generate_trajectory(y0=np.zeros(self.n_dimensions), duration=self.duration)

        # noisy trajectory
        noise = np.random.randn(self.n_dimensions, self.n_basis)
        y, yd, ydd = self.primitive.generate_trajectory(noise=noise, y0=np.zeros(self.n_dimensions), duration=self.duration)

        # trajectories are the same size, but different paths
        self.assertEqual(x.shape, y.shape)
        self.assertNotEqual(x.sum(), y.sum())

        # actual parameters are still zero
        self.assertEqual(np.sum([trans.weights.sum() for trans in self.primitive.transformation]), 0.0)

        # check it reaches the goal
        self.assertTrue(np.square(x[:, -1] - self.goal).sum() < 0.025)
        self.assertTrue(np.square(y[:, -1] - self.goal).sum() < 0.025)

    def test_basis_functions(self):

        for basis in self.primitive.basis_functions(self.duration):
            self.assertEqual(basis.shape, (self.duration / 0.01, self.n_basis))


if __name__ == "__main__":
    unittest.main()
