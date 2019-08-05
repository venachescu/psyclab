#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from psyclab.movements.dmp.rhythmic import RhythmicMovementPrimitive, RhythmicCanonicalSystem, RhythmicTransformationSystem


class TestRhythmicCanonicalSystem(unittest.TestCase):

    def setUp(self):
        self.canonical = RhythmicCanonicalSystem()

    def tearDown(self):
        pass

    def test_step(self):
        self.canonical.step(0.01)
        self.assertTrue(self.canonical.x < 1.0)
        self.assertTrue(self.canonical.v != 0.0)

    def test_state(self):
        self.assertEqual(self.canonical.state, {'x': 1.0, 'v': 0.0, 'vd': 0.0, 'xd': 0.0})

    def test_reset(self):
        self.canonical.step(0.01)
        self.canonical.reset()
        self.assertEqual(self.canonical.x, 1.0)
        self.assertEqual(self.canonical.v, 0.0)
        self.assertEqual(self.canonical.xd, 0.0)
        self.assertEqual(self.canonical.vd, 0.0)


class TestRhythmicTransformationSystem(unittest.TestCase):

    def setUp(self):
        self.transformation = RhythmicTransformationSystem()
        pass

    def tearDown(self):
        pass


class TestRhythmicMovementPrimitive(unittest.TestCase):

    duration = 0.75
    n_dimensions = 2
    n_basis = 12
    goal = [-1.5, 3.14]

    def setUp(self):
        self.primitive = RhythmicMovementPrimitive(
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
        self.assertEqual(self.primitive.goal, self.goal)

    def test_generate_trajectory(self):

        data0 = self.primitive.generate_trajectory()

        noise = np.random.randn(self.n_dimensions, self.n_basis)
        data1 = self.primitive.generate_trajectory(noise=noise)

        self.assertEqual(data0['t'].sum(), data1['t'].sum())
        self.assertNotEqual(data0['y'].sum(), data1['y'].sum())
        self.assertEqual(np.sum([trans.weights.sum() for trans in self.primitive.transformation]), 0.0)

    def test_basis_functions(self):

        for basis in self.primitive.basis_functions(self.duration):
            self.assertEqual(basis.shape, (self.duration / 0.01, self.n_basis))


if __name__ == "__main__":
    unittest.main()
