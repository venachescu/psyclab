
import unittest
import numpy as np

from psyclab.neural import stimulus


class TestStimuli(unittest.TestCase):

    n = np.arange(-1, 2)

    def test_step_stimuli(self):
        stim = stimulus.StepStimulus(center=0, width=2, duration=0.1)
        self.assertEqual((stim.step(self.n) - np.array((0.0, 1.0, 0.0))).sum(), 0.0)

    def test_delay(self):
        stim = stimulus.StepStimulus(center=0, width=2, duration=0.1, delay=0.1)
        self.assertAlmostEqual(np.sum(stim.step(self.n, dt=0.1)), 0.0)
        self.assertAlmostEqual(np.sum(stim.step(self.n, dt=0.1)), 1.0)
        self.assertAlmostEqual(np.sum(stim.step(self.n, dt=0.1)), 0.0)

    def test_duration(self):
        stim = stimulus.StepStimulus(center=0, width=2, duration=0.1)
        self.assertFalse(stim.expired)
        self.assertAlmostEqual(np.sum(stim.step(self.n, dt=0.1)), 1.0)
        self.assertAlmostEqual(np.sum(stim.step(self.n, dt=0.1)), 0.0)
        self.assertTrue(stim.expired)


if __name__ == "__main__":
    test = TestStimuli()
    test.test_step_stimuli()
    test.test_delay()
    test.test_duration()
