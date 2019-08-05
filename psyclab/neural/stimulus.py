#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/neural/stimulus.py
Vince Enachescu 2019
"""


import numpy as np

from inspect import getmembers


class NeuralInput:
    """
    Base class to represent some input stimulus to a neural simulation; each
    time this object is called, it steps forward an internal model which
    determines the intensity of signal over time.
    """

    def __init__(self, duration=None, delay=0.0, **kwargs):

        self.duration = duration
        self.delay = delay
        self._time = 0.0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._kernel = np.vectorize(self.kernel)

    def step(self, x, dt=0.01):

        self._time += dt
        if self._time <= self.delay or self.expired:
            return np.zeros_like(x)

        return self._kernel(x)

    def reset(self):
        self._time = 0.0

    def kernel(self, x):
        """
        Neural input kernel stub, this should be replaced with mathematical
        function describing its output

        :param x: Array of neuron tuning center values
        :return:
        """
        raise NotImplementedError

    @property
    def expired(self):
        if self.duration is not None:
            if self._time > (self.duration + self.delay):
                    return True
        return False

    @property
    def parameters(self):
        """ Dictionary of all numerical parameters for the neural field """

        names, _ = zip(*getmembers(self.__class__, predicate=lambda m: isinstance(m, (int, float))))
        return dict((name, getattr(self, name)) for name in names)

    def __call__(self, x, dt=0.01):
        return self.step(x, dt=dt)


class StepStimulus(NeuralInput):
    """
    Step function of neural input - for a number of input neurons, the signal
    is at full strength and zero everywhere else.
    """

    center = 90
    width = 10
    amplitude = 1.0

    def kernel(self, n):
        if abs(n - self.center) < self.width // 2:
            return self.amplitude
        return 0


class GaussianStimulus(NeuralInput):
    """
    Step function of neural input - for a number of input neurons, the signal
    is at full strength and zero everywhere else.
    """

    center = 90
    width = 10
    amplitude = 1.0

    def kernel(self, n):
        return np.exp(-np.square(n - self.center) / (2 * np.square(self.sigma))) * self.amplitude

    @property
    def sigma(self):
        return self.width / np.sqrt(2 * np.log(2))


class StepInhibition(NeuralInput):
    """
    Step function of neural input - for a number of input neurons, the signal
    is at full strength and zero everywhere else.
    """

    center = 90
    width = 10
    amplitude = 1.0
    slope = 0.0

    def kernel(self, n):

        if abs(n - self.center) < self.width // 2:
            return 0

        if n > self.center:
            d = abs(n - self.center + self.width // 2)
        else:
            d = abs(n - self.center - self.width // 2)

        return min([-self.amplitude + self.slope * d, 0])


class InhibitoryStimulus(NeuralInput):
    """
    An inhibitory stimulus that is zero at the center and then sweeps down in
    a curve to a constant value everywhere else
    """

    amplitude = 1.0
    center = 0

    # omega - (0.01, 0.3)
    omega = 0.02
    ceiling = 1.0

    def kernel(self, n):
        return self.amplitude * (np.minimum(np.square(1.0 / np.sinh(self.omega * (n - self.center + 1.0e-9))), self.ceiling) - self.ceiling)


class InhibitoryMask(NeuralInput):

    center = 0
    width = 10
    amplitude = 1.0

    alpha1 = 0.1
    alpha2 = 0.2

    sigma1 = 1.0
    sigma2 = 1.0

    def kernel(self, n):

        if np.abs(n - self.center) < self.width / 2:
            return 0.0

        return -(
            np.exp(-np.square(self.alpha1 * (n - self.center)) / self.sigma1) -
            np.exp(-np.square(self.alpha2 * (n - self.center)) / self.sigma2)
        )


class CosineTuning(NeuralInput):
    """
    Cosine tuning of neural input
    """

    center = 0
    width = 90          # width in radians
    amplitude = 1.0

    def kernel(self, n):

        return self.amplitude * np.cos((n - self.center) * np.pi / self.width)


class NeuralInputBuffer:
    """
    Input buffer for simulated neural stimulus, used to delay input signals by a set amount of time
    """

    def __init__(self, neurons, length=0.5, time_step=0.01):

        self.shape = (neurons, int(length / time_step))
        self.index = -1
        self.time_step = time_step
        self._buffer = np.zeros(self.shape)

    def step(self, inputs=None, time_step=0.01):

        if inputs is None:
            inputs = np.zeros(self.shape[0])

        steps = int(time_step / self.time_step)
        index = np.mod(self.index + steps, self.shape[1])
        indices = np.mod(np.arange(index, index + steps), self.shape[1])
        output = np.sum(self._buffer[:, indices], axis=1)
        self._buffer[:, indices] = np.zeros((self.shape[0], steps))
        self._buffer[:, self.index] = inputs
        self.index = index
        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # neurons = np.arange(181)
    # stim1 = GaussianStimulus()
    # stim2 = StepStimulus()
    # stim3 = StepInhibition(slope=0.01)
    # v1 = stim1(neurons)
    # v2 = stim2(neurons)
    # v3 = stim3(neurons)
    # plt.plot(v1)
    # plt.plot(v2)
    # plt.plot(v3)
    # plt.show()
    mask = InhibitoryStimulus(omega=0.05, center=30)
    print(mask.parameters)
    x = np.arange(-45, 45)
    # plt.plot(x, mask(x))
    # plt.show()
