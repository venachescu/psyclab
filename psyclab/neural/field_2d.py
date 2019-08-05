#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Neural Field model

based on code and examples from Nicolas P. Rougier and Eric J. Nichols

References
----------
[1] Rougier, N. P. (2005). Dynamic neural field with local inhibition.
Biological Cybernetics, 94(3), 169–179. http://doi.org/10.1007/s00422-005-0034-8
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


class NeuralField(object):

    def __init__(self, neurons=512, time_step=0.01):

        self._n = neurons

        self._size = 30.0
        self._gamma = 1.0
        self._axon_speed = 500.0

        self._values = np.zeros((self._n, self._n))
        self._input = np.zeros((self._n, self._n))
        self._voltage = np.zeros((self._n, self._n))

        self._steps = 0
        self._elapsed = 0.0
        self._time_step = time_step
        self._activity = np.zeros((1, self._n, self._n))

    def initialize(self, size):

        a, b = np.meshgrid(
            np.arange(-size / 2.0, size / 2.0, size / float(self._n)),
            np.arange(-size / 2.0, size / 2.0, size / float(self._n)))
        x = np.sqrt(a**2 + b**2)
        self._values = x

        self._noise_kernel = np.exp(-(a**2/32.0 + b**2/32.0))/(np.pi*32.0) * 0.1 * np.sqrt(self._time_step)
        self._kernel = -4 * np.exp(-x / 3) / (18 * np.pi)
        self._kerneli = self.precompute_kernel(self._n, size, self._time_step, self._axon_speed)
        self._nrings = len(self._kerneli)

        self._activity = np.stack([fft2(self.S),] * self._nrings)
        # self._rate = fftshift(fft2(ifftshift(self.S))).real

    def precompute_kernel(self, n, size, time_step, axon_speed):

        radius = np.sqrt((n / 2.0)**2 + (n / 2.0)**2)
        self._synaptic = size ** 2 / float(n ** 2)

        # width of a ring in # of grid intervals
        width = max(1.0, self._axon_speed * time_step * n / size)
        n_rings = 1 + int(radius / width)

        def disc(step, n):
            def distance(x, y):
                return np.sqrt((x - n // 2)**2 + (y - n // 2)**2)
            D = np.fromfunction(distance, (n, n))
            return np.where(D < (step * width), True, False).astype(np.float32)

        # Generate 1+int(d/r) rings
        disc1 = disc(1, n)
        L = [disc1 * self.K]
        for i in range(1, n_rings):
            disc2 = disc(i + 1, n)
            L.append(((disc2 - disc1) * self.K))
            disc1 = disc2

        # Precompute Fourier transform for each kernel ring since they're
        # only used in the Fourier domain
        Ki = np.zeros((n_rings, n, n))  # self.Ki is our kernel in layers in Fourier space
        for i in range(n_rings):
            Ki[i, :, :] = np.real(fftshift(fft2(ifftshift(L[i]))))

        return Ki

    def step(self, dt=None):

        dt = dt or self._time_step
        self._steps += int(round(dt / self._time_step))
        self._elapsed += dt

        L = np.sum([k * u for k, u in zip(self.Ki, self.U)], axis=0)
        L = self._synaptic * (fftshift(ifft2(ifftshift(L)))).real

        e = np.random.normal(0, 1.0, (self._n, self._n)) * self._noise_kernel
        dV = dt / self._gamma * (-self.V + L + self.I) + e

        self.V += dV
        self._activity = np.roll(self._activity, 1, axis=0)
        self._activity[0] = fft2(self.S)
        # self.U = fftshift(fft2(ifftshift(self.S)))

    def firing_rate(self, V=None):
        if V is None:
            V = self.V
        S0    = 1.0 # S: maximum frequency
        alpha = 10000.0 # α: steepness at the threshold
        theta = 0.005 # θ: firing threshold
        return S0 / (1.0 + np.exp(-1*alpha*(V-theta)))

    @property
    def V(self):
        return self._voltage

    @V.setter
    def V(self, value):
        self._voltage = value

    @property
    def I(self):
        return self._input

    @I.setter
    def I(self, value):
        self._input = value

    @property
    def K(self):
        return self._kernel

    @property
    def Ki(self):
        return self._kerneli

    @property
    def S(self):
        return self.firing_rate(self.V)

    @property
    def U(self):
        return self._rate

    @U.setter
    def U(self, value):
        self._rate = [value] + self._rate[:-1]

    @property
    def active_neurons(self):
        """ Indices of neurons with activity above threshold """
        indices, = np.where(self.output > self.threshold)
        if not len(indices):
            return
        return indices

    @property
    def activity_center(self):
        """ Index of the neuron at the center of highest activity density """
        indices = self.active_neurons
        if indices is None:
            return int(np.argmax(self.activity))
        return int(np.median(indices))

    @property
    def parameters(self):
        """ Dictionary of all numerical parameters for the neural field """

        names, _ = zip(*getmembers(self.__class__, predicate=lambda m: isinstance(m, (int, float))))
        return dict((name, getattr(self, name)) for name in names)


if __name__ == "__main__":

    '''This is the input from external source, I.
    You can delete/add/change variables but you must initialize an I that uses x.'''
    # Gamma = 20.0
    # sigma = 5.65685425
    # I = Gamma * (np.exp(-1 * x**2 / sigma**2) / (sigma**2 * np.pi))
    nf = NeuralField()
    nf.initialize(64)
