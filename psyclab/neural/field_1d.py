#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-Dimensional Dynamic Neural Field


References
----------
Christopoulos, V., Bonaiuto, J., & Andersen, R. A. (2015).
A Biologically Plausible Computational Theory for Value Integration and Action Selection
in Decisions with Competing Alternatives.
PLoS Computational Biology, 11(3), e1004104.
http://doi.org/10.1371/journal.pcbi.1004104.s007

"""

import numpy as np

from inspect import getmembers
from scipy.stats import mode


class NeuralField1D(object):
    """
    Dynamic Neural Field

    This class models a set of dynamical equations simulating a 'neural field'.
    """

    tau = 20                   # time constant of dynamic field
    tau_trace_build = 500      #
    tau_trace_decay = 2000     # time constants of memory traces

    local_excitation = 10.0
    local_excitation_sigma = 5

    local_inhibition = 25.0
    local_inhibition_sigma = 40
    global_inhibition = 0.8

    trace_gain = 0.0
    trace_sigma = 5.0

    noise_gain = 5.0            # noise levels
    # noise_gain = 1.0
    noise_sigma = 15.0          # width of the noise kernel

    threshold = 0.5
    resting_level = -5.0        # steepness parameter of sigmoid function
    beta_output = 1             # resting level

    def __init__(self, neurons=181, centers=None, **kwargs):

        # number of neurons in the field (should be odd)
        if neurons % 2 == 0:
            neurons += 1
            # raise Warning('number of neurons in field must be odd')
        self._n = neurons

        if centers is None:
            self._centers = np.arange(-self.half_field, self.half_field + 1)
        else:
            self._centers = centers

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.noise = np.zeros(self.n)
        self.activity = np.zeros(self.n)
        self.trace = np.zeros(self.n)
        self.output = np.zeros(self.n)

        self.noise_kernel = None
        self.interaction_kernel = None
        self.trace_kernel = None

        self.compute_kernels()

    @property
    def n(self):
        """ Number of neurons in the field """
        return self._n

    @property
    def centers(self):
        """ Values of the center of each neurons tuning curve, i.e. the value they represent """
        return self._centers

    @property
    def half_field(self):
        """ Half of the number of neurons in the field """
        return self._n // 2

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

    def compute_kernels(self):
        """ Compute the kernels used to convolve neuronal input and output """

        # self.kSize_uu = min(round(3 * max(self.sigma_exc, self.sigma_inh)), self.half_field)
        self.interaction_kernel = difference_of_gaussians(
            self.centers, self.local_excitation, self.local_excitation_sigma,
            self.local_inhibition, self.local_inhibition_sigma
        ) - self.global_inhibition

        self.trace_kernel = self.trace_gain * gaussian(self.centers, 0, self.trace_sigma)

        if self.noise_sigma > 0:
            self.noise_kernel = gaussian(self.centers, 0, self.noise_sigma)
        else:
            self.noise_kernel = None

    def step(self, stimulus=None, time_step=0.01):
        """ Step forward the dynamic neural field model """

        if stimulus is None:
            stimulus = np.zeros(self.n)

        # calculation of field outputs
        self.output = sigmoid(self.activity, self.beta_output, 0)

        # get endogenous input to fields by convolving outputs with interaction kernels
        cv_output = convolve(self.interaction_kernel, self.output)
        cv_trace = convolve(self.trace_kernel, self.trace)

        self.activity += self.compute_field_update(self.activity, stimulus, cv_output, cv_trace, self.noise)
        self.trace += self.memory_trace_update(self.activity, self.trace, self.output)
        self.noise = self.noise_update(gain=self.noise_gain)

    def noise_update(self, gain=1.0):
        """ Generate new, spatially correlated noise """

        noise = gain * np.random.randn(self.n)
        if self.noise_kernel is not None:
            return convolve(self.noise_kernel, noise)

        return noise

    def compute_field_update(self, field, stimulus, cv_output, cv_trace, noise):
        """ Compute the change in neural field activity (its derivative) """

        return (-field + self.resting_level + stimulus + cv_output + cv_trace) / self.tau + noise

    def memory_trace_update(self, field, trace, output):
        """ Update memory trace (only if there is activity in the field) """

        active = self.activity > 0
        if not any(active):
            return (-trace + output) / self.tau_trace_build - trace / self.tau_trace_decay

        return (-trace + output) * active / self.tau_trace_build + \
            (-trace) * (1.0 - active) / self.tau_trace_decay

    def reset(self):
        """ Reset all the dynamical variables to their initial state """

        self.noise = np.zeros(self.n)
        self.activity = np.zeros(self.n)
        self.trace = np.zeros(self.n)
        self.output = np.zeros(self.n)

    @classmethod
    def simulation(cls, *stimuli, duration=2.0, time_step=0.01, plots=False, seed=None, **kwargs):
        """
        Run a simulation of the 1d neural field, supplying whatever stimuli and
        parameters desired.

        """

        model = cls(time_step=time_step, **kwargs)

        steps = int(duration / time_step)
        results = {
            'time': time_step * np.arange(steps),
            'attention': np.zeros(steps),
            'active': np.zeros(steps),
            'activity': np.zeros((model.n, steps)),
            'output': np.zeros((model.n, steps)),
            'stimulus': np.zeros((model.n, steps)),
            'trace': np.zeros((model.n, steps)),
            'parameters': dict(model.parameters)
        }

        # include parameters from stimuli
        results['parameters'].update({
            f'{name}{i:02d}': value for i, stimulus in enumerate(stimuli)
            for name, value in stimulus.parameters.items()
        })

        if seed is not None:
            np.random.seed(seed)
            results['parameters']['seed'] = seed

        for i, t in enumerate(results['time']):

            if len(stimuli):
                stimulus = np.sum([stimulus(model.centers, dt=time_step) for stimulus in stimuli], axis=0)
            else:
                stimulus = np.zeros(model.n)

            model.step(stimulus=stimulus, time_step=time_step)

            if model.active_neurons is None:
                active = 0.0
            else:
                active = len(model.active_neurons)

            results['attention'][i] = model.activity_center
            results['active'][i] = active
            results['activity'][:, i] = model.activity
            results['output'][:, i] = model.output
            results['trace'][:, i] = model.trace
            results['stimulus'][:, i] = stimulus

        results['statistics'] = cls.compute_statistics(results)

        if not plots:
            return results

        fig, axs = cls.plot_results(results, model, time_step)
        return results, (fig, axs)

    @staticmethod
    def compute_statistics(results):
        """
        Compute important features like reaction time from simulation results

        :param results:
        :param values:
        :return:
        """

        stats = {}
        indices, = np.where(results['active'])
        if len(indices):
            stats['reaction'] = results['time'][min(indices)]
            stats['first_active'] = int(results['attention'][min(indices)])
        else:
            stats['first_active'] = None

        stats['total_active'] = results['active'].sum()
        stats['most_active'] = int(mode(results['attention'][np.nonzero(results['attention'])]).mode)
        return stats

    @staticmethod
    def plot_results(results, model, time_step, color_map='viridis'):
        """
        Plot the results of a neural field simulation using matplotlib

        :param results:
        :param values:
        :param model:
        :param time_step:
        :return:
        """

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axs[0, 0].set_title('Stimulus')
        axs[0, 0].imshow(results['stimulus'], cmap=color_map)
        axs[0, 0].set_yticks(np.arange(0, model.n, 15))
        axs[0, 0].set_yticklabels(model.centers[np.array(axs[0, 0].get_yticks(), dtype=np.int32)])

        axs[0, 1].set_title('Activity')
        axs[0, 1].imshow(results['activity'], cmap=color_map)
        axs[0, 1].set_yticks(np.arange(0, model.n, 15))
        axs[0, 1].set_yticklabels(model.centers[np.array(axs[0, 1].get_yticks(), dtype=np.int32)])

        indices, = np.where(results['active'] > model.threshold)
        if len(indices):
            centers = np.argmax(results['output'][:, indices], axis=0)
            axs[0, 1].scatter(indices, centers, results['active'][indices] * 2, marker='.', color=(0, 0, 0))

        axs[1, 0].set_title('Trace')
        axs[1, 0].imshow(results['trace'], cmap=color_map)
        axs[1, 0].set_xticklabels(np.array(axs[1, 0].get_xticks()) * time_step)

        axs[1, 1].set_title('Output')
        axs[1, 1].imshow(results['output'], cmap=color_map)
        axs[1, 1].set_xticklabels(np.array(axs[1, 1].get_xticks()) * time_step)
        axs[1, 1].plot(results['attention'], linestyle='--', color='white')

        fig.set_tight_layout('tight')
        return fig, axs


def gaussian(x, mu=0.0, sigma=1.0, normed=True):
    """ A gaussian kernel """

    y = np.exp(-np.square(x - mu) / (2.0 * np.square(sigma))) \
        * (1.0 / (sigma * np.sqrt(2 * np.pi)))

    if normed:
        return y / y.sum()

    return y / y.max()


def difference_of_gaussians(x, gain1=1.0, sigma1=1.0, gain2=0.5, sigma2=4.0):
    """ A difference of gaussians """
    return gain1 * gaussian(x, 0, sigma1, normed=True) - gain2 * gaussian(x, 0, sigma2, normed=True)


def sigmoid(x, beta=0.0, x0=1.0, amplitude=1.0):
    """ A sigmoid kernel """
    return amplitude / (1.0 + np.exp(-beta * (x - x0)))


def convolve(kernel, signal):
    """ Convolve a signal with itself using a specified kernel """
    return np.convolve(kernel, np.pad(signal, len(kernel) // 2, 'reflect'), 'valid')


if __name__ == "__main__":

    # dnf = NeuralField1D()
    # dnf.initialize()
    stats = NeuralField1D.simulation()
