#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual field decision model
"""

import numpy as np

# from scipy.stats import mode

from psyclab.neural.stimulus import GaussianStimulus, InhibitoryStimulus, NeuralInputBuffer, CosineTuning
from psyclab.neural.field_1d import NeuralField1D


class DualFieldDecisionModel:

    def __init__(
            self, stimuli=None, neurons=91, time_step=0.01,
            arousal_gain=1.0, inhibition_gain=0.1,
            attention_tau=10.0,
            **kwargs
    ):
        """
        Dual field decision model

        :param neurons:
        """

        self.reflexive = NeuralField1D(neurons=neurons, **kwargs)
        self.reflective = NeuralField1D(neurons=self.reflexive.n, centers=self.reflexive.centers, **kwargs)
        self.neurons = self.reflexive.n

        self.reflexive_input = NeuralInputBuffer(self.neurons, length=0.02, time_step=time_step)
        self.reflective_input = NeuralInputBuffer(self.neurons, length=0.1, time_step=time_step)

        if isinstance(stimuli, (tuple, list)):
            self.stimuli = dict(zip(range(len(stimuli)), stimuli))
        else:
            self.stimuli = stimuli or {}

        self.frontal_inhibition = InhibitoryStimulus(omega=0.07)
        self.frontal_bias = GaussianStimulus(amplitude=10.0)
        # self.frontal_bias = CosineTuning(amplitude=1.0)

        self.stimulus = np.zeros(self.neurons)
        self.inhibition = np.zeros(self.neurons)
        self.bias = np.zeros(self.neurons)

        # index of neuron at maximum attention
        self.attention = 0
        self.attention_tau = 10.0

        # index of neuron at maximum activation in reflexive field
        self.intention = 0
        # variance of activity in the reflexive field
        self.conflict = 0

        self.arousal_gain = arousal_gain
        self.inhibition_gain = inhibition_gain

        # level of psychological arousal (stimulus to reflective)
        self.arousal = 0.0

    def step_stimulus(self, time_step=0.01):

        output = np.sum([stimulus.step(self.centers, dt=time_step) for stimulus in self.stimuli.values()], axis=0)
        return output
        # optionally pop off old expired stimuli
        # for name in self.stimuli.keys():
        #     if self.stimuli[name].expired:
        #         self.stimuli.pop(name)

    def step_reflexive(self, time_step=0.01):

        inputs = self.reflexive_input.step(self.stimulus, time_step=time_step)
        self.reflexive.step(self.inhibition + inputs, time_step=time_step)
        self.conflict = np.var(self.reflexive.activity)

        # self.frontal_inhibition.omega = (1.0 / (1.0 + np.exp(-self.conflict / 1000.0))) * 0.3 + 0.01
        self.frontal_inhibition.omega = np.exp(-self.conflict / 5.0) * 0.3 + 0.01
        self.intention = self.reflexive.activity_center
        return np.mean(self.reflexive.activity)

    def step_reflective(self, time_step=0.01):

        self.bias = self.frontal_bias.step(self.centers, dt=time_step)
        inputs = self.reflective_input.step(self.stimulus, time_step=time_step) * self.arousal_gain

        self.reflective.step(stimulus=inputs + self.bias, time_step=time_step)
        self.attention += (self.centers[self.reflective.activity_center] - self.attention) / self.attention_tau * self.reflective.output[self.reflective.activity_center]

        # step the frontal inhibition forward
        # self.frontal_inhibition.amplitude = self.arousal
        self.frontal_inhibition.center = self.attention
        return self.frontal_inhibition.step(self.centers, dt=time_step) * self.inhibition_gain

    def step(self, time_step=0.01):
        """

        :param time_step:
        :return:
        """

        self.stimulus = self.step_stimulus(time_step=time_step)

        self.inhibition = self.step_reflective(time_step=time_step)

        self.arousal = self.step_reflexive(time_step=time_step)

    def reset(self):

        self.reflexive_input.reset()
        self.reflective_input.reset()

        self.reflexive.reset()
        self.reflective.reset()

        self.stimulus = np.zeros(self.neurons)
        self.inhibition = np.zeros(self.neurons)
        self.bias = np.zeros(self.neurons)

        self.attention = 0
        self.intention = 0
        self.conflict = 0
        self.arousal = 0.0

    @property
    def centers(self):
        return self.reflexive.centers

    @property
    def parameters(self):

        return self.reflexive.parameters, self.reflective.parameters

    @classmethod
    def simulation(cls, *stimuli, duration=2.0, time_step=0.01, plots=False, seed=None, **kwargs):
        """
        Run a simulation of the 1d neural field, supplying whatever stimuli and
        parameters desired.

        """

        model = cls(stimuli=stimuli, neurons=181, time_step=time_step, **kwargs)
        model.frontal_bias.center = -40
        model.inhibition_gain = 5.0

        steps = int(duration / time_step)
        results = {
            'time': time_step * np.arange(steps),
            'intention': np.zeros(steps),
            'attention': np.zeros(steps),
            'active': np.zeros(steps),
            'arousal': np.zeros(steps),
            'conflict': np.zeros(steps),
            'reflexive': {
                'activity': np.zeros((model.neurons, steps)),
                'output': np.zeros((model.neurons, steps)),
            },
            'reflective': {
                'activity': np.zeros((model.neurons, steps)),
                'output': np.zeros((model.neurons, steps)),
            },
            'stimulus': np.zeros((model.neurons, steps)),
            'inhibition': np.zeros((model.neurons, steps)),
            'bias': np.zeros((model.neurons, steps)),
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

            model.step(time_step=time_step)

            if model.reflexive.active_neurons is None:
                active = 0.0
            else:
                active = len(model.reflexive.active_neurons)

            results['intention'][i] = model.intention
            results['attention'][i] = model.reflective.activity_center
            results['active'][i] = active
            results['arousal'][i] = model.arousal
            results['conflict'][i] = model.conflict
            results['stimulus'][:, i] = model.stimulus
            results['inhibition'][:, i] = model.inhibition
            results['bias'][:, i] = model.bias

            for field in ('reflexive', 'reflective'):
                results[field]['activity'][:, i] = getattr(model, field).activity
                results[field]['output'][:, i] = getattr(model, field).output

        results['statistics'] = NeuralField1D.compute_statistics(results)

        if not plots:
            return results

        fig, axs = cls.plot_results(results, model, time_step)
        return results, (fig, axs)

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

        fig, axs = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True)

        axs[0, 0].set_title('Stimulus')
        axs[0, 0].imshow(results['stimulus'], cmap=color_map)
        axs[0, 0].set_ylim(0, 181)
        axs[0, 0].set_yticks(np.arange(0, model.neurons, 30))
        axs[0, 0].set_yticklabels(model.centers[np.array(axs[0, 0].get_yticks(), dtype=np.int32)])

        axs[0, 1].set_title('Reflexive Activity')
        axs[0, 1].imshow(results['reflexive']['activity'], cmap=color_map)
        axs[0, 1].set_xticklabels(np.array(axs[1, 0].get_xticks()) * time_step)

        indices, = np.where(results['active'])
        if len(indices):
            centers = np.argmax(results['reflexive']['output'][:, indices], axis=0)
            axs[0, 1].scatter(indices, centers, results['active'][indices] * 2, marker='.', color=(0, 0, 0))

        axs[0, 2].set_title('Reflexive Output')
        axs[0, 2].imshow(results['reflexive']['output'], cmap=color_map)
        axs[0, 2].set_xticklabels(np.array(axs[1, 1].get_xticks()) * time_step)
        axs[0, 2].plot(results['intention'], linestyle='--', color='white')

        axs[1, 0].set_title('Bias')
        axs[1, 0].set_ylim(0, 181)
        axs[1, 0].imshow(results['bias'], cmap=color_map)
        # axs[1, 0].set_yticks(np.arange(0, model.neurons, 15))
        # axs[1, 0].set_yticklabels(model.centers[np.array(axs[0, 1].get_yticks(), dtype=np.int32)])

        axs[1, 0].set_title('Frontal Inhibition')
        axs[1, 0].imshow(results['inhibition'], cmap=color_map)
        # axs[1, 0].set_yticklabels(np.array(axs[1, 0].get_yticks(), dtype=np.int32) - 45)

        axs[1, 1].set_title('Reflective Activity')
        axs[1, 1].imshow(results['reflective']['activity'], cmap=color_map)
        axs[1, 1].set_xticklabels(np.array(axs[1, 0].get_xticks()) * time_step)

        centers = np.argmax(results['reflective']['output'][:, indices], axis=0)
        axs[1, 1].scatter(indices, centers, results['active'][indices] * 2, marker='.', color=(0, 0, 0))

        axs[1, 2].set_title('Reflective Output')
        axs[1, 2].imshow(results['reflective']['output'], cmap=color_map)
        axs[1, 2].set_xticklabels(np.array(axs[1, 1].get_xticks()) * time_step)
        axs[1, 2].plot(results['attention'], linestyle='--', color='white')

        # indices, = np.where(results['active'] > model.threshold)
        # if len(indices):
        #     centers = np.argmax(results['reflexive']['output'][:, indices], axis=0)
        #     axs[0, 1].scatter(indices, centers, results['active'][indices] * 2, marker='.', color=(0, 0, 0))

        # axs[1, 2].set_title('Frontal Bias')
        # axs[1, 2].imshow(results['bias'], cmap=color_map)
        # axs[1, 2].set_yticklabels(np.array(axs[2, 0].get_yticks(), dtype=np.int32) - 45)

        fig.set_tight_layout('tight')
        return fig, axs


if __name__ == "__main__":
    DualFieldDecisionModel.simulation(plots=True)
