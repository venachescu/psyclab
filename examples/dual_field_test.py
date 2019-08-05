
import numpy as np
import pandas as pd

from psyclab.models.dual_field import DualFieldDecisionModel
from psyclab.neural.stimulus import GaussianStimulus
from psyclab import Color


class DualFieldTest:

    def __init__(self, neurons=91, time_step=0.01, duration=2.0, **kwargs):

        self.time_step = time_step
        self.duration = duration
        self.neurons = neurons

        self.results = pd.DataFrame(np.linspace(0.0, duration, int(duration / time_step)), columns=('time',))
        self.values = np.zeros((neurons, int(duration / time_step), 6))
        self.stats = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        stimuli = self.make_stimuli()
        self.model = DualFieldDecisionModel(stimuli, neurons=neurons, time_step=time_step)
        self.model.frontal_bias.center = 15

    def run(self):

        self.results['arousal'] = 0.0
        self.results['attention'] = 0.0
        self.results['active'] = 0.0

        for i, t in enumerate(self.results['time']):

            self.step(time_step=self.time_step)

            # if t > 0.1:
            #     self.model.frontal_bias.center = -self.neurons // 4

            active = self.model.reflexive.active_neurons
            if active is None:
                active = 0.0
            else:
                active = len(active)

            self.results.loc[i, ['arousal', 'attention', 'active']] = np.array((self.model.arousal, self.model.reflexive.activity_center, active))
            self.values[:, i, 0] = self.model.reflexive.activity
            self.values[:, i, 1] = self.model.reflexive.output
            self.values[:, i, 2] = self.model.reflective.activity
            self.values[:, i, 3] = self.model.reflective.output
            self.values[:, i, 4] = self.model.bias
            self.values[:, i, 5] = self.model.inhibition

        indices, = np.where(self.results['active'])
        if len(indices):
            self.stats['reaction'] = self.results.loc[min(indices), 'time']

    def step(self, time_step=0.01):
        self.model.step(time_step=time_step)

    def make_stimuli(self):
        return {}

    def make_plots(self):

        fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey=True)
        axs[0, 0].set_title('Reflexive Activity')
        axs[0, 0].imshow(self.values[:, :, 0], cmap='viridis')
        axs[0, 0].set_yticklabels(np.array(axs[0, 0].get_yticks(), dtype=np.int32) - 45)

        axs[0, 1].set_title('Reflexive Output')
        axs[0, 1].imshow(self.values[:, :, 1], cmap='viridis')
        axs[0, 1].set_yticklabels(np.array(axs[0, 1].get_yticks(), dtype=np.int32) - 45)

        indices, = np.where(self.results['active'] > self.model.reflexive.threshold)
        if len(indices):
            centers = np.argmax(self.values[:, indices, 1], axis=0)
            axs[0, 1].scatter(indices, centers, self.results.loc[indices, 'active'] * 2, marker='.', color=Color('bright purple').rgb)

        axs[1, 0].set_title('Reflective Activity')
        axs[1, 0].imshow(self.values[:, :, 2], cmap='viridis')
        axs[1, 0].set_xticklabels(np.array(axs[1, 0].get_xticks()) * self.time_step)
        axs[1, 0].plot(self.results['attention'], linestyle='--', color='white')

        axs[1, 1].set_title('Reflective Output')
        axs[1, 1].imshow(self.values[:, :, 3], cmap='viridis')
        axs[1, 1].set_xticklabels(np.array(axs[1, 1].get_xticks()) * self.time_step)

        axs[2, 0].set_title('Frontal Bias')
        axs[2, 0].imshow(self.values[:, :, 4], cmap='viridis')
        axs[2, 0].set_yticklabels(np.array(axs[2, 0].get_yticks(), dtype=np.int32) - 45)

        axs[2, 1].set_title('Frontal Inhibition')
        axs[2, 1].imshow(self.values[:, :, 5], cmap='viridis')
        axs[2, 1].set_yticklabels(np.array(axs[2, 1].get_yticks(), dtype=np.int32) - 45)

        fig.set_tight_layout('tight')
        return fig, axs


class SingleStimulusTest(DualFieldTest):

    amplitude = 5.0

    def make_stimuli(self):
        return dict(stimulus=GaussianStimulus(center=0, amplitude=self.amplitude))


class DoubleStimuliTest(DualFieldTest):

    def make_stimuli(self):
        return dict(
            left=GaussianStimulus(center=-self.neurons // 4, amplitude=5.0, width=5, delay=0.25),
            right=GaussianStimulus(center=self.neurons // 4, amplitude=5.0, width=5, delay=0.25),
        )


if __name__ == "__main__":

    import matplotlib
    matplotlib.use('macosx')
    import matplotlib.pyplot as plt

    results = []

    test = DoubleStimuliTest(amplitude=5.0)
    # test = SingleStimulusTest(amplitude=5.0)
    test.run()
    test.make_plots()
    plt.show()

    # for _ in range(100):
    #     test = SingleStimulusTest(amplitude=5.0)
    #     test.run()
    #     results.append(test.stats)
        # test.make_plots()
        # plt.show()

    # plt.hist(pd.DataFrame(results)['reaction'])
