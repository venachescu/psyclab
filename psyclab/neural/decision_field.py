

import numpy as np

from random import shuffle

from psyclab.neural.field_1d import NeuralField1D
from psyclab.neural import stimulus


class DecisionField(NeuralField1D):

    def __init__(self, *args, **kwargs):

        NeuralField1D.__init__(self)
        self._stimuli = []
        self.stimulus = np.zeros(self.n)
        self.values = np.linspace(0, self.n, self.n)
        self.threshold = 0.75

    def step(self, time_step=0.001):

        self.stimulus = self.step_stimuli(time_step=time_step)
        NeuralField1D.step(self, stimulus=self.stimulus, time_step=time_step)

        indices, = np.where(self.output > self.threshold)
        if not len(indices):
            return

        shuffle(indices)
        index = next(iter(indices))
        # print('Index!', index)
        return self.values[index]

    def step_stimuli(self, time_step=0.001):

        return np.sum([s(self.values, dt=time_step) for s in self._stimuli], axis=0)


class DecisionSimulation:

    def __init__(self, plot=None, time_step=0.001, **kwargs):

        self.field = DecisionField(**kwargs)
        self.data = []
        self.plot = plot
        self.results = []
        self.time = 0.0
        self.time_step = time_step

    def simulation(self):

        while True:
            output = self.field.step(time_step=self.time_step)
            if output is not None:
                break

            self.time += self.time_step

        # print(f'Result: {output}, Time: {self.time:03f}')
        self.results.append({'time': self.time, 'index': output})

    def step(self):

        output = self.field.step(time_step=self.time_step)
        self.time += self.time_step
        return output

    def animate(self, i):

        output = self.step(time_step=self.time_step)
        if output is not None:
            self.results.append({'time': self.time, 'index': output})
            self.reset()

        self.plot['output'].set_data((self.field.values, self.field.output))
        self.plot['activity'].set_data((self.field.values, self.field.activity))
        self.plot['text'].set_text(f'time = {self.time:03f}')

    def initialize(self):
        self.plot['output'].set_data((self.field.values, self.field.output))
        self.plot['activity'].set_data((self.field.values, self.field.activity))
        self.plot['text'].set_text('')
        return self.plot

    def reset(self):

        self.time = 0.0
        self.field.reset()
        shuffle(self._stimuli)


def run_simulations(i, a=30, b=150, count=250, gaussian=True, **kwargs):

    sim = DecisionSimulation()
    for _ in range(count):
        if gaussian:
            sim.field._stimuli = [
                stimulus.GaussianStimulus(center=a, **kwargs),
                stimulus.GaussianStimulus(center=b, **kwargs)
            ]
        else:
            sim.field._stimuli = [
                stimulus.StepStimulus(center=a, **kwargs),
                stimulus.StepStimulus(center=b, **kwargs)
            ]
        sim.simulation()
        sim.results[-1]['iteration'] = i
        sim.results[-1].update(kwargs)

    return sim.results


if __name__ == "__main__":

    # from psyclab.utilities.tools import parallel_map
    #
    # results = parallel_map(run_simulations, range(10), amplitude=15, width=45, count=250)
    #
    # import pandas as pd
    # df = pd.concat(list(map(pd.DataFrame, results)))

    # sim = DecisionSimulation(plot=dict(figure=fig, ax=ax, output=line_a, activity=line_b, text=time_text))

    # df = pd.DataFrame(sim.results)
    # df['time'].hist()

    import matplotlib
    matplotlib.use('MacOSX')

    import matplotlib.pyplot as plt
    from matplotlib import animation

    field = DecisionField(n=180)
    field._stimuli = [
        stimulus.GaussianStimulus(center=60, amplitude=10.0, width=25),
        stimulus.GaussianStimulus(center=130, amplitude=10.0, width=25)
    ]
    field.step()
    t, dt = 0.0, 0.002

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 181), ylim=(-10, 10))

    line_a, = ax.plot([], [], '-', lw=2, mew=5, label='output')
    line_b, = ax.plot([], [], '-', lw=2, mew=5, label='activity')
    line_c, = ax.plot([], [], '-', lw=2, mew=5, label='trace')
    line_d, = ax.plot([], [], '-', lw=2, mew=5, label='noise')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.legend()

    def init():
        global field
        line_a.set_data((field.values, field.output))
        line_b.set_data((field.values, field.activity / 10.0))
        line_c.set_data((field.values, field.trace))
        line_d.set_data((field.values, field.noise))
        time_text.set_text('')
        return line_a, line_b, line_c, line_d, time_text

    def animate(i):
        global field, t, dt
        field.step(time_step=dt)
        line_a.set_data((field.values, field.output))
        line_b.set_data((field.values, field.activity / 10.0))
        line_c.set_data((field.values, field.trace))
        line_d.set_data((field.values, field.noise))
        time_text.set_text('time = %.2f' % t)
        t += dt
        return line_a, line_b, line_c, line_d, time_text

    ani = animation.FuncAnimation(fig, animate, frames=range(100), interval=5, blit=True, init_func=init)
    fig.show()
