
import numpy as np

from scipy.signal import find_peaks

from glumpy.transforms import Position, OrthographicProjection

from psyclab import Color
from psyclab.display.glumpy_app import GlumpyApp
from psyclab.display.shapes import Trace
from psyclab.utilities.osc import route
from psyclab.display.shapes import Lines, Markers
from psyclab.neural.stimulus import GaussianStimulus
from psyclab.models.dual_field import DualFieldDecisionModel


class DualFieldSimulation(GlumpyApp):

    def __init__(self, neurons=91, time_step=0.01, **kwargs):

        sensory_input = dict(
            left=GaussianStimulus(center=-(neurons // 4), amplitude=0.8),
            right=GaussianStimulus(center=(neurons // 4), amplitude=0.7)
        )
        self.model = DualFieldDecisionModel(sensory_input, neurons=neurons, time_step=time_step)

        GlumpyApp.__init__(self, 'Dual Field Decision Model', width=960, height=600, **kwargs)

    def make_programs(self, window):

        transform = OrthographicProjection(Position(), aspect=None, normalize=True)

        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.shapes = {
            'reflexive': Lines(n=self.model.neurons, color=Color('baby blue').rgb, linewidth=10, transform=transform, window=window),
            'reflective': Lines(n=self.model.neurons, color=Color('light purple').rgb, linewidth=10, transform=transform, window=window),
            'stimulus': Lines(n=self.model.neurons, color=Color('red').rgb, linewidth=5, transform=transform, window=window),
            'inhibition': Lines(n=self.model.neurons, color=Color('dark blue').rgb, linewidth=5, transform=transform, window=window),
            'bias': Lines(n=self.model.neurons, color=Color('soft pink').rgb, linewidth=5, transform=transform, window=window),
            'reflexive_out': Lines(n=self.model.neurons, color=Color('ocean blue').rgb, linewidth=10, transform=transform, window=window),
            'reflective_out': Lines(n=self.model.neurons, color=Color('violet').rgb, linewidth=10, transform=transform, window=window),
            'attention': Markers(x, window=window, size=15, linewidth=2, color=(1, 1, 1), bg_color=(0, 0, 0)),
            'intention': Markers(x, window=window, size=15, linewidth=2, color=(1, 1, 1), bg_color=(0, 0, 0))
        }

    def step(self, dt, **kwargs):

        self.model.step(time_step=dt)
        indices = self.model.reflexive.active_neurons
        if indices is not None:
            [self.send('/neurons', int(i)) for i in indices]

        index = self.model.reflective.activity_center
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        x[:, 0] = self.model.reflective.centers[index] / self.model.neurons
        # x[:, 0] = (index / self.model.neurons)
        # self.shapes['attention'].x = x
        self.shapes['attention'].update(x)

        index = self.model.reflexive.activity_center
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        x[:, 0] = self.model.reflective.centers[index] / self.model.neurons
        # x[:, 0] = (index / self.model.neurons)
        # self.shapes['attention'].x = x
        self.shapes['intention'].update(x)

        self.shapes['reflexive'].y = self.model.reflexive.activity / 10.
        self.shapes['reflexive'].update()

        self.shapes['reflective'].y = self.model.reflective.activity / 10.
        self.shapes['reflective'].update()

        self.shapes['reflexive_out'].y = self.model.reflexive.output
        self.shapes['reflexive_out'].update()

        self.shapes['reflective_out'].y = self.model.reflective.output
        self.shapes['reflective_out'].update()

        self.shapes['stimulus'].y = self.model.stimulus / 10.
        self.shapes['stimulus'].update()

        self.shapes['inhibition'].y = self.model.inhibition / 2.
        self.shapes['inhibition'].update()

        self.shapes['bias'].y = self.model.bias / 1.
        self.shapes['bias'].update()

        #[shape.update() for shape in self.shapes.values()]

    def start(self):

        self._client = self.connect('localhost', 7402)
        GlumpyApp.start(self)

    def receive(self, route, source, message):

        parameter = route.strip('/')

        value = float(message)
        setattr(self.model.reflexive, parameter, value)
        setattr(self.model.reflective, parameter, value)
        # return

    @route('/left/amplitude', '/right/amplitude', '/left/sigma', '/right/sigma', '/left/width', '/right/width')
    def receive_amplitude(self, route, source, message):

        _, side, parameter = route.split('/')
        setattr(self.model.stimuli[side], parameter, float(message))
        # index = 0 if side == 'left' else 1
        # setattr(self.model.sensory_input[index], parameter, float(message))

    @route('/bias/amplitude', '/bias/sigma', '/bias/width')
    def receive_amplitude(self, route, source, message):

        _, _, parameter = route.split('/')
        setattr(self.model.frontal_bias, parameter, float(message))

    @route('/inhibition/gain', '/inhibition/center', '/inhibition/sigma1', '/inhibition/sigma2', '/inhibition/input', '/inhibition/width', '/inhibition/alpha1', '/inhibition/alpha2')
    def route_inhibition(self, route, source, message):

        *_, parameter = route.split('/')
        value = float(message)

        if parameter == 'gain':
            self.model.inhibition_gain = value
        elif parameter == 'input':
            self.model.arousal_gain = value
        elif parameter == 'center':
            self.model.frontal_inhibition.center = value
        elif parameter == 'width':
            self.model.frontal_inhibition.omega = value
            # self.model.frontal_inhibition.width = value
        # elif parameter == 'sigma1':
        #     self.model.frontal_inhibition.sigma1 = value
        # elif parameter == 'sigma2':
        #     self.model.frontal_inhibition.sigma2 = value
        # elif parameter == 'alpha1':
        #     self.model.frontal_inhibition.alpha1 = value
        # elif parameter == 'alpha2':
        #     self.model.frontal_inhibition.alpha2 = value


if __name__ == "__main__":
    sim = DualFieldSimulation()
    sim.start()
