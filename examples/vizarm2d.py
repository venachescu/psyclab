
import numpy as np

from glumpy.transforms import Position, OrthographicProjection
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection

from psyclab.display.glumpy_app import GlumpyApp
# from psyclab.models.arm2d import Arm2DModel
from psyclab.models.upper_arm import UpperArm

from psyclab.display.shapes import Lines, Markers


class Arm2DVisualization(GlumpyApp):

    # model = Arm2DModel()
    model = UpperArm()

    def make_programs(self, window):

        transform = OrthographicProjection(Position(), aspect=None, normalize=True)

        x = self.model.positions(z=True).T
        self.shapes = {
            'links': Lines(x[:, 0], x[:, 1], window=window, transform=transform, linewidth=20.0, color=(0.75, 0.75, 0.75)),
            'joints': Markers(x, window=window, size=15, linewidth=2, color=(0, 0, 0))
        }

    def step(self, dt, **kwargs):

        # self.model.apply_torque([1.0, 0.0])
        # u = np.abs(np.random.randn(6)) / 10000.0
        # self.model.step(u, dt=dt)
        x = self.model.positions(z=True).T

        self.shapes['links'].x = x[:, 0]
        self.shapes['links'].y = x[:, 1]
        self.shapes['links'].update()

        self.shapes['joints'].update(x)


if __name__ == "__main__":
    sim = Arm2DVisualization()
    sim.start()
