#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/display/shapes.py
Vince Enachescu 2019
"""

import numpy as np

from glumpy import gl
from glumpy.gloo import Program
from glumpy.transforms import Position, OrthographicProjection
from glumpy.graphics.collections.agg_segment_collection import AggSegmentCollection
from glumpy.graphics.collections.marker_collection import MarkerCollection

from psyclab import Color


class Trace(Program):

    vertex_source = """
    attribute float x, y, intensity;

    varying float v_intensity;
    void main (void)
    {
        v_intensity = intensity;
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
    """

    fragment_source = """
    uniform vec3 color;
    varying float v_intensity;
    void main()
    {
        gl_FragColor = vec4(color, v_intensity);
    }
    """

    def __init__(self, values=None, color=(0, 0, 0), n=100, scale=1.0, origin=(0, 0), **kwargs):

        Program.__init__(self, self.vertex_source, self.fragment_source, count=n)

        self.scale = scale
        self.origin = origin

        self['x'] = np.linspace(-1, 1, n)
        self['y'] = (values or np.zeros(n)) * scale + origin[1]
        self['intensity'] = 1.0
        self['color'] = color
        # self['color'] = Color(color).rgb

    def update(self, trace=None):

        if trace is not None:
            y = np.roll(np.copy(self.y), -1)
            y[-1] = trace
            self.y = y

    def draw(self):
        Program.draw(self, gl.GL_LINE_STRIP)

    @property
    def x(self):
        return (self['x'] - self.origin[0]) / self.scale

    @x.setter
    def x(self, x):
        self['x'] = x * self.scale + self.origin[0]

    @property
    def y(self):
        return (self['y'] - self.origin[1]) / self.scale

    @y.setter
    def y(self, y):
        self['y'] = y * self.scale + self.origin[1]


class Lines(AggSegmentCollection):

    def __init__(self, *args, transform=None, window=None, linewidth=10, n=101, color=(0, 0, 0), scale=1.0, origin=(0, 0), **kwargs):

        self.scale = scale
        self.origin = origin

        if len(args) == 1:
            self.x, self.y = np.linspace(-1, 1, len(args[0])), args[0]
        elif len(args) == 2:
            self.x, self.y = args
            n = len(args[0])
        else:
            self.x, self.y = np.linspace(-1, 1, n), np.zeros(n)
            # self.x, self.y = np.linspace(-1, 1, n - 1), np.zeros(n - 1)

        if transform is None:
            transform = OrthographicProjection(Position(), normalize=True)

        AggSegmentCollection.__init__(self, linewidth='local', transform=transform)
        self.append(self.P0, self.P1, linewidth=linewidth, color=np.tile(Color(color).rgba, (n + 1, 1)))
        self['antialias'] = 1

        if window is not None:
            window.attach(self['transform'])
            window.attach(self['viewport'])

    def update(self, trace=None):
        """ """

        if trace is not None:
            y = np.roll(np.copy(self.y), -1)
            y[-1] = trace
            self.y = y

        self['P0'] = np.repeat(self.P0, 4, axis=0)
        self['P1'] = np.repeat(self.P1, 4, axis=0)

    @property
    def n(self):
        return self._x.shape[0] - 1

    @property
    def x(self):
        return self._x[1:-1]

    @x.setter
    def x(self, x):
        self._x = np.pad(x, (1, 1), mode='edge')

    @property
    def y(self):
        return self._y[1:-1]

    @y.setter
    def y(self, y):
        self._y = np.pad(y, (1, 1), mode='edge')

    @property
    def P0(self):
        x = self.scale * self._x[:-1] + self.origin[0]
        y = self.scale * self._y[:-1] + self.origin[1]
        return np.dstack((x, y, np.zeros(self.n))).reshape(self.n, 3)

    @property
    def P1(self):
        x = self.scale * self._x[1:] + self.origin[0]
        y = self.scale * self._y[1:] + self.origin[1]
        return np.dstack((x, y, np.zeros(self.n))).reshape(self.n, 3)


class Markers(MarkerCollection):

    def __init__(self, x, transform=None, window=None, linewidth=2, color=(0, 0, 0), scale=1.0, origin=(0, 0), **kwargs):

        self.scale = scale
        self.origin = origin

        if transform is None:
            transform = OrthographicProjection(Position(), normalize=True)

        MarkerCollection.__init__(self, marker='disc', transform=transform)
        self.append(x, size=15, linewidth=linewidth, itemsize=1, fg_color=Color(color).rgba, bg_color=(1, .5, .5, 1))
        # self['antialias'] = 1

        if window is not None:
            window.attach(self['transform'])
            window.attach(self['viewport'])

    def update(self, x=None):
        """ """

        if x is not None:
            self['position'] = x


class Pixels(Program):

    vertex_source = """"""
    shader_source = """"""

    def __init__(self):
        return
