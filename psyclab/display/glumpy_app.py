#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/display/glumpy_app.py
Vince Enachescu 2019
"""

# import numpy as np

from functools import partial

from glumpy import app
from psyclab.utilities.osc import OSCResponder


class GlumpyApp(OSCResponder):
    """
    Template Class to make a basic glumpy powered app, including a built-in
    OSC responder.

    """

    def __init__(self, title='psyclab', width=800, height=800, bg_color=(1, 1, 1, 1), **kwargs):

        self.title = title
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.shapes = {}
        self.window = None

        OSCResponder.__init__(self, **kwargs)

    def start(self):
        """ Start the application and run (blocking) """

        OSCResponder.start(self)

        self.window = self.make_window()
        self.make_programs(self.window)

        try:
            app.run()
        except Exception as error:
            OSCResponder.stop(self)
            raise error

    def make_window(self):
        """ Create the application window """

        window = app.Window(title=self.title, width=self.width, height=self.height, color=self.bg_color)
        window.set_handler('on_draw', self.on_draw)
        window.set_handler('on_init', self.on_init)
        window.set_handler('on_resize', self.on_resize)
        window.set_handler('on_close', partial(self.on_close, caller=self.title))
        return window

    def make_programs(self, window):
        """ Initialize the OpenGL shader programs (placeholder function) """
        pass

    def step(self, dt, **kwargs):

        for shape in self.shapes.values():
            shape.update()

    def on_init(self):
        return

    def on_draw(self, dt):
        """ Update graphics callback - """

        self.step(dt)

        self.window.clear(color=self.bg_color)

        for shape in self.shapes.values():
            shape.draw()

    def on_close(self, caller='window'):
        """ On window close - close responder thread """

        OSCResponder.stop(self)

    def on_resize(self, width, height):
        """ On window resize """

        self.width, self.height = width, height
        self.window.clear(color=self.bg_color)
        self.window.swap()
        self.window.clear(color=self.bg_color)
