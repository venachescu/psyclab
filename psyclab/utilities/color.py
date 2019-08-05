#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/graphics/color.py
Vince Enachescu 2019
"""

import numpy as np
import hsluv

from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS


class Color:
    """
    Color object to make conversions and comparisons between color spaces
    easier.
    """

    def __init__(self, *args, hsl=None, bits=8, alpha=1.0):

        self.alpha = alpha
        self.bits = bits

        if hsl is not None:
            self.from_hsl(hsl)
            return

        if len(args) == 1:
            args, = args

        if isinstance(args, str):
            self.from_string(args)
            return

        if isinstance(args, Color):
            args = args.values

        self.from_rgb(*args)

    def from_hsl(self, *args):
        """ Specify a color from a hue, saturation and luminosity triplet """

        if len(args) == 1:
            args, = args

        h, s, l = args
        self.from_rgb(*hsluv.hsluv_to_rgb((h, s, l)))

    def from_rgb(self, r, g, b):

        x = (r, g, b)
        if min(x) >= 0.0 and max(x) <= 1.00001:
            x = tuple(round((2 ** self.bits - 1) * v) for v in x)

        self.values = tuple(map(int, x))

    def from_string(self, string):

        if not string.startswith('#'):
            for pre, colors in (
                ('', BASE_COLORS),
                ('tab:', TABLEAU_COLORS),
                ('', CSS4_COLORS),
                ('xkcd:', XKCD_COLORS)
            ):
                name = ''.join((pre, string))
                if name in colors:
                    string = colors[name]
                    break
            else:
                raise Exception(f'could not find your color! {string}')

        if isinstance(string, tuple):
            self.from_rgb(*string)
        else:
            self.from_hexadecimal(string)

    def from_hexadecimal(self, string):

        value = string.lstrip('#')
        if len(value) == 3:
            value = ''.join(c * 2 for c in value)

        n = len(value) // 3
        self.values = tuple(int(value[i:i + n], 16) for i in range(0, len(value), n))

    def distance(self, color, space='rgb', norm='l2'):
        return

    @property
    def hex(self):
        return '#' + ''.join(map('{:02x}'.format, self.values))

    @property
    def rgb(self):
        return self.as_float()

    @property
    def RGB(self):
        return self.values

    def as_float(self):
        return tuple(float(x / (2 ** self.bits - 1)) for x in self.values)

    @property
    def rgba(self):
        return (self.rgb + (self.alpha,))

    @property
    def r(self):
        return self.rgb[0]

    @property
    def g(self):
        return self.rgb[1]

    @property
    def b(self):
        return self.rgb[2]

    @property
    def hsl(self):
        return hsluv.rgb_to_hsluv(self.rgb)

    @property
    def h(self):
        return self.hsl[0]

    @property
    def s(self):
        return self.hsl[1]

    @property
    def l(self):
        return self.hsl[2]

    def __getattr__(self, key):
        return

    def __getitem__(self, index):
        return self.values[index]

    def __iter__(self):
        for value in self.values:
            yield value
