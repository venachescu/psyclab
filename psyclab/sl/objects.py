#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/objects.py
Vince Enachescu 2019
"""

import ctypes

from psyclab.sl.cstruct import CStruct


class SLObjectMessage(CStruct):

    _fields_ = [
        ('name', ctypes.c_char * 100),
        ('position', ctypes.c_double * 4),
        ('rotation', ctypes.c_double * 4),
    ]


class SLObject(object):

    def __init__(self, name, index=1):
        return

    @staticmethod
    def parse_text(text):

        lines = filter(None, map(str.strip, text.split('\n')))
        objects = []

        while True:
            try:
                name, type_id = next(lines.split())
                color = map(float, next(lines).split())
                position = map(float, next(lines).split())
                rotation = map(float, next(lines).split())
                scale = map(float, next(lines).split())
                contact_model = int(next(lines).strip())
                object_parameters = next(lines)
                contact_parameters = next(lines)
                # objects.append(SLObject(
                # ))
            except StopIteration:
                break

        return


if __name__ == "__main__":

    import re

    from os import path
    from psyclab.sl import robot_user_path

    cfg_path = path.join(robot_user_path('master'), 'config', 'Objects.cf')
    with open(cfg_path, 'r') as fp:
        body = fp.read()

    for comment in re.findall('\/\*.*?\*\/', body, re.DOTALL):
        body = body.replace(comment, '')

    lines = SLObject.parse_text(body)
