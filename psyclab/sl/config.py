#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/config.py
Vince Enachescu 2019
"""


import re
import ctypes

from os import path, environ
from sys import modules

from CppHeaderParser import CppHeader

from .cstruct import CStruct, parse_defines, replace_defines, read_struct_format


try:
    prog_root = environ['PROG_ROOT']
    src_path = path.join(prog_root, 'SL')
except KeyError:
    # do something to handle the new workspace
    # workspace_root = environ['SL_WORKSPACE']
    # src_path = path.join(prog_root, 'SL')
    raise Exception('SL Error: $PROG_ROOT not defined.')

try:
    mach_type = environ['MACHTYPE']
except KeyError:
    raise Exception('SL Error: $MACHTYPE not defined.')

_defines = {}
_types = {}


def header_path(header='SL.h', robot_name=None):
    if robot_name is None:
        return path.join(src_path, 'include', header)
    elif path.isdir(robot_name):
        return path.join(robot_name, 'include', header)
    else:
        return path.join(robot_path(robot_name), 'include', header)


def robot_path(robot_name, workspace=None):
    robot_path = path.join(prog_root, robot_name)
    if not path.isdir(robot_path):
        raise Exception('robot {} not found'.format(robot_name))
    return robot_path


def robot_user_path(robot_name, suffix='User'):
    user_path = path.join(prog_root, '{}{}'.format(robot_name, suffix))
    if not path.isdir(user_path):
        raise Exception('user path not found: {}{} '.format(robot_name, suffix))
    return user_path


def _load(module):

    if len(_defines):
        return

    with open(header_path(), 'r') as fp:
        header_file = fp.read()

    header = CppHeader(header_file, argType='string')
    defines = parse_defines(header)

    setattr(modules[module], 'SEM_ID', ctypes.c_long)

    types = {'SEM_ID': ctypes.c_long}
    for struct_name in header.classes.keys():
        fields = read_struct_format(header_file, struct_name, defines=defines)
        struct_type = type(struct_name, (CStruct,), {'_fields_': fields})
        setattr(modules[module], struct_name, struct_type)
        types[struct_name] = struct_type

    _defines.update(defines)
    _types.update(types)


def read_robot_config(robot_name):

    header = CppHeader(header_path('SL_user.h', robot_name))
    defines, enums = parse_defines(header, include_enums=True)
    defines = replace_defines(defines, _defines)
    defines.update(_defines)

    return {
        'joints': [
            enums['RobotDOFs'][i + 1] for i in range(defines['N_DOFS'])
        ],
        'endeffectors': [
            enums['RobotEndeffectors'][i + 1] for i in range(defines['N_ENDEFFS'])
        ],
        'links': [
            enums['RobotLinks'][i + 1] for i in range(defines['N_LINKS'])
        ],
        'defines': defines,
        'enums': enums,
    }


def read_sample_prefs(user_path, script='default_script'):
    """ read in the sampling preferences saved in the prefs folder """

    if not path.isdir(user_path) and path.isdir(path.join(environ['PROG_ROOT'], user_path)):
        user_path = path.join(environ['PROG_ROOT'], user_path)

    prefs_path = path.join(user_path, 'prefs')
    if not path.isdir(prefs_path):
        print('[Error] prefs path does not exist', prefs_path)
        return

    lines = []
    with open(path.join(prefs_path, script), 'r') as fp:
        lines = fp.read()

    if not lines:
        print('[Error] prefs script not found')
        return

    prefs = {}
    for line in lines.split('\n'):
        if line == '' or line == '\n':
            continue
        key, value = line.split()
        name = key[0:key.find('_')]
        key = key.replace('{}_default_'.format(name), '')
        prefs.setdefault(name, {})[key] = value

    for name, pref in prefs.items():
        for key in pref.keys():
            if re.match('script'.format(name), key):
                file_path = path.join(prefs_path, pref[key])
                if not path.isfile(file_path):
                    continue

                fp = open(file_path, 'r')
                pref[key] = [l for l in fp.read().split('\n') if l]
                fp.close()

    return prefs


def read_config_file(user_path, file_name):

    with open(path.join(user_path, 'config', file_name), 'r') as fp:
        data = fp.read()

    match = re.search('\/\* format: (.*?) \*\/', data)
    if match is not None:
        variables = match.group(1).replace(',', '').split()
    else:
        variables = None

    for comment in re.findall('\/\*.*?\*\/', data, re.DOTALL):
        data = data.replace(comment, '')

    lines = filter(lambda line: line != '', data.split('\n'))
    return [dict(zip(variables, line.strip().split())) for line in lines]
