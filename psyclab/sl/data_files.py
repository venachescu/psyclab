#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/data_files.py
Vince Enachescu 2019
"""

import numpy as np

from math import ceil
from re import match
from os import path, getcwd, remove, makedirs, listdir
from shutil import copyfile


class InvalidSLDataFile(Exception):
    pass


def last_data_file(directory=None, as_int=True):
    """ Index of the current data file from SL data collection """
    try:
        with open(path.join(directory or getcwd(), '.last_data'), 'r') as fp:
            index = int(fp.read().strip())
            if as_int:
                return index
            index -= 1
    except FileNotFoundError:
        print('File not found error')
        if as_int:
            return 0
        index = 0

    return f'd{index:05d}'


def read_sl_data_file(file_path=None, clip=True):
    """ read an SL data file into dict of np arrays
    clip - option will truncate np vectors to end of data recording """

    file_path = file_path or last_data_file()
    if file_path is None or not path.isfile(file_path):
        raise InvalidSLDataFile('cannot find data file', file_path)

    # make sure it is in fact an SL data file
    data_file = open(file_path, 'rb')

    try:
        dimensions = map(eval, data_file.readline().strip().decode('ascii').split())
        n_values, n_variables, time_steps, sample_rate = tuple(dimensions)
    except (TypeError, ValueError):
        raise InvalidSLDataFile('not a valid SL data file: {}'.format(file_path))

    header = data_file.readline().strip().decode().split()
    names, units = tuple(header[0::2]), tuple(header[1::2])

    # read binary data
    raw_data = data_file.read(n_values * 4)
    data_file.close()

    # check for malformed (or files not fully written)
    if len(raw_data) % 4:
        raise InvalidSLDataFile(f'buffer size not multiple of float size: {file_path}')

    raw_data = np.frombuffer(raw_data, '>f')
    raw_data = raw_data.reshape(time_steps, n_variables)

    # clip data file at the end of time signal
    if clip is True and 'time' in names:
        t_stop = 1
        while t_stop < time_steps and raw_data[t_stop, names.index('time')] > 0.0:
            t_stop += 1

    # if clip parameter is a float use that as clipping time is seconds
    elif isinstance(clip, float):
        t_stop = int(min(time_steps, ceil(clip * sample_rate)))

    else:
        t_stop = time_steps

    # pack into dictionary
    info = {
        'time_steps': t_stop,
        'time_step': float(1.0 / sample_rate),
        'sample_rate': sample_rate,
        'variables': names,
        'units': units
    }
    return info, {names[n]: raw_data[0:t_stop, n] for n in range(n_variables)}


def write_sl_data_file(file_path, data, variables, units, time_steps, sample_rate):
    """ write data set to SL data file format
    file_path - output data file path
    data - matrix of each variables time series data
    time_steps - number of times steps in data file
    sample_rate - sampling rate for the data file """

    # open data file
    try:
        data_file = open(file_path, 'wb')
    except OSError:
        print('[Error] cannot open %s to export' % file_path)
        return

    n_variables = len(variables)

    # data file dimensions
    data_file.write('%d %d %d %f\n' % (time_steps * n_variables, n_variables, time_steps, sample_rate))

    # variable names and units
    variable_names = ''
    for name in variables:
        variable_names += '%s  %s  ' % (name, units[name])
    data_file.write(variable_names + '\n')

    # write binary data (32bit big endian floats)
    data = data.reshape(data.size).astype('>f')
    data_file.write(data.tostring())
    data_file.close()

    return file_path


def archive_data_files(user_path, archive_path, delete=False):

    if not path.isdir(archive_path):
        makedirs(archive_path)

    for file_name in listdir(user_path):
        if match('^d\d{5}$', file_name):
            copyfile(path.join(archive_path, file_name))
            if delete:
                remove(path.join(user_path, file_name))
