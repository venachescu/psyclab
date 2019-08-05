

import sys
import yaml
import hashlib

# import numpy as np
# import pandas as pd
import multiprocessing as mp

# from glob import glob
from functools import partial, wraps
from inspect import isroutine, getmembers, signature
# from datetime import datetime, timedelta


def file_hash(file_path):
    """ Compute the md5 hash of a file """

    digest = hashlib.md5()
    with open(file_path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(4096), b''):
            digest.update(chunk)

    return digest.hexdigest()


def parallel_map(func, *args, parallel=True, **kwargs):
    """ Just like the map function - but with as many workers as you have cpus. """

    arguments = list(zip(*args)) if len(args) != 1 else list(args[0])

    if not parallel:
        return list(map(partial(func, **kwargs), arguments))

    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(partial(func, **kwargs), arguments)


def lazy_property(function):
    """ Decorator to make a lazily executed property """

    attribute = '_' + function.__name__

    @property
    @wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


def parse_function(func, kind=None):
    """
    Parse the doc string of a function for specifically formatted labels
    """

    values = {
        'function': func.__name__,
        'object': func,
        'arguments': tuple(signature(func).parameters.keys())
    }
    if kind is not None:
        values['kind'] = kind

    doc = func.__doc__
    if doc is None:
        return values

    desc, *labels = doc.split('\n\n')
    values['description'] = desc.lstrip().replace('\n    ', ' ')

    if not labels:
        return values

    labels = yaml.load(''.join(labels))
    if labels:
        values.update(labels)
    return values


def labeled_functions(kind):
    """
    Find any labeled functions in a module and return them with any labels
    from the doc string
    """

    def is_labeled(args):
        _, func = args
        if isroutine(func) and hasattr(func, 'label'):
            if getattr(func, 'label') == kind:
                return True

    module = sys.modules['Pointing.analysis.{}'.format(kind)]
    _, funcs = zip(*filter(is_labeled, getmembers(module)))
    return tuple(map(parse_function, funcs))
