#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/utilities/hashish.py
Vince Enachescu 2019
"""

import hashlib
import six
import math

LONG_SCALE = float(0xFFFFFFFFFFFFFFF)


def hash_bernoulli(*args, p=0.5, salt=None):
    """ integer up to max_val randomized by hash space of the args strings """

    return 1 if hash_float(*args, salt=salt) <= p else 0


def hash_choice(*args, choices=None, salt=None):
    """ integer up to max_val randomized by hash space of the args strings """

    if not len(choices):
        raise Exception('no choices to choose from!', args)

    elif len(choices) == 1:
        return choices[0]

    index = hash_integer(*args, max_val=len(choices)-1)
    return choices[index]


def hash_weighted_choice(*args, choices=[], weights=[], salt=None):
    """ integer up to max_val randomized by hash space of the args strings """

    if len(choices) != len(weights):
        raise Exception('each choice must have a weight')

    if not len(choices):
        raise Exception('no choices to choose from!')

    elif len(choices) == 1:
        return choices[0]

    choices = [c
        for choice, weight in zip(choices, weights)
        for c in [choice] * int(round(weight*100))
    ]

    return hash_choice(*args, choices=choices, salt=salt)


def hash_boolean(*args, salt=None):
    """ boolean randomized by hash space of the args strings """

    return bool(hash_integer(*args, max_val=1, salt=salt))


def hash_integer(*args, max_val=1, salt=None):
    """ integer up to max_val randomized by hash space of the args strings """

    return hash_number(*args, salt=salt) % (max_val+1)


def hash_float(*args, min_val=0.0, max_val=1.0, salt=None):
    """ integer up to max_val randomized by hash space of the args strings """

    value = hash_number(*args, salt=salt) / LONG_SCALE
    return min_val + (max_val - min_val) * value


def hash_number(*args, salt=None):
    """ transform the strings in args to a hashed number """

    return int(hash_digest(*args, salt=salt), 16)


def hash_normal(*args, salt=None):
    """ generate a hash randomized number from a normal distribution using a box-mueller transform """

    f1 = hash_float(*args, salt=salt)
    f2 = hash_float(*reversed(args), salt=salt)
    z1 = math.sqrt(-2 * math.log(f1)) * math.cos(2 * math.pi * f2)
    z2 = math.sqrt(-2 * math.log(f1)) * math.sin(2 * math.pi * f2)
    return hash_choice(*args, choices=[z1, z2], salt=salt)


def hash_digest(*args, salt=None, length=16):
    """ transform the strings in args to a hash hexdigest """

    hash_str = ':'.join(['{}'.format(arg) for arg in args])
    if salt is not None:
        hash_str += str(salt)

    if not isinstance(hash_str, six.binary_type):
        hash_str = hash_str.encode('ascii')

    return hashlib.sha1(hash_str).hexdigest()[:length - 1]


if __name__ == "__main__":

    choices = 'abcdefghij'
    weights = [float(i) / sum(range(10)) for i in range(10)]

    for i in range(10):
        print(i, hash_weighted_choice(choices, weights, chr(i)))
        # print(i, hash_number(chr(i)))
        # print(i, hash_boolean(chr(i)))
        # print(i, hash_integer(chr(i), max_val=100))
        # print(i, hash_bernoulli(0.6, chr(i)))
