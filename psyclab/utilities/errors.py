#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/utilities/errors.py
Vince Enachescu 2018

Custom exception classes used within the psyclab package
"""


class PsyclabException(Exception):
    pass


class DimensionMismatch(PsyclabException):
    pass


class InvalidParameterValue(PsyclabException):
    pass
