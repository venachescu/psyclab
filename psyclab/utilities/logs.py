#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/utilities/logs.py
Vince Enachescu 2019
"""

import logging

from psyclab.utilities.tools import lazy_property


class Logged:

    _logging_level = logging.DEBUG
    _log = None

    def __init__(self, log_level='debug'):

        if Logged._log is not None:
            return

        Logged._logging_level = getattr(logging, log_level.upper())

    @lazy_property
    def _logger(self, *args, **kwargs):

        if Logged._log is not None:
            return Logged._log

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self._logging_level)

        # create console handler and set level to debug
        handler = logging.StreamHandler()
        handler.setLevel(self._logging_level)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        Logged._log = logger
        return logger

    def debug(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def log(self, *args, **kwargs):
        return self._logger.log(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self._logger.warning(*args, **kwargs)

    def info(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)
