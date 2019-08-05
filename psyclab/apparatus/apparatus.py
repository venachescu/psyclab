#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import re

import multiprocessing as mp

from os import path, makedirs, listdir, remove
from shutil import copyfile
from threading import Event

from psyclab.utilities.osc import OSCResponder, route
from psyclab.apparatus.osc_controller import OSCController

from psyclab.sl import Robot, robot_user_path
from psyclab.sl.data_files import read_sl_data_file, last_data_file, InvalidSLDataFile


class Apparatus(OSCResponder):
    """
    Apparatus class used to connect an SL robot model to an experiment controller
    """

    data_file = Event()

    _controller_class = OSCController

    def __init__(self, robot_name, configuration='User', host='0.0.0.0', port=7401, **kwargs):

        OSCResponder.__init__(self, host=host, port=port, **kwargs)

        # to handle multiple udp clients
        self._client = {}

        self.n_data_file = None
        self.configuration = configuration
        self.robot_name = robot_name

        self.controller = None
        self.robot = Robot(robot_name, user_path=robot_user_path(robot_name, configuration))

    def start_robot(self, *args, **kwargs):

        if self.robot is not None:
            if self.robot.running:
                return

        return self.robot.start(**kwargs)

    def stop(self, *args):

        if self.controller:
            self._client['controller'].send_message('/stop', True)
            self.controller.terminate()

        self.robot.stop()
        OSCResponder.stop(self)

    def set_task(self, task_name):

        if not self.running:
            return

        self.send('/task', task_name, to='apparatus')

    @route('/apparatus/connect')
    def connect_apparatus(self, route, source, *messages):
        """ Incoming connection request from apparatus """

        pid, port = messages
        host, _ = source

        self._client['apparatus'] = self.connect(host, port)
        self.debug(f'apparatus pid:{pid} connecting from {host}:{port}')

        self.connected_apparatus(pid, host, port)

    def connected_apparatus(self, *args):
        """ Connected as client to apparatus callback function """

        pid, host, port = args
        self.send('/apparatus', port, to='apparatus')

    @route('/sl/data_file')
    def receive_data_file(self, route, source, n_data_file):

        host, port = source
        self.n_data_file = int(n_data_file)
        self.info(f'data file d{n_data_file:05d} saved; apparatus {host}:{port}')

        self.data_file.set()
        self.data_file.clear()

    @property
    def user_path(self):
        """ Active user path for the SL robot model """
        if hasattr(self, '_user_path'):
            return self._user_path
        return self.robot._user_path

    @property
    def last_data(self):
        """ Index of the current data file from SL data collection """
        return last_data_file(self.user_path)

    def reset_data_index(self):
        """ Reset the data file index """
        try:
            remove(path.join(self.user_path, '.last_data'))
        except FileNotFoundError:
            pass

    def load_data_file(self, n_data_file, retry=3, pause=0.25):
        """ Read an SL data file into a metadata header and a dictionary of numpy arrays """

        if n_data_file is None:
            return

        while retry:

            # if the data file has not been written (or not fully written), handle exception and retry
            try:
                header, data = read_sl_data_file(f'd{n_data_file:05d}')
                return header, data

            except (ValueError, FileNotFoundError, InvalidSLDataFile):

                # self.warning(f'failed to read d{n_data_file:05d}, retrying... ({retry})')
                time.sleep(pause)
                retry -= 1
                pause *= 2

    def remove_data_files(self, archive_path=None, confirm=False):
        """ Remove all saved data files in the user path; optional archiving """

        for data_file in listdir(self.user_path):
            if not re.match('d\d{5}', data_file):
                continue
            if confirm:
                remove(path.join(self.user_path, data_file))
            else:
                print('rm ' + path.join(self.user_path, data_file))

    def archive_data_files(self, archive_path, make_paths=True):
        """ Archive all data files in the user path """

        if make_paths:
            makedirs(archive_path)

        for data_file in listdir(self.user_path):
            if not re.match('d\d{5}', data_file):
                continue
            copyfile(data_file, path.join(archive_path, data_file))

    def start_controller(self, server_port, client_port, apparatus_host, pid=None):

        kwargs = {
            'robot_name': self.robot_name,
            'robot_pid': pid or self._pid,
            'apparatus_port': server_port,
            'apparatus_host': apparatus_host,
            'configuration': self.configuration,
            'start': True,
        }

        ctx = mp.get_context('spawn')
        controller = ctx.Process(target=self._controller_class, kwargs=kwargs)
        controller.start()

        client = self.connect(apparatus_host, client_port)

        self.debug(f'started {self._controller_class.__name__}, listening {server_port}, sending {client_port}')
        return client, controller

    @route('/controller/connect')
    def connect_controller(self, route, source, pid, port):

        host, source_port = source
        self.debug(f'apparatus pid {pid} controller connecting; {port}:{host}:{source_port}')
        if self.controller:
            self.error('new controller not connected - controller already running!')
            return

        client, controller = self.start_controller(int(port), int(port) + 1, host, pid=pid)
        self.controller = controller
        self._client['controller'] = client

