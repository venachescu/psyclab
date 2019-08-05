#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/apparatus/parallel_apparatus.py
Vince Enachescu 2019
"""

import time
# import multiprocessing as mp

from queue import Queue
from threading import Lock, Condition, Semaphore

from psyclab.apparatus.apparatus import Apparatus
from psyclab.utilities.osc import OSCResponder, route
from psyclab.sl import Robot, robot_user_path, last_data_file

# magic number to make sure shared memory segments do not overlap
MYTOK_MAX = 19045


class ParallelApparatus(Apparatus):
    """
    This class allows for the parallel simulation and management of multiple
    SL robot models.

    """

    robot_launch = Lock()
    robot_queue = Queue()
    data_file = Condition()

    def __init__(self, robot_name, configuration='User', port=7401, host='0.0.0.0', **kwargs):

        self.robot_name = robot_name
        self.configuration = configuration
        self._user_path = robot_user_path(robot_name, configuration)

        self.robots = []
        self.controllers = {}
        self.ports = {}
        self.data_files = {}

        OSCResponder.__init__(self, host=host, port=port, **kwargs)

    def start(self, n_robots=1, headless=True, timeout=15, pause=0.1, **kwargs):
        """ Start apparatus and launch robot models """

        OSCResponder.start(self)

        for n in range(n_robots):
            self.start_robot(headless=headless, **kwargs)

        while self.robot_queue.qsize() < n_robots:

            time.sleep(pause)
            timeout -= pause
            if timeout <= 0.0:
                self.stop()
                self.warning('failed to start!')
                return

    def start_robot(self, **kwargs):

        self.robot_launch.acquire(blocking=True, timeout=1)

        if len(self.robots):
            pid = self.robots[-1]['pid'] + MYTOK_MAX
        else:
            pid = None

        robot = Robot(self.robot_name, user_path=self.user_path, pid=pid)
        self.robots.append({'robot': robot, 'pid': robot._pid})
        robot.start(**kwargs)

    def stop(self):

        for model in self.robots:
            model['robot'].stop()

        for client, controller in self.controllers.values():
            client.send_message('/stop', True)
            controller.terminate()

        OSCResponder.stop(self)

    def send_message(self, route, message, data=None, pause=None, pid=None):

        if pid is not None:
            robot = self.robot(pid=pid)
            robot['client'].send_message(route, message)
            return

        for model in self.robots:
            model['client'].send_message(route, message)
            if pause is not None:
                time.sleep(pause)

    @route('/apparatus/connect')
    def connect_apparatus(self, route, source, pid, port):

        host, _ = source
        self.debug(f'apparatus pid: {pid}, connection request {host}:{port}')

        client = self.connect(host, port)
        client.send_message('/apparatus', port)
        self.robots[-1]['client'] = client
        self.robots[-1]['port'] = port

        self.connected_apparatus(pid, host, port)

        self.robot_launch.release()
        # self.robot_queue.put(pid)

    @route('/controller/connect')
    def connect_controller(self, route, source, pid, port):

        host, source_port = source
        self.debug(f'apparatus pid {pid} controller connecting; {port}:{host}:{source_port}')
        self.controllers[pid] = self.start_controller(int(port), int(port) + 1, host, pid=pid)
        self.ports[source_port] = pid

        # self.robot_launch.release()
        time.sleep(2.0)
        self.robot_queue.put(pid)

    @route('/sl/data_file')
    def receive_data_file(self, route, source, n_data_file):

        _, port = source
        with self.data_file:
            self.data_files[self.ports[port]] = int(n_data_file)
            self.data_file.notify_all()

        self.debug(f'data file d{n_data_file:05d} saved; apparatus {source}')

    def robot(self, pid=None, port=None):

        if port is not None:
            return next(filter(lambda r: r['pid'] == self.ports[port], self.robots))

        return next(filter(lambda r: r['pid'] == pid, self.robots))

    @property
    def n_robots(self):
        return len(self.robots)

    @property
    def task_running(self):
        return [robot.get('task_running', False) for robot in self.robots]

    @property
    def last_data(self):
        """ Index of the current data file from SL data collection """
        return last_data_file(self.user_path)
