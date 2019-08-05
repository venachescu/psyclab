#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/robot.py
Vince Enachescu 2019
"""

from os import path, environ, chdir, getcwd, getpid, setpgrp, kill
from subprocess import Popen
from sysv_ipc import ExistentialError

from .servo import Servo
from .cstruct import replace_defines
from .shared_memory import SharedMemoryObject, get_semaphore, get_process_pid, shared_memory_names, semaphore_names
from .config import robot_path, robot_user_path, read_robot_config, read_sample_prefs


class Robot(object):
    """
    Robot class provides a wrapper around an SL robot model and manage a running
    instance.
    """

    joints = []
    links = []
    endeffectors = []

    def __init__(self, robot_name, user_path=None, pid=None):
        """
        Args:
            robot_name (str): name of robot model, or full path to the model
            user_path (str, optional): path to the user directory of the robot
            pid (int, optional): parent pid process for the specific robot -
                used to seed shared memory keys and uniquely identify robot
        """

        if path.isabs(robot_name) and path.isdir(robot_name):
            self._model_path = robot_name
            robot_name = path.basename(robot_name)
        else:
            self._model_path = robot_path(robot_name)

        self._user_path = user_path or robot_user_path(robot_name)
        self._robot_name = robot_name
        self._pid = pid or getpid()

        self._servos = {}
        self.semaphores = {}
        self.shared_memory = {}
        self.prefs = {}

        # robot configuration
        config = read_robot_config(self._model_path)
        [setattr(self, name, value) for name, value in config.items()]

        # sampling preferences
        self.prefs = read_sample_prefs(self._user_path)

    def start(self, headless=False, debug=False, xrobot=False, **kwargs):
        """
        Start the robot simulation, each servo can be turned on, off or to 'debug'

        Args:
            [task,motor,opengl,simulation] (bool|str): state of the associated servo process; True to launch, False
                to skip launching the subprocess or the string `'debug'` to run the process within a debugger
            xrobot (bool): launch the robot using its xrobot process, default `False`
            headless (bool): launch each of the subprocess in its own x11 terminal window, default `False`
        """

        if self.running:
            # TODO: make a warning
            print('[Error] {} already running'.format(self._robot_name))
            return

        # always make sure to start in working path
        if getcwd() != self._user_path:
            chdir(self._user_path)

        if xrobot:
            if not self._start_xrobot(**kwargs):
                return False

        positions = (
            {'x': 10, 'y': 160 * i}
            for i in range(len(self._servo_args))
        )

        for args in self._servo_args:

            args.update(dict(debug=debug, headless=headless))

            if xrobot:
                args['launch'] = False
            elif args['name'] in kwargs:
                launch = kwargs.pop(args['name'])
                if launch in (True, False):
                    args['launch'] = launch
                elif launch == 'debug':
                    args['debug'] = launch
                elif launch == 'headless':
                    args['headless'] = launch

            if args['launch']:
                args.update(next(positions))

            args.update(kwargs)
            self._start_servo(**args)

        # attach to shared memory and semaphores
        self.attach(self._pid)

        for servo in self.servos:
            servo.read_messages()

        return True

    def _start_xrobot(self, opengl=True, **kwargs):
        """ start the robot externally using the xrobot executable """

        xpath = path.join(self._user_path, environ['MACHTYPE'], 'x{}'.format(self._robot_name))
        args = [xpath,]
        if not opengl:
            args.append('-ng')

        xprocess = Popen(xpath, preexec_fn=setpgrp)
        self._pid = xprocess.pid
        if xprocess.wait():
            print('[Error] problem starting robot')
            return False

        self._xrobot = True
        # self._xrobot = get_process_pid(self._robot_name)
        return self._pid

    def _start_servo(self, name, label, launch=True, headless=False, debug=False, **kwargs):
        """ create and start a servo object """

        args = {
            'xpath': path.join(self._user_path, environ['MACHTYPE']),
            'robot_name': self._robot_name,
            'parent_pid': self._pid,
            'headless': headless,
            'launch': launch,
            'debug': debug
        }
        args.update(kwargs)

        self._servos[label] = Servo(name, label, **args)
        setattr(self, name, self._servos[label])
        return True

    def attach(self, pid):
        """ attach to the shared memory and semaphores of a running robot """

        for name in semaphore_names:
            self.semaphores[name] = get_semaphore('{}.{}'.format(self._robot_name, name), pid, time_out=0.2)

        for name, struct_name, n_data in shared_memory_names:
            n_data = replace_defines(n_data, self.defines)
            if not n_data:
                continue
            self.shared_memory[name] = SharedMemoryObject(name, self._robot_name, pid, struct_name=struct_name, n_data=n_data)

    @property
    def servos(self):
        return self._servos.values()

    @property
    def running(self):
        if hasattr(self, '_xrobot'):
            return self._xrobot
        if not len(self._servos):
            return False
        return any(servo.running for servo in self.servos)

    def stop(self, hard=False):
        """ stop the SL robot simulation """

        names = list(self.shared_memory.keys())
        for name in names:
            shm = self.shared_memory.pop(name)
            if shm.attached:
                shm.detach()
            try:
                shm.remove()
            except:
                continue
            del shm

        names = list(self.semaphores.keys())
        for name in names:
            try:
                self.semaphores.pop(name).remove()
            except ExistentialError:
                pass

        if self.running:
            # if hasattr(self, '_xrobot'):
                # if hard:
                    # kill(self._xrobot, 9)
                # else:
                    # kill(self._xrobot, 15)
            # else:
            [servo.stop(hard=hard) for servo in self.servos]

        return True

    def pause(self):
        """ pause the simulation using the pause semaphore """

        if not self.running:
            return

        semaphore = self.semaphores['smPauseSem']
        return semaphore.release()

    @property
    def joint_state(self):
        """ get current joint states from shared memory, defaults to """

        shm = self.shared_memory.get('smJointState')
        if shm is None:
            return

        struct = shm.data
        return {
            joint: {
                'th': state.th,
                'thd': state.thd,
                'thdd': state.thdd,
                'load': state.load,
                'u': state.u,
                'ufb': state.ufb
            } for joint, state in zip(self.joints, struct.joint_state[1:])
        }

    @property
    def joint_desired_state(self):
        """ get current joint desired states, defaults to SL_DJstate form """
        shm = self.shared_memory.get('smJointDesState')
        if shm is None:
            return

        struct = shm.data
        return {
            joint: {
                'th': state.th,
                'thd': state.thd,
                'thdd': state.thdd,
                'uff': state.uff,
                'uex': state.uex
            } for joint, state in zip(self.joints, struct.joint_des_state[1:])
        }

    @property
    def cart_state(self):
        """ get current endeffector state, defaults to SL_Cstate """

        shm = self.shared_memory.get('smCartStates')
        if shm is None:
            return

        struct = shm.data
        return {
            endeffector: {
                'x': state.x[1:],
                'xd': state.xd[1:],
                'xdd': state.xdd[1:],
            } for endeffector, state in zip(self.endeffectors, struct.state[1:])
        }

    def change_real_time(self, toggle=True):
        """ toggle the real time simulation speed """

        if not self.running:
            return

        self.simulation.write_message('changeRealTime', int(toggle))

    def set_task(self, task_name):
        """ set task by name from the user tasks installed """

        if not self.running:
            return

        self.task.write_message('setTask', task_name)

    def sl_command(self, command, data=None):
        """ send a command to the task servo's message queue """

        if not self.running:
            return

        self.task.write_message('setCommand', command)

    def __del__(self):
        self.stop()

    _servo_args = [
        {'name': 'task', 'label': 'Task', 'launch': True, 'text_color': 'LimeGreen'},
        {'name': 'motor', 'label': 'Motor', 'launch': True, 'text_color': 'SteelBlue1'},
        {'name': 'opengl', 'label': 'OpenGL', 'launch': (environ.get('DISPLAY') is not None), 'text_color': 'MediumOrchid1'},
        {'name': 'simulation', 'label': 'Sim', 'launch': True, 'text_color': 'DeepPink1'},
        {'name': 'vision', 'label': 'Vision', 'launch': False, 'text_color': 'Chatreuse1'},
        {'name': 'ROS', 'label': 'ROS', 'launch': False, 'text_color': 'HotPink'}
    ]
