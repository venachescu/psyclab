#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/servo.py
Vince Enachescu 2019
"""

import ctypes

from os import path, setpgrp
from subprocess import Popen, PIPE, DEVNULL
from struct import unpack, unpack_from, pack, calcsize
from platform import platform
from sysv_ipc import ExistentialError

from .shared_memory import SharedMemoryObject, get_semaphore
from .config import robot_user_path, mach_type, _defines


class Servo(SharedMemoryObject):
    """
    Servo class is a wrapper to manage an sl servo subprocess, including a
    mapping to its shared memory. Messages can be written and read from the
    shared memory to control and collect data from the running servo

    """

    def __init__(self, servo_name, label, robot_name, parent_pid=0, xpath=None, launch=False,
                 text_color='Green', **kwargs):

        self.name = servo_name
        self.label = label
        self.robot_name = robot_name
        self.parent_pid = parent_pid
        self.text_color = text_color

        if xpath is None:
            self._xpath = path.join(robot_user_path(self.robot_name), mach_type)
        else:
            self._xpath = xpath

        self._xprocess = None
        self._max_messages = _defines['MAX_N_MESSAGES'] + 1
        self._max_bytes = _defines['MAX_BYTES_MESSAGES']

        if launch:
            self.start_xprocess(parent_pid, **kwargs)

        self._servo_semaphore = get_semaphore(self.servo_semname, parent_pid)

        shm_name = 'sm{}Message'.format(self.label)
        SharedMemoryObject.__init__(self, shm_name, robot_name, parent_pid, struct_name='smMessage')

        self._message_ready = get_semaphore(self.message_ready_semname, parent_pid)

    @property
    def servo_semname(self):
        return '{}.sm{}ServoSem'.format(self.robot_name, self.label)

    @property
    def message_ready_semname(self):
        return '{}.sm{}MsgReadySem'.format(self.robot_name, self.label)

    def start_xprocess(self, parent_pid, debug=False, w=80, h=10, x=0, y=0, graphics=True, headless=False, **kwargs):
        """ launch the servo process """

        xpath = path.join(self._xpath, 'x{}'.format(self.name))
        if not path.isfile(xpath):
            raise Exception('cannot start executable: {}'.format(xpath))

        args = [
            'xterm',
            '-wf',
            '-leftbar',
            '-geometry', '{}x{}-{}+{}'.format(w, h, x, y),
            '-bg', 'black',
            '-fg', self.text_color,
            '-title', '{} {} Servo'.format(self.robot_name, self.name.title()),
            '-e'
        ] if not headless else []

        args.extend([xpath] if not debug else _debug_command(xpath))
        if not graphics:
            args.extend('-ng')
        args.extend(['-pid', str(self.parent_pid)])

        self._xprocess = Popen(args, stdin=DEVNULL if headless else PIPE, stdout=DEVNULL if headless else PIPE, preexec_fn=setpgrp, close_fds=True)
        if self._xprocess.poll() is not None:
            print('[Error] could not start {}'.format(self.name))
            return self._xprocess.returncode

        # servo will create this semaphore when done initializing
        ready = get_semaphore('{}.smInitProcReadySem'.format(self.robot_name), self.parent_pid)
        ready.acquire()

    def stop(self, hard=False):
        """ stop the servo if running """

        if self.attached:
            self.detach()

        if self.running:
            self._xprocess.terminate() if not hard else self._xprocess.kill()
            self._xprocess = None

            try:
                self._message_ready.remove()
            except ExistentialError:
                pass

            try:
                self._servo_semaphore.remove()
            except ExistentialError:
                pass

    @property
    def running(self):
        if self._xprocess is not None:
            return (self._xprocess.poll() is None)
        return False

    def __del__(self):
        return self.stop()

    def __repr__(self):
        return '<{} {} Servo: Message Queue {}>'.format(
            self.robot_name, self.name, self.key)

    def read_messages(self):
        """ read in any current messages in the servo's message queue """

        # data = self.get_data()
        data = self.read(self.size)
        _, n_messages, n_bytes_used = unpack_from('lii', data, 0)

        offset = ctypes.sizeof(ctypes.c_long) + 2 * ctypes.sizeof(ctypes.c_int) + 20
        moffset = ctypes.sizeof(ctypes.c_long) + 2 * ctypes.sizeof(ctypes.c_int) + 2020 + 4

        messages = []
        for i in range(n_messages):

            raw_bytes = unpack_from('c' * 20, data, offset)
            name = b''.join(raw_bytes[:raw_bytes.index(b'\x00')]).decode()
            offset += 20

            boffset, = unpack_from('i', data, moffset)
            moffset += calcsize('i')

            if i == n_messages:
                size = n_bytes_used - boffset
            else:
                size = unpack_from('i', data, moffset)[0] - boffset

            content = b''.join(unpack_from('c' * size, data, 8 + 4 + 4 + 2020 + 404))
            messages.append(dict(name=name, data=content))

        return messages

    def write_message(self, name, data=None):
        """ send a message some data to servo's message queue """

        self._semaphore.acquire()

        offset = calcsize('l')
        isize = calcsize('i')
        name_length = 20

        n_messages = unpack('i', self.read(isize, offset=offset))[0]
        if n_messages >= self._max_messages:
            print('[Error] message buffer full')
            self._semaphore.release()
            return

        n_messages += 1
        self.write(pack('i', n_messages), offset=offset)
        offset += isize

        if data is None:
            size = 0
        else:
            if isinstance(data, int):
                size = isize
                data = pack('i', data)

            elif isinstance(data, float):
                size = isize
                data = pack('f', data)

            elif isinstance(data, str):
                data += '\0'
                size = len(data)

            elif isinstance(data, bytes):
                size = len(data)

        n_bytes_used = unpack('i', self.read(isize, offset=offset))[0]
        if n_bytes_used + size >= self._max_bytes:
            print('[Error] message buffer full')
            self._semaphore.release()
            return

        # new memory offset is the current total byte count
        moffset = n_bytes_used
        n_bytes_used += size
        self.write(pack('i', n_bytes_used), offset=offset)
        offset += isize

        name_offset = (n_messages) * name_length + offset
        self.write(pack('B' * 20, *(0,) * 20), offset=name_offset)
        self.write(name, offset=name_offset)

        moffset_offset = (n_messages) * isize + self._max_messages * name_length + offset
        self.write(pack('i', moffset), offset=moffset_offset)

        if data is not None:
            data_offset = self._max_messages * (isize + name_length) + offset
            self.write(data, offset=(moffset + data_offset))

        self._semaphore.release()
        self._message_ready.release()

    def wait_for_message(self, clear=True):
        """ wait for a message sent to this message queue """

        self._message_ready.acquire()
        messages = self.read_messages()

        if clear:
            self.clear_messages()
        return messages

    def clear_messages(self, zero=False):
        """ empty the message queue - writes 0 to the message and byte count """

        self._semaphore.acquire()
        self.write(pack('i', 0), offset=8)
        self.write(pack('i', 0), offset=12)

        # print zeros to the whole shared memory buffer
        if zero:
            n = self.size - 12
            self.write(pack('B'*n, *(0,)*n), offset=12)

        if self._message_ready.value:
            self._message_ready.acquire()

        self._semaphore.release()


def _debug_command(executable):
    if 'Darwin' in platform():
        return ['lldb', '-f', executable, '--']
    else:
        return ['gdb', '-ex=r', '--args', executable]
