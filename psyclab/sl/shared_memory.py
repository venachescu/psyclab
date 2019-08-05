#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/shared_memory.py
Vince Enachescu 2019
"""

import ctypes

from time import sleep

from subprocess import check_output, STDOUT, CalledProcessError
from sysv_ipc import Semaphore, SharedMemory, ExistentialError

from .cstruct import CStruct, read_struct_format
from .config import header_path, _defines, _types

_header_path = header_path('SL_shared_memory.h')


class SharedMemoryObject(SharedMemory):

    _data_format = []

    def __init__(self, name, robot_name, pid, struct_name=None, n_data=1):

        self._name = name
        self._robot_name = robot_name
        self._struct_name = struct_name

        self._data_type = None
        self._struct_type = None
        self._semaphore = None

        fields = self.read_format(n_data=n_data)
        self._struct_type = type(name, (CStruct,), {'_fields_': fields})

        try:
            key = get_semaphore_key(self.shmemory_name, pid)

            # weird bug - only smMiscSimSensors finds a different key
            if 'smMiscSimSensors' in self.shmemory_name:
                key = key + 1

            SharedMemory.__init__(self, key, size=self.size)
        except ExistentialError:
            raise Exception('Cannot access shared memory {}'.format(self.shmemory_name))

        self._semaphore = get_semaphore(self.semaphore_name, pid)

    @property
    def shmemory_name(self):
        return '{0}.{1}'.format(self._robot_name, self._name)

    @property
    def struct_name(self):
        if self._struct_name is None:
            return self._name
        return self._struct_name

    @property
    def semaphore_name(self):
        return '{}.{}_sem'.format(self._robot_name, self._name)

    def read_format(self, n_data=1):

        fields = read_struct_format(_header_path, self.struct_name, defines=_defines, types=_types)
        field_name, array_type = fields[-1]
        self._data_type = array_type._type_

        if n_data == 1:
            return fields

        fields[-1] = (field_name, self._data_type * n_data)
        return fields

    @property
    def size(self):
        return ctypes.sizeof(self._struct_type)

    @property
    def data(self):

        self._semaphore.acquire()
        raw_data = self.read(self.size)
        self._semaphore.release()

        return self._struct_type.from_buffer_copy(raw_data)

    def __del__(self):
        if self.attached:
            self.detach()


def get_process_pid(robot_name):
    """ find the process id for a currently running sl robot """

    try:
        result = check_output(['pgrep', 'x{0}'.format(robot_name)])
        return int(result.strip())
    except:
        return None


def get_semaphore_key(name, pid):
    """ get the key for a semaphore from its name string """

    key = pid
    for i in range(0, len(name)):
        key += i * 100 + ord(name[i])
    return key


def get_semaphore(name, pid, time_out=None, pause=0.1):
    """ get semaphore by name, will wait until time out (default forever) """

    semaphore_key = get_semaphore_key(name, pid)
    while True:
        try:
            return Semaphore(semaphore_key)

        except ExistentialError:
            sleep(pause)

        if time_out is not None:
            time_out -= pause
            if time_out <= 0.0:
                return None


def check_semaphore(semaphore_key):
    """ confirm if semaphore exists by key """

    if isinstance(semaphore_key, str):
        semaphore_key = eval(semaphore_key)

    keys = [eval(sem['key']) for sem in list_semaphores()]
    return semaphore_key in keys


def clear_shared_memory():
    """ clear all semaphores, shared memory and message queues """

    for shmtype in ('-m', '-s', '-q'):
        items = _list_ipcs(shmtype)
        for item in items:
            try:
                check_output(('ipcrm', shmtype, item['id']), stderr=STDOUT)
            except CalledProcessError:
                pass


def list_shared_memory():
    return _list_ipcs('-m')


def list_semaphores():
    return _list_ipcs('-s')


def _list_ipcs(shmtype):

    output = check_output(('ipcs', shmtype)).decode()

    # mac os (dariwn), bsd, etc.
    if output.startswith('IPC'):
        _, fields, _, *values = output.split('\n')
        fields = [field.lower() for field in fields.split()]

    # linux
    else:
        _, _, fields, *values = output.split('\n')
        fields = [field.lower().replace('semid', 'id').replace('shmid', 'id') for field in fields.split()]

    return [dict(zip(fields, value.split())) for value in filter(None, values)]


# unused shared memory segments are commented out, but may be re-enabled as needed
shared_memory_names = [
        ('smJointState', 'smJointStates', 'N_DOFS+1'),
        ('smJointDesState', 'smJointDesStates', 'N_DOFS+1'),
        # ('smSJointDesState', 'smSJointDesStates', 'N_DOFS+1'),
        # ('smJointSimState', 'smJointSimStates', 'N_DOFS+1'),
        # ('smVisionBlobs', 'smVisionBlobs', 'MAX_BLOBS+1'),
        # ('smVisionBlobsaux', 'smVisionBlobsaux', 'MAX_BLOBS+1'),
        # ('smRawBlobs', 'smRawBlobs', 'MAX_BLOBS+1'),
        # ('smRawBlobs2D', 'smRawBlobs2D', 'MAX_BLOBS+1'),
        ('smCartStates', 'smCartStates', 'N_ENDEFFS+1'),
        ('smMiscSensors', 'smMiscSensors', 'N_MISC_SENSORS+1'),
        ('smMiscSimSensors', 'smMiscSimSensors', 'N_MISC_SENSORS+1'),
        # ('smContacts', 'smContacts', 'CONTACTS+1'),
        # ('smBaseState', 'smBaseState', '2'),
        # ('smBaseOrient', 'smBaseOrient', '2'),
        ('smUserGraphics', 'smUserGraphics', '2'),
        ('smDCommands', 'smDCommands', 'N_DOFS+1')
    ]

semaphore_names = (
    'smJointDSReadySem',
    'smSJointDSReadySem',
    # 'smRawBlobsReadySem',
    # 'smLearnInvdynSem',
    # 'smLearnBlob2BodySem',
    # 'smOscilloscopeSem',
    'smPauseSem',
    # 'smUGraphReadySem',
    # 'smObjectsReadySem',
    'smInitProcReadySem',
)
