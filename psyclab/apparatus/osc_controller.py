

# import ctypes
import struct

import numpy as np

from pythonosc.osc_server import BlockingOSCUDPServer

from psyclab.utilities.osc import OSCResponder, route
from psyclab.sl import Robot


class OSCController(Robot, OSCResponder):
    """
    OSCController provides a base class to write controllers for SL robot
    models, which communicate state and commands through OSC messages.
    """

    def __init__(self, robot_name, robot_pid=None, apparatus_port=0, apparatus_host='127.0.0.1', start=False, attach=False, **kwargs):

        self._apparatus_host = apparatus_host
        self._apparatus_port = apparatus_port

        OSCResponder.__init__(self, host='0.0.0.0', port=apparatus_port + 1)
        Robot.__init__(self, robot_name, pid=robot_pid)

        if start:
            self.start(attach=attach)

    def start(self, attach=True):

        # connect to shared memory
        if attach:
            self.attach(self._pid)

        self._client = self.connect(self._apparatus_host, self._apparatus_port)
        self.debug(f'controller receiving from apparatus on {self._apparatus_host}, returning commands on {self._apparatus_port}')

        self._server = BlockingOSCUDPServer((self._host, self._port), self._dispatcher)
        self._host, self._port = self._server.socket.getsockname()
        self._server.serve_forever()

    @route('/stop')
    def stop(self, *args):

        OSCResponder.stop(self)

    @route('/step')
    def step_controller(self, route, source, time_step):

        time_step, = struct.unpack('f', time_step)
        data = struct.pack('f', time_step)
        self.send('/next', data)


if __name__ == "__main__":
    controller = OSCController('arm2D')
    controller.start()
