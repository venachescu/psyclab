#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/utilities/osc.py
Vince Enachescu 2019
"""

from functools import wraps
from itertools import chain
from threading import Thread

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from psyclab.utilities.logs import Logged


def route(*paths):
    """ Decorator to add functions as listeners for a set of OSC routes """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 0:
                return func(None, None, **kwargs)
            return func(*args, **kwargs)

        wrapper.paths = paths
        return wrapper

    return decorator


def is_route(obj):
    def check_route(name):
        member = getattr(obj, name, None)
        if callable(member) and hasattr(member, 'paths'):
            return [(p, member) for p in member.paths]
    return check_route


def find_routes(obj):
    return tuple(chain(*filter(None, map(is_route(obj), dir(obj)))))


class OSCResponder(Logged):

    def __init__(self, host='0.0.0.0', port=7401, **kwargs):

        self._host = host
        self._port = port
        self._client = None
        self._server = None
        self._server_thread = None

        Logged.__init__(self, **kwargs)

        self._dispatcher = Dispatcher()
        self._dispatcher.set_default_handler(self.receive)

        self._routes = {}
        for osc_path, callback in find_routes(self):
            self._dispatcher.map(osc_path, callback)
            self._routes[osc_path] = callback

    def start(self, *args):

        if self._server is not None:
            return

        self._server = ThreadingOSCUDPServer((self._host, self._port), self._dispatcher, )
        self._server_thread = Thread(target=self._server.serve_forever, name=str(self), daemon=True)
        self._server_thread.start()

        self._host, self._port = self._server.socket.getsockname()

        self.debug('responder thread started.')

    def stop(self, *args):

        if self._server is not None:
            self._server.shutdown()

        if self._server_thread is not None:
            self._server_thread.join()
            self.debug('responder thread stopped.')

    def send(self, route, message, to=None):

        if self._client is None:
            return

        if not isinstance(self._client, dict):
            self._client.send_message(route, message)
            return

        if to is not None:
            if not isinstance(to, (tuple, list)):
                to = (to,)
            for name in filter(lambda key: key in self._client, to):
                self._client[name].send_message(route, message)
            return

        for client in self._client.values():
            client.send_message(route, message)

    def receive(self, route, source, *messages):

        self.info(f'[osc] {source[0]}:{source[1]} {route}')
        for message in messages:
            self.info(message)

    # @route('/connect')
    # def on_connect(self, route, source, data):
    #     """ Handle a connection call over OSC """
    #
    #     host, port = source
    #     client = self.connect(host, data)
    #     if not isinstance(self._client, dict):
    #         self._client = client
    #     else:
    #         self._client[port] = client
    #
    #     self.connected(host, port, data)

    @staticmethod
    def connect(host='localhost', port=7402):
        """ Open an OSC client for network communication """
        return SimpleUDPClient(host, port)

    def connected(self, host, port, data):
        """ Callback after an OSC connection created """
        self.info(f'client connection, from {host}:{port}, port {data}')

    @property
    def client(self):
        return self._client

    def __str__(self):
        return f'{self.__class__.__name__}@{self._host}:{self._port}'

    def __repr__(self):
        return f'<{self}>'


if __name__ == "__main__":
    responder = OSCResponder()
    responder.start()
