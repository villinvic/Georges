import binascii
import zmq
import socket
import os
import platform
import signal

def chunk(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def parseMessage(message):
    lines = message.splitlines()

    assert (len(lines) % 2 == 0)

    diffs = chunk(lines, 2)

    for diff in diffs:
        diff[1] = binascii.unhexlify(diff[1].zfill(8))

    return diffs


class MemoryWatcherZMQ:
    """Reads and parses game memory changes.

    Pass the location of the socket to the constructor, then either manually
    call next() on this class to get a single change, or else use it like a
    normal iterator.
    """

    def __init__(self, path, ID):
        super(MemoryWatcherZMQ, self).__init__()

        self.path = path + '/MemoryWatcher'
        self.id = ID
        self.messages = None
        self.windows = platform.system() == 'Windows'

        # write_with_folder(self.path, '5555')

        context = zmq.Context()
        if self.windows:
            self.socket = context.socket(zmq.PULL)
            self.socket.bind("tcp://127.0.0.1:%d" % 5555)
        else:
            self.socket = context.socket(zmq.REP)
            #self.socket.bind("ipc://" + self.path+str(self.id))
            self.socket.setsockopt(zmq.RCVTIMEO, 15000)
            self.socket.setsockopt(zmq.LINGER, 0)

    def __del__(self):
        """Closes the socket."""
        try:
            self.socket.unbind("ipc://" + self.path + str(self.id))
        except Exception:
            pass

    def bind(self):
        self.socket.bind("ipc://" + self.path + str(self.id))

    def unbind(self):
        try:
            self.socket.unbind("ipc://" + self.path + str(self.id))
        except zmq.ZMQError:
            pass


    def __iter__(self):
        """Iterate over this class in the usual way to get memory changes."""
        return self

    def __next__(self):
        """Returns the next (address, value) tuple, or None on timeout.

        address is the string provided by dolphin, set in Locations.txt.
        value is a four-byte string suitable for interpretation with struct.
        """
        self.advance()
        return self.get_messages()

    def get_messages(self):
        if self.messages is None:
            try:
                message = self.socket.recv()
                message = message.decode('utf-8')
                self.messages = parseMessage(message)
            except zmq.ZMQError as e:
                print(e)
                return -1

        return self.messages

    def advance(self):
        if not self.windows:
            try:
                self.socket.send(b'')
            except zmq.ZMQError:
                pass
        self.messages = None


class MemoryWatcher:
    """Reads and parses game memory changes.

    Pass the location of the socket to the constructor, then either manually
    call next() on this class to get a single change, or else use it like a
    normal iterator.
    """

    def __init__(self, path):
        try:
            os.unlink(path)
        except OSError:
            pass
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.settimeout(1)
        self.socket.bind(path)

    def __exit__(self, *args):
        """Closes the socket."""
        pass

    def unbind(self):
        self.socket.close()

    def __iter__(self):
        """Iterate over this class in the usual way to get memory changes."""
        return self

    def __next__(self):
        """Returns the next (address, value) tuple, or None on timeout.

        address is the string provided by dolphin, set in Locations.txt.
        value is a four-byte string suitable for interpretation with struct.
        """

        return self.get_messages()

    def get_messages(self):
        try:
            message = self.socket.recv(1024).decode('utf-8')
            message = message.strip('\x00')
            messages = parseMessage(message)
        except socket.timeout:
            return []

        return messages

    def advance(self):
        pass