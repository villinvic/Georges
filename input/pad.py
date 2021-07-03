from config.loader import Default
from input.enums import Stick, Button, Trigger
from input.action_space import ControllerState

import zmq
import os
import platform


class Pad(Default):
    """Writes out controller inputs."""
    action_dim = 50

    def __init__(self, path, player_id, port=None):
        super(Pad, self).__init__()
        """Create, but do not open the fifo."""
        self.pipe = None
        self.path = path + 'georges_' + str(player_id)
        self.windows = port is not None
        self.port = port
        self.player_id = player_id
        self.message = ""
        self.action_space = []

        self.previous_state = ControllerState()

    def connect(self):
        if self.windows:
            context = zmq.Context()
            with open(self.path, 'w') as f:
                f.write(str(self.port))

            self.pipe = context.socket(zmq.PUSH)
            address = "tcp://127.0.0.1:%d" % self.port
            print("Binding pad %s to address %s" % (self.path, address))
            self.pipe.bind(address)
        else:
            try:
                os.unlink(self.path)
            except:
                pass

            os.mkfifo(self.path)

            self.pipe = open(self.path, 'w', buffering=1)

    def unbind(self):
        if not self.windows:
            self.pipe.close()
            try:
                os.unlink(self.path)
            except Exception:
                pass

        self.message = ""

    def flush(self):
        if self.windows:
            self.pipe.send_string(self.message)
        else:
            self.pipe.write(self.message)
        self.message = ""

    def write(self, command, buffering=False):
        self.message += command + '\n'
        if not buffering:
            self.flush()

    def press_button(self, button, buffering=False):
        """Press a button."""
        #assert button in Button or button in UsefullButton
        self.write('PRESS {}'.format(button), buffering)

    def release_button(self, button, buffering=False):
        """Release a button."""
        #assert button in Button or button in UsefullButton
        self.write('RELEASE {}'.format(button), buffering)

    def press_trigger(self, trigger, amount, buffering=False):
        """Press a trigger. Amount is in [0, 1], with 0 as released."""
        #assert trigger in Trigger or trigger in UsefullButton
        #assert 0 <= amount <= 1
        self.write('SET {} {:.2f}'.format(trigger, amount), buffering)

    def tilt_stick(self, stick, x, y, buffering=False):
        """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
        #assert stick in Stick
        #assert 0 <= x <= 1 and 0 <= y <= 1
        self.write('SET {} {:.2f} {:.2f}'.format(stick, x, y), buffering)

    def reset(self):
        for button in Button:
            self.release_button(button)
        for trigger in Trigger:
            self.press_trigger(trigger, 0)
        for stick in Stick:
            self.tilt_stick(stick, 0.5, 0.5)

