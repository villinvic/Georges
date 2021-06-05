from input.enums import UsefullButton

import itertools
import numpy as np



class ControllerState(dict):
    def __init__(self, button=None, stick=(0.5, 0.5), c_stick=(0.5, 0.5), duration=3, no_op=False):
        dict.__init__(self)
        for item in UsefullButton:
            if button and item.name == button:
                self[button] = True
            else:
                self[item.name] = False

        self.to_press = np.array([button]) if button is not None else np.array([], np.int32)

        self['stick'] = stick
        self['c_stick'] = c_stick
        self['duration'] = duration
        self['id'] = None
        self.no_op = no_op

    def __str__(self):

        string = ""
        for item in UsefullButton:
            string += "%s(%d) " % (item.name, self[item.name])
        string += "S(%s,%s) " % self['stick']
        string += "C(%s,%s)" % self['c_stick']

        return string

    def send_controller(self, pad):
        if not self.no_op:
            for button in np.setdiff1d(pad.previous_state.to_press, self.to_press):
                pad.release_button(button, buffering=True)
            for button in self.to_press:
                pad.press_button(button, buffering=True)
            pad.tilt_stick('MAIN', *self['stick'], buffering=True)
            pad.tilt_stick('C', *self['c_stick'], buffering=True)
            pad.flush()
            pad.previous_state = self


class ActionSpace:
    neutral = [
        (0.5, 0.5)
    ]

    back_front = [
        (0., 0.5),
        (1., 0.5)
    ]

    cardinal = [
        (0.5 * (1 + np.cos(x)), 0.5 * (1 + np.sin(x))) for x in np.arange(0, 4) * np.pi * 0.5
    ]

    diagonal = [
        (0.5 * (1 + np.cos(x)), 0.5 * (1 + np.sin(x))) for x in np.arange(1, 8, 2) * np.pi * 0.25
    ]

    other_angles = [
        (0.5 * (1 + np.cos(x)), 0.5 * (1 + np.sin(x))) for x in np.arange(1, 17, 2) * np.pi * 0.125
    ]

    all_stick_states = neutral + cardinal + diagonal + other_angles

    stick_states_upB = [
        (0.853, 0.853),
        (0.146, 0.853),
    ]

    smash_states = cardinal

    tilt_stick_states = [
        (0.3, 0.5),
        (0.7, 0.5),
        (0.5, 0.3),
        (0.5, 0.7)
    ]

    wave_dash_sticks = [
        (0.023, 0.353),
        (0.977, 0.353),

    ]

    def __getitem__(self, item):
        return self.controller_states[item]

    def __init__(self, short_hop=3):
        self.len = 0

        """
        1 directions with no button
        2 tap b, lazer, plumbers downb
        3 tilts
        4 smashes with c stick and cardinals
        5 jumps, normal, front, back, full hop, short hop
        6 upbs
        7 L with all directions (rolls, spot dodge, shield, air dodges...)
        8 wavedashes
        9 shield grab, jump grab
        10 tap down b, tap side b
        11 noop

        """

        all_states = [ControllerState(stick=stick) for stick in self.all_stick_states] + \
                     [[ControllerState(stick=self.cardinal[-1], button='B', duration=1), ControllerState(duration=1)]] + \
                     [ControllerState(stick=stick, button='A') for stick in (self.tilt_stick_states + self.neutral)] + \
                     [ControllerState(stick=stick, c_stick=c_stick) for stick, c_stick in
                      itertools.product(self.cardinal + self.neutral, self.cardinal)] + \
                     [ControllerState(stick=stick, button='X', duration=short_hop + 1) for stick in
                      (self.back_front + self.neutral)] + \
                     [[ControllerState(stick=stick, button='X', duration=short_hop),
                       ControllerState(stick=stick, duration=1)] for stick in (self.back_front + self.neutral)] + \
                     [ControllerState(stick=upB_stick, button='B') for upB_stick in self.stick_states_upB] + \
                     [ControllerState(stick=stick, button='L') for stick in
                      (self.neutral + self.cardinal + self.diagonal)] + \
                     [[ControllerState(stick=stick, button='X', duration=short_hop),
                       ControllerState(stick=stick, duration=1), ControllerState(stick=stick, duration=1)] for stick in
                      (self.wave_dash_sticks)] + \
                     [[ControllerState(stick=stick, button='B', duration=1), ControllerState(stick=stick, duration=1)]
                      for stick in (self.back_front)] + \
                     [ControllerState(button='B')] + \
                     [[ControllerState(button='X', duration=1), ControllerState(button='Z', duration=2)]] + \
                     [ControllerState(no_op=True)]

        shield_grab = ControllerState()
        shield_grab.to_press = np.array(['A', 'L'])
        all_states += [shield_grab]

        self.controller_states = np.array(
            all_states
        , dtype=ControllerState)

        self.dim = len(self.controller_states)