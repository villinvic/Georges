import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List

from game.enums import *

def int_handler(state, name, shift=0, mask=0xFFFFFFFF, wrapper=None, default=0):
    """Returns a handler that sets an attribute for a given object.

    obj is the object that will have its attribute set. Probably a State.
    name is the attribute name to be set.
    shift will be applied before mask.
    Finally, wrapper will be called on the value if it is not None. If wrapper
    raises ValueError, sets attribute to default.

    This sets the attribute to default when called. Note that the actual final
    value doesn't need to be an int. The wrapper can convert int to whatever.
    This is particularly useful for enums.
    """

    def handle(value):
        transformed = (struct.unpack('>i', value)[0] >> shift) & mask
        state[name] = generic_wrapper(transformed, wrapper, default)

    state[name] = default
    return handle


def float_handler(state, name, wrapper=None, default=0.0):
    """Returns a handler that sets an attribute for a given object.

    Similar to int_handler, but no mask or shift.
    """

    def handle(value):
        as_float = struct.unpack('>f', value)[0]
        state[name] = generic_wrapper(as_float, wrapper, default)

    state[name] = default
    return handle


def generic_wrapper(value, wrapper, default):
    if wrapper is not None:
        try:
            value = wrapper(value)
        except ValueError:
            value = default
    return value


def add_address(x, y):
    """Returns a string representation of the sum of the two parameters.

    x is a hex string address that can be converted to an int.
    y is an int.
    """
    return "{0:08X}".format(int(x, 16) + y)


class StateManager:
    """Converts raw memory changes into attributes in a State object."""

    '''
    TODO : projectiles
    '''

    def __init__(self, state, test=False):
        """Pass in a State object. It will have its attributes zeroed."""
        self.state = state
        self.addresses = {}

        if test:
            self.addresses['804D7420'] = int_handler(self.state, 'frame')
        else:  # If not testing, we are running with speedhack gecko code, which requires slippi's frame counter
            self.addresses['804d6cf4'] = int_handler(self.state, 'frame')  # slippi

        for player_id in range(4):
            prefix = 'p' + str(player_id) + '_'

            self.addresses[add_address('8045310C', 0xE90 * player_id)] = int_handler(self.state, prefix+'stocks', 8, 0xFF)

            type_address = add_address('803F0E08', 0x24 * player_id)
            type_handler = int_handler(self.state, prefix+'type', 24, 0xFF, PlayerType, PlayerType.Unselected)
            character_handler = int_handler(self.state, prefix+'character', 8, 0xFF, Character, Character.Unselected)
            self.addresses[type_address] = [type_handler, character_handler]

            data_pointer = add_address('80453130', 0xE90 * player_id)
            self.addresses[data_pointer + ' 70'] = int_handler(self.state, prefix+'action_state', 0, 0xFFFF, ActionState,
                                                               ActionState.Unselected)
            self.addresses[data_pointer + ' 8C'] = float_handler(self.state, prefix+'facing')
            self.addresses[data_pointer + ' E0'] = float_handler(self.state, prefix+'vel_x')
            self.addresses[data_pointer + ' E4'] = float_handler(self.state, prefix+'vel_y')
            self.addresses[data_pointer + ' EC'] = float_handler(self.state, prefix+'attack_vel_x')
            self.addresses[data_pointer + ' F0'] = float_handler(self.state, prefix+'attack_vel_y')
            self.addresses[data_pointer + ' 14C'] = float_handler(self.state, prefix+'ground_vel')
            self.addresses[data_pointer + ' 110'] = float_handler(self.state, prefix+'x')
            self.addresses[data_pointer + ' 114'] = float_handler(self.state, prefix+'y')
            self.addresses[data_pointer + ' 140'] = int_handler(self.state, prefix+'on_ground', 0, 0xFFFF, lambda x: x == 0, True)
            self.addresses[data_pointer + ' 8F4'] = float_handler(self.state, prefix+'action_frame')
            self.addresses[data_pointer + ' 1890'] = float_handler(self.state, prefix+'percent')
            self.addresses[data_pointer + ' 19BC'] = float_handler(self.state, prefix+'hitlag_left')
            # self.addresses[data_pointer + ' 20CC'] = int_handler(self.state, prefix+'action_counter', 16, 0xFFFF)
            self.addresses[data_pointer + ' 23A0'] = float_handler(self.state, prefix+'hitstun_left')
            self.addresses[data_pointer + ' 19F8'] = float_handler(self.state, prefix+'shield_size')
            self.addresses[data_pointer + ' 2174'] = int_handler(self.state, prefix+'charging_smash', 0, 0x2)
            self.addresses[data_pointer + ' 19C8'] = int_handler(self.state, prefix+'jumps_used', 24, 0xFF)
            self.addresses[data_pointer + ' 19EC'] = int_handler(self.state, prefix+'body_state', 0, 0xFF, BodyState,
                                                                 BodyState.Normal)

            # item
            # pointer to first item is located at 0x24 of the pointer @ -0x3E74
            # item_address =

    def handle(self, messages):
        for address, value in messages:
            """Convert the raw address and value into changes in the State."""
            if address in self.addresses:
                handlers = self.addresses[address]
                if isinstance(handlers, list):
                    for handler in handlers:
                        handler(value)
                else:
                    handlers(value)
            else:
                print(address)

    def locations(self):
        """Returns a list of addresses for exporting to Locations.txt."""
        return self.addresses.keys()