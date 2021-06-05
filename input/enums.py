import enum

@enum.unique
class UsefullButton(enum.IntEnum):
    A = 0
    B = 1
    X = 2
    Z = 4
    START = 5
    L = 6



@enum.unique
class Button(enum.IntEnum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11


@enum.unique
class Trigger(enum.IntEnum):
    L = 0
    R = 1


@enum.unique
class Stick(enum.IntEnum):
    MAIN = 0
    C = 1