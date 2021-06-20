import numpy as np
from math import floor, ceil

class Name:
    _base = np.array([
        'Hungrybox',
        'Leffen',
        'Mango',
        'Axe',
        'Wizzrobe',
        'Zain',
        'aMSa',
        'Plup',
        'iDBW',
        'Mew2King',
        'S2J',
        'Fiction',
        'SFAT',
        'moky',
        'n0ne',
        'Trif',
        'Captain Faceroll',
        'Swedish Delight',
        'Hax$',
        'Lucky',
        'Ginger',
        'Spark',
        'ChuDat',
        'PewPewU',
        'IIoD',
        'ARMY',
        'AbsentPage',
        'Bananas',
        'KJH',
        'Shroomed'
    ])

    _custom = np.array([
        'nihyru',
        'Syquo',
        'Goji',
        'Punchim',

        'dark siko',
        'El Moustacho',
        
        'GanonNose',
        '0-to-SD',
        'CarpalSyndrome',
        'SheikIsTrap',
        'MarthArm',
        'Botser',
        'Jigglybuff',
        'Mang0ne',
        'MarioMario',
        '4ParralelUniversesAheadOfU',
    ])

    _all = np.concatenate([_base, _custom])

    def __init__(self, name=None):
        if name is not None:
            self._name = name
        else:
            component1, component2 = np.random.choice(self._all, 2)
            offset = np.random.choice([ceil, floor])
            self._name = component1[:offset(len(component1)/2)] + component2[offset(len(component2)/2):]

    def inerit_from(self, *other_names):
        if len(other_names) == 1:
            addition = np.random.choice(self._all)
        else:
            addition = other_names[1]._name
        offset = np.random.choice([ceil, floor])
        if np.random.random() < 0.5:
            self._name = other_names[0]._name[:offset(len(other_names[0]._name)/2)] + addition[offset(len(addition)/2):]
        else:
            self._name = addition[:offset(len(addition) /2)] + other_names[0]._name[offset(len(other_names[0]._name) / 2):]

    def get(self):
        return self._name

    def __repr__(self):
        return self._name
