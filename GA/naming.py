import numpy as np
from math import floor, ceil

class Name:
    _base = np.array([
        'Nojima',
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
        'Shroomed',
        'Westballz',
        'Medz',
        'MikeHaze',
        'Professor Pro',
        '2saint',
        'Gahtzu',
        'Albert',
        'Spud',
        'FatGoku',
        'Rishi',
        'Bimbo',
        'Magi',
        'Morsecode762',
        'Jakenshaken',
        'HugS',
        'Stango',
        'Zamu',
        'Drephen',
        'Michael',
        'Ice',
        'billybopeep',
        'La Luna',
        'Colbol',
        'Overtriforce',
        'Slox',
        'Kalamazhu',
        'Nickemwit',
        'Jerry',
        'Aura',
        'Nut',
        'Kalvar',
        'Polish',
        'Kevin Maples',
        'Bladewise',
        'Tai',
        'Squid',
        'Forrest',
        'Joyboy',
        'koDoRiN',
        'Ryan Ford',
        'Free Palestine',
        'Ryobeat',
        'Ka-Master',
        'Kũrv',
        'Frenzy',
        'MoG',
        'Boyd',
        'Cool Lime',
        'bobby big ballz',
        'Nintendude',
        'Franz',
        'Nicki',
        'lint',
        'King Momo',
        'TheRealThing',
        'Umarth',
        'Zeo',
        'Pricent',
        'Prince Abu',
        'Amsah',
        'Rocky',
        'Sharkz',
        'HTwa',
        'Kage',
        'Schythed',
        'Panda',
        'Soonsay',
        'TheSWOOPER',
        'Snowy',

        'Azen',

    ])

    _custom = np.array([
        'nihyru',
        'Syquo',
        'Goji',
        'Punchim',
        'Jolie-Edeweiss',
        'Bellin',

        'dark siko',
        'El Moustacho',

        'chevrier',

        'HowToJV5'
        'GanonNose',
        '0-to-SD',
        'CarpalSyndrome',
        'SheikIsTrap',
        'MarthArm',
        'Botser',
        'Jigglybuff',
        'MarioLuigi',
        'DontTestMe',
        'BossTheme',
        'FD is lame',
        'taking out trash',
        'CringeBoi',
        'TurboInt',
        'Megaman',
        'random fan420'

        'Georges',
    ])


    _all = np.concatenate([_base, _custom])



    def __init__(self, name=None):
        if name is not None:
            self._name = name
        else:
            component1, component2 = np.random.choice(self._all, 2)
            self._name = self.clip_name_prefix(component1) + self.clip_name_suffix(component2)

    @staticmethod
    def clip_name_prefix(name, low=0.3, high=0.7):
        l = floor(np.clip(len(name)*low, 1, np.inf))
        h = ceil(len(name)*high + 1)

        if l == h :
            x = l
        else:
            x = np.random.randint(l, h)

        return name[:x]

    @staticmethod
    def clip_name_suffix(name, low=0.3, high=0.7):
        l = floor(np.clip(len(name) * (1-high), 1, np.inf))
        h = ceil(len(name ) * (1-low) + 1)

        if l == h :
            x = l
        else:
            x = np.random.randint(l, h)

        return name[x:]

    def inerit_from(self, *other_names):
        if len(other_names) == 1:
            addition = np.random.choice(self._all)
        else:
            addition = other_names[1]._name

        self._name = ''
        while self._name == '' or len(self._name) > 17:
            if np.random.random() < 0.5:
                self._name = self.clip_name_prefix(other_names[0]._name, low=0.25, high=0.75)\
                             + self.clip_name_suffix(addition, low=0.25, high=0.75)
            else:
                self._name = self.clip_name_prefix(addition, low=0.25, high=0.75)\
                             + self.clip_name_suffix(other_names[0]._name, low=0.25, high=0.75)

    def get(self):
        return self._name

    def set(self, name):
        self._name = name

    def __repr__(self):
        return self._name
