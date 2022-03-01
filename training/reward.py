import numpy as np

from game.state import GameState
from game.state import PlayerState


class Rewards:
    base = {
        'hit_dmg': PlayerState.scales[PlayerState.indexes['percent']],
        'hurt_dmg': PlayerState.scales[PlayerState.indexes['percent']],
        'hurt_ally_dmg': PlayerState.scales[PlayerState.indexes['percent']],
        'kill': PlayerState.scales[PlayerState.indexes['stocks']],
        'death': PlayerState.scales[PlayerState.indexes['stocks']],
        'death_ally': PlayerState.scales[PlayerState.indexes['stocks']],
        'distance': PlayerState.scales[PlayerState.indexes['x']],
        'win': 1.,
    }

    special = np.array([
        'combo'
        #'negative_scale'
        'action_state_entropy'
    ])

    main = np.array([
        'win'
    ])


    def __init__(self, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, states, reward_shape):
        """
        states[b,t, state]
        rewards[b,t-1, rewards]

        'hit_dmg',
        'hurt_dmg',
        'hurt_ally_dmg',
        'kill',
        'death',
        'death_ally',
        'is_combo',
        'win'
        """

        combo_p2_multiplier = 1. + np.float32(states[:, 1:, GameState.indexes['p2_hitstun_left']] >= 1) * reward_shape['combo']
        combo_p3_multiplier = 1. + np.float32(states[:, 1:, GameState.indexes['p3_hitstun_left']] >= 1) * reward_shape['combo']

        self['hit_dmg'] =  (np.maximum(states[:, 1:, GameState.indexes['p2_percent']] - states[:, :-1, GameState.indexes['p2_percent']], 0.) * combo_p2_multiplier +
                   np.maximum(states[:, 1:, GameState.indexes['p3_percent']] - states[:, :-1, GameState.indexes['p3_percent']], 0.) * combo_p3_multiplier)

        self['hurt_dmg'] = -np.maximum(states[:, 1:, GameState.indexes['p0_percent']] - states[:, :-1, GameState.indexes['p0_percent']], 0.)
        self['hurt_ally_dmg'] = -np.maximum(states[:, 1:, GameState.indexes['p1_percent']] - states[:, :-1, GameState.indexes['p1_percent']], 0.)
        self['kill'] = np.maximum(states[:, :-1, GameState.indexes['p2_stocks']] - states[:, 1:, GameState.indexes['p2_stocks']], 0.) +\
                       np.maximum(states[:, :-1, GameState.indexes['p3_stocks']] - states[:, 1:, GameState.indexes['p3_stocks']], 0.)

        self['death'] = -np.maximum(states[:, :-1, GameState.indexes['p0_stocks']] - states[:, 1:, GameState.indexes['p0_stocks']], 0.)
        self['death_ally'] = -np.maximum(states[:, :-1, GameState.indexes['p1_stocks']] - states[:, 1:, GameState.indexes['p1_stocks']], 0.)

        dp2 = -np.sqrt( np.square(states[:, 1:, GameState.indexes['p2_x']] - states[:, 1:, GameState.indexes['p0_x']] )
                                    + np.square(states[:, 1:, GameState.indexes['p2_y']] - states[:, 1:, GameState.indexes['p0_y']])) \
                           + np.sqrt( np.square(states[:, :-1, GameState.indexes['p2_x']] - states[:, :-1, GameState.indexes['p0_x']])
                                    + np.square(states[:, :-1, GameState.indexes['p2_y']] - states[:, :-1, GameState.indexes['p0_y']]))
        dp3 = -np.sqrt(np.square(states[:, 1:, GameState.indexes['p3_x']] - states[:, 1:, GameState.indexes['p0_x']] )
                                    + np.square(states[:, 1:, GameState.indexes['p3_y']] - states[:, 1:, GameState.indexes['p0_y']])) \
                           + np.sqrt( np.square(states[:, :-1, GameState.indexes['p3_x']] - states[:, :-1, GameState.indexes['p0_x']])
                                    + np.square(states[:, :-1, GameState.indexes['p3_y']] - states[:, :-1, GameState.indexes['p0_y']]))

        self['distance'] = np.maximum(dp2, dp3)

        win = np.logical_and(
            states[:, 1:, GameState.indexes['p2_stocks']] + states[:, 1:, GameState.indexes['p3_stocks']] < 1e-4,
            self['kill'] > 0)
        loss = np.logical_and(
            states[:, 1:, GameState.indexes['p0_stocks']] + states[:, 1:, GameState.indexes['p1_stocks']] < 1e-4,
            self['death'] + self['death_ally'] < 0)

        self['win'] = np.float32(win) - np.float32(loss)

        # use np arrays instead of dicts...
        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)
        #self.values[:, :] = (1.0 - reward_shape['negative_scale']) * np.maximum(total, 0.) + total

        return self.values





