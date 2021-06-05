from GA.ranking import Elo
from game.enums import PlayerType
from characters.characters import Character
from input.pad import Pad
from input.enums import Button

from training.reward import Rewards

import numpy as np
import threading
from collections import deque
from time import time
class Player:
    def __init__(self,
                 index,
                 pad_path='dolphin/User/Pipes/daboy',
                 trajectory_length = 80,
                 state_dim = None,
                 ):

        self.index = index
        self.pad = Pad(pad_path, player_id=index)

        self.next_possible_move_frame = 0
        self.last_action_id = 0

        self.state_dim = state_dim
        self.trajectory_length = trajectory_length
        self.trajectory_index = 0
        self.trajectory = {
            'state' : np.zeros((trajectory_length, state_dim), dtype=np.float32),
            'action' : np.zeros((trajectory_length,), dtype=np.int32),
            'probs': None, # to set when char attributed (to match action_dim)
            'time': None,
        }

        self.char = None
        self.type = None
        self.costumes = None
        self.policy = None
        self.action_space = None
        self.action_queue = deque()

        self.r = Rewards(1, self.trajectory_length)
        self.scales = {
            'hit_dmg' : 1.,
            'hurt_dmg' :1.,
            'hurt_ally_dmg': 1.,
            'kill' : 1.,
            'death': 1.,
            'death_ally' : 1.,
            'win': 1.,
            'combo':1.,
            'negative_scale':1.,
            'action_state_entropy':1.,
        }

    def attribute_individual(self, individual):
        self.char = individual.char
        self.type = individual.type
        self.costumes = self.char.costumes
        self.policy = individual.policy
        self.action_space = self.char.action_space
        self.trajectory['probs'] = np.zeros((self.trajectory_length, self.action_space.dim), dtype=np.float32)

    def press_A(self):
        self.pad.press_button(Button.A)

    def act(self, state):
        if self.type == PlayerType.Human:
            if not self.action_queue:
                traj_index = self.trajectory_index % self.trajectory_length
                state.get(self.trajectory['state'][traj_index], self.index, self.last_action_id)
                action_id, distribution = self.policy(self.trajectory['state'][traj_index])
                action = self.action_space[action_id]
                if isinstance(action, list):
                    self.action_queue.extend(reversed(action))
                else:
                    self.action_queue.append(action)
                self.last_action_id = action_id
                self.trajectory['action'][traj_index] = action_id
                self.trajectory['probs'][traj_index] = distribution

                if traj_index == 0:
                    self.trajectory['time'] = time()


                self.trajectory_index += 1

            if state.frame >= self.next_possible_move_frame:
                action = self.action_queue.pop()
                self.next_possible_move_frame = state.frame + action['duration']
                action.send_controller(self.pad)

    def finalize(self):
        if self.type == PlayerType.Human:
            traj_index = self.trajectory_index % self.trajectory_length
            if traj_index>0:
                self.trajectory['state'][traj_index:] = self.trajectory['state'][traj_index-1]
                self.trajectory_index = 0

                # send traj


    def link(self, individual_id):
        pass

    def unlink(self):
        pass

    def send_trajectory(self):
        pass


class PlayerGroup:
    def __init__(self, players):
        self.size = len(players)
        self.players = np.array(players, dtype=Player)

        self.t = [None] * self.size

        self.act_v = np.vectorize(lambda x, state: x.act(state))

    def connect_pads(self):
        for i in range(self.size):
            self.t[i] = threading.Thread(target=self.players[i].pad.connect)
            self.t[i].start()

        for i in range(self.size):
            self.t[i].join()

    def disconnect_pads(self):
        for p in self.players:
            p.pad.unbind()

    def attribute_individuals(self, players):
        for i,p in enumerate(players):
            self.players[i].attribute_individual(p)

    def act(self, state):
        for player in self.players:
            player.act(state)

    def finalize(self):
        for player in self.players:
            player.finalize()

    def __getitem__(self, item):
        return self.players[item]

    def __iter__(self, *args, **kwargs):
        return self.players.__iter__(*args, **kwargs)
