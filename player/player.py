from GA.ranking import Elo
from game.enums import PlayerType
from game.state import GameState
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
                 pad_path,
                 trajectory_length,
                 state_dim,
                 ):

        self.index = index
        self.pad = Pad(pad_path, player_id=index)

        self.next_possible_move_frame = 0
        self.last_action_id = 0

        self.state_dim = state_dim
        self.training_connection = None
        self.param_updater = None
        self.trajectory_length = trajectory_length
        self.trajectory_index = 0
        self.trajectory = {
            'state' : np.zeros((trajectory_length, state_dim), dtype=np.float32),
            'action' : np.zeros((trajectory_length,), dtype=np.int32),
            'probs': None, # to set when char attributed (to match action_dim)
            'hidden_states': np.zeros((2, 256),  dtype=np.float32),
            'time': None,
            'id': 0
        }

        self.char = None
        self.type = None
        self.costumes = None
        self.policy = None
        self.action_space = None
        self.action_queue = deque()
        self.individual = None

        self.is_dead = False

    def attribute_individual(self, individual):
        self.char = individual.char().get()
        self.type = individual.type
        self.costumes = self.char.costumes
        self.policy = individual.policy
        self.action_space = self.char.action_space
        self.trajectory['probs'] = np.zeros((self.trajectory_length, self.action_space.dim), dtype=np.float32)
        self.trajectory['id'] = individual.id
        self.training_connection = None
        if hasattr(individual, 'connection'):
            self.training_connection = individual.connection
            self.param_updater = individual.update_params

    def press_A(self):
        self.pad.press_button(Button.A)

    def update_death(self):
        self.is_dead = self.trajectory['state'][self.trajectory_index % self.trajectory_length][GameState.stock_indexes[0]] == 0

    def act(self, state):
        if self.type == PlayerType.Human:
            if not self.action_queue and not self.is_dead:
                traj_index = self.trajectory_index % self.trajectory_length
                state.get(self.trajectory['state'][traj_index], self.index, self.last_action_id)
                self.update_death()

                action_id, distribution, hidden_h, hidden_c = self.policy(self.trajectory['state'][traj_index])
                if action_id >= self.action_space.dim:
                    print(distribution)
                    action_id = self.action_space.dim-1
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
                    self.trajectory['hidden_states'][:] = hidden_h, hidden_c

                    if self.trajectory_index > 0 and self.training_connection is not None \
                            and not self.training_connection.ssh:
                        self.training_connection.send_exp(self.trajectory)
                        self.param_updater()

                self.trajectory_index += 1

            if not self.is_dead and state.frame >= self.next_possible_move_frame:
                action = self.action_queue.pop()
                self.next_possible_move_frame = state.frame + action['duration']
                action.send_controller(self.pad)

    def finalize(self, state):
        if self.type == PlayerType.Human:
            traj_index = self.trajectory_index % self.trajectory_length
            if traj_index>0:
                state.get(self.trajectory['state'][traj_index], self.index, self.last_action_id)
                if traj_index < self.trajectory_length-1:
                    self.trajectory['state'][traj_index+1:] = self.trajectory['state'][traj_index]
                self.trajectory_index = 0

                if self.training_connection is not None and not self.training_connection.ssh:
                    self.training_connection.send_exp(self.trajectory)
                    self.param_updater()

            self.action_queue.clear()
            self.next_possible_move_frame = -np.inf
            self.is_dead = False

            #from training.reward import Rewards
            #from GA.genotype import RewardShape

            #r_shape = RewardShape()
            #for variable in r_shape._variables.values():
            #    variable._current_value = 0.01

            #r_shape._variables['win']._current_value = 100
            #r = Rewards(1, self.trajectory_length)
            #print(r.compute(self.trajectory['state'][np.newaxis, :], r_shape))


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

        #for i in range(self.size):
        #    self.t[i].join()

    def disconnect_pads(self):
        for p in self.players:
            p.pad.unbind()
            p.pad.pipe.close()

    def attribute_individuals(self, players):
        for i,p in enumerate(players):
            self.players[i].attribute_individual(p)

    def act(self, state):
        for player in self.players:
            player.act(state)

    def finalize(self, state):
        for player in self.players:
            player.finalize(state)

    def __getitem__(self, item):
        return self.players[item]

    def __iter__(self, *args, **kwargs):
        return self.players.__iter__(*args, **kwargs)
