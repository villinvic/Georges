import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import zmq
import zmq.ssh as zmq_ssh
import zmq.decorators

from time import sleep
import signal
import fire
import pickle
import sys

from config.loader import Default
from logger.logger import Logger
from game.console import Console
from population.individual import Individual
from characters import characters
from game.enums import PlayerType
from visualization.parameters import IndividualVisualizer


class Arena(Default, Logger):
    def __init__(self,
                 hub_ip='127.0.0.1',
                 ID=0,
                 test=False,
                 obs_streaming=False,
                 ssh="",
                 ):

        super(Arena, self).__init__()
        self.id = ID
        self.zmq_context = zmq.Context()
        self.trainer_ip = self.hub_ip = hub_ip
        self.hub_pop_info = dict()

        if ssh:
            self.connect = lambda socket: zmq_ssh.tunnel_connection(socket, "tcp://%s:%d" % (self.hub_ip, self.MATCH_PORT),
                                  "isys3@%s" % self.hub_ip, password=ssh, timeout=0)

        else:
            self.connect = lambda socket: socket.connect("tcp://%s:%d" % (self.hub_ip, self.MATCH_PORT))

        self.players = np.empty((4,), dtype=IndividualManager)
        for i in range(len(self.players)):
            self.players[i] = IndividualManager(hub_ip, ssh, self.zmq_context)

        self.console = Console(
            ID,
            self.TRAJECTORY_LENGTH,
            self.mw_path,
            self.exe_path,
            self.iso_path,
            self.pad_path,
            test,
        )

        self.obs_streaming = obs_streaming
        if obs_streaming:
            self.stream_path = "obs/match_info{player_index}.txt"
            self.individual_visualizer = IndividualVisualizer()

        else:
            self.stream_path = None

        signal.signal(signal.SIGINT, self.exit)
        self.exited = False

    def update_players(self, player_ids):

        successes = np.full(4, False)
        for i, p in enumerate(player_ids):
            if p < self.POP_SIZE:
                self.players[i].link(p)
            else:
                self.players[i].type = PlayerType.CPU
                self.players[i].name._name = self.REFERENCE_NAME
                self.players[i].genotype['type']._character = characters.string2char[self.REFERENCE_CHAR]
                successes[i] = True
        tries = 0

        while not np.all(successes):
            for i, player in enumerate(self.players):
                if not successes[i]:
                    successes[i] = player.update_params(init=True)
            tries += 1
            sleep(1.)
            if tries > 30 :
                self.logger.warning('CANT ACCESS TRAINER')

        if self.obs_streaming:
            self.individual_visualizer.observe(self.players[0])

    @zmq.decorators.socket(zmq.REQ)
    def request_match(self, match_socket, last_match_result=(None, (None, None, None, None, None))):
        self.connect(match_socket)
        match_socket.setsockopt(zmq.RCVTIMEO, 30000)
        match_socket.setsockopt(zmq.LINGER, 0)
        try:

            match_socket.send_pyobj(last_match_result + (self.obs_streaming,))
            if self.obs_streaming:
                match_type, new_player_ids, pop_dict = match_socket.recv_pyobj()
                print(match_type)
                self.hub_pop_info.update(pop_dict)
                self.update_stream_info(new_player_ids, pop_dict)
            else:
                match_type, new_player_ids = match_socket.recv_pyobj()
            self.update_players(new_player_ids)
            return match_type, new_player_ids
        except zmq.ZMQError as e:
            print(e)
            return self.request_match(last_match_result=last_match_result)


    def play_game(self):
        return self.console.play_game(*self.players)

    def __call__(self):
        # request match, update players, play game
        last_match_result = None
        match_type = None
        player_ids = (None, None, None, None)
        while True:

            self.logger.debug('Arena %d requesting Match...' % self.id)
            match_type, player_ids = self.request_match(last_match_result=(match_type, (*player_ids, last_match_result)))
            if player_ids[0] is not None:
                self.logger.debug('Arena %d received match : [%d,%d]vs[%d,%d]' % (self.id, *player_ids))
                last_match_result = self.play_game()
            else:
                self.logger.debug('Weird exit')
                break

        return 0

    def exit(self, *args):
        self.exited = True
        self.console.close()
        self.logger.info('Arena %d closed' % self.id)


        #import pprint
        #for char in characters.available_chars:
        #    print(char)
            #pprint.pprint(char.frame_data)

        sys.exit(0)

    # if we are streaming the arena
    def update_stream_info(self, player_ids, pop_dict):
        for i, player_id in enumerate(player_ids):
            if player_id == 28:
                p_info = {
                    'name': '20XX',
                    'elo': 1000,
                }
            else:
                p_info = pop_dict[player_id]
            comment = "[%s](%.0f)" % (p_info['name'], p_info['elo'])
            with open(self.stream_path.format(player_index=i), 'w+') as f:
                f.write(comment)


class TrainerConnection(Default):
    def __init__(self, individual_id, trainer_ip, ssh, zmq_c: zmq.Context):
        super(TrainerConnection, self).__init__()

        self.individual_id = individual_id

        self.param_port = self.PARAM_PORT_BASE + individual_id// (self.POP_SIZE // self.N_TRAINERS)
        self.exp_port = self.EXP_PORT_BASE + individual_id // (self.POP_SIZE // self.N_TRAINERS)

        self.param_socket = zmq_c.socket(zmq.SUB)
        self.exp_socket = zmq_c.socket(zmq.PUSH)
        self.ssh = ssh != ""

        if ssh:
            zmq_ssh.tunnel_connection(self.param_socket, "tcp://%s:%d" % (trainer_ip, self.param_port),
                                      "isys3@%s" % trainer_ip, password=ssh, timeout=30)
            zmq_ssh.tunnel_connection(self.exp_socket, "tcp://%s:%d" % (trainer_ip, self.exp_port),
                                      "isys3@%s" % trainer_ip, password=ssh, timeout=30)
            self.param_socket.setsockopt(zmq.RCVTIMEO, 15000)
            self.param_socket.setsockopt(zmq.LINGER, 0)
        else:
            self.param_socket.connect("tcp://%s:%d" % (trainer_ip, self.param_port))
            self.exp_socket.connect("tcp://%s:%d" % (trainer_ip, self.exp_port))

        self.param_socket.setsockopt(zmq.SUBSCRIBE, str(self.individual_id).encode())

    def send_exp(self, traj):
        self.exp_socket.send_pyobj(traj)

    def recv_params(self):
        try:
            if self.ssh :
                topic, data = self.param_socket.recv_multipart()
            else:
                topic, data = self.param_socket.recv_multipart(zmq.NOBLOCK)
            return pickle.loads(data)
        except zmq.ZMQError:
            pass

        return None

    def close(self):
        self.param_socket.close()
        self.exp_socket.close()


class IndividualManager(Individual):
    def __init__(self, trainer_ip, ssh, zmq_c):
        super(IndividualManager, self).__init__(-1, characters.Character)
        self.connection = None
        self.trainer_ip = trainer_ip
        self.ssh = ssh
        self.zmq_c = zmq_c

    def link(self, individual_id):
        self.id = individual_id
        if self.connection is not None:
            self.connection.close()
            del self.connection
        self.connection = TrainerConnection(individual_id, self.trainer_ip, self.ssh, self.zmq_c)

    def close_connection(self):
        self.connection.close()

    def update_params(self, init=False):
        if init and self.genotype['brain'] is not None:
            if self.genotype['brain'].has_lstm:
                self.genotype['brain'].lstm.reset_states()
        if self.connection is not None:
                arena_genes = self.connection.recv_params()
                if arena_genes is None:
                    return False
                self.set_arena_genes(arena_genes)
                return True
        else:
            return True

    def send_exp(self, traj):
        self.connection.send_exp(traj, zmq.NOBLOCK)




if __name__ == '__main__':

    sys.exit(fire.Fire(Arena))
