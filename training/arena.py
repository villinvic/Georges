import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import zmq
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


class Arena(Default, Logger):
    def __init__(self,
                 hub_ip=None,
                 ID=0,
                 mw_path='dolphin/User/MemoryWatcher',
                 test=True,
                 exe_path='dolphin/dolphin-emu-nogui-id',
                 iso_path='../isos/game.iso',
                 ):
        super(Arena, self).__init__()
        self.id = ID
        self.zmq_context = zmq.Context()
        self.trainer_ip = self.hub_ip = hub_ip

        self.match_socket = self.zmq_context.socket(zmq.REQ)
        self.match_socket.connect("tcp://%s:%d" % (self.hub_ip, self.MATCH_PORT))
        self.players = np.empty((4,), dtype=IndividualManager)
        for i in range(len(self.players)):
            self.players[i] = IndividualManager(hub_ip, self.zmq_context)

        self.console = Console(
            ID,
            self.TRAJECTORY_LENGTH,
            mw_path,
            test,
            exe_path,
            iso_path
        )

        signal.signal(signal.SIGINT, self.exit)

    def update_players(self, player_ids):

        for i, p in enumerate(player_ids):
            self.players[i].link(p)

        tries = 0
        successes = np.full(4, False)
        while not np.all(successes):
            for i, player in enumerate(self.players):
                if not successes[i]:
                    successes[i] = player.update_params()
            tries += 1
            sleep(1.)
            if tries > 30 :
                self.logger.warning('CANT ACCESS TRAINER')

    def request_match(self, last_match_result=(None, (None, None, None, None, None))):
        try:
            self.match_socket.send_pyobj(last_match_result)
            match_type, new_player_ids = self.match_socket.recv_pyobj()
            self.logger.debug(match_type)
            self.update_players(new_player_ids)
            return match_type, new_player_ids
        except zmq.ZMQError as e:
            print(e)

        return None, (None, None, None , None)



    def play_game(self):
        return self.console.play_game(*self.players)

    def __call__(self):
        # request match, update players, play game
        try:
            last_match_result = None
            match_type = None
            player_ids = (None, None, None, None)
            while True:

                match_type, player_ids = self.request_match((match_type, (*player_ids, last_match_result)))
                last_match_result = self.play_game()

        except KeyboardInterrupt:
            pass

    def exit(self, frame, sig):
        self.console.close()
        self.logger.info('Arena %d closed' % self.id)
        sys.exit(0)


class TrainerConnection(Default):
    def __init__(self, individual_id, trainer_ip, zmq_c: zmq.Context):
        super(TrainerConnection, self).__init__()

        self.individual_id = individual_id

        self.param_port = self.PARAM_PORT_BASE + individual_id// (self.POP_SIZE // self.N_TRAINERS)
        self.exp_port = self.EXP_PORT_BASE + individual_id // (self.POP_SIZE // self.N_TRAINERS)

        self.param_socket = zmq_c.socket(zmq.SUB)
        self.exp_socket = zmq_c.socket(zmq.PUSH)

        self.param_socket.connect("tcp://%s:%d" % (trainer_ip, self.param_port))
        self.exp_socket.connect("tcp://%s:%d" % (trainer_ip, self.exp_port))
        self.param_socket.setsockopt(zmq.SUBSCRIBE, str(self.individual_id).encode())

    def send_exp(self, traj):
        self.exp_socket.send_pyobj(traj)

    def recv_params(self):
        try:
            topic, data = self.param_socket.recv_multipart(zmq.NOBLOCK)
            return pickle.loads(data)
        except zmq.ZMQError:
            pass

        return None

    def close(self):
        self.param_socket.close()
        self.exp_socket.close()


class IndividualManager(Individual):
    def __init__(self, trainer_ip, zmq_c):
        super(IndividualManager, self).__init__(-1, characters.Character)
        self.connection = None
        self.trainer_ip = trainer_ip
        self.zmq_c = zmq_c

    def link(self, individual_id):
        self.id = individual_id
        if self.connection is not None:
            self.connection.close()
            del self.connection
        self.connection = TrainerConnection(individual_id, self.trainer_ip, self.zmq_c)

    def close_connection(self):
        self.connection.close()

    def update_params(self):
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
