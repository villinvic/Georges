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
from game.enums import PlayerType


class Arena(Default, Logger):
    def __init__(self,
                 hub_ip=None,
                 ID=0,
                 mw_path='dolphin/User/MemoryWatcher',
                 test=False,
                 exe_path='dolphin/dolphin-emu-nogui-id',
                 iso_path='../isos/game.iso',

                 obs_streaming=False
                 ):

        print(hub_ip, ID, mw_path, test, exe_path, iso_path)

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

        self.obs_streaming = obs_streaming
        if obs_streaming:
            from population.population import Population
            self.stream_path = "obs/match_info{player_index}.txt"
            self.stream_pipe = self.zmq_context.socket(zmq.SUB)
            self.stream_pipe.subscribe('')
            self.stream_pipe.connect("tcp://%s:%d" %(self.hub_ip, self.STREAMING_PORT))
            self.population = Population(self.POP_SIZE, n_reference=1)
            self.population.initialize()
            self.logger.info('Synchronization with Hub...')
            c = 0
            while True:
                try:
                    self.population.read_pickled(self.stream_pipe.recv_pyobj(zmq.NOBLOCK))
                    break
                except zmq.ZMQError:
                    pass
                sleep(1)
                c += 1
                if c > 30:
                    self.logger.debug('Can\'t connect to hub for streaming!')
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
                    successes[i] = player.update_params()
            tries += 1
            sleep(1.)
            if tries > 30 :
                self.logger.warning('CANT ACCESS TRAINER')
                # 28

    def request_match(self, last_match_result=(None, (None, None, None, None, None))):
        try:
            self.match_socket.send_pyobj(last_match_result)
            match_type, new_player_ids = self.match_socket.recv_pyobj()
            self.update_players(new_player_ids)
            if self.obs_streaming:
                self.update_stream_info(new_player_ids)
            return match_type, new_player_ids
        except zmq.ZMQError as e:
            print(e)

        return None, (None, None, None , None)



    def play_game(self):
        return self.console.play_game(*self.players)

    def __call__(self):
        # request match, update players, play game
        last_match_result = None
        match_type = None
        player_ids = (None, None, None, None)
        while not self.exited:

            self.logger.debug('Arena %d requesting Match...' % self.id)
            match_type, player_ids = self.request_match((match_type, (*player_ids, last_match_result)))
            if not self.exited:
                self.logger.debug('Arena %d received match : [%d,%d]vs[%d,%d]' % (self.id, *player_ids))
                last_match_result = self.play_game()

        sys.exit(0)


    def exit(self, frame, sig):
        self.exited = True
        self.console.close()
        self.logger.info('Arena %d closed' % self.id)
        sys.exit(0)

    # if we are streaming the arena
    def update_stream_info(self, player_ids):
        try :
            self.population.read_pickled(self.stream_pipe.recv_pyobj(zmq.NOBLOCK))
        except zmq.ZMQError:
            pass

        for i, player_id in enumerate(player_ids):
            comment = "[%s](%.0f)" % (self.population[player_id].name.get(),
                                    self.population[player_id].elo())
            with open(self.stream_path.format(player_index=i), 'w+') as f:
                f.write(comment)



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
