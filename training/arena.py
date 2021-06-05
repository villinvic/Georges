import numpy as np
import zmq
from time import sleep

from config.loader import Default
from game.console import Console


class Arena(Default):
    def __init__(self,
                 hub_ip,
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
        self.players = np.empty((4,), dtype=PlayerManager)

        self.console = Console(
            ID,
            mw_path,
            test,
            exe_path,
            iso_path
        )

    def update_players(self, player_ids):

        for i, p in enumerate(player_ids):
            self.players[i].link(p)

        success = False
        tries = 0
        successes = np.full(4, False)
        while not success:
            for i, player in enumerate(self.players):
                if not successes[i]:
                    successes[i] = player.update_params()
            tries += 1
            sleep(1)
            if tries > 30 :
                print('CANT ACCESS TRAINER')

    def request_match(self, last_match_result=None):
        try:
            self.match_socket.send_pyobj(last_match_result)

            new_player_ids = self.match_socket.recv_pyobj(zmq.NOBLOCK)

            self.update_players(new_player_ids)
        except zmq.ZMQError:
            pass

    def play_game(self):
        pass

    def play_match(self):
        pass


class TrainerConnection(Default):
    def __init__(self, individual_id, trainer_ip, zmq_c: zmq.Context):
        super(TrainerConnection, self).__init__()

        self.individual_id = individual_id
        self.topic = individual_id % (self.POP_SIZE//self.N_TRAINERS)

        self.param_port = self.PARAM_PORT_BASE + individual_id// (self.POP_SIZE // self.N_TRAINERS)
        self.exp_port = self.EXP_PORT_BASE + individual_id // (self.POP_SIZE // self.N_TRAINERS)

        self.param_socket = zmq_c.socket(zmq.SUB)
        self.exp_socket = zmq_c.socket(zmq.PUSH)

        self.param_socket.connect("tcp://%s:%d" % (trainer_ip, self.param_port))
        self.exp_socket.connect("tcp://%s:%d" % (trainer_ip, self.exp_port))

        self.param_socket.subscribe(str(self.topic).encode())

    def send_exp(self, traj):
        self.exp_socket.send_pyobj(traj)

    def recv_params(self):
        try:
            params = self.param_socket.recv_pyobj(zmq.NOBLOCK)
            return params
        except zmq.ZMQError:
            pass

        return None

    def close(self):
        self.param_socket.close()
        self.exp_socket.close()


class PlayerManager(Default):
    def __init__(self, player, trainer_ip, zmq_c):
        super(PlayerManager, self).__init__()
        self.player = player
        self.trainer_ip = trainer_ip
        self.zmq_c = zmq_c

        self.trainer_connection : TrainerConnection = None


    def link(self, individual_id):
        if self.trainer_connection is not None:
            self.trainer_connection.close()
            del self.trainer_connection

        self.trainer_connection = TrainerConnection(individual_id, self.trainer_ip, self.zmq_c)

    def send_exp(self, traj):
        self.trainer_connection.send_exp(traj)

    def update_params(self):
        params =  self.trainer_connection.recv_params()
        if params is None:
            return False
        else:
            self.player.update(params)
            return True






