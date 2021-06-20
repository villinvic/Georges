from game.dolphin import DolphinInstance
from game.state import GameState
from player.player import Player, PlayerGroup
from characters.characters import *
from population.individual import Individual

from time import sleep, time
import sys

class Console:
    def __init__(self,
                 ID=0,
                 trajectory_length=100,
                 mw_path='dolphin/User/MemoryWatcher',
                 test=True,
                 exe_path='dolphin/dolphin-emu-nogui-id',
                 iso_path='../isos/game.iso',
                 ):
        self.id = ID
        self.dolphin = DolphinInstance(exe_path, iso_path, test, ID)
        self.state = GameState(mw_path=mw_path, instance_id=ID, test=test)
        self.players = PlayerGroup([Player(i, trajectory_length=trajectory_length, state_dim=self.state.size) for i in range(4)])

    def play_game(self, *individuals):

        self.state.mw.bind()
        self.players.attribute_individuals(individuals)
        self.state.update_players(individuals)
        self.dolphin.run(*self.players)
        self.players.connect_pads()



        frames = 0
        t = time()
        done = False
        result = None
        self.state.init()
        while not done and time() - t < 10*60:
            # Player hold a trajectory instance
            # choose action based on state, send to controller
            # data required : time of traj, states, actions, probabilities
            done, result = self.state.update()
            self.players.act(self.state)
            frames += 1

        self.players.finalize()

        self.dolphin.close()
        self.state.mw.unbind()
        self.players.disconnect_pads()
        return result

    def close(self):
        self.dolphin.close()
        self.state.mw.unbind()


if __name__ == '__main__':
    c = Console()

    players = [Individual(i, char, False, test=True) for i, char in enumerate(np.random.choice(available_chars, 4))]

    try:
        for _ in range(2):
            c.play_game(*players)
    except KeyboardInterrupt:
        c.dolphin.close()
        c.state.mw.unbind()