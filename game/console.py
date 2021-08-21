from game.dolphin import DolphinInstance
from game.state import GameState
from player.player import Player, PlayerGroup
from characters.characters import *
from population.individual import Individual

from time import sleep, time
import os
import sys

class Console:
    def __init__(self,
                 ID,
                 trajectory_length,
                 mw_path,
                 exe_path,
                 iso_path,
                 pad_path,
                 test,
                 ):
        self.id = ID
        self.dolphin = DolphinInstance(exe_path, iso_path, test, ID)
        self.state = GameState(mw_path=mw_path, instance_id=ID, test=test)

        instance_pad_path = pad_path + str(ID) + '/'
        if not os.path.isdir(instance_pad_path):
            os.makedirs(instance_pad_path)

        self.players = PlayerGroup([Player(i, trajectory_length=trajectory_length, state_dim=self.state.size,
                                           pad_path=instance_pad_path) for i in range(4)])

    def play_game(self, *individuals):

        self.state.mw.bind()
        self.players.attribute_individuals(individuals)
        self.state.update_players(individuals)
        self.players.connect_pads()
        self.dolphin.run(*self.players)
        #sleep(3)




        frames = 0
        done = False
        result = None
        self.state.init()
        while not done and frames < 7*60*60:
            # Player hold a trajectory instance
            # choose action based on state, send to controller
            # data required : time of traj, states, actions, probabilities
            done, result = self.state.update()
            if result is None:
                self.players.finalize()
                self.dolphin.close()
                self.state.mw.unbind()
                self.players.disconnect_pads()
                sleep(1)
                return self.play_game(*individuals)


            self.players.act(self.state)
            frames += 1


        self.players.finalize(self.state)
        self.dolphin.close()
        self.state.mw.unbind()
        self.players.disconnect_pads()

        return result

    def close(self):
        self.dolphin.close()
        self.state.mw.unbind()
        self.players.disconnect_pads()

if __name__ == '__main__':
    c = Console()

    players = [Individual(i, char, False, test=True) for i, char in enumerate(np.random.choice(available_chars, 4))]

    try:
        for _ in range(100):
            c.play_game(*players)
    except KeyboardInterrupt:
        c.dolphin.close()
        c.state.mw.unbind()