import zmq
from time import time, sleep
from copy import deepcopy
import numpy as np

from config.loader import Default
from GA.tournament import Tournament
from population.population import Population


class Hub(Default):
    def __init__(self, ip):
        super(Hub, self).__init__()

        self.ip = ip
        self.population = Population(self.POP_SIZE)

        # read config for ports
        # socket for sending matches or starting training session modes
        # receiving match results

        c = zmq.Context()

        self.match_socket = c.socket(zmq.REP)
        self.match_socket.bind("tcp://%s:%d" % (self.ip, self.MATCH_PORT))

        self.trainer_PUB = c.socket(zmq.PUB)
        self.trainer_REQs = [c.socket(zmq.PULL) for _ in range(self.N_TRAINERS)]
        # Hub and Trainer should have the same IP
        self.trainer_PUB.bind("icp://%s" % (self.HUB_PUBSUB))
        for i, socket in enumerate(self.trainer_REQs):
            socket.bind("icp://%s_%d" % (self.HUB_REQREP, i))

    def handle_actor_requests(self, tournament_match=None):
        try:
            if tournament_match is not None:
                match_type, msg = self.match_socket.recv_pyobj()
            else:
                match_type, msg = self.match_socket.recv_pyobj(flags=zmq.NOBLOCK)

            if msg is not None:
                self.update_elos(*msg)

            if tournament_match is None:
                self.match_socket.send_pyobj(
                    ('normal', self.sample_players())
                )
            else:
                self.match_socket.send_pyobj(
                    *tournament_match
                )

            return match_type, msg

        except zmq.ZMQError:
            pass

        return None

    def update_elos(self, p0, p1, p2, p3, result):
        mean_team1 = (self.population[p0].elo() + self.population[p1].elo()) / 2.
        mean_team2 = (self.population[p1].elo() + self.population[p2].elo()) / 2.

        self.population[p0].elo.update(mean_team1, mean_team2, result)
        self.population[p1].elo.update(mean_team1, mean_team2, result)
        self.population[p2].elo.update(mean_team2, mean_team1, 1 - result)
        self.population[p3].elo.update(mean_team2, mean_team1, 1 - result)


    def publicate_population(self):
        self.trainer_PUB.send_pyobj(self.population)

    def update_population(self):
        for socket in self.trainer_REQs:

            socket.send(b'update')
            players = socket.recv_pyobj()
            for p in players:
                self.population[p.id] = p

        self.last_update_time = time()

    def sample_players(self):
        p0 = np.random.random_integers(0, self.population.total_size)

        indexes = np.arange(self.population.total_size)
        mask = np.ones(self.population.total_size, dtype=bool)
        mask[p0] = False

        probabilities = np.array([self.population[p0].win_prob(individual.elo()) for individual
                                  in self.population[indexes[mask]]])
        probabilities /= np.sum(probabilities)
        other_players = np.random.choice(indexes[mask], 3, replace=False, p=probabilities)
        np.random.shuffle(other_players)

        return p0, *other_players

    def do_tournament(self):
        tournament = Tournament(self.population.size)
        for match in tournament():
            tournament.step(*self.handle_actor_requests(tournament_match=match))
        winners = tournament.result()
        self.population.ranking()[0].inerit_from(*winners) # worst individual gets replaced by a crossover between the two winners

    def evolve(self):
        evolved = []
        for individual in self.population:
            if individual.data_used > self.evolve_period or individual.mean_entropy < self.minimum_entropy:
                if self.last_update_time - time() > 60 * 1:
                    self.update_population()

                individual.data_used = 0

                indexes = np.arange(self.POP_SIZE)
                mask = np.ones(self.POP_SIZE, dtype=bool)
                mask[individual.id] = False
                other = np.random.choice(self.population[indexes[mask]])

                if other.elo() > individual.elo() + self.required_elo_difference:
                    individual.inerit_from(other)
                    individual.perturb()
                    evolved.append(individual)

        if evolved:
            self.trainer_PUB.send_pyobj(evolved)

    def __call__(self, *args, **kwargs):
        self.last_update_time = time()
        last_tournament_time = time()
        # start trainers...

        self.publicate_population()

        try:
            while True:
                self.handle_actor_requests()
                self.evolve()

                current = time()
                if current - self.last_update_time > self.pop_update_freq_minutes*60:
                    self.update_population()

                if current - last_tournament_time > self.tournament_freq_minutes*60:
                    last_tournament_time = current
                    self.do_tournament()

        except KeyboardInterrupt:
            pass

        # serialize pop
        print('Hub exited.')