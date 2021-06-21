import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import signal
import zmq
from time import time, sleep
from subprocess import Popen
import numpy as np
from socket import gethostname, gethostbyname
import datetime


from config.loader import Default
from logger.logger import Logger
from GA.tournament import Tournament
from population.population import Population


class Hub(Default, Logger):
    def __init__(self, is_localhost=False):
        super(Hub, self).__init__()

        self.ip = '127.0.0.1' if is_localhost else gethostbyname(gethostname())

        self.population = Population(self.POP_SIZE)
        # Load checkpoint if specified...
        self.logger.info('Population Initialization started...')
        self.population.initialize()
        self.logger.info(self.population)
        self.trainers = [None] * self.N_TRAINERS

        self.running_instance_identifier = datetime.datetime.now().strftime("Georges_%Y-%m-%d_%H-%M")

        # read config for ports
        # socket for sending matches or starting training session modes
        # receiving match results

        c = zmq.Context()

        self.match_socket = c.socket(zmq.REP)
        self.match_socket.bind("tcp://%s:%d" % (self.ip, self.MATCH_PORT))

        self.trainer_PUB = c.socket(zmq.PUB)
        self.trainer_UPDATE = c.socket(zmq.PULL)
        # Hub and Trainer should have the same IP
        self.trainer_PUB.bind("ipc://%s" % (self.HUB_PUBSUB))
        self.trainer_UPDATE.bind("ipc://%s" % self.HUB_PUSHPULL)

        self.logger.info('Hub initialized at %s' %self.ip)

    def handle_actor_requests(self, tournament_match=None):
        try:
            if tournament_match is not None:
                match_type, msg = self.match_socket.recv_pyobj()
            else:
                match_type, msg = self.match_socket.recv_pyobj(flags=zmq.NOBLOCK)
                self.logger.debug(match_type)

            if match_type is not None:
                self.update_elos(*msg)

            if tournament_match is None:
                self.match_socket.send_pyobj(
                    ('normal', self.sample_players())
                )
            else:
                self.match_socket.send_pyobj(
                    tournament_match
                )

            return match_type, msg

        except zmq.ZMQError:
            pass

        return None, None

    def update_elos(self, p0, p1, p2, p3, result):
        mean_team1 = (self.population[p0].elo() + self.population[p1].elo()) / 2.
        mean_team2 = (self.population[p1].elo() + self.population[p2].elo()) / 2.

        self.population[p0].elo.update(mean_team1, mean_team2, result)
        self.population[p1].elo.update(mean_team1, mean_team2, result)
        self.population[p2].elo.update(mean_team2, mean_team1, 1 - result)
        self.population[p3].elo.update(mean_team2, mean_team1, 1 - result)


    def publicate_population(self):
        self.trainer_PUB.send_pyobj(self.population.to_serializable())

    def update_population(self):
        try:
            while True :
                individuals = self.trainer_UPDATE.recv_pyobj(zmq.NOBLOCK)
                for individual in individuals:
                    self.logger.debug('received update! %d' % individual['id'])

                    # Check age in case we receive an older version of an individual
                    # (in case of tournament of evolution)
                    self.population[individual['id']].set_all(individual, check_age=True)
        except zmq.ZMQError:
            pass


        self.last_update_time = time()

    def sample_players(self):
        p0 = np.random.random_integers(0, self.population.total_size-1)

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
        tournament = Tournament(self.POP_SIZE)
        for match in tournament():
            self.update_population()
            tournament.step(*self.handle_actor_requests(tournament_match=match))
        winners = tournament.result()
        self.population.ranking()[0].inerit_from(*winners) # worst individual gets replaced by a crossover between the two winners
        self.trainer_PUB.send_pyobj([self.population.ranking()[0].get_all()])

    def evolve(self):
        evolved = []
        for individual in self.population:
            if individual.data_used > self.evolve_period or individual.mean_entropy < self.minimum_entropy:

                #individual.data_used = 0

                indexes = np.arange(self.POP_SIZE)
                mask = np.ones(self.POP_SIZE, dtype=bool)
                mask[individual.id] = False
                other = np.random.choice(self.population[indexes[mask]])

                if other.elo() > individual.elo() + self.required_elo_difference:
                    self.logger.debug('Evolving individual %d...' % individual.id)
                    self.logger.debug('From...')
                    self.logger.debug(individual)

                    individual.inerit_from(other)
                    individual.perturb()

                    self.logger.debug('To...')
                    self.logger.debug(individual)

                    evolved.append(individual.get_all())

        if evolved:
            self.trainer_PUB.send_pyobj(evolved)

    def start_trainers(self):
        # id, ip, individual_ids
        cmd = "python3 training/trainer.py --ID=%d --ip=%s --individual_ids=%s --instance_id=%s"
        if self.POP_SIZE % self.N_TRAINERS != 0:
            print('POP_SIZE', self.POP_SIZE, 'is not a multiple of N_TRAINERS', self.N_TRAINERS)
            print('N_TRAINERS must divide POP_SIZE for optimal and fair GPU usage.')
            raise ValueError

        for ID in range(self.N_TRAINERS):
            individual_ids = list(range(ID*self.POP_SIZE // self.N_TRAINERS, (1+ID)*self.POP_SIZE // self.N_TRAINERS))
            self.trainers[ID] = Popen((cmd % (ID, self.ip, str(individual_ids).replace(" ", ""),
                                              self.running_instance_identifier)).split(),
                                      env={'PYTHONPATH': os.getcwd()})

    def exit(self):
        for trainer in self.trainers:
            trainer.send_signal(signal.SIGINT)

        for trainer in self.trainers:
            trainer.wait()

    def __call__(self, *args, **kwargs):
        self.last_update_time = time()
        last_tournament_time = time()
        self.start_trainers()
        sleep(15)
        self.publicate_population()

        try:
            while True:
                self.handle_actor_requests()
                self.evolve()

                current = time()
                if current - self.last_update_time > self.pop_update_freq:
                    self.update_population()

                if current - last_tournament_time > self.tournament_freq_minutes*60:
                    last_tournament_time = current
                    self.do_tournament()

        except KeyboardInterrupt:
            pass

        # serialize pop
        self.exit()
        self.logger.info('Exited.')
