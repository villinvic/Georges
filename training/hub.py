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
import pickle
import matplotlib.pyplot as plt


from config.loader import Default
from logger.logger import Logger
from GA.tournament import Tournament
from population.population import Population
from characters.characters import string2char
import visualization.ranking


class Hub(Default, Logger):
    def __init__(self, ip=None, ckpt=""):
        super(Hub, self).__init__()

        self.ip = '127.0.0.1' if ip is None else ip
        self.logger.info('Hub started at ' + self.ip)
        self.running_instance_identifier = datetime.datetime.now().strftime("Georges_%Y-%m-%d_%H-%M")

        self.population = Population(self.POP_SIZE)
        # Load checkpoint if specified...
        self.logger.info('Population Initialization started...')
        self.population.initialize(trainable=True, reference_char=string2char[self.REFERENCE_CHAR], reference_name=self.REFERENCE_NAME)
        if ckpt:
            self.load(ckpt)

        self.logger.info(self.population)
        self.trainers = [None] * self.N_TRAINERS

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
                match_type, msg, streaming = self.match_socket.recv_pyobj()
            else:
                match_type, msg, streaming = self.match_socket.recv_pyobj(flags=zmq.NOBLOCK)
                self.logger.debug(match_type)

            if match_type is not None:
                self.update_elos(*msg)

            if tournament_match is None:
                msg_2 = ('normal', self.sample_players())
                if streaming:
                    msg_2 += (self.population.to_dict(),)
                self.match_socket.send_pyobj(
                    msg_2
                )
            else:
                if streaming:
                    tournament_match += (self.population.to_dict(),)
                self.match_socket.send_pyobj(
                    tournament_match
                )

            return match_type, msg

        except zmq.ZMQError:
            pass

        return None, None

    def update_elos(self, p0, p1, p2, p3, result):
        mean_team1 = (self.population[p0].elo() + self.population[p1].elo()) / 2.
        mean_team2 = (self.population[p2].elo() + self.population[p3].elo()) / 2.

        self.population[p0].elo.update(mean_team1, mean_team2, result)
        self.population[p1].elo.update(mean_team1, mean_team2, result)
        self.population[p2].elo.update(mean_team2, mean_team1, 1 - result)
        self.population[p3].elo.update(mean_team2, mean_team1, 1 - result)

        print(self.population[p1].elo())


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
        self.logger.info('--Starting Tournament--')
        tournament_start_time = time()
        tournament = Tournament(self.POP_SIZE)
        for match in tournament():
            type, msg = self.handle_actor_requests(tournament_match=match)
            if msg is not None:
                tournament.step(type, *msg)

            if time() - tournament_start_time > self.tournament_timeout_minutes * 60:
                self.logger.warning('Tournament was interrupted : Timeout')
                return
        w1, w2 = tournament.result()
        self.update_population()
        self.logger.info('-- Tournament winners --')
        self.logger.info(self.population[w1])
        self.logger.info(self.population[w2])
        self.population[w1].tournaments_won += 1
        self.population[w2].tournaments_won += 1

        worst_individual = self.population.ranking()[0]
        worst_individual.inerit_from(self.population[w1], self.population[w2]) # worst individual gets replaced by a crossover between the two winners
        self.trainer_PUB.send_pyobj([worst_individual.get_all()])

    def evolve(self):
        evolved = []
        for individual in self.population:
            if individual.data_used > self.evolve_period or individual.mean_entropy < self.minimum_entropy:

                individual.data_used = 0

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

        self.save()

    def load(self, ckpt_path):
        self.logger.info("Loading checkpoint %s ..." % ckpt_path)
        self.population.load(ckpt_path)

        path_dirs = ckpt_path.split('/')
        for dir in path_dirs:
            if 'Georges' in dir:
                self.running_instance_identifier = dir

    def save(self):
        # save pop
        self.logger.info('Retrieving latest versions of individuals from trainer...')
        for _ in range(15):
            self.update_population()
            sleep(0.1)

        self.logger.info('Saving population and parameters...')


        ckpt_path = 'checkpoints/'+self.running_instance_identifier+'/'
        full_path = ckpt_path + 'ckpt_' + str(self.population.checkpoint_index) + '/'
        self.population.checkpoint_index += 1
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except OSError as exc:
                print(exc)

        visualization.ranking.visualize(self.population, full_path)
        self.population.save(full_path)

        _, dirs, _ = next(os.walk(ckpt_path))
        if len(dirs) > self.ckpt_keep:
            oldest = sorted(dirs)[0]
            _, _, files = next(os.walk(ckpt_path+oldest))
            for f in files:
                if '.pkl' in f or '.params' in f:
                    os.remove(ckpt_path+oldest+'/'+f)
            try:
                os.rmdir(ckpt_path+oldest)
            except Exception:
                self.logger.warning("Tried to delete a non empty checkpoint directory")
        # build diagram

        ranking = self.population.ranking()
        cols = 'Player tag', 'Main', 'Elo', 'Games played', 'Tournaments won', 'Lineage prestige'
        char_data = np.empty((len(ranking)+1, len(cols)), dtype=object)
        char_data[0, :] = cols
        for i, p in enumerate(reversed(ranking)):
            char_data[i+1][0] = p.name.get()
            char_data[i+1][1] = p.genotype['type'].__repr__()
            char_data[i+1][2] = "%.0f" % p.elo()
            char_data[i+1][3] = str(p.elo.games_played)
            char_data[i+1][4] = str(p.tournaments_won)
            char_data[i+1][5] = str(p.lineage_prestige)

            # age ?
            # lineage prestige
            # prestige
            # icons ?

        np.savetxt(ckpt_path+'population_table.csv', char_data, delimiter=",", fmt="%s")




    def __call__(self, *args, **kwargs):
        self.last_update_time = time()
        last_tournament_time = time()
        last_save_time = time()
        self.start_trainers()
        sleep(5)
        self.publicate_population()

        try:
            while True:
                self.handle_actor_requests()
                self.evolve()

                current = time()
                if current - self.last_update_time > self.pop_update_freq_minutes*60:
                    self.last_update_time = current
                    self.update_population()

                if current - last_tournament_time > self.tournament_freq_minutes*60:
                    last_tournament_time = current
                    self.do_tournament()

                if current - last_save_time > self.save_time_freq_minutes*60:
                    last_save_time = current
                    self.save()


        except KeyboardInterrupt:
            pass

        # serialize pop
        self.exit()
        self.logger.info('Exited.')
