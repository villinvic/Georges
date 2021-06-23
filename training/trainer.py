import zmq
import numpy as np
import pandas as pd
import signal
from time import time, sleep
import sys
import fire
import os
import pickle

import tensorflow as tf
from training import RL
from config.loader import Default
from logger.logger import Logger
from game.state import GameState
from training.reward import Rewards
from population.population import Population


class Trainer(Default, Logger):
    def __init__(self, ID=0, ip='', individual_ids=[], instance_id='Georges_'):
        super(Trainer, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        N_GPUS = len(gpus)
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.gpu_id = 0 if ID < N_GPUS else -1


        log_dir = 'logs/' + instance_id + '/train_' + str(ID)
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        #==============================================================#

        self.rewards = Rewards(self.batch_size, self.TRAJECTORY_LENGTH)
        self.id = ID
        self.ip = ip
        self.individual_ids = {individual_id : i for i, individual_id in enumerate(individual_ids)}

        self.trained = Population(len(individual_ids), n_reference=0)
        self.trained.initialize(individual_ids=individual_ids, trainable=True)

        self.param_port = self.PARAM_PORT_BASE + ID
        self.exp_port = self.EXP_PORT_BASE + ID

        c = zmq.Context()
        self.param_socket = c.socket(zmq.PUB)
        self.param_socket.bind("tcp://%s:%d" % (self.ip, self.param_port))
        self.exp_socket = c.socket(zmq.PULL)
        self.exp_socket.bind("tcp://%s:%d" % (self.ip, self.exp_port))

        self.hub_pub_pipe= c.socket(zmq.SUB)
        self.hub_pub_pipe.subscribe(b'')

        self.hub_update_pipe = c.socket(zmq.PUSH)
        # Hub and Trainer should have the same IP
        self.hub_update_pipe.connect("ipc://%s" % self.HUB_PUSHPULL)
        self.hub_pub_pipe.connect("ipc://%s" % (self.HUB_PUBSUB))

        self.exp = [[] for _ in range(len(self.individual_ids))]
        self.rcved = 0
        self.train_cntr = 0
        self.time = time()

        signal.signal(signal.SIGINT, self.exit)

        self.logger.info('Trainer %d bound to ports (%d, %d) initialized' % (self.id, self.param_port, self.exp_port))


    def read_pop_update(self):
        try:
            individuals = self.hub_pub_pipe.recv_pyobj(zmq.NOBLOCK)
            for individual in individuals:
                if individual['id'] in self.individual_ids:
                    self.trained[self.individual_ids[individual['id']]].set_all(individual)
                    # self.pub_params(individual.id)
                    self.logger.debug('Trainer %d updated individual %d from Hub' % (self.id, individual['id']))
            return True
        except zmq.ZMQError:
            pass
        return False

    def update_hub(self):
        try:
            # If we received something, then we must submit our trained individuals
            self.hub_update_pipe.send_pyobj(self.trained.to_serializable(), zmq.NOBLOCK)
            self.logger.debug('sent update to hub')
        except zmq.ZMQError as e:
            pass

    def recv_training_data(self):
        received = 0
        try:
            while True:
                traj = self.exp_socket.recv_pyobj(zmq.NOBLOCK)
                self.exp[self.individual_ids[traj['id']]].append(traj)
                received += 1
        except zmq.ZMQError:
            pass
        self.rcved += received

    def pub_params(self):
        for individual in self.trained:
            try:
                #self.param_socket.send_string(str(individual.id), zmq.SNDMORE)
                #self.param_socket.send_pyobj(individual.get_arena_genes(), flags=zmq.NOBLOCK)
                self.param_socket.send_multipart([str(individual.id).encode(), pickle.dumps(individual.get_arena_genes())])
            except zmq.ZMQError:
                pass

    def train(self, individual_index):
        if self.trained[individual_index] is not None and len(self.exp[individual_index]) >= self.batch_size:
            # Get experience from the queue
            trajectory = pd.DataFrame(self.exp[individual_index][:self.batch_size]).values
            self.exp[individual_index] = self.exp[individual_index][self.batch_size:]

            # Cook data
            states = np.float32(np.stack(trajectory[:, 0], axis=0))
            actions = np.float32(np.stack(trajectory[:, 1], axis=0)[:, :-1])
            probs = np.float32(np.stack(trajectory[:, 2], axis=0)[:, :-1])
            hidden_states = np.float32(np.stack(trajectory[:, 3], axis=0)[:, :-1])
            is_old = np.any(time()- np.stack(trajectory[:, 4], axis=0)> self.batch_age_limit)
            rews = self.rewards.compute(states, self.trained[individual_index].get_reward_shape())[:, :, np.newaxis]

            states *= GameState.scales

            if not is_old :
                # Train
                self.logger.debug('train!')
                with tf.summary.record_if(self.train_cntr % self.write_summary_freq == 0):
                    self.trained[individual_index].train(states, actions, rews, probs, hidden_states, 0)

                self.trained[individual_index].data_used += self.batch_size * self.TRAJECTORY_LENGTH
                return True
            else:
                print('Experience too old !', individual_index, self.train_cntr)

        return False

    def train_each(self):
        tf.summary.experimental.set_step(self.train_cntr)
        self.train_cntr += 1

        success = False
        for index in self.individual_ids.values():
            success = success or self.train(index)

        if success:
            self.train_cntr += 1

        dt = time() - self.time
        if self.train_cntr % (self.write_summary_freq * 5) == 0:
            self.time = time()
            if dt < 3600:
                tps = float(self.TRAJECTORY_LENGTH * self.rcved) / dt
                tf.summary.scalar(name="misc/TPS", data=tps)
                self.rcved = 0
                total_waiting = 0
                for exp in self.exp:
                    total_waiting += len(exp)

                if total_waiting > len(self.individual_ids) * self.batch_size:
                    self.logger.debug('exp waiting at Trainer %d : %d' % (self.id, total_waiting))

    def __call__(self):
        try:
            success = False
            c = 0
            while not success:
                success = self.read_pop_update()
                c += 1
                sleep(1)
                if c > 15 :
                    self.logger.warning('Trainer %d Cant connect to Hub !' %self.id)

            self.pub_params()
            last_pub_time = time()
            last_update_time = time()

            while True:
                self.read_pop_update()
                self.recv_training_data()
                self.train_each()

                current = time()
                if current - last_pub_time > 10:
                    self.pub_params()
                    last_pub_time = current
                if current - last_update_time > self.hub_update_freq_minutes*60:
                    self.update_hub()
                    last_update_time = current

        except KeyboardInterrupt:
            self.logger.info('Trainer %d exited.' % self.id)

    def exit(self, sig, frame):
        try :
            self.exp_socket.unbind("tcp://%s:%d" % (self.ip, self.exp_port))
        except zmq.ZMQError as e:
            pass
        try:
            self.param_socket.unbind("tcp://%s:%d" % (self.ip, self.param_port))
        except zmq.ZMQError as e:
            pass

        self.update_hub()

        self.logger.info('Trainer %d exited.' % self.id)
        sys.exit(0)


if __name__ == '__main__':

    sys.exit(fire.Fire(Trainer))