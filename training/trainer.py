import zmq
import numpy as np
import pandas as pd
import tensorflow as tf
import signal
from time import time
import sys
import fire

from config.loader import Default
from game.state import GameState
from training.reward import Rewards


class Trainer(Default):
    def __init__(self, id, ip, individual_ids):
        super(Trainer, self).__init__()

        self.rewards = Rewards(self.batch_size, self.TRAJECTORY_LENGTH)

        self.id = self.gpu_id = id
        self.ip = ip
        self.individual_ids = {individual_id : i for i, individual_id in enumerate(individual_ids)}

        self.trained = np.empty((len(individual_ids)), dtype=object)
        for i in range(len(self.trained)):
            self.trained[i] = None

        c = zmq.Context()
        self.param_socket = c.socket(zmq.PUB)
        self.param_socket.bind("tcp://%s:%d" % (self.ip, self.PARAM_PORT))
        self.exp_socket = c.socket(zmq.PULL)
        self.exp_socket.bind("tcp://%s:%d" % (self.ip, self.EXP_PORT))

        self.hub_pub_pipe= c.socket(zmq.SUB)
        self.hub_pub_pipe.subscribe(b'')

        self.hub_reply_pipe = c.socket(zmq.REP)
        # Hub and Trainer should have the same IP
        self.hub_reply_pipe.connect("icp://%s_%d" % (self.HUB_REQREP, id))
        self.hub_pub_pipe.connect("icp://%s" % (self.HUB_PUBSUB))

        self.exp = [[] for _ in range(len(self.individual_ids))]
        self.rcved = 0
        self.train_cntr = 0
        self.time = time()

        signal.signal(signal.SIGINT, self.exit)

    def read_pop_update(self):
        try:
            individuals =  self.hub_pub_pipe.recv_pyobj(zmq.NOBLOCK)
            for individual in individuals:
                if individual.id in self.individual_ids:
                    self.trained[self.individual_ids[individual.id]] = individual
                    # self.pub_params(individual.id)
            return True
        except zmq.ZMQError:
            pass
        return False

    def reply_hub(self):
        try:
            self.hub_reply_pipe.recv(flag=zmq.NOBLOCK)
            # If we received something, then we must submit our trained individuals
            self.hub_reply_pipe.send_pyobj(self.trained)
        except zmq.ZMQError as e:
            print(e)

    def recv_training_data(self):
        received = 0
        try:
            while True:
                traj = self.exp_socket.recv_pyobj(zmq.NOBLOCK)
                self.exp[self.individual_ids[traj['id']]].append(traj['traj'])
                received += 1
        except zmq.ZMQError:
            pass
        self.rcved += received

    def pub_params(self, specified=None):
        if specified is not None:
            try:
                self.param_socket.send(str(specified).encode(), zmq.SNDMORE)
                self.param_socket.send_pyobj(self.trained[self.individual_ids[specified]].get_params(), flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

        else:
            for individual in self.trained:
                try:
                    self.param_socket.send(str(individual.id).encode(), zmq.SNDMORE)
                    self.param_socket.send_pyobj(individual.get_params(), flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    pass

    def train(self, individual_index):
        if self.trained[individual_index] is not None and len(self.exp[individual_index]) >= self.batch_size:
            # Get experience from the queue
            trajectory = pd.DataFrame(self.exp[individual_index][:self.batch_size]).values
            self.exp = self.exp[individual_index][self.batch_size:]

            # Cook data
            states = np.float32(np.stack(trajectory[:, 0], axis=0))
            actions = np.float32(np.stack(trajectory[:, 1], axis=0)[:, :-1])
            probs = np.float32(np.stack(trajectory[:, 2], axis=0)[:, :-1])
            is_old = np.any(time() - np.stack(trajectory[:, 3], axis=0)[:, 0, :] > self.batch_age_limit)

            rews = self.rewards.compute(states, self.trained[individual_index].get_reward_shape())[:, :, np.newaxis]

            states *= GameState.scales

            if not is_old :
                # Train
                with tf.summary.record_if(self.train_cntr % self.write_summary_freq == 0):
                    self.trained[individual_index].train(states, actions, rews, probs, gpu=self.gpu_id)

                self.trained[individual_index].data_used += self.batch_size * self.TRAJECTORY_LENGTH
            else:
                print('Experience too old !')

    def train_each(self):
        tf.summary.experimental.set_step(self.train_cntr)
        self.train_cntr += 1

        for index in self.individual_ids.values():
            self.train(index)

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

                print('exp waiting : ', total_waiting)

    def run(self):
        success = False
        c = 0
        while not success:
            success = self.read_pop_update()
            c += 1
            if c > 30 :
                print('Cant connect to Hub !')

        self.pub_params()
        last_pub_time = time()

        while True:
            self.reply_hub()
            self.read_pop_update()
            self.recv_training_data()
            self.train_each()

            if time() - last_pub_time > 10:
                self.pub_params()
                last_pub_time = time()

    def exit(self, sig, frame):
        self.exp_socket.unbind()
        self.param_socket.unbind()
        print('Trainer', self.id, 'exited.')
        sys.exit(0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.exp_socket.unbind()
            self.param_socket.unbind()
        except:
            pass


if __name__ == '__main__':
    fire.Fire(Trainer)