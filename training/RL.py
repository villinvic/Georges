import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.backend import set_value
import numpy as np
from tensorflow.keras.activations import relu, softmax
from copy import deepcopy

from config.loader import Default
from game.state import PlayerState

class Distribution(object):
    def __init__(self, dim):
        self._dim = dim
        self._tiny = 1e-8

    @property
    def dim(self):
        raise self._dim

    def kl(self, old_dist, new_dist):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist, new_dist):
        raise NotImplementedError

    def entropy(self, dist):
        raise NotImplementedError

    def log_likelihood_sym(self, x, dist):
        raise NotImplementedError

    def log_likelihood(self, xs, dist):
        raise NotImplementedError


class Categorical(Distribution):
    def kl(self, old_prob, new_prob):
        """
        Compute the KL divergence of two Categorical distribution as:
            p_1 * (\log p_1  - \log p_2)
        """
        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob + self._tiny) - tf.math.log(new_prob + self._tiny)))

    def likelihood_ratio(self, x, old_prob, new_prob):
        return (tf.reduce_sum(new_prob * x) + self._tiny) / (tf.reduce_sum(old_prob * x) + self._tiny)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            \log \sum(p_i * x_i)

        :param x (tf.Tensor or np.ndarray): Values to compute log likelihood
        :param param (Dict): Dictionary that contains probabilities of outputs
        :return (tf.Tensor): Log probabilities
        """
        probs = param["prob"]
        assert probs.shape == x.shape, \
            "Different shape inputted. You might have forgotten to convert `x` to one-hot vector."
        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, probs, amount=1):
        # NOTE: input to `tf.random.categorical` is log probabilities
        # For more details, see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/categorical
        # [probs.shape[0], 1]
        # tf.print(probs, tf.math.log(probs), tf.random.categorical(tf.math.log(probs), amount), summarize=-1)
        return tf.cast(tf.map_fn(lambda p: tf.cast(tf.random.categorical(tf.math.log(p), amount), tf.float32), probs),
                       tf.int64)

    def entropy(self, probs):
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)


class CategoricalActor(tf.keras.Model):
    '''
    Actor model class
    '''

    def __init__(self, action_dim, epsilon, layer_dims, default_activation='elu',
                 name="CategoricalActor"):
        super().__init__(name=name)
        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim
        self.epsilon = epsilon

        self.denses = [Dense(dim, activation=default_activation, dtype="float32", name='dense_%d' % i)
                       for i, dim in enumerate(layer_dims)]

        self.prob = Dense(action_dim, dtype='float32', name="prob", activation="softmax")

    def _compute_feature(self, features):
        for layer in self.denses:
            features = layer(features)

        return features

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """

        features = self._compute_feature(states)

        probs = self.prob(features) * (1.0 - self.epsilon) + self.epsilon / np.float32(self.action_dim)

        return probs

    def get_action(self, state):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == 1

        state = state[np.newaxis][np.newaxis].astype(
            np.float32) if is_single_state else state
        action, probs = self._get_action_body(tf.constant(state))

        return (action.numpy()[0][0], probs.numpy()) if is_single_state else (action, probs)

    @tf.function
    def _get_action_body(self, state):
        probs = self._compute_dist(state)
        action = tf.squeeze(self.dist.sample(probs), axis=1)
        return action, probs

    def get_probs(self, states):
        return self._compute_dist(states)

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)


class Policy(tf.keras.Model, Default):
    '''
    Actor model class
    '''

    def __init__(self, action_dim, layer_dims, lstm_dim, default_activation='elu',
                 name="CategoricalActor"):
        super().__init__(name=name)
        Default.__init__(self)

        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim
        self.has_lstm = lstm_dim > 0
        if lstm_dim > 0:

            self.lstm = LSTM(lstm_dim, time_major=False, dtype='float32', stateful=True, return_sequences=True,
                         return_state=True, name='lstm')

        else :
            self.lstm = None

        self.denses = [Dense(dim, activation=default_activation, dtype="float32", name='dense_%d' % i)
                       for i, dim in enumerate(layer_dims)]

        self.prob = Dense(action_dim, dtype='float32', name="prob", activation="softmax")

    @tf.function
    def init_body(self, features):

        if self.has_lstm:
            features, hidden_h, hidden_c = self.lstm(features)
            for layer in self.denses:
                features = layer(features)
            features = self.prob(features)
            return hidden_h, hidden_c
        else:
            for layer in self.denses:
                features = layer(features)
            features = self.prob(features)
            return None, None

    def _compute_feature(self, features):
        if self.has_lstm:
            features, hidden_h, hidden_c = self.lstm(features)
            for layer in self.denses:
                features = layer(features)
            return features, (hidden_h, hidden_c)
        else:
            for layer in self.denses:
                features = layer(features)
            return features, (None, None)

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """

        features, hidden_states = self._compute_feature(states)

        probs = self.prob(features) * (1.0 - self.EPSILON_GREEDY) + self.EPSILON_GREEDY / np.float32(self.action_dim)

        return probs, hidden_states

    @tf.function
    def _get_action_body(self, state):
        probs, hidden_states = self._compute_dist(state)
        action = tf.squeeze(self.dist.sample(probs), axis=1)
        return action, probs, hidden_states

    def __call__(self, state):
        action, probs, (hidden_h, hidden_c) = self._get_action_body(state[np.newaxis][np.newaxis])
        return action.numpy()[0][0], probs.numpy(), hidden_h.numpy()[0], hidden_c.numpy()[0]

    def set_params(self, params):
        for dense, param in zip(self.denses, params['actor_core']):
            dense.set_weights(param)
        self.prob.set_weights(params['actor_head'])
        if self.has_lstm:
            self.lstm.set_weights(params['lstm'])

    def get_params(self):
        actor_weights = [dense.get_weights() for dense in self.denses]
        return {
            'lstm'      : self.lstm.get_weights() if self.has_lstm else None,
            'actor_core': actor_weights,
            'actor_head': self.prob.get_weights(),
        }


class ActionStateProbs(tf.keras.Model):
    '''
    Unconditional model representing probability distribution for each action state.
    '''

    def __init__(self, name='action_state_probs'):
        super().__init__(name=name)
        self.probs = tf.Variable(tf.fill([1, PlayerState.action_state_dim], 1.), dtype=tf.float32)

    def get(self):
        return softmax(self.probs)


class V(tf.keras.Model):
    """
    Value model function
    """

    def __init__(self, layer_dims, default_activation='elu', name='vf'):
        super().__init__(name=name)

        # self.l1 = Dense(128, activation='elu', dtype='float32', name="v_L1")
        #self.l1 = GRU(512, time_major=False, dtype='float32', stateful=True, return_sequences=True)
        self.denses = [Dense(dim, activation=default_activation, dtype='float32') for dim in layer_dims]

        self.v = Dense(1, activation='linear', dtype='float32', name="v")

    def call(self, states):
        features = states
        for layer in self.denses:
            features = layer(features)

        return self.v(features)


class AC(tf.keras.Model, Default):
    def __init__(self, action_dim, layer_dims,
                 lstm_dim):
        super(AC, self).__init__(name='AC')
        Default.__init__(self)

        self.action_dim = action_dim
        self.has_lstm = lstm_dim > 0

        if lstm_dim == 0:
            self.lstm = None
        else:
            self.lstm = LSTM(lstm_dim, time_major=False, dtype='float32', stateful=False, return_sequences=True,
                         return_state=False, name='lstm')
        self.V = V(layer_dims)
        self.policy = CategoricalActor(action_dim, self.epsilon_greedy, layer_dims)
        self.as_probs = ActionStateProbs()
        self.p_optim = tf.keras.optimizers.SGD(learning_rate=1.)

        self.optim = tf.keras.optimizers.RMSprop(rho=0.99, epsilon=1e-5) # Learning rate is affected when training

        self.step = tf.Variable(0, dtype=tf.int32)

        self.range_ = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(self.TRAJECTORY_LENGTH-1, dtype=tf.int32), axis=0), [self.BATCH_SIZE, 1]),
                                     axis=2)
        self.pattern = tf.expand_dims([tf.fill((self.TRAJECTORY_LENGTH-1,), i) for i in range(self.BATCH_SIZE)], axis=2)

    def train(self, log_name, training_params, as_entropy_scale, states, actions, rewards, probs, hidden_states, gpu):
        # do some stuff with arrays
        # print(states, actions, rewards, dones)
        # Set both networks with corresponding initial recurrent state
        self.optim.learning_rate.assign(training_params['learning_rate'])

        v_loss, mean_entropy, min_entropy, max_entropy, min_logp, max_logp, grad_norm, as_entropy \
            = self._train(np.float32(training_params['entropy_cost']), np.float32(training_params['gamma']),
                          np.float32(as_entropy_scale), states, actions, rewards, probs, hidden_states , gpu)

        tf.summary.scalar(name=log_name + "/v_loss", data=v_loss)
        tf.summary.scalar(name=log_name + "/as_ent", data=as_entropy)
        tf.summary.scalar(name=log_name + "/min_entropy", data=min_entropy)
        tf.summary.scalar(name=log_name + "/max_entropy", data=max_entropy)
        tf.summary.scalar(name=log_name + "/mean_entropy", data=mean_entropy)
        tf.summary.scalar(name=log_name + "/ent_scale", data=training_params['entropy_cost'])
        tf.summary.scalar(name=log_name + "/gamma", data=training_params['gamma'])
        tf.summary.scalar(name=log_name + "/learning_rate", data=training_params['learning_rate'])
        tf.summary.scalar(name=log_name + "/min_logp", data=min_logp)
        tf.summary.scalar(name=log_name + "/max_logp", data=max_logp)
        tf.summary.scalar(name=log_name + "/grad_norm", data=grad_norm)
        #tf.summary.scalar(name="misc/distance", data=tf.reduce_mean(states[:, :, -1]))
        tf.summary.scalar(name=log_name + "/reward", data=tf.reduce_sum(tf.reduce_mean(rewards, axis=0)))


    @tf.function
    def _train(self, alpha, gamma, as_entropy_scale, states, actions, rewards, probs, hidden_states, gpu):
        '''
        Main training function
        '''

        if tf.reduce_any(tf.math.is_nan(states)):
            print('states')
        if tf.reduce_any(tf.math.is_nan(actions)):
            print('actions')
        if tf.reduce_any(tf.math.is_nan(rewards)):
            print('rewards')
        if tf.reduce_any(tf.math.is_nan(probs)):
            print('probs')
        if tf.reduce_any(tf.math.is_nan(hidden_states)):
            print('hidden_states')


        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):

            actions = tf.cast(actions, dtype=tf.int32)
            with tf.GradientTape() as tape:
                # Update the action_state probability distribution

                as_probs = self.as_probs.get()[0]

                action_states = tf.cast(tf.argmax(states[:, :-1,
                    PlayerState.onehot_offsets['action_state']:
                    PlayerState.onehot_offsets['action_state'] + PlayerState.action_state_dim], axis=2), dtype=tf.int32)

                taken_as = tf.gather_nd(as_probs, tf.expand_dims(action_states, axis=2), batch_dims=0)
                NLL = -tf.math.log(taken_as + 1e-8)
                loss = tf.reduce_mean(NLL)

            grad = tape.gradient(loss, self.as_probs.trainable_variables)
            self.p_optim.apply_gradients(zip(grad, self.as_probs.trainable_variables))

            # Reward rare action_states with their negative log likelihood
            rewards += tf.clip_by_value(NLL - tf.math.log(400.0), clip_value_min=0.0,
                                        clip_value_max=10.0) * as_entropy_scale


            as_ent = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.math.log(as_probs+1e-8), as_probs), -1))

            with tf.GradientTape() as tape:
                # Optimize the actor and critic
                if self.has_lstm:
                    lstm_states = self.lstm(states, initial_state=[hidden_states[:,0],hidden_states[:, 1]])
                else:
                    lstm_states = states

                v_all = self.V(lstm_states)[: ,:, 0]
                p = self.policy.get_probs(lstm_states[:, :-1])
                kl = tf.divide(p, probs+1e-3)#tf.reduce_sum(p * tf.math.log(tf.divide(p, probs)), axis=-1)
                indices = tf.concat(values=[self.pattern, self.range_, tf.expand_dims(actions, axis=2)], axis=2)
                rho_mu = tf.minimum(1., tf.gather_nd(kl, indices, batch_dims=0))
                targets = self.compute_trace_targets(v_all, rewards, rho_mu, gamma)
                #targets = self.compute_gae(v_all[:, :-1], rewards[:, :-1], v_all[:, -1])
                advantage = tf.stop_gradient(targets) - v_all
                v_loss = tf.reduce_mean(tf.square(advantage))

                p_log = tf.math.log(p + 1e-8)

                ent = - tf.reduce_sum(tf.multiply(p_log, p), -1)


                taken_p_log = tf.gather_nd(p_log, indices, batch_dims=0)

                p_loss = - tf.reduce_mean( tf.stop_gradient(rho_mu) * taken_p_log * tf.stop_gradient(targets[:, 1:]*gamma + rewards - v_all[:, :-1]) + alpha * ent)
                    #taken_p_log * tf.stop_gradient(advantage) + self.entropy_scale * ent)

                total_loss = 0.5 * v_loss + p_loss

            grad = tape.gradient(total_loss, self.policy.trainable_variables
                                 + self.V.trainable_variables
                                 + self.lstm.trainable_variables)

            # x is used to track the gradient size
            x = 0.0
            c = 0.0
            for gg in grad:
                c += 1.0
                x += tf.reduce_mean(tf.abs(gg))
            x /= c

            self.optim.apply_gradients(zip(grad, self.policy.trainable_variables
                                           + self.V.trainable_variables + self.lstm.trainable_variables))

            self.step.assign_add(1)
            mean_entropy = tf.reduce_mean(ent)
            min_entropy = tf.reduce_min(ent)
            max_entropy = tf.reduce_max(ent)
            return v_loss, mean_entropy, min_entropy, max_entropy, tf.reduce_min(
                p_log), tf.reduce_max(p_log), x, as_ent

    def compute_gae(self, v, rewards, last_v, gamma):
        v = tf.transpose(v)
        rewards = tf.transpose(rewards)
        reversed_sequence = [tf.reverse(t, [0]) for t in [v, rewards]]

        def bellman(future, present):
            val, r = present
            return (1. - self.gae_lambda) * val + self.gae_lambda * (
                        r + gamma * future)

        returns = tf.scan(bellman, reversed_sequence, last_v)
        returns = tf.reverse(returns, [0])
        returns = tf.transpose(returns)
        return returns

    def compute_trace_targets(self, v, rewards, rho_mu, gamma):
        # coefs set to 1
        vals_s = tf.transpose(v[:, :-1])
        vals_sp1 = tf.transpose(v[:, 1:])
        last_vr = v[:, -1]# + rewards[:, -1]
        rewards = tf.transpose(rewards) #  rewards[:, :-1]
        rho_mu = tf.transpose(rho_mu)
        reversed_sequence = [tf.reverse(t, [0]) for t in [vals_s, vals_sp1, rewards, rho_mu]]

        def bellman(future, present):
            val_s, val_sp1, r, rm = present

            return val_s+ rm * (r + gamma * val_sp1 - val_s) + gamma * rm \
                   * (future - val_sp1)

        returns = tf.scan(bellman, reversed_sequence, last_vr)
        returns = tf.reverse(returns, [0])
        returns = tf.transpose(returns)
        return tf.concat([returns, tf.expand_dims(last_vr, axis=1)], axis=1)

    def get_params(self):
        actor_weights = [dense.get_weights() for dense in self.policy.denses]
        return {
            'lstm': self.lstm.get_weights() if self.has_lstm else None,
            'actor_core': actor_weights,
            'actor_head': self.policy.prob.get_weights(),
        }

    def get_training_params(self):
        actor_weights = [dense.get_weights() for dense in self.policy.denses]
        value_weights = [dense.get_weights() for dense in self.V.denses]
        return {
            'lstm'      : self.lstm.get_weights() if self.has_lstm else None,
            'actor_core': actor_weights,
            'actor_head': self.policy.prob.get_weights(),
            'value_core': value_weights,
            'value_head': self.V.v.get_weights()
        }

    def set_training_params(self, params):

        if self.has_lstm:
            self.lstm.set_weights(params['lstm'])
        for dense_layer_weights, dense in zip(params['actor_core'], self.policy.denses):
            dense.set_weights(dense_layer_weights)
        self.policy.prob.set_weights(params['actor_head'])
        for dense_layer_weights, dense in zip(params['value_core'], self.V.denses):
            dense.set_weights(dense_layer_weights)
        self.V.v.set_weights(params['value_head'])




    @tf.function
    def init_body(self, lstm):
        if self.has_lstm:
            lstm = self.lstm(lstm)
        x = self.policy.get_probs(lstm[:, 1:])
        self.V(lstm)
        
    def perturb(self):
        pass
    
    def crossover(self, other_policy):
        model_size = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        cross_point = np.random.randint(0, model_size)
        c = 0
        parents = [self.get_training_params(), other_policy.get_training_params()]
        np.random.shuffle(parents)

        if parents[0]['lstm']:
            for i in range(len(parents[0]['lstm'])):
                if parents[0]['lstm'][i].ndim > 1 :
                    for j in range(len(parents[0]['lstm'][i])):
                            if c > cross_point:
                                parents[0]['lstm'][i][j][:] = parents[1]['lstm'][i][j]
                            elif c + len(parents[0]['lstm'][i][j]) >= cross_point:
                                parents[0]['lstm'][i][j][cross_point-c:] = parents[1]['lstm'][i][j][cross_point-c:]
                            c += len(parents[0]['lstm'][i][j])
                else:
                    if c > cross_point:
                        parents[0]['lstm'][i][:] = parents[1]['lstm'][i]
                    elif c + len(parents[0]['lstm'][i]) >= cross_point:
                        parents[0]['lstm'][i][cross_point - c:] = parents[1]['lstm'][i][cross_point - c:]
                    c += len(parents[0]['lstm'][i])

            for i in range(len(parents[0]['actor_core'])):
                for j in range(len(parents[0]['actor_core'][i])):
                    if parents[0]['actor_core'][i][j].ndim > 1 :
                        for k in range(len(parents[0]['actor_core'][i][j])):
                                if c > cross_point:
                                    parents[0]['actor_core'][i][j][k][:] = parents[1]['actor_core'][i][j][k]
                                elif c + len(parents[0]['actor_core'][i][j][k]) >= cross_point:
                                    parents[0]['actor_core'][i][j][k][cross_point-c:] = parents[1]['actor_core'][i][j][k][cross_point-c:]
                                c += len(parents[0]['actor_core'][i][j][k])
                    else:
                        if c > cross_point:
                            parents[0]['actor_core'][i][j][:] = parents[1]['actor_core'][i][j]
                        elif c + len(parents[0]['actor_core'][i][j]) >= cross_point:
                            parents[0]['actor_core'][i][j][cross_point - c:] = parents[1]['actor_core'][i][j][cross_point - c:]
                        c += len(parents[0]['actor_core'][i][j])

            for i in range(len(parents[0]['actor_head'])):
                if parents[0]['actor_head'][i].ndim > 1:
                    for j in range(len(parents[0]['actor_head'][i])):
                        if c > cross_point:
                            parents[0]['actor_head'][i][j][:] = parents[1]['actor_head'][i][j]
                        elif c + len(parents[0]['actor_head'][i][j]) >= cross_point:
                            parents[0]['actor_head'][i][j][cross_point - c:] = parents[1]['actor_head'][i][j][cross_point - c:]
                        c += len(parents[0]['actor_head'][i][j])
                else:
                    if c > cross_point:
                        parents[0]['actor_head'][i][:] = parents[1]['actor_head'][i]
                    elif c + len(parents[0]['actor_head'][i]) >= cross_point:
                        parents[0]['actor_head'][i][cross_point - c:] = parents[1]['actor_head'][i][cross_point - c:]
                    c += len(parents[0]['actor_head'][i])
                    
            
            for i in range(len(parents[0]['value_core'])):
                for j in range(len(parents[0]['value_core'][i])):
                    if parents[0]['value_core'][i][j].ndim > 1 :
                        for k in range(len(parents[0]['value_core'][i][j])):
                                if c > cross_point:
                                    parents[0]['value_core'][i][j][k][:] = parents[1]['value_core'][i][j][k]
                                elif c + len(parents[0]['value_core'][i][j][k]) >= cross_point:
                                    parents[0]['value_core'][i][j][k][cross_point-c:] = parents[1]['value_core'][i][j][k][cross_point-c:]
                                c += len(parents[0]['value_core'][i][j][k])
                    else:
                        if c > cross_point:
                            parents[0]['value_core'][i][j][:] = parents[1]['value_core'][i][j]
                        elif c + len(parents[0]['value_core'][i][j]) >= cross_point:
                            parents[0]['value_core'][i][j][cross_point - c:] = parents[1]['value_core'][i][j][cross_point - c:]
                        c += len(parents[0]['value_core'][i][j])
                    
            
            for i in range(len(parents[0]['value_head'])):
                if parents[0]['value_head'][i].ndim > 1:
                    for j in range(len(parents[0]['value_head'][i])):
                        if c > cross_point:
                            parents[0]['value_head'][i][j][:] = parents[1]['value_head'][i][j]
                        elif c + len(parents[0]['value_head'][i][j]) >= cross_point:
                            parents[0]['value_head'][i][j][cross_point - c:] = parents[1]['value_head'][i][j][cross_point - c:]
                        c += len(parents[0]['value_head'][i][j])
                else:
                    if c > cross_point:
                        parents[0]['value_head'][i][:] = parents[1]['value_head'][i]
                    elif c + len(parents[0]['value_head'][i]) >= cross_point:
                        parents[0]['value_head'][i][cross_point - c:] = parents[1]['value_head'][i][cross_point - c:]
                    c += len(parents[0]['value_head'][i])
                    
            

        self.set_training_params(parents[0])