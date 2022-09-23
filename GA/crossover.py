import tensorflow as tf
import numpy as np


def DCX(a, b, c, states, ratio, eps_kl, max_iter):
    # TODO eps kl not good
    # data is as follows
    # [B, T, [value]]

    apply_a = np.float32(np.random.random(size=states.shape[:-1]+(1,)) > ratio)


def DCX_rewards(a, b, c, states, rewards_a, rewards_b, eps_kl, max_iter):
    # data is as follows
    # [B, T, [value]]
    kl = np.inf
    iter = 0
    while kl > eps_kl and iter < max_iter:
        iter += 1


@tf.function()
def distil_loop(a,b,c, states, mask, eps_kl, max_iter):
    kl = np.inf
    iter = 0
    while kl > eps_kl and iter < max_iter:
        iter += 1
        kl = bi_distillation(a,b,c, states, mask)
        # compute grads, apply grads
        with tf.GradientTape() as tape:
            kl = bi_distillation(a,b,c, states, mask)

        grad = tape.gradient(kl, c.trainable_variables)
        c.optim.apply_gradients(zip(grad, c.trainable_variables))



def bi_distillation(a,b,c, states, mask):
    # distill both on value and policy

    v_a = a.V(states)[:, :, 0]
    p_a = a.policy.get_probs(states[:, :])

    v_b = b.V(states)[:, :, 0]
    p_b = b.policy.get_probs(states[:, :])

    v_c = c.V(states)[:, :, 0]
    p_c = c.policy.get_probs(states[:, :])

    v_loss = 0.5 * tf.reduce_mean(
        tf.square(v_a-v_b)
    )





