import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def log_uniform(low=0, high=1, size=None, base=10):
    assert low < high
    low = np.log(low + 1e-8)/np.log(base)
    high = np.log(high + 1e-8)/np.log(base)
    return np.power(base, np.random.uniform(low, high, size))


# NN crossover

def nn_crossover(a, b, architecture={}):
    pairs = [pairwise_cross_corr(ax, bx) for ax,bx in zip(a,b)]
    W_permuted = [np.zeros((2,)+ax.shape, dtype=np.float32) for ax in a]
    for layer_index, previous_index in architecture.items():
        if previous_index is not None:
            W_permuted[layer_index][:, :-1, :] = a[layer_index][pairs[previous_index][0], :], b[layer_index][pairs[previous_index][1], :]

        layer_crossover(a, b, pairs[layer_index], layer_index, previous_index, W_permuted[layer_index])

    offspring = [(safe_crossover(*np.rollaxis(W_permuted[d], 0))) for d in range(len(a))]

    return offspring


def layer_crossover(a, b, pairs, layer_index, previous_index, permuted):
    ax = a[layer_index]
    bx = b[layer_index]

    if previous_index is None:
        permuted[:] = ax[:, pairs[0]], bx[:, pairs[1]]
    else:
        permuted[:] = permuted[0][:, pairs[0]], permuted[1][:, pairs[1]]


def corr_matrix(X,Y):
    xvar = tf.reduce_sum(tf.math.squared_difference(X, tf.reduce_mean(X, axis=0)), axis=0)
    yvar = tf.reduce_sum(tf.math.squared_difference(Y, tf.reduce_mean(Y, axis=0)), axis=0)
    dot = tf.tensordot(tf.transpose(X), Y, axes=1)

    M = dot / tf.sqrt(xvar * yvar)

    return M


def pairwise_cross_corr(La, Lb):
    n = La.shape[1]
    scaler = StandardScaler()
    n_La = scaler.fit_transform(La)
    n_Lb = scaler.fit_transform(Lb)

    m = corr_matrix(n_La, n_Lb).numpy()

    m[np.isnan(m)] = -1
    argmax_columns = np.flip(np.argsort(m, axis=0), axis=0)
    #print(np.max(argmax_columns))
    #print(argmax_columns.shape)
    dead_neurons = np.sum(m, axis=0) == - n

    pairs = np.full((2, n), fill_value=np.nan, dtype=np.int32)
    pairs[1:] = np.arange(n)
    index_add = 0

    for index in range(n):
        if not dead_neurons[index]:
            for count in range(n):
                if argmax_columns[count, index] not in pairs[0]:
                    pairs[0, index_add] = argmax_columns[count, index]
                    index_add += 1
                    break

    for index in range(n):
        if index not in pairs[0] and index_add < n:
            pairs[0, index_add] = index
            index_add += 1

    #print(pairs)
    return pairs


def get_indice_pairs(ax, bx):
    l = np.empty((2,len(ax)), dtype=np.int32)
    for index in range(len(ax)):
        sm = np.abs(np.min(ax[index]) + np.min(bx[index]))
        sp = np.abs(np.max(ax[index]) + np.max(bx[index]))
        if sp > sm:
            l[0, index] = np.argmax(ax[index])
            l[1, index] = np.argmax(bx[index])
        else:
            l[0, index] = np.argmin(ax[index])
            l[1, index] = np.argmax(bx[index])

    return l


def safe_crossover(ax, bx):
    t = np.random.uniform(-0.1, 1.1) # 0.25
    return (1-t) * ax + t * bx

