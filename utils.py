import tensorflow as tf


def fc(input, num_out, activation=tf.nn.relu, w_init=tf.zeros, b_init=tf.random_normal):
    layer, w = fc_with_weight(input, num_out, activation, w_init, b_init)
    return layer


def fc_with_weight(input, num_out, activation=tf.nn.relu, w_init=tf.zeros, b_init=tf.random_normal):
    _, num_input = shape(input)
    w = tf.Variable(w_init(shape=(num_input, num_out)))
    b = tf.Variable(b_init(shape=(1, num_out)))
    u = tf.matmul(input, w) + b
    return activation(u), w


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def xavier_init(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=shape, stddev=xavier_stddev)