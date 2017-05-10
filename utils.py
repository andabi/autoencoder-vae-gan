import tensorflow as tf


def fc(input, num_out, is_training, act=tf.nn.relu, w_init=tf.random_normal, b_init=tf.zeros, name='fc_layer'):
    layer, w, b, h1, h2 = fc_with_variables(input, num_out, is_training, act, w_init, b_init, name)
    return layer


def fc_with_variables(input, num_out, is_training, act=tf.nn.relu, w_init=tf.random_normal, b_init=tf.zeros,
                      name='fc_layer'):
    with tf.name_scope(name):
        _, num_input = shape(input)
        w = tf.Variable(w_init(shape=(num_input, num_out)), name='weights')
        b = tf.Variable(b_init(shape=(1, num_out)), name='biases')
        h1 = tf.matmul(input, w) + b
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training)
        return act(h2), w, b, h1, h2


def xavier_init(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=shape, stddev=xavier_stddev)


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


class Diff(object):
    def __init__(self, v=0.):
        self.value = v
        self.diff = 0.

    def value(self):
        return self.value

    def diff(self):
        return self.diff

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v
