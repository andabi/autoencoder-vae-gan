import tensorflow as tf


def get_scope_variable(var, shape, initializer):
    try:
        v = tf.get_variable(var, shape=shape, initializer=initializer)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        v = tf.get_variable(var)
    return v


def fc(name, input, num_out, is_training, act=tf.nn.relu, w_init=tf.random_normal_initializer,
       b_init=tf.zeros_initializer):
    layer, w, b, h1, h2 = fc_with_variables(name, input, num_out, is_training, act, w_init, b_init)
    return layer


def fc_with_variables(name, input, num_out, is_training, act=tf.nn.relu, w_init=tf.random_normal_initializer,
                      b_init=tf.zeros_initializer):
    _, num_input = shape(input)
    with tf.variable_scope(name) as scope:
        w = get_scope_variable('weights', (num_input, num_out), w_init)
        b = get_scope_variable('biases', (1, num_out), b_init)
        h1 = tf.matmul(input, w) + b
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training, scope=scope,
                                          reuse=scope.reuse)
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

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v
