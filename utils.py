import tensorflow as tf


def get_scope_variable(var, shape, initializer):
    try:
        v = tf.get_variable(var, shape=shape, initializer=initializer)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        v = tf.get_variable(var)
    return v


def fc(name, input, num_out, act=tf.nn.relu, w_init=tf.random_normal_initializer,
       b_init=tf.zeros_initializer, is_training=False, bn=True):
    _, num_input = shape(input)
    with tf.variable_scope(name) as scope:
        w = get_scope_variable('w', (num_input, num_out), w_init)
        b = get_scope_variable('b', (1, num_out), b_init)
        h1 = tf.matmul(input, w) + b
        if bn:
            h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training, scope='bn',
                                              reuse=scope.reuse)
            out = act(h2, name='out')
        else:
            out = act(h1, name='out')
    return out


def conv2d(name, input, filter, strides, padding, act=tf.nn.relu, is_training=False, w_init=tf.random_normal_initializer, bn=True):
    with tf.variable_scope(name) as scope:
        w = get_scope_variable('w', filter, w_init)
        h1 = tf.nn.conv2d(input, w, strides, padding)
        if bn:
            h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training, scope='bn',
                                          reuse=scope.reuse)
            out = act(h2, name='out')
        else:
            out = act(h1, name='out')
    return out


def conv2d_transpose(name, input, filter, strides, output_shape, padding, act=tf.nn.relu, is_training=False, w_init=tf.random_normal_initializer, bn=True):
    with tf.variable_scope(name) as scope:
        w = get_scope_variable('weights', filter, w_init)
        h1 = tf.nn.conv2d_transpose(input, w, output_shape, strides, padding)
        if bn:
            h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training, scope='bn',
                                          reuse=scope.reuse)
            out = act(h2)
        else:
            out = act(h1)
    return out


def xavier_init(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=shape, stddev=xavier_stddev)


def leaky_relu(x, name=None):
    return tf.maximum(x, 0.01 * x, name=name)


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
