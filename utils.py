import tensorflow as tf


def fc(input, num_out, activation=tf.nn.relu):
    layer, w = fc_with_weight(input, num_out, activation)
    return layer


def fc_with_weight(input, num_out, activation=tf.nn.relu):
    _, num_input = input.get_shape()
    w = tf.Variable(tf.zeros(shape=(num_input, num_out)))
    b = tf.Variable(tf.random_normal(shape=(1, num_out)))
    u = tf.matmul(input, w) + b
    return activation(u), w