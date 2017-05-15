import tensorflow as tf
from utils import *
import os
import numpy as np

X_SIZE = 784
Z_SIZE = 128


class Generator(object):
    def __init__(self, code_size=Z_SIZE):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.code_size = code_size

    def __call__(self, z):
        H_SIZE = 256
        H2_SIZE = 512
        with tf.variable_scope('gen'):
            out_1 = fc_bn('out_1', z, H_SIZE, is_training=self.is_training)
            out_2 = fc_bn('out_2', out_1, H2_SIZE, is_training=self.is_training)
            x = fc_bn('x', out_2, X_SIZE, act=tf.nn.sigmoid, is_training=self.is_training)
        return x


class Discriminator(object):
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def __call__(self, x):
        H_SIZE = 512
        H2_SIZE = 256
        with tf.variable_scope('disc'):
            out_1 = fc_bn('out_1', x, H_SIZE, is_training=self.is_training)
            out_2 = fc_bn('out_2', out_1, H2_SIZE, is_training=self.is_training)
            d = fc_bn('out', out_2, 1, act=tf.nn.sigmoid, is_training=self.is_training)
        return d

    def discriminate(self, sess, data, batch_size):
        x = tf.placeholder(tf.float32, (None, X_SIZE), name='x')
        d = sess.run(self(x), feed_dict={x: data, self.is_training: False})
        return d


class GD(object):
    def __init__(self, gen, disc, ckpt_path='checkpoints'):
        self.ckpt_path = ckpt_path
        self.gen = gen
        self.disc = disc

    def _loss_gen(self, z):
        loss = -1. * tf.reduce_mean(tf.log(self.disc(self.gen(z))))
        return loss

    def _loss_disc(self, z, data):
        loss_real = tf.log(self.disc(data))
        loss_fake = tf.log(1. - self.disc(self.gen(z)))
        loss = -1. * tf.reduce_mean(loss_real + loss_fake)
        return loss

    def _load(self, sess):
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

    def sample_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.gen.code_size))

    def train(self, sess, data, final_step, lr, batch_size, writer, k_disc=1, ckpt_step=1):
        z = tf.placeholder(tf.float32, (None, self.gen.code_size), name='z')
        x = tf.placeholder(tf.float32, (None, X_SIZE), name='x_data')
        loss_gen_op = self._loss_gen(z)
        loss_disc_op = self._loss_disc(z, x)

        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gen')):
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_gen_op, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'gen'), global_step=global_step)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='disc')):
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_disc_op, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'disc'))

        # grad_loss = tf.gradients(self.loss, [self.w_mu, self.w_var])
        # grad_encoder = tf.gradients([self.mu, self.log_var], [self.h2_mu, self.h2_var])

        self._load(sess)

        summary_loss_gen_op = tf.summary.scalar('summary_loss_gen', loss_gen_op)
        summary_loss_disc_op = tf.summary.scalar('summary_loss_disc', loss_disc_op)

        # = tf.summary.scalar('summary_loss_disc', loss_disc_op)
        # summary_op = tf.summary.merge_all()

        loss_gen, loss_disc = Diff(), Diff()
        for step in range(global_step.eval(), final_step):
            for k in range(k_disc):
                input, _ = data.next_batch(batch_size)
                _, curr_loss_disc, summary_loss_disc = sess.run([optimizer_disc, loss_disc_op, summary_loss_disc_op],
                                                                feed_dict={x: input,
                                                                           z: self.sample_noise(batch_size),
                                                                           self.disc.is_training: True,
                                                                           self.gen.is_training: True})
                # _, loss, g_loss, g_encoder, summary = sess.run([optimizer_gen, self.loss, grad_loss, grad_encoder, summary_op],
                #                                       feed_dict={self.input: input_batch, self.is_training: True, self.batch_size: batch_size})

            _, curr_loss_gen, summary_loss_gen = sess.run([optimizer_gen, loss_gen_op, summary_loss_gen_op],
                                                          feed_dict={z: self.sample_noise(batch_size),
                                                                     self.gen.is_training: True,
                                                                     self.disc.is_training: True})

            if (step + 1) % ckpt_step == 0:
                loss_gen.update(curr_loss_gen)
                loss_disc.update(curr_loss_disc)

                tf.train.Saver().save(sess, self.ckpt_path + '/gan', global_step=step)
                print 'step-{}\td_loss_gen={:2.2f}%\td_loss_disc={:2.2f}%\tloss_gen={}\tloss_disc={}'.format(step,
                                                                                                             loss_gen.diff * 100,
                                                                                                             loss_disc.diff * 100,
                                                                                                             loss_gen.value,
                                                                                                             loss_disc.value)
                # print 'grad_loss={}, grad_encoder={}'.format(g_loss, g_encoder)
                # writer.add_summary([summary_loss_gen, summary_loss_disc], global_step=step)

    def generate(self, sess, batch_size):
        self._load(sess)
        z = tf.placeholder(tf.float32, (None, self.gen.code_size), name='z')
        x = sess.run(self.gen(z), feed_dict={z: self.sample_noise(batch_size), self.gen.is_training: False})
        return x
