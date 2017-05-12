import tensorflow as tf
from utils import *
import os
import numpy as np

X_SIZE = 784
Z_SIZE = 128


class Generator(object):
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool)
        self.z = tf.placeholder(tf.float32, (None, Z_SIZE))
        self.net = self._net(self.z)

    def _net(self, z):
        H_SIZE = 256
        with tf.variable_scope('gen'):
            out_1 = fc('out_1', z, H_SIZE, self.is_training)
            x = fc('x', out_1, X_SIZE, self.is_training, act=tf.nn.sigmoid)
        return x

    def generate(self, sess, z):
        x = sess.run(self.net, feed_dict={self.z: z, self.is_training: True})
        return x


class Discriminator(object):
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.Variable(dtype=tf.float32, trainable=False)
        self.net = self._net(self.x)

    def _net(self, x):
        H_SIZE = 256
        with tf.variable_scope('disc'):
            out_1 = fc('out_1', x, H_SIZE, self.is_training)
            d = fc('out', out_1, 1, self.is_training, act=tf.nn.sigmoid)
        return d

    def discriminate(self, sess, x):
        d = sess.run(self._net(x), feed_dict={self.is_training: False})
        return d


class GD(object):
    def __init__(self, gen, disc, ckpt_path='checkpoints'):
        self.ckpt_path = ckpt_path
        self.gen = gen
        self.disc = disc

    def _loss_gen(self, batch_size):
        loss = -1. * tf.reduce_sum(tf.log(self.disc.net(self.gen.net))) / batch_size
        return loss

    def _loss_disc(self, batch_size):

        loss_data = tf.log(self.disc.net)

        loss_gen = tf.log(1. - self.disc.net(self.gen.net))
        loss = -1. * tf.reduce_sum(loss_data + loss_gen) / batch_size
        return loss

    def _load(self, sess):
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

    def train(self, sess, data, final_step, lr, batch_size, writer, k_disc=1, ckpt_step=1):
        loss_gen_op = self._loss_gen(batch_size)
        loss_disc_op = self._loss_disc(batch_size)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_gen, optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                [loss_gen_op, loss_disc_op], global_step=global_step)

            # optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss_disc)
        # grad_loss = tf.gradients(self.loss, [self.w_mu, self.w_var])
        # grad_encoder = tf.gradients([self.mu, self.log_var], [self.h2_mu, self.h2_var])

        self._load(sess)

        summary_loss_gen_op, summary_loss_disc_op = tf.summary.scalar(['summary_loss_gen', 'summary_loss_disc'],
                                                                      [loss_gen_op, loss_disc_op])
        # = tf.summary.scalar('summary_loss_disc', loss_disc_op)
        # summary_op = tf.summary.merge_all()

        loss_gen, loss_disc = Diff(), Diff()
        for step in range(global_step.eval(), final_step):
            for k in range(k_disc):
                noise = np.random.normal(0, 1, (batch_size, Z_SIZE))
                input_batch, _ = data.next_batch(batch_size)
                _, curr_loss_disc, summary_loss_disc = sess.run(
                    [optimizer_disc, loss_disc_op, summary_loss_disc_op],
                    feed_dict={self.disc.x: input_batch, self.gen.z: noise,
                               self.disc.is_training: True,
                               self.gen.is_training: True})
                # _, loss, g_loss, g_encoder, summary = sess.run([optimizer_gen, self.loss, grad_loss, grad_encoder, summary_op],
                #                                       feed_dict={self.input: input_batch, self.is_training: True, self.batch_size: batch_size})
            noise = np.random.normal(0, 1, (batch_size, Z_SIZE))
            _, curr_loss_gen, summary_loss_gen = sess.run([optimizer_gen, loss_gen_op, summary_loss_gen_op],
                                                          feed_dict={self.gen.z: noise, self.gen.is_training: True,
                                                                     self.disc.is_training: True})

            if (step + 1) % ckpt_step == 0:
                loss_gen.update(curr_loss_gen)
                loss_disc.update(curr_loss_disc)

                tf.train.Saver().save(sess, self.ckpt_path + '/gan', global_step=step)
                print 'step-{}\td_loss_gen={:2.2f}%\td_loss_disc={:2.2f}%\tloss_gen{}\tloss_disc={}'.format(step,
                                                                                                            loss_gen.diff * 100,
                                                                                                            loss_disc.diff * 100,
                                                                                                            loss_gen.value,
                                                                                                            loss_disc.value)
                # print 'grad_loss={}, grad_encoder={}'.format(g_loss, g_encoder)
                writer.add_summary([summary_loss_gen, summary_loss_disc], global_step=step)
