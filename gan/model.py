import tensorflow as tf
from utils import *
import os
import numpy as np

X_SIZE = 784
Z_SIZE = 128


class Generator(object):
    def __init__(self, code_size=Z_SIZE, batch_size=1):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.code_size = code_size
        self.z = tf.placeholder(tf.float32, (batch_size, code_size), name='z')
        self()

    # def __call__(self):
    #     H_SIZE = 128
    #     H2_SIZE = 256
    #     H3_SIZE = 512
    #     with tf.variable_scope('gen'):
    #         out_1 = fc_bn('out_1', self.z, H_SIZE, act=leaky_relu, is_training=self.is_training)
    #         out_2 = fc_bn('out_2', out_1, H2_SIZE, act=leaky_relu, is_training=self.is_training)
    #         out_3 = fc_bn('out_3', out_2, H3_SIZE, act=leaky_relu, is_training=self.is_training)
    #         # out_4 = fc_bn('out_4', out_3, H3_SIZE, is_training=self.is_training)
    #         # out_5 = fc_bn('out_5', out_4, H3_SIZE, is_training=self.is_training)
    #         x = fc('x', out_3, X_SIZE, act=tf.nn.sigmoid)
    #     return x


    def __call__(self):
        H_SIZE = 32
        H2_SIZE = 128
        H3_SIZE = 169
        batch_size = shape(self.z)[0]
        with tf.variable_scope('gen'):
            out_1 = fc_bn('out_1', self.z, H_SIZE, act=leaky_relu, is_training=self.is_training)
            out_2 = fc_bn('out_2', out_1, H2_SIZE, act=leaky_relu, is_training=self.is_training)
            out_3 = fc_bn('out_3', out_2, H3_SIZE, act=leaky_relu, is_training=self.is_training)
            out_3 = tf.reshape(out_3, [batch_size, 13, 13, 1])
            out_4 = conv2d_transpose_bn('out_4', out_3, filter=[6, 6, 10, 1], output_shape=[batch_size, 18, 18, 10],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        act=leaky_relu, is_training=self.is_training)
            out_5 = conv2d_transpose_bn('out_5', out_4, filter=[6, 6, 10, 10], output_shape=[batch_size, 23, 23, 10],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        act=leaky_relu, is_training=self.is_training)
            x = conv2d_transpose('x', out_5, filter=[6, 6, 1, 10], output_shape=[batch_size, 28, 28, 1],
                                 strides=[1, 1, 1, 1], padding='VALID',
                                 act=tf.nn.sigmoid)
        return x


class Discriminator(object):
    def __init__(self, in_size=X_SIZE, batch_size=1):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.x = tf.placeholder(tf.float32, (batch_size, in_size), name='x')
        self()

    def __call__(self, x=None):
        H_SIZE = 128
        H2_SIZE = 32
        H3_SIZE = 16
        if x is None:
            x = self.x
        batch_size = shape(x)[0]
        with tf.variable_scope('disc'):
            x = tf.reshape(x, [batch_size, 28, 28, 1])
            out_1 = conv2d_bn('out_1', x, filter=[6, 6, 1, 10], strides=[1, 1, 1, 1], padding='VALID', act=leaky_relu,
                              is_training=self.is_training)
            out_2 = conv2d_bn('out_2', out_1, filter=[6, 6, 10, 10], strides=[1, 1, 1, 1], padding='VALID',
                              act=leaky_relu, is_training=self.is_training)
            out_3 = conv2d_bn('out_3', out_2, filter=[6, 6, 10, 1], strides=[1, 1, 1, 1], padding='VALID',
                              act=leaky_relu, is_training=self.is_training)
            out_3 = tf.reshape(out_3, [batch_size, 169])
            out_4 = fc_bn('out_4', out_3, H_SIZE, act=leaky_relu, is_training=self.is_training)
            out_5 = fc_bn('out_5', out_4, H2_SIZE, act=leaky_relu, is_training=self.is_training)
            # out_5 = fc_bn('out_5', out_4, H3_SIZE, act=leaky_relu, is_training=self.is_training)
            d = fc('out', out_5, 1, act=tf.nn.sigmoid)
        return d


class GD(object):
    def __init__(self, gen, disc, ckpt_path='checkpoints'):
        self.ckpt_path = ckpt_path
        self.gen = gen
        self.disc = disc

    # def _loss_disc(self, batch_size):
    #     loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones((batch_size, 1)), logits=self.disc())
    #     loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros((batch_size, 1)),
    #                                                         logits=self.disc(self.gen()))
    #     loss = 0.5 * tf.reduce_mean(loss_real + loss_fake)
    #     return loss

    def _loss_gen(self, batch_size):
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones((batch_size, 1)),
        #                                                logits=self.disc(self.gen()))
        loss = -tf.log(self.disc(self.gen()))
        return tf.reduce_mean(loss)

    def _loss_disc(self, batch_size):
        loss_real = -tf.log(self.disc())
        loss_fake = -tf.log(1. - self.disc(self.gen()))
        loss = 0.5 * (loss_real + loss_fake)
        return tf.reduce_mean(loss)

    def _load(self, sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

    def sample_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.gen.code_size))

    def train(self, sess, data, final_step, lr_gen, lr_disc, batch_size, writer, k_disc=1, ckpt_step=1):

        loss_gen_op = self._loss_gen(batch_size)
        loss_disc_op = self._loss_disc(batch_size)

        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gen')):
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_gen).minimize(loss_gen_op,
                                                                                  var_list=tf.get_collection(
                                                                                      tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                      'gen'), global_step=global_step)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='disc')):
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_disc).minimize(loss_disc_op,
                                                                                    var_list=tf.get_collection(
                                                                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                        'disc'))

        # grad_loss = tf.gradients(self.loss, [self.w_mu, self.w_var])
        # grad_encoder = tf.gradients([self.mu, self.log_var], [self.h2_mu, self.h2_var])

        sess.run(tf.global_variables_initializer())
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
                                                                feed_dict={self.disc.x: input,
                                                                           self.gen.z: self.sample_noise(batch_size),
                                                                           self.disc.is_training: True,
                                                                           self.gen.is_training: True})
                # _, loss, g_loss, g_encoder, summary = sess.run([optimizer_gen, self.loss, grad_loss, grad_encoder, summary_op],
                #                                       feed_dict={self.input: input_batch, self.is_training: True, self.batch_size: batch_size})

            _, curr_loss_gen, summary_loss_gen = sess.run([optimizer_gen, loss_gen_op, summary_loss_gen_op],
                                                          feed_dict={self.gen.z: self.sample_noise(batch_size),
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

                x = sess.run(self.gen(),
                             feed_dict={self.gen.z: self.sample_noise(batch_size), self.gen.is_training: False})
                x = tf.reshape(x, [batch_size, 28, 28, 1])
                image_summary = tf.summary.image('generated_image', x, 1)
                summary = sess.run(image_summary)
                writer.add_summary(summary, global_step=step)

                # print 'grad_loss={}, grad_encoder={}'.format(g_loss, g_encoder)
                # writer.add_summary([summary_loss_gen, summary_loss_disc], global_step=step)

    def generate(self, sess, batch_size):
        sess.run(tf.global_variables_initializer())
        self._load(sess)
        return sess.run(self.gen(), feed_dict={self.gen.z: self.sample_noise(batch_size), self.gen.is_training: False})

    def discriminate(self, sess, data):
        sess.run(tf.global_variables_initializer())
        self._load(sess)
        return sess.run(tf.sigmoid(self.disc()), feed_dict={self.disc.x: data, self.disc.is_training: False})
