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

    def __call__(self):
        batch_size = shape(self.z)[0]
        with tf.variable_scope('gen'):
            out_1 = fc('out_1', self.z, 392, act=leaky_relu, is_training=self.is_training)
            out_1 = tf.reshape(out_1, [-1, 7, 7, 8])
            out_2 = conv2d_transpose('out_2', out_1, filter=[3, 3, 4, 8], output_shape=[batch_size, 14, 14, 4],
                                        strides=[1, 2, 2, 1], padding='SAME',
                                        act=leaky_relu, is_training=self.is_training)
            out = conv2d_transpose('out', out_2, filter=[3, 3, 1, 4], output_shape=[batch_size, 28, 28, 1],
                                        strides=[1, 2, 2, 1], padding='SAME',
                                        act=tf.nn.sigmoid, bn=False)
            out = tf.reshape(out, [-1, 784])
        return out


class Discriminator(object):
    def __init__(self, in_size=X_SIZE, batch_size=1):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.x = tf.placeholder(tf.float32, (batch_size, in_size), name='x')
        self()

    def __call__(self, x=None):
        if x is None:
            x = self.x
        with tf.variable_scope('disc'):
            x = tf.reshape(x, [-1, 28, 28, 1])
            out_1 = conv2d('out_1', x, filter=[3, 3, 1, 32], strides=[1, 2, 2, 1], padding='SAME',
                           act=leaky_relu, is_training=self.is_training)
            out_2 = conv2d('out_2', out_1, filter=[3, 3, 32, 64], strides=[1, 2, 2, 1], padding='SAME',
                           act=leaky_relu, is_training=self.is_training)
            out_2 = tf.reshape(out_2, [-1, 7 * 7 * 64])
            out_3 = fc('out_3', out_2, 1024, act=leaky_relu, is_training=self.is_training)
            out_4 = fc('out_4', out_3, 10, act=leaky_relu, is_training=self.is_training)
            d = fc('out', out_4, 1, act=tf.nn.sigmoid, bn=False)
        return d


class GD(object):
    def __init__(self, gen, disc, ckpt_path='checkpoints'):
        self.ckpt_path = ckpt_path
        self.gen = gen
        self.disc = disc

    def _loss_gen(self):
        loss = -tf.log(self.disc(self.gen()) + 1e-9)
        return tf.reduce_mean(loss)

    def _loss_disc(self):
        loss_real, loss_fake = -tf.log(self.disc() + 1e-9), -tf.log(1. - self.disc(self.gen()) + 1e-9)
        loss = 0.5 * (loss_real + loss_fake)
        return tf.reduce_mean(loss)

    def _load(self, sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

    def _summaries_gen(self):
        tf.get_variable_scope().reuse_variables()
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen'):
            tf.summary.histogram(v.name, v, collections=['gen'])
            tf.summary.histogram('grad/' + v.name, tf.gradients(self._loss_gen(), v), collections=['gen'])

        tf.summary.scalar('gen/loss', self._loss_gen(), collections=['gen'])

        tf.summary.histogram('gen/disc', self.disc(self.gen()), collections=['disc'])
        tf.summary.scalar('gen/disc', tf.reduce_mean(self.disc(self.gen())), collections=['gen'])

        return tf.summary.merge_all(key='gen')

    def _summaries_disc(self):
        tf.get_variable_scope().reuse_variables()
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc'):
            tf.summary.histogram(v.name, v, collections=['disc'])
            tf.summary.histogram('grad/' + v.name, tf.gradients(self._loss_disc(), v), collections=['disc'])

        tf.summary.scalar('disc/loss', self._loss_disc(), collections=['disc'])

        tf.summary.histogram('disc/disc_real', self.disc(), collections=['disc'])
        tf.summary.scalar('disc/disc_real', tf.reduce_mean(self.disc()), collections=['disc'])
        tf.summary.histogram('disc/disc_fake', self.disc(self.gen()), collections=['disc'])
        tf.summary.scalar('disc/disc_fake', tf.reduce_mean(self.disc(self.gen())), collections=['disc'])

        return tf.summary.merge_all(key='disc')

    def sample_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.gen.code_size))

    def train(self, sess, data, final_step, lr_gen, lr_disc, batch_size, writer, k_gen=1, k_disc=1, ckpt_step=1):

        loss_gen_op = self._loss_gen()
        loss_disc_op = self._loss_disc()

        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gen')):
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_gen, beta2=0.5).minimize(loss_gen_op,
                                                                                  var_list=tf.get_collection(
                                                                                      tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                      'gen'))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='disc')):
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_disc).minimize(loss_disc_op,
                                                                                    var_list=tf.get_collection(
                                                                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                        'disc'), global_step=global_step)

        sess.run(tf.global_variables_initializer())
        self._load(sess)

        s_gen_all_op = self._summaries_gen()
        s_disc_all_op = self._summaries_disc()
        s_gen_img_op = tf.summary.image('generated_image_training', tf.reshape(self.gen(), [batch_size, 28, 28, 1]), 1)

        loss_gen, loss_disc = Diff(), Diff()
        for step in range(global_step.eval(), final_step):
            curr_loss_disc, s_disc_all = 0, None
            for k in range(k_disc):
                input, _ = data.next_batch(batch_size)
                _, curr_loss_disc, s_disc_all = sess.run([optimizer_disc, loss_disc_op, s_disc_all_op],
                                                          feed_dict={self.disc.x: input,
                                                                           self.gen.z: self.sample_noise(batch_size),
                                                                           self.disc.is_training: True,
                                                                           self.gen.is_training: True})

            curr_loss_gen, s_gen_all = 0, None
            for k in range(k_gen):
                _, curr_loss_gen, s_gen_all = sess.run([optimizer_gen, loss_gen_op, s_gen_all_op],
                                                              feed_dict={self.gen.z: self.sample_noise(batch_size),
                                                                         self.gen.is_training: True,
                                                                         self.disc.is_training: True})

            loss_gen.update(curr_loss_gen)
            loss_disc.update(curr_loss_disc)
            print 'step-{}\td_loss_gen={:2.2f}%\td_loss_disc={:2.2f}%\tloss_gen={}\tloss_disc={}'.format(step,
                                                                                                         loss_gen.diff * 100,
                                                                                                         loss_disc.diff * 100,
                                                                                                         loss_gen.value,
                                                                                                         loss_disc.value)

            if step % ckpt_step == 0:
                tf.train.Saver().save(sess, self.ckpt_path + '/gan', global_step=step)

                writer.add_summary(s_gen_all, global_step=step)
                writer.add_summary(s_disc_all, global_step=step)

                s_gen_img = sess.run(s_gen_img_op, feed_dict={self.gen.z: self.sample_noise(batch_size), self.gen.is_training: False})
                writer.add_summary(s_gen_img, global_step=step)

    def generate(self, sess, batch_size):
        sess.run(tf.global_variables_initializer())
        self._load(sess)
        return sess.run(self.gen(), feed_dict={self.gen.z: self.sample_noise(batch_size), self.gen.is_training: False})

    def discriminate(self, sess, data):
        sess.run(tf.global_variables_initializer())
        self._load(sess)
        return sess.run(self.disc(), feed_dict={self.disc.x: data, self.disc.is_training: False})
